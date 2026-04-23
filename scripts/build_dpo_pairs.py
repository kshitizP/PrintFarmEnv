"""
build_dpo_pairs.py
==================
Constructs DPO preference pairs (chosen, rejected) from baseline + teacher
trajectories using heuristic labellers targeting the 9 failure modes and
operator-trust scenarios.

Inputs:
    data/baselines/clairvoyant/{task_id}/episodes.jsonl   (as "chosen" source)
    data/baselines/naive/{task_id}/episodes.jsonl          (as "rejected" source)
    data/teacher/{task_id}/episodes.jsonl                  (optional extra)

Outputs:
    data/dpo/preferences.jsonl
    data/dpo/stats.json

Row format (TRL DPOTrainer compatible):
    {
      "prompt": "<system>\\n<user obs JSON>",
      "chosen":   "<good action JSON>",
      "rejected": "<bad action JSON>",
      "meta": {"label_source": "...", "task_id": "...", "step": ...}
    }

Label sources (heuristics):
    fault_ignored         — naive did WAIT while a printer had an active fault
                            (clairvoyant dispatched diag/maintenance/unjam)
    spool_runout_ignored  — naive let spool <= 0 while clairvoyant swapped
    fatigue_ignored       — naive printed while fatigue >= 7 (clairvoyant maint'd)
    error_ignored         — naive didn't dispatch unjam_printer on ERROR state
    queue_stall           — naive WAITed with idle printer + pending job
    paired_action         — same state, clairvoyant chose action A, naive chose B ≠ A

Target: ≥ 2,000 pairs (per IMPLEMENTATION_PLAN §2)

Usage:
    python scripts/build_dpo_pairs.py
    python scripts/build_dpo_pairs.py --min-pairs 2000 --max-pairs 10000
    python scripts/build_dpo_pairs.py --tasks task_1 task_2 task_3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_sft_dataset import SFT_SYSTEM_PROMPT, _compact_obs, _compact_action


# ---------------------------------------------------------------------------
# Observation fingerprinting — map step to a stable key so naive + clairv
# trajectories can be aligned at equivalent states.
# ---------------------------------------------------------------------------

def _state_key(obs: dict) -> str:
    """Coarse fingerprint of the observation for cross-policy alignment.

    We hash (time_step, tuple of printer states + fatigue buckets + job ids)
    so that two policies hitting "equivalent" states get the same key.
    """
    printers = tuple(
        (p["printer_id"], p["state"], round(p.get("fatigue_level", 0.0)),
         p.get("current_job_id"))
        for p in obs.get("printers", [])
    )
    queue = tuple(
        (j["job_id"], j["state"], j.get("progress_steps"))
        for j in obs.get("active_queue", [])[:5]
    )
    return json.dumps({"t": obs.get("time_step"), "p": printers, "q": queue})


def _iter_episodes(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Heuristic classifiers — label the "rejected" reason
# ---------------------------------------------------------------------------

def _classify_rejection(obs: dict, rejected_action: dict) -> Optional[str]:
    """Why was this naive action bad? Returns label or None (= unclassified)."""
    act = rejected_action.get("action")

    # --- ERROR state ignored ---
    for p in obs.get("printers", []):
        if p["state"] == "ERROR" and act in (None, "WAIT"):
            return "error_ignored"

    # --- Fatigue ignored ---
    for p in obs.get("printers", []):
        if p.get("fatigue_level", 0) >= 7 and act in (None, "WAIT", "ASSIGN_JOB"):
            return "fatigue_ignored"

    # --- Spool runout ignored ---
    for p in obs.get("printers", []):
        if p["state"] in ("PRINTING", "PAUSED_RUNOUT") and p.get("spool_weight_g", 1000) < 10:
            if act in (None, "WAIT", "ASSIGN_JOB"):
                return "spool_runout_ignored"

    # --- Telemetry anomalies ignored ---
    t = obs.get("time_step", 0)
    for p in obs.get("printers", []):
        stale = (t - p.get("telemetry_ts", t)) > 5
        temp_bad = p.get("hotend_temp", 200) < 50 and p["state"] == "PRINTING"
        fan_bad  = p.get("fan_rpm", 3000) < 500 and p["state"] == "PRINTING"
        if (stale or temp_bad or fan_bad) and act in (None, "WAIT"):
            return "fault_ignored"

    # --- Queue stall (idle printer with pending job, but WAIT) ---
    idle_p = [p for p in obs.get("printers", []) if p["state"] == "IDLE"]
    pending_j = [j for j in obs.get("active_queue", []) if j["state"] == "PENDING"]
    if idle_p and pending_j and act == "WAIT":
        return "queue_stall"

    return None


# ---------------------------------------------------------------------------
# Pair builder
# ---------------------------------------------------------------------------

def _load_steps_by_state(root: Path, tasks: list[str]) -> dict[tuple[str, str], list[dict]]:
    """Return {(task_id, state_key) -> [step, ...]} index."""
    idx: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for task_id in tasks:
        path = root / task_id / "episodes.jsonl"
        for ep in _iter_episodes(path):
            for step in ep.get("steps", []):
                key = _state_key(step["observation"])
                idx[(task_id, key)].append(step)
    return idx


def main():
    ap = argparse.ArgumentParser(description="Build DPO preference pairs")
    ap.add_argument("--clairvoyant-root", default="data/baselines/clairvoyant")
    ap.add_argument("--naive-root", default="data/baselines/naive")
    ap.add_argument("--teacher-root", default="data/teacher")
    ap.add_argument("--output-root", default="data/dpo")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--min-pairs", type=int, default=2000)
    ap.add_argument("--max-pairs", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    clair_root = Path(args.clairvoyant_root).resolve()
    naive_root = Path(args.naive_root).resolve()

    if args.tasks:
        tasks = args.tasks
    else:
        tasks = sorted({p.name for p in clair_root.iterdir() if p.is_dir()} if clair_root.exists() else [])

    print(f"Tasks: {tasks}")
    print("Indexing clairvoyant (chosen) trajectories...")
    chosen_idx = _load_steps_by_state(clair_root, tasks)
    print(f"  {sum(len(v) for v in chosen_idx.values()):,} clairvoyant steps across "
          f"{len(chosen_idx):,} unique state-keys")

    print("Indexing naive (rejected) trajectories...")
    rejected_idx = _load_steps_by_state(naive_root, tasks)
    print(f"  {sum(len(v) for v in rejected_idx.values()):,} naive steps across "
          f"{len(rejected_idx):,} unique state-keys")

    pairs = []
    label_counts: dict[str, int] = defaultdict(int)

    # --- Aligned pairs: same (task, state_key), different actions ---
    print("\nBuilding aligned pairs...")
    aligned_keys = set(chosen_idx.keys()) & set(rejected_idx.keys())
    print(f"  {len(aligned_keys):,} overlapping state-keys")

    for key in aligned_keys:
        task_id, _ = key
        for chosen_step in chosen_idx[key]:
            rejected_step = rng.choice(rejected_idx[key])
            chosen_action   = chosen_step["action"]
            rejected_action = rejected_step["action"]
            if chosen_action == rejected_action:
                continue  # no preference signal
            label = _classify_rejection(rejected_step["observation"], rejected_action)
            label = label or "paired_action"
            pairs.append({
                "prompt": SFT_SYSTEM_PROMPT + "\n\n" + _compact_obs(chosen_step["observation"]),
                "chosen": _compact_action(chosen_action),
                "rejected": _compact_action(rejected_action),
                "meta": {
                    "label_source": label,
                    "task_id": task_id,
                    "step": chosen_step["step"],
                },
            })
            label_counts[label] += 1
            if len(pairs) >= args.max_pairs:
                break
        if len(pairs) >= args.max_pairs:
            break

    # --- Fallback pairs: heuristic-classified naive steps vs forced good action ---
    if len(pairs) < args.min_pairs:
        print(f"\nOnly {len(pairs)} aligned pairs — adding heuristic pairs...")
        for task_id in tasks:
            for ep in _iter_episodes(naive_root / task_id / "episodes.jsonl"):
                for step in ep.get("steps", []):
                    label = _classify_rejection(step["observation"], step["action"])
                    if label is None:
                        continue
                    # Forced "chosen" action based on label
                    chosen = _forced_chosen_for_label(label, step["observation"])
                    if chosen is None:
                        continue
                    rejected = {k: v for k, v in step["action"].items()
                                if v is not None and k != "metadata"}
                    if chosen == rejected:
                        continue
                    pairs.append({
                        "prompt": SFT_SYSTEM_PROMPT + "\n\n" + _compact_obs(step["observation"]),
                        "chosen": json.dumps(chosen, separators=(",", ":")),
                        "rejected": json.dumps(rejected, separators=(",", ":")),
                        "meta": {
                            "label_source": f"heuristic:{label}",
                            "task_id": task_id,
                            "step": step["step"],
                        },
                    })
                    label_counts[f"heuristic:{label}"] += 1
                    if len(pairs) >= args.max_pairs:
                        break
                if len(pairs) >= args.max_pairs:
                    break
            if len(pairs) >= args.max_pairs:
                break

    rng.shuffle(pairs)

    out_path = out_root / "preferences.jsonl"
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    stats = {
        "total_pairs": len(pairs),
        "by_label": dict(label_counts),
        "tasks": tasks,
        "min_pairs_target": args.min_pairs,
        "max_pairs_cap": args.max_pairs,
    }
    with (out_root / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTotal pairs: {len(pairs):,}  →  {out_path}")
    print(f"By label:    {dict(label_counts)}")
    if len(pairs) < args.min_pairs:
        print(f"WARNING: below target of {args.min_pairs}")


def _forced_chosen_for_label(label: str, obs: dict) -> Optional[dict]:
    """Produce a 'good' action for a heuristically labeled bad naive step."""
    printers = obs.get("printers", [])

    if label == "error_ignored":
        for p in printers:
            if p["state"] == "ERROR":
                op_id = _pick_operator(obs)
                if op_id:
                    return {"action": "DISPATCH_TICKET", "printer_id": p["printer_id"],
                            "operator_id": op_id, "ticket_type": "unjam_printer"}
        return None

    if label == "fatigue_ignored":
        for p in printers:
            if p.get("fatigue_level", 0) >= 7:
                return {"action": "REQUEST_MAINTENANCE",
                        "printer_id": p["printer_id"],
                        "maintenance_type": "basic"}
        return None

    if label == "spool_runout_ignored":
        for p in printers:
            if p.get("spool_weight_g", 1000) < 10 and p.get("current_material"):
                return {"action": "REQUEST_SPOOL_SWAP",
                        "printer_id": p["printer_id"],
                        "material": p["current_material"]}
        return None

    if label == "fault_ignored":
        t = obs.get("time_step", 0)
        for p in printers:
            stale = (t - p.get("telemetry_ts", t)) > 5
            temp_bad = p.get("hotend_temp", 200) < 50 and p["state"] == "PRINTING"
            fan_bad  = p.get("fan_rpm", 3000) < 500 and p["state"] == "PRINTING"
            if stale or temp_bad or fan_bad:
                return {"action": "RUN_DIAGNOSTIC", "printer_id": p["printer_id"]}
        return None

    if label == "queue_stall":
        idle = next((p for p in printers if p["state"] == "IDLE"), None)
        pending = next((j for j in obs.get("active_queue", []) if j["state"] == "PENDING"), None)
        if idle and pending:
            return {"action": "ASSIGN_JOB",
                    "printer_id": idle["printer_id"],
                    "job_id": pending["job_id"]}

    return None


def _pick_operator(obs: dict) -> Optional[str]:
    """Pick any on-shift operator with queue capacity."""
    for o in obs.get("operators", []):
        if o["is_on_shift"] and o["queue_size"] < o["queue_capacity"]:
            return o["operator_id"]
    # fallback: any operator
    if obs.get("operators"):
        return obs["operators"][0]["operator_id"]
    return None


if __name__ == "__main__":
    main()
