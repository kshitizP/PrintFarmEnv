"""
build_sft_dataset.py
====================
Merges clairvoyant + teacher trajectories into a single ChatML SFT dataset,
filtered to keep only profitable episodes (above per-task naive floor).

Inputs:
    data/baselines/clairvoyant/{task_id}/episodes.jsonl
    data/teacher/{task_id}/episodes.jsonl

Outputs:
    data/sft/train.jsonl    # ChatML: {"messages": [system, user, assistant]}
    data/sft/eval.jsonl     # 5% holdout
    data/sft/stats.json     # counts per source, per task, filter reasons

Row format:
    {
      "messages": [
        {"role": "system", "content": "<SFT_SYSTEM_PROMPT>"},
        {"role": "user", "content": "<compact observation JSON>"},
        {"role": "assistant", "content": "<action JSON>"}
      ],
      "meta": {"source": "clairvoyant|teacher", "task_id": "task_1", "episode": 3, "step": 17}
    }

Usage:
    python scripts/build_sft_dataset.py
    python scripts/build_sft_dataset.py --min-profit-percentile 50 --eval-fraction 0.05
    python scripts/build_sft_dataset.py --only-sources clairvoyant
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

from printfarm_env.tasks import TaskGrader

SFT_SYSTEM_PROMPT = """You are the Dispatcher for a 4-printer 3D print farm. Respond with a single JSON action object (no prose, no markdown).

Actions:
- {"action":"ASSIGN_JOB","printer_id":<int>,"job_id":"<str>"}
- {"action":"CANCEL_JOB","job_id":"<str>"}
- {"action":"PAUSE_JOB","printer_id":<int>}
- {"action":"RESUME_JOB","printer_id":<int>}
- {"action":"RUN_DIAGNOSTIC","printer_id":<int>}
- {"action":"DISPATCH_TICKET","printer_id":<int>,"operator_id":"<str>","ticket_type":"<str>"}
- {"action":"REQUEST_SPOOL_SWAP","printer_id":<int>,"material":"<str>"}
- {"action":"REQUEST_MAINTENANCE","printer_id":<int>,"maintenance_type":"basic|full_rebuild"}
- {"action":"OVERRIDE_OPERATOR","ticket_id":"<str>","reason":"<str>"}
- {"action":"WAIT"}"""


def _compact_obs(obs: dict) -> str:
    """Same compaction logic as generate_teacher_rollouts for consistency."""
    compact = {
        "time_step": obs.get("time_step"),
        "max_steps": obs.get("max_steps"),
        "net_profit_usd": round(obs.get("net_profit_usd", 0.0), 2),
        "inventory": {k: round(v, 1) for k, v in obs.get("inventory", {}).items()},
        "active_queue": [
            {
                "job_id": j["job_id"],
                "state": j["state"],
                "material": j["material_required"],
                "weight_g": j["weight_required_g"],
                "print_time_steps": j["print_time_steps"],
                "progress_steps": j["progress_steps"],
                "priority": j["priority"],
                "deadline_steps": j["deadline_steps"],
                "price_usd": j["price_usd"],
            }
            for j in obs.get("active_queue", [])[:20]
        ],
        "printers": [
            {
                "printer_id": p["printer_id"],
                "profile_id": p.get("profile_id"),
                "state": p["state"],
                "current_material": p.get("current_material"),
                "current_job_id": p.get("current_job_id"),
                "spool_weight_g": round(p.get("spool_weight_g", 0.0), 1),
                "reliability": round(p.get("reliability", 0.0), 3),
                "maintenance_due_in": p.get("maintenance_due_in"),
                "fatigue_level": round(p.get("fatigue_level", 0.0), 2),
                "warmup_remaining": p.get("warmup_remaining"),
                "offline_remaining": p.get("offline_remaining"),
                "hotend_temp": round(p.get("hotend_temp", 0.0), 1),
                "fan_rpm": p.get("fan_rpm"),
                "telemetry_ts": p.get("telemetry_ts"),
                "outstanding_ticket_id": p.get("outstanding_ticket_id"),
            }
            for p in obs.get("printers", [])
        ],
        "operators": [
            {
                "operator_id": o["operator_id"],
                "skill_level": o["skill_level"],
                "is_on_shift": o["is_on_shift"],
                "queue_size": o["queue_size"],
                "queue_capacity": o["queue_capacity"],
                "current_fatigue": round(o.get("current_fatigue", 0.0), 2),
                "busy_until": o.get("busy_until"),
            }
            for o in obs.get("operators", [])
        ],
    }
    return json.dumps(compact, separators=(",", ":"))


def _compact_action(action: dict) -> str:
    """Strip None-valued fields from action dict."""
    return json.dumps(
        {k: v for k, v in action.items() if v is not None and k != "metadata"},
        separators=(",", ":"),
    )


def _iter_episodes(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _episode_profitable(ep: dict, floor: float) -> bool:
    """Keep episodes whose final profit beats the naive floor."""
    return ep.get("final_profit_usd", 0.0) > floor


def main():
    ap = argparse.ArgumentParser(description="Build SFT ChatML dataset")
    ap.add_argument("--clairvoyant-root", default="data/baselines/clairvoyant")
    ap.add_argument("--teacher-root", default="data/teacher")
    ap.add_argument("--output-root", default="data/sft")
    ap.add_argument("--only-sources", nargs="+", default=["clairvoyant", "teacher"])
    ap.add_argument("--eval-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Restrict to these tasks (default: all)")
    ap.add_argument("--drop-wait-fraction", type=float, default=0.8,
                    help="Drop this fraction of WAIT-action steps to reduce label skew")
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Discover tasks
    clair_root = Path(args.clairvoyant_root).resolve()
    teach_root = Path(args.teacher_root).resolve()
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = sorted({p.name for p in clair_root.iterdir() if p.is_dir()} |
                       {p.name for p in teach_root.iterdir() if p.is_dir()}) \
                if (clair_root.exists() or teach_root.exists()) else []

    stats = {
        "per_task": {},
        "per_source": {"clairvoyant": 0, "teacher": 0},
        "filtered": {"low_profit": 0, "parse_fail": 0, "wait_dropped": 0},
        "total_train": 0,
        "total_eval": 0,
    }

    train_path = out_root / "train.jsonl"
    eval_path  = out_root / "eval.jsonl"

    all_rows = []

    for task_id in tasks:
        try:
            g = TaskGrader(task_id)
            floor = g._NAIVE_FLOOR[task_id]
        except Exception:
            floor = float("-inf")
        stats["per_task"][task_id] = {"clairvoyant": 0, "teacher": 0, "floor": floor}

        for source, root in [("clairvoyant", clair_root), ("teacher", teach_root)]:
            if source not in args.only_sources:
                continue
            ep_path = root / task_id / "episodes.jsonl"
            for ep in _iter_episodes(ep_path):
                if not _episode_profitable(ep, floor):
                    stats["filtered"]["low_profit"] += 1
                    continue
                for step in ep.get("steps", []):
                    if source == "teacher" and not step.get("parse_ok", True):
                        stats["filtered"]["parse_fail"] += 1
                        continue
                    action = step["action"]
                    # Drop some WAIT steps to reduce class imbalance
                    if (action.get("action") == "WAIT"
                            and rng.random() < args.drop_wait_fraction):
                        stats["filtered"]["wait_dropped"] += 1
                        continue

                    user_content = _compact_obs(step["observation"])
                    asst_content = _compact_action(action)
                    row = {
                        "messages": [
                            {"role": "system",    "content": SFT_SYSTEM_PROMPT},
                            {"role": "user",      "content": user_content},
                            {"role": "assistant", "content": asst_content},
                        ],
                        "meta": {
                            "source": source,
                            "task_id": task_id,
                            "episode": ep.get("episode"),
                            "step": step["step"],
                            "episode_profit": ep.get("final_profit_usd"),
                        },
                    }
                    all_rows.append(row)
                    stats["per_source"][source] += 1
                    stats["per_task"][task_id][source] += 1

    # Shuffle and split
    rng.shuffle(all_rows)
    n_eval = int(len(all_rows) * args.eval_fraction)
    eval_rows  = all_rows[:n_eval]
    train_rows = all_rows[n_eval:]

    with train_path.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with eval_path.open("w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")

    stats["total_train"] = len(train_rows)
    stats["total_eval"]  = len(eval_rows)

    with (out_root / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"Train:  {len(train_rows):>7,}  →  {train_path}")
    print(f"Eval:   {len(eval_rows):>7,}  →  {eval_path}")
    print(f"By source:   {stats['per_source']}")
    print(f"Filtered:    {stats['filtered']}")
    print(f"Stats:  {out_root / 'stats.json'}")


if __name__ == "__main__":
    main()
