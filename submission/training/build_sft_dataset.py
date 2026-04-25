"""
build_sft_dataset.py — Generate the SFT warm-start dataset.

Produces ~120 (prompt, oracle-action) pairs stratified per the senior's
prescription:

  40% True Positives  — observation contains a real anomaly + maybe a message.
                        Oracle: investigate the anomaly (Safety > Customer).
  30% True Negatives  — observation contains a customer message, no anomaly.
                        Oracle: respond to the message correctly.
  30% False Positives — observation contains a "resolved alarm" decoy
                        (looks like a fault but already resolved).
                        Oracle: handle messages/queue, ignore the decoy.

Output format: HuggingFace conversational JSONL
    {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": <observation text>},
        {"role": "assistant", "content": "<action>{...}</action>"}
    ]}

Usage:
    python -m submission.training.build_sft_dataset \
        --out submission/data/sft_warm.jsonl \
        --n 120 --seed 42
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make submission importable
_here = Path(__file__).resolve().parent
for _candidate in [_here.parent, _here.parent.parent]:
    if (_candidate / "submission" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from submission.env.decision_point import (
    DecisionPointEnv, K_HORIZON, SIGNAL_TYPES, signal_present,
)
from submission.shared.serialize import serialize_obs
from submission.shared.obs_formatter import format_observation_as_text
from submission.shared.prompt import SYSTEM_PROMPT
from submission.training.rollout import _compress_obs_json


# ── Oracle: pick the correct action given the new hierarchy ───────────────────

def oracle_action(decision_obs, ground_truth_tags: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the correct action dict per the Safety > Throughput > Service hierarchy.

    Priority order:
      1. Real anomaly visible (anomaly_flags or predictive note) → INVESTIGATE that printer
      2. Resolved-alarm decoy → ignore it; handle messages or wait
      3. Customer message → respond correctly per category
      4. No signal → WAIT
    """
    note_tags = ground_truth_tags.get("operator_notes", [])
    msg_tags = ground_truth_tags.get("customer_messages", [])
    anomaly_tags = ground_truth_tags.get("anomaly_flags", [])

    # 1. Real novel-fault anomaly → investigate
    for tag in anomaly_tags:
        if tag.get("correct_action") == "investigate":
            pid = tag.get("printer_id")
            if pid is not None:
                return {"action_type": "RUN_DIAGNOSTIC", "printer_id": int(pid)}

    # 2. Real predictive operator note → investigate
    for tag in note_tags:
        if (tag.get("category") == "predictive"
                and tag.get("ground_truth_action") == "investigate"):
            pid = tag.get("printer_id")
            if pid is not None:
                return {"action_type": "RUN_DIAGNOSTIC", "printer_id": int(pid)}

    # 3. Customer messages — handle by category
    if msg_tags:
        tag = msg_tags[0]
        gt = tag.get("ground_truth_action")
        job_id = tag.get("job_id")
        if gt == "accept_rush":
            return {"action_type": "ASSIGN_JOB", "job_id": job_id, "printer_id": _pick_idle_printer(decision_obs)}
        if gt == "standard_queue":
            return {"action_type": "WAIT"}
        if gt == "decline":
            return {"action_type": "CANCEL_JOB", "job_id": job_id}
        if gt == "accept_substitute":
            return {"action_type": "REQUEST_SPOOL_SWAP",
                    "printer_id": _pick_idle_printer(decision_obs) or 1}
        return {"action_type": "WAIT"}

    # 4. Substitution operator notes
    for tag in note_tags:
        if tag.get("ground_truth_action") == "accept_substitute":
            return {"action_type": "REQUEST_SPOOL_SWAP",
                    "printer_id": _pick_idle_printer(decision_obs) or 1}

    # 5. No urgent signal → WAIT
    return {"action_type": "WAIT"}


def _pick_idle_printer(obs) -> Optional[int]:
    """Find an IDLE printer to assign to, fallback to 1."""
    if obs is None:
        return 1
    printers = obs.printers if hasattr(obs, "printers") else obs.get("printers", [])
    for p in printers:
        state_val = getattr(getattr(p, "state", None), "value", None)
        if state_val is None and isinstance(p, dict):
            state_val = p.get("state")
        if state_val == "IDLE":
            pid = getattr(p, "printer_id", None) or (p.get("printer_id") if isinstance(p, dict) else None)
            if pid is not None:
                return int(pid)
    return 1


# ── Scenario classification ────────────────────────────────────────────────────

def classify_scenario(ground_truth_tags: Dict[str, Any]) -> str:
    """Return scenario class for stratification.

    Returns one of:
      "true_positive"  — has real fault (any printer has correct_action=investigate)
      "false_positive" — has resolved-alarm decoy (is_resolved flag) + no real fault
      "true_negative"  — has customer message, no fault, no decoy
      "other"          — substitution-only / structured-only / nothing actionable
    """
    note_tags = ground_truth_tags.get("operator_notes", [])
    msg_tags = ground_truth_tags.get("customer_messages", [])
    anomaly_tags = ground_truth_tags.get("anomaly_flags", [])

    has_real_fault = any(
        t.get("correct_action") == "investigate"
        and not t.get("is_resolved", False)
        for t in anomaly_tags
    )
    has_real_fault = has_real_fault or any(
        t.get("category") == "predictive" and t.get("ground_truth_action") == "investigate"
        for t in note_tags
    )
    has_decoy = any(t.get("is_resolved", False) for t in anomaly_tags)
    has_message = bool(msg_tags)

    if has_real_fault:
        return "true_positive"
    if has_decoy:
        return "false_positive"
    if has_message:
        return "true_negative"
    return "other"


# ── Generation loop ────────────────────────────────────────────────────────────

def generate_examples(
    n_target: int,
    target_mix: Dict[str, float],
    tasks: List[str],
    seed: int = 42,
    max_attempts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate examples until we hit the target count + class mix."""
    rng = random.Random(seed)
    target_counts = {k: int(round(v * n_target)) for k, v in target_mix.items()}
    # any rounding leftover goes to true_positive
    total = sum(target_counts.values())
    if total < n_target:
        target_counts["true_positive"] += n_target - total

    counts = Counter()
    examples: List[Dict[str, Any]] = []
    if max_attempts is None:
        max_attempts = n_target * 20

    sig_idx = 0
    for attempt in range(max_attempts):
        if sum(counts[k] for k in target_counts) >= n_target:
            break

        # Bias the env toward producing each scenario type
        target_signal = SIGNAL_TYPES[sig_idx % len(SIGNAL_TYPES)]
        sig_idx += 1
        task_id = tasks[attempt % len(tasks)]
        ep_seed = rng.randint(0, 2**31)

        env = DecisionPointEnv(k_horizon=K_HORIZON)
        try:
            serialized, obs = env.reset(
                seed=ep_seed, task_id=task_id, target_signal=target_signal,
            )
        except Exception:
            continue

        if not signal_present(obs, target_signal):
            continue

        gt_tags = env.get_decision_tags()
        scenario = classify_scenario(gt_tags)

        # Skip if we don't need this class anymore
        if scenario == "other":
            continue
        if counts[scenario] >= target_counts.get(scenario, 0):
            continue

        # Get oracle action
        action_dict = oracle_action(obs, gt_tags)
        if action_dict is None:
            continue

        # Format observation text (same path the GRPO trainer uses)
        compressed = _compress_obs_json(serialized)
        obs_text = format_observation_as_text(compressed)
        action_str = "<action>" + json.dumps(action_dict, separators=(",", ":")) + "</action>"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current state:\n{obs_text}"},
                {"role": "assistant", "content": action_str},
            ],
            "scenario": scenario,
            "task_id": task_id,
            "seed": ep_seed,
            "target_signal": target_signal,
            "oracle_action_type": action_dict["action_type"],
        })
        counts[scenario] += 1

    return examples


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build SFT warm-start dataset")
    parser.add_argument("--out", default="submission/data/sft_warm.jsonl")
    parser.add_argument("--n", type=int, default=120, help="Target dataset size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp_pct", type=float, default=0.40)
    parser.add_argument("--tn_pct", type=float, default=0.30)
    parser.add_argument("--fp_pct", type=float, default=0.30)
    parser.add_argument("--tasks", nargs="+",
                        default=["task_1", "task_2", "task_3", "task_4", "task_5"])
    args = parser.parse_args()

    target_mix = {
        "true_positive": args.tp_pct,
        "true_negative": args.tn_pct,
        "false_positive": args.fp_pct,
    }

    print(f"Target: {args.n} examples, mix = {target_mix}")
    examples = generate_examples(
        n_target=args.n, target_mix=target_mix,
        tasks=args.tasks, seed=args.seed,
    )

    # Stats
    counts = Counter(e["scenario"] for e in examples)
    action_counts = Counter(e["oracle_action_type"] for e in examples)
    signal_counts = Counter(e["target_signal"] for e in examples)

    print(f"\nGenerated {len(examples)} examples:")
    for k, v in counts.most_common():
        pct = v / max(len(examples), 1) * 100
        print(f"  {k:18s}: {v:4d}  ({pct:.0f}%)")
    print(f"\nOracle action distribution:")
    for k, v in action_counts.most_common():
        print(f"  {k:25s}: {v}")
    print(f"\nSignal type coverage:")
    for k, v in signal_counts.most_common():
        print(f"  {k:15s}: {v}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"\nWrote {out_path}")

    # Show first example for sanity
    if examples:
        print(f"\n--- first example ({examples[0]['scenario']}) ---")
        print(f"USER:\n{examples[0]['messages'][1]['content'][:400]}...")
        print(f"\nASSISTANT:\n{examples[0]['messages'][2]['content']}")


if __name__ == "__main__":
    main()
