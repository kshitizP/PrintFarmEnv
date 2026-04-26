"""
Build forced-balance GRPO prompt dataset.

Generates env-based prompts then classifies each by inferred optimal action.
Keeps 112 per action (1008 total), splits 80/20 into train/val.
Also writes a 90-prompt stratified eval set (10 per action from val).

Run: python3 -m submission.training.build_grpo_dataset [--n_per_action N]
Writes:
  submission/data/grpo_train.jsonl   (~806 prompts)
  submission/data/grpo_val.jsonl     (~202 prompts)
  submission/data/grpo_eval.jsonl    (90 prompts, 10 per action)

This is a one-time offline step (~10-20 min). Results are committed to the
repo so training jobs load them directly without re-generating.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from submission.env.decision_point import DecisionPointEnv, SIGNAL_TYPES, signal_present
from submission.env.models import FarmObservation, FarmActionEnum
from submission.shared.obs_formatter import format_observation_as_text
from submission.shared.prompt import SYSTEM_PROMPT
from submission.training.rollout import _compress_obs_json

DATA_DIR = ROOT / "submission" / "data"
N_PER_ACTION = 112
# task_0_3 starts P1 with fatigue_level=7, making REQUEST_MAINTENANCE reachable
TASKS = ["task_0_3", "task_1", "task_2", "task_3", "task_4"]
SCORABLE_ACTIONS = [
    "ASSIGN_JOB", "CANCEL_JOB", "DISPATCH_TICKET", "PAUSE_JOB",
    "REQUEST_MAINTENANCE", "REQUEST_SPOOL_SWAP", "RESUME_JOB",
    "RUN_DIAGNOSTIC", "WAIT",
]


def infer_action(obs: FarmObservation) -> str:
    """Classify observation → most likely optimal action.

    Priority order matches expected difficulty: rare/text signals first,
    structured fallbacks last.
    """
    printers = obs.printers
    jobs = obs.active_queue

    # 1. Customer message: explicit cancel → CANCEL_JOB
    cancel_words = ["cancel", "abort", "stop", "nevermind", "pull", "halt",
                    "withdraw", "don't need", "changed mind", "shelved", "resubmit",
                    "on hold", "wrong design", "overordered", "cheaper supplier",
                    "pulled the contract"]
    if obs.customer_messages:
        for msg in obs.customer_messages:
            text = msg.lower() if isinstance(msg, str) else msg.get("text", "").lower()
            if any(w in text for w in cancel_words):
                return "CANCEL_JOB"

    # 2. Anomaly flag + printer PRINTING → DISPATCH_TICKET (needs human)
    #    (novel compound faults need escalation, not just pause)
    if obs.anomaly_flags:
        for p in printers:
            if p.state.value == "PRINTING":
                return "DISPATCH_TICKET"

    # 3. Operator note with SEVERE mechanical symptom → PAUSE_JOB
    severe_words = ["grinding", "burning", "smoke", "spark", "bang", "shak",
                    "clunk", "jam", "crash", "wobbl", "clog", "crac", "overheat",
                    "stall", "miss", "layer shift", "seizure"]
    if obs.operator_notes:
        for note in obs.operator_notes:
            note_l = note.lower()
            if any(w in note_l for w in severe_words):
                for p in printers:
                    if p.state.value == "PRINTING":
                        return "PAUSE_JOB"

    # 4. Operator note with AMBIGUOUS sensor reading → RUN_DIAGNOSTIC
    ambiguous_words = ["temp", "thermistor", "webcam", "hash", "telemetry",
                       "fan_rpm", "rpm", "progress stuck", "stale", "reliability",
                       "freeze", "seems slower", "hiccup", "fluctuat", "drift",
                       "oscillat", "irregular", "unusual"]
    if obs.operator_notes:
        for note in obs.operator_notes:
            note_l = note.lower()
            if any(w in note_l for w in ambiguous_words):
                for p in printers:
                    if p.state.value == "PRINTING":
                        return "RUN_DIAGNOSTIC"
        # Any note about printer issues while printing → diagnostic
        for p in printers:
            if p.state.value == "PRINTING":
                return "RUN_DIAGNOSTIC"

    # 5. Suspicious sensor values on a printer → RUN_DIAGNOSTIC
    for p in printers:
        if p.state.value == "PRINTING":
            if p.hotend_temp == 0 or p.hotend_temp > 400:
                return "RUN_DIAGNOSTIC"
            if p.fan_rpm == 0:
                return "RUN_DIAGNOSTIC"

    # 6. PAUSED_RUNOUT → REQUEST_SPOOL_SWAP
    for p in printers:
        if p.state.value == "PAUSED_RUNOUT":
            return "REQUEST_SPOOL_SWAP"

    # 7. Low spool while PRINTING (< 180g) → REQUEST_SPOOL_SWAP
    for p in printers:
        if p.state.value == "PRINTING" and p.spool_weight_g < 180:
            return "REQUEST_SPOOL_SWAP"

    # 8a. Agent-paused printer (PAUSED state, spool OK) → RESUME_JOB
    for p in printers:
        if p.state.value == "PAUSED" and p.spool_weight_g > 200:
            return "RESUME_JOB"

    # 8b. Post-runout: printer IDLE with a PAUSED job (spool swap completed) → RESUME_JOB
    paused_job_ids = {j.job_id for j in jobs if j.state.value == "PAUSED"}
    for p in printers:
        if (p.state.value == "IDLE" and p.current_job_id is not None
                and p.current_job_id in paused_job_ids):
            return "RESUME_JOB"

    # 9. High fatigue IDLE → REQUEST_MAINTENANCE (task_0_3 starts with fatigue=7)
    for p in printers:
        if p.fatigue_level >= 7.0 and p.state.value == "IDLE" and p.outstanding_ticket_id is None:
            return "REQUEST_MAINTENANCE"

    # 10. IDLE + matching PENDING job → ASSIGN_JOB
    idle = [p for p in printers if p.state.value == "IDLE" and p.current_job_id is None
            and p.fatigue_level < 7.5 and p.outstanding_ticket_id is None]
    pending = [j for j in jobs if j.state.value == "PENDING"]
    for j in pending:
        for p in idle:
            if p.current_material == j.material_required and p.spool_weight_g >= j.weight_required_g:
                return "ASSIGN_JOB"

    # 11. Customer message (non-cancel) → context-dependent
    if obs.customer_messages:
        # Rush order → ASSIGN_JOB if printer available
        if idle and pending:
            return "ASSIGN_JOB"

    # 12. Default: nothing urgent → WAIT
    return "WAIT"


def generate_balanced_prompts(n_per_action: int = N_PER_ACTION,
                               seed: int = 42,
                               max_attempts: int = 80_000) -> list:
    """Generate env-based prompts classified by inferred action.

    Returns list of prompt dicts compatible with evaluate_completion().
    """
    rng = random.Random(seed)
    buckets: dict[str, list] = defaultdict(list)
    attempts = 0
    signal_idx = 0

    print(f"Generating {n_per_action} prompts per action ({n_per_action * 9} total)...")
    print("Progress: ", end="", flush=True)

    while attempts < max_attempts:
        # Stop when all buckets are full
        if all(len(buckets[a]) >= n_per_action for a in SCORABLE_ACTIONS):
            break

        attempts += 1
        task_id = TASKS[attempts % len(TASKS)]
        ep_seed = rng.randint(0, 2**31)
        target_signal = SIGNAL_TYPES[signal_idx % len(SIGNAL_TYPES)]
        signal_idx += 1

        try:
            dp_env = DecisionPointEnv()
            serialized, obs = dp_env.reset(
                seed=ep_seed, task_id=task_id, target_signal=target_signal,
            )
        except Exception:
            continue

        if not signal_present(obs, target_signal):
            continue

        inferred = infer_action(obs)

        # Skip if this bucket is already full
        if len(buckets[inferred]) >= n_per_action:
            continue

        compressed = _compress_obs_json(serialized)
        obs_text = format_observation_as_text(compressed)

        # Build ground_truth_tags for reward computation
        gt_tags = {
            "has_notes": bool(obs.operator_notes),
            "has_messages": bool(obs.customer_messages),
            "has_anomaly": bool(obs.anomaly_flags),
            "inferred_action": inferred,
        }

        prompt_info = {
            "prompt": serialized,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current state:\n{obs_text}"},
            ],
            "task_id": task_id,
            "seed": ep_seed,
            "target_signal": target_signal,
            "inferred_action": inferred,
            "ground_truth_tags": gt_tags,
            "observation_text": obs_text,
            "decision_obs": obs.model_dump() if hasattr(obs, "model_dump") else obs,
        }
        buckets[inferred].append(prompt_info)

        total_collected = sum(len(v) for v in buckets.values())
        if total_collected % 100 == 0:
            print(f"{total_collected}", end=" ", flush=True)

    print()
    print(f"\nAttempts: {attempts}")
    for action in SCORABLE_ACTIONS:
        print(f"  {action}: {len(buckets[action])}")

    # Flatten and verify
    all_prompts = []
    for action in SCORABLE_ACTIONS:
        all_prompts.extend(buckets[action][:n_per_action])

    return all_prompts


def _serialize_prompt(p: dict) -> dict:
    """Make prompt JSON-serializable (convert Pydantic objects to dicts)."""
    out = {k: v for k, v in p.items()}
    if "decision_obs" in out and hasattr(out["decision_obs"], "model_dump"):
        out["decision_obs"] = out["decision_obs"].model_dump()
    return out


def build_eval_set(val_prompts: list, n_per_action: int = 10) -> list:
    """Create stratified eval set: n_per_action per action from val."""
    by_action: dict[str, list] = defaultdict(list)
    for p in val_prompts:
        by_action[p["inferred_action"]].append(p)

    eval_set = []
    for action in SCORABLE_ACTIONS:
        available = by_action[action]
        if len(available) < n_per_action:
            print(f"  Warning: only {len(available)} val prompts for {action}, want {n_per_action}")
        eval_set.extend(available[:n_per_action])
    return eval_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_action", type=int, default=N_PER_ACTION)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_prompts = generate_balanced_prompts(args.n_per_action, seed=args.seed)

    # Shuffle before split
    random.shuffle(all_prompts)

    n = len(all_prompts)
    split = int(n * 0.8)
    train_prompts = all_prompts[:split]
    val_prompts = all_prompts[split:]

    print(f"\nSplit: {len(train_prompts)} train / {len(val_prompts)} val")

    # 90-prompt stratified eval set from val
    eval_prompts = build_eval_set(val_prompts, n_per_action=10)
    print(f"Eval set: {len(eval_prompts)} prompts ({len(eval_prompts) // 9} per action)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(_serialize_prompt(r)) + "\n")
        print(f"Wrote {len(rows)} prompts → {path}")

    write_jsonl(DATA_DIR / "grpo_train.jsonl", train_prompts)
    write_jsonl(DATA_DIR / "grpo_val.jsonl", val_prompts)
    write_jsonl(DATA_DIR / "grpo_eval.jsonl", eval_prompts)


if __name__ == "__main__":
    main()
