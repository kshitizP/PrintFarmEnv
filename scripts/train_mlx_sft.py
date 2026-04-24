"""
train_mlx_sft.py — MLX-native SFT training for Apple Silicon
=============================================================

Converts clairvoyant rollouts to chat-format JSONL, then runs
mlx-lm LoRA fine-tuning. Much faster than PyTorch GRPO on MPS.

Usage:
    # Gemma 3 4B
    python scripts/train_mlx_sft.py --model google/gemma-3-4b-it \
        --data dagger_gemma3/round_1/data/grpo_steps.jsonl \
        --output mlx_gemma3 --iters 500

    # Gemma 4 E4B
    python scripts/train_mlx_sft.py --model google/gemma-4-E4B-it \
        --data dagger_gemma3/round_1/data/grpo_steps.jsonl \
        --output mlx_gemma4 --iters 500

    # Quick smoke test
    python scripts/train_mlx_sft.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="MLX SFT training")
    p.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    p.add_argument("--data", type=str, default="dagger_gemma3/round_1/data/grpo_steps.jsonl",
                   help="Path to grpo_steps.jsonl rollout file")
    p.add_argument("--output", type=str, default="mlx_sft_output")
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--num-layers", type=int, default=16,
                   help="Number of layers to LoRA fine-tune (-1 for all)")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Quick smoke test (10 iters)")
    return p.parse_args()


def _compress_observation(user_content: str) -> str:
    """
    Compress the observation JSON to reduce token count.

    Removes:
    - Completed jobs from active_queue (not actionable)
    - Zero/default/null fields from printers
    - oversight_log trimmed to last 3 entries
    - ticket_events, reward_breakdown, total_labor_billed (noise)
    - Redundant printer fields at defaults
    """
    prefix = "Current State:\n"
    if not user_content.startswith(prefix):
        return user_content

    try:
        obs = json.loads(user_content[len(prefix):])
    except (json.JSONDecodeError, ValueError):
        return user_content

    # 1. Remove completed jobs from active_queue
    if "active_queue" in obs:
        obs["active_queue"] = [
            j for j in obs["active_queue"]
            if j.get("state") not in ("COMPLETED", "CANCELLED")
        ]
        # Strip zero/default fields from remaining jobs
        for j in obs["active_queue"]:
            for k in list(j.keys()):
                if j[k] in (0, 0.0, False, None, ""):
                    del j[k]

    # 2. Compress printer entries — drop default/zero fields
    printer_defaults = {
        "warmup_remaining": 0, "offline_remaining": 0,
        "bed_drift_counter": 0.0, "reliability_penalty_active": False,
        "outstanding_ticket_id": None, "current_job_id": None,
        "revealed_this_step": False,
    }
    if "printers" in obs:
        for p in obs["printers"]:
            for k, default_val in printer_defaults.items():
                if p.get(k) == default_val:
                    p.pop(k, None)
            # Remove null material
            if p.get("current_material") is None:
                p.pop("current_material", None)

    # 3. Trim oversight_log to last 3 entries
    if "oversight_log" in obs:
        obs["oversight_log"] = obs["oversight_log"][-3:]

    # 4. Remove low-value top-level fields
    for drop_key in ("ticket_events", "reward_breakdown", "total_labor_billed"):
        obs.pop(drop_key, None)

    # 5. Compress operator entries — drop defaults
    if "operators" in obs:
        for op in obs["operators"]:
            if op.get("current_ticket_id") is None:
                op.pop("current_ticket_id", None)
            if not op.get("pattern_recommendations"):
                op.pop("pattern_recommendations", None)
            if op.get("printer_visit_counts") == {}:
                op.pop("printer_visit_counts", None)

    return prefix + json.dumps(obs, separators=(",", ":"))


def _is_psychic_action(rec: dict) -> bool:
    """
    Return True if the clairvoyant used hidden env state (active_faults)
    that has NO observable signal in the observation.

    These are RUN_DIAGNOSTIC or DISPATCH_TICKET/diagnostic_physical on
    printers that look perfectly healthy — IDLE, webcam OK, no bed drift,
    no reliability penalty.  The model can never learn to replicate these.
    """
    try:
        action = json.loads(rec["completion"])
    except (json.JSONDecodeError, KeyError):
        return False

    act = action.get("action")
    if act not in ("RUN_DIAGNOSTIC", "DISPATCH_TICKET"):
        return False
    # Only diagnostic_physical dispatches are psychic; unjam/maintenance are observable
    if act == "DISPATCH_TICKET" and action.get("ticket_type") != "diagnostic_physical":
        return False

    # Parse observation to inspect the target printer
    user_msg = rec["prompt_messages"][-1]["content"]
    try:
        start = user_msg.index("{")
        obs = json.loads(user_msg[start:])
    except (ValueError, json.JSONDecodeError):
        return False

    pid = action.get("printer_id")
    target = None
    for p in obs.get("printers", []):
        if p.get("printer_id") == pid:
            target = p
            break
    if target is None:
        return False

    # Check if there's any observable signal justifying the action
    has_signal = (
        target.get("bed_drift_counter", 0) > 0.001
        or target.get("reliability_penalty_active", False)
        or target.get("state") in ("ERROR", "FAULTED")
        or target.get("webcam_hash", "") != f'cam_{pid}_ok'
    )
    return not has_signal


def _filter_bad_examples(records: list) -> list:
    """
    Remove genuinely misleading training examples while preserving
    the true expert action distribution.

    Removes:
    1. Failed spool swaps (reward < -0.15) — clairvoyant retries when no
       operator is available, costing -$0.20 per attempt.
    2. Psychic actions — RUN_DIAGNOSTIC / DISPATCH_TICKET diagnostic_physical
       on healthy-looking printers where the clairvoyant used hidden
       active_faults.  The model cannot learn these from observations.
    3. Entire low-reward episodes (ep_reward < 0.3) — confusing trajectories
       where clairvoyant cascaded into repeated failures.
    """
    import collections

    # Build episode reward map
    ep_rewards: dict[tuple, float] = {}
    for r in records:
        key = (r.get("task_id"), r.get("episode"))
        ep_rewards[key] = r.get("episode_reward", 1.0)
    low_eps = {k for k, v in ep_rewards.items() if v < 0.3}

    cleaned = []
    removed_spool = 0
    removed_psychic = 0
    removed_low_ep = 0
    for r in records:
        ep_key = (r.get("task_id"), r.get("episode"))
        if ep_key in low_eps:
            removed_low_ep += 1
            continue
        try:
            action = json.loads(r["completion"])
            if action.get("action") == "REQUEST_SPOOL_SWAP" and r["reward"] < -0.15:
                removed_spool += 1
                continue
        except (json.JSONDecodeError, KeyError):
            pass
        if _is_psychic_action(r):
            removed_psychic += 1
            continue
        cleaned.append(r)

    # Report
    action_counts = collections.Counter()
    for r in cleaned:
        try:
            action_counts[json.loads(r["completion"]).get("action", "?")] += 1
        except (json.JSONDecodeError, KeyError):
            action_counts["PARSE_FAIL"] += 1
    print(f"  Data filtering: {len(records)} → {len(cleaned)}")
    print(f"    Removed: {removed_spool} failed spool swaps, "
          f"{removed_psychic} psychic actions, "
          f"{removed_low_ep} low-episode-reward steps")
    print(f"  Action distribution: {dict(action_counts.most_common())}")

    return cleaned


def convert_rollouts_to_chat(
    input_path: str,
    output_dir: str,
    val_split: float = 0.05,
    seed: int = 42,
    min_reward: float = -10.0,
) -> dict:
    """
    Convert grpo_steps.jsonl → MLX chat-format {train,valid}.jsonl.

    Each record becomes:
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Current State:\n{compressed_obs_json}"},
            {"role": "assistant", "content": "{action_json}"}
        ]}

    Filters out low-reward steps, deduplicates, balances classes,
    and compresses observations to fit token budgets.
    """
    import random

    rng = random.Random(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records = []
    with open(input_path) as f:
        for line in f:
            rec = json.loads(line)
            # Filter: skip very negative rewards (catastrophic steps)
            if rec["reward"] < min_reward:
                continue
            records.append(rec)

    # Remove genuinely bad examples (failed spool swaps)
    records = _filter_bad_examples(records)

    rng.shuffle(records)
    val_count = max(1, int(len(records) * val_split))
    val_records = records[:val_count]
    train_records = records[val_count:]

    stats = {
        "total": len(records),
        "train": len(train_records),
        "valid": len(val_records),
        "filtered_out": 0,
    }

    for split_name, split_data in [("train", train_records), ("valid", val_records)]:
        out_path = Path(output_dir) / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for rec in split_data:
                # Build chat messages — compress the observation
                messages = []
                for msg in rec["prompt_messages"]:
                    content = msg["content"]
                    if msg["role"] == "user":
                        content = _compress_observation(content)
                    messages.append({
                        "role": msg["role"],
                        "content": content,
                    })
                # The expert completion is the assistant response
                # Clean it: strip metadata fields that aren't useful for the model
                try:
                    action_data = json.loads(rec["completion"])
                    # Keep only the action-relevant fields
                    clean_action = {"action": action_data.get("action", "WAIT")}
                    for key in ["printer_id", "job_id", "operator_id", "ticket_type",
                                "ticket_id", "material", "maintenance_type", "reason"]:
                        if action_data.get(key) is not None:
                            clean_action[key] = action_data[key]
                    completion = json.dumps(clean_action)
                except (json.JSONDecodeError, KeyError):
                    completion = rec["completion"]

                messages.append({
                    "role": "assistant",
                    "content": completion,
                })

                f.write(json.dumps({"messages": messages}) + "\n")

        print(f"  {split_name}: {len(split_data)} examples → {out_path}")

    return stats


def _parse_training_log(log_path: str) -> dict:
    """
    Parse the captured mlx_lm training log to extract loss curves.

    Returns dict with keys:
      train_losses: list of (iter, loss) tuples
      val_losses:   list of (iter, loss) tuples
      final_train_loss: float | None
      final_val_loss:   float | None
      has_nan: bool
      is_learning: bool  — True if val loss decreased from first to last eval
    """
    import re
    train_losses = []
    val_losses = []
    has_nan = False

    with open(log_path) as f:
        for line in f:
            # Train: "Iter 10: Train loss 3.456, ..."
            m = re.search(r"Iter\s+(\d+).*Train loss\s+([\d.]+|nan|inf)", line)
            if m:
                it = int(m.group(1))
                val = m.group(2)
                if val in ("nan", "inf"):
                    has_nan = True
                    train_losses.append((it, float("nan")))
                else:
                    train_losses.append((it, float(val)))

            # Val: "Iter 10: Val loss 2.345, ..."
            m = re.search(r"Iter\s+(\d+).*Val loss\s+([\d.]+|nan|inf)", line)
            if m:
                it = int(m.group(1))
                val = m.group(2)
                if val in ("nan", "inf"):
                    has_nan = True
                    val_losses.append((it, float("nan")))
                else:
                    val_losses.append((it, float(val)))

    final_train = train_losses[-1][1] if train_losses else None
    final_val = val_losses[-1][1] if val_losses else None

    # Learning check: val loss should decrease from first to last
    is_learning = False
    if len(val_losses) >= 2:
        first_val = val_losses[0][1]
        last_val = val_losses[-1][1]
        if not (has_nan or first_val != first_val or last_val != last_val):
            is_learning = last_val < first_val * 0.95  # at least 5% drop

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "has_nan": has_nan,
        "is_learning": is_learning,
    }


def _check_action_accuracy(data_dir: str, model_path: str, adapter_path: str,
                            n_samples: int = 50) -> dict:
    """
    Quick post-training sanity check: run the model on validation examples
    and measure action-type accuracy (does it predict the right action name?).
    """
    import mlx_lm

    val_path = Path(data_dir) / "valid.jsonl"
    if not val_path.exists():
        return {"error": "no validation file"}

    samples = []
    with open(val_path) as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= n_samples:
                break

    if not samples:
        return {"error": "empty validation set"}

    print(f"\n  Action accuracy check on {len(samples)} val samples...")
    model, tokenizer = mlx_lm.load(model_path, adapter_path=adapter_path)
    from mlx_lm.sample_utils import make_sampler
    greedy_sampler = make_sampler(temp=0.0)

    correct_action = 0
    correct_full = 0
    valid_json = 0
    total = len(samples)

    for i, sample in enumerate(samples):
        # Build prompt from system + user messages (exclude assistant)
        prompt_msgs = [m for m in sample["messages"] if m["role"] != "assistant"]
        expected = sample["messages"][-1]["content"]

        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        response = mlx_lm.generate(
            model, tokenizer, prompt=prompt,
            max_tokens=100, sampler=greedy_sampler, verbose=False,
        )

        # Parse expected and predicted
        try:
            exp_action = json.loads(expected)
            exp_name = exp_action.get("action", "?")
        except json.JSONDecodeError:
            continue

        try:
            # Response may have trailing text — extract first JSON object
            resp_clean = response.strip()
            if "{" in resp_clean:
                start = resp_clean.index("{")
                depth = 0
                end = start
                for ci, ch in enumerate(resp_clean[start:], start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = ci + 1
                            break
                resp_clean = resp_clean[start:end]
            pred_action = json.loads(resp_clean)
            valid_json += 1
            pred_name = pred_action.get("action", "??")
        except (json.JSONDecodeError, ValueError):
            pred_name = "PARSE_FAIL"

        if pred_name == exp_name:
            correct_action += 1
        if response.strip() == expected.strip():
            correct_full += 1

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{total}: action_acc={correct_action/(i+1)*100:.0f}% "
                  f"json_valid={valid_json/(i+1)*100:.0f}%")

    results = {
        "total": total,
        "valid_json_pct": valid_json / total * 100,
        "action_accuracy_pct": correct_action / total * 100,
        "exact_match_pct": correct_full / total * 100,
    }
    print(f"  Action accuracy: {results['action_accuracy_pct']:.1f}% "
          f"| Valid JSON: {results['valid_json_pct']:.1f}% "
          f"| Exact match: {results['exact_match_pct']:.1f}%")
    return results


def run_mlx_training(
    model: str,
    data_dir: str,
    output_dir: str,
    iters: int,
    batch_size: int,
    learning_rate: float,
    num_layers: int,
    max_seq_length: int,
    seed: int,
    val_batches: int = -1,
) -> dict:
    """Run mlx_lm.lora training via subprocess, capture and parse output."""
    adapter_path = str(Path(output_dir) / "adapters")
    log_path = str(Path(output_dir) / "training.log")

    eval_every = max(min(iters // 10, 50), 5)  # eval ~10 times during training
    save_every = max(iters // 4, 25)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model,
        "--train",
        "--data", data_dir,
        "--adapter-path", adapter_path,
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--num-layers", str(num_layers),
        "--max-seq-length", str(max_seq_length),
        "--steps-per-report", "5",
        "--steps-per-eval", str(eval_every),
        "--val-batches", str(val_batches),
        "--save-every", str(save_every),
        "--seed", str(seed),
        "--mask-prompt",
        "--grad-checkpoint",
    ]

    print(f"\n  Running: {' '.join(cmd)}")
    print(f"  Logging to: {log_path}\n")

    # Capture output to both terminal and log file
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=str(Path(__file__).parent.parent),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"\n  ERROR: mlx_lm.lora exited with code {proc.returncode}")
        print(f"  Check {log_path} for details")
        sys.exit(1)

    # Parse the training log for loss curves
    metrics = _parse_training_log(log_path)

    # Print summary
    print(f"\n  {'='*50}")
    print(f"  TRAINING SUMMARY")
    print(f"  {'='*50}")
    print(f"  Adapters saved → {adapter_path}")
    print(f"  Log saved → {log_path}")
    if metrics["has_nan"]:
        print(f"  ⚠ WARNING: NaN/Inf detected in loss!")
    if metrics["final_train_loss"] is not None:
        print(f"  Final train loss: {metrics['final_train_loss']:.4f}")
    if metrics["final_val_loss"] is not None:
        print(f"  Final val loss:   {metrics['final_val_loss']:.4f}")
    if metrics["val_losses"]:
        first_v = metrics["val_losses"][0][1]
        last_v = metrics["val_losses"][-1][1]
        if first_v == first_v and last_v == last_v:  # not NaN
            drop = (1 - last_v / first_v) * 100
            print(f"  Val loss change:  {first_v:.4f} → {last_v:.4f} ({drop:+.1f}%)")
    if metrics["is_learning"]:
        print(f"  ✓ Model IS learning (val loss decreased >5%)")
    elif not metrics["has_nan"] and metrics["val_losses"]:
        print(f"  ✗ Model may NOT be learning (val loss did not decrease >5%)")
    print(f"  {'='*50}")

    # Save metrics for later analysis
    metrics_path = Path(output_dir) / "training_metrics.json"
    serializable = {
        "train_losses": metrics["train_losses"],
        "val_losses": metrics["val_losses"],
        "final_train_loss": metrics["final_train_loss"],
        "final_val_loss": metrics["final_val_loss"],
        "has_nan": metrics["has_nan"],
        "is_learning": metrics["is_learning"],
    }
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Metrics → {metrics_path}")

    return metrics


def fuse_model(model: str, adapter_path: str, output_dir: str) -> str:
    """Fuse LoRA adapters into the base model."""
    fused_path = str(Path(output_dir) / "fused")

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model,
        "--adapter-path", adapter_path,
        "--save-path", fused_path,
    ]

    print(f"\n  Fusing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))

    if result.returncode != 0:
        print(f"  WARNING: Fuse failed (exit {result.returncode}), using adapter path")
        return adapter_path

    print(f"  Fused model → {fused_path}")
    return fused_path


def quick_eval_mlx(model_path: str, adapter_path: str | None, tasks: list, seed: int, n_episodes: int = 5):
    """Evaluate using mlx_lm.generate for inference."""
    import mlx_lm

    print(f"\n  Loading model for eval: {model_path}")
    if adapter_path:
        model, tokenizer = mlx_lm.load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = mlx_lm.load(model_path)

    from mlx_lm.sample_utils import make_sampler
    eval_sampler = make_sampler(temp=0.1)

    from printfarm_env.env import PrintFarmEnvironment
    from printfarm_env.models import FarmAction, FarmActionEnum
    from scripts.grpo_rollout import obs_to_prompt_messages, parse_action
    from baselines.naive_greedy import naive_action
    from baselines.clairvoyant_greedy import clairvoyant_action

    env = PrintFarmEnvironment()
    results = {}

    for task_id in tasks:
        scores = {"naive": [], "clairvoyant": [], "model": []}
        for ep in range(n_episodes):
            s = seed + ep

            # Naive
            obs = env.reset(seed=s, task_id=task_id)
            while not obs.done:
                obs = env.step(naive_action(obs))
            scores["naive"].append(obs.reward)

            # Clairvoyant
            obs = env.reset(seed=s, task_id=task_id)
            while not obs.done:
                obs = env.step(clairvoyant_action(env))
            scores["clairvoyant"].append(obs.reward)

            # Model (MLX inference)
            obs = env.reset(seed=s, task_id=task_id)
            step_count = 0
            while not obs.done:
                messages = obs_to_prompt_messages(obs)
                # Compress user content to match training format
                for m in messages:
                    if m["role"] == "user":
                        m["content"] = _compress_observation(m["content"])
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                response = mlx_lm.generate(
                    model, tokenizer, prompt=prompt,
                    max_tokens=150, sampler=eval_sampler, verbose=False,
                )
                action = parse_action(response)
                obs = env.step(action)
                step_count += 1
            scores["model"].append(obs.reward)

            print(f"    {task_id} ep {ep+1}/{n_episodes}: "
                  f"naive={scores['naive'][-1]:.4f} "
                  f"clairv={scores['clairvoyant'][-1]:.4f} "
                  f"model={scores['model'][-1]:.4f} ({step_count} steps)")

        means = {k: sum(v) / len(v) for k, v in scores.items()}
        gap = (means["model"] - means["naive"]) / max(means["clairvoyant"] - means["naive"], 1e-6) * 100
        results[task_id] = {**means, "gap": gap}
        print(f"  {task_id}: naive={means['naive']:.4f}  clairv={means['clairvoyant']:.4f}  "
              f"model={means['model']:.4f}  gap={gap:.1f}%")

    return results


def main():
    args = parse_args()

    if args.smoke:
        args.iters = 10
        args.batch_size = 2
        val_batches = 10  # fast eval for smoke test
        print("SMOKE TEST — 10 iters, batch_size=2\n")
    else:
        val_batches = -1  # full validation set for reliable signal

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = str(output_dir / "sft_data")

    # --- Step 1: Convert rollouts to chat format ---
    print("Step 1: Converting rollouts to chat format")
    stats = convert_rollouts_to_chat(
        input_path=args.data,
        output_dir=data_dir,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(f"  Total: {stats['total']} → train: {stats['train']}, valid: {stats['valid']}")

    # --- Step 2: Train ---
    print(f"\nStep 2: MLX LoRA training ({args.iters} iters)")
    start = time.time()
    metrics = run_mlx_training(
        model=args.model,
        data_dir=data_dir,
        output_dir=str(output_dir),
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        val_batches=val_batches,
    )
    elapsed = time.time() - start
    print(f"  Training took {elapsed / 60:.1f} min")

    if metrics["has_nan"]:
        print("\n  ABORTING: NaN detected during training. Check training.log.")
        sys.exit(1)

    # --- Step 3: Action accuracy check ---
    adapter_path = str(output_dir / "adapters")
    print(f"\nStep 3: Post-training action accuracy check")
    acc_results = _check_action_accuracy(
        data_dir=data_dir,
        model_path=args.model,
        adapter_path=adapter_path,
        n_samples=min(50, stats["valid"]),
    )
    # Save accuracy results
    acc_path = output_dir / "action_accuracy.json"
    with open(acc_path, "w") as f:
        json.dump(acc_results, f, indent=2)

    # --- Step 4: Fuse adapters ---
    print(f"\nStep 4: Fusing adapters")
    fused_path = fuse_model(args.model, adapter_path, str(output_dir))

    # --- Step 5: Evaluate ---
    if not args.smoke:
        tasks = ["task_1", "task_2", "task_3", "task_4", "task_5"]
        print(f"\nStep 5: Environment evaluation on {tasks}")
        results = quick_eval_mlx(
            model_path=args.model,
            adapter_path=adapter_path,
            tasks=tasks,
            seed=args.seed,
        )

        # Save results
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results → {results_path}")
        print(f"  Fused model → {fused_path}")
    else:
        print("\n  Smoke test complete — skipping eval")
        print(f"  Adapters → {adapter_path}")


if __name__ == "__main__":
    main()
