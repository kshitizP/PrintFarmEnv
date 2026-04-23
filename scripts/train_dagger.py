"""
train_dagger.py — DAgger-style iterative GRPO training
======================================================

Iteratively:
  1. Collect rollouts using a MIX of the current model + clairvoyant baseline
  2. Train GRPO on the mixed data
  3. Evaluate
  4. Repeat, shifting the mix toward the model's own policy

Each round the model sees more states from its OWN behaviour — including
the messy states that arise from its mistakes — while the clairvoyant mix
prevents catastrophic forgetting.

Schedule (configurable):
  Round 1: 100% clairvoyant (bootstrap — model is untrained)
  Round 2:  50% clairvoyant + 50% model
  Round 3:  30% clairvoyant + 70% model
  Round 4:  10% clairvoyant + 90% model

Usage:
    # Smoke test (verifies the loop runs end-to-end)
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. python scripts/train_dagger.py --smoke

    # Full training
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. python scripts/train_dagger.py \
        --tasks task_1 task_2 task_3 --episodes 20 --grpo-steps 50 --rounds 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmActionEnum
from scripts.grpo_rollout import (
    collect_rollouts,
    collect_mixed_rollouts,
    save_dataset,
    load_as_hf_dataset,
    make_env_reward_fn,
    obs_to_prompt_messages,
    parse_action,
    _clairvoyant_policy_fn,
    _build_model_policy_fn,
)

# ─── Device ──────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ─── Default mixing schedule ────────────────────────────────────────────────

# (clairvoyant_weight, model_weight) per round
DEFAULT_SCHEDULE = [
    (1.0, 0.0),   # Round 1: pure clairvoyant (bootstrap)
    (0.5, 0.5),   # Round 2: 50/50
    (0.3, 0.7),   # Round 3: 30/70
    (0.1, 0.9),   # Round 4: 10/90
]

VALID_ACTIONS = {a.value for a in FarmActionEnum}


def parse_args():
    p = argparse.ArgumentParser(description="DAgger-style iterative GRPO training")
    p.add_argument("--model",       type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--tasks",       nargs="+", default=["task_1", "task_2", "task_3"])
    p.add_argument("--episodes",    type=int, default=20, help="Episodes per task per round")
    p.add_argument("--grpo-steps",  type=int, default=50, help="GRPO training steps per round")
    p.add_argument("--rounds",      type=int, default=4,  help="Number of DAgger rounds")
    p.add_argument("--group-size",  type=int, default=4)
    p.add_argument("--batch-size",  type=int, default=2)
    p.add_argument("--grad-accum",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=5e-6)
    p.add_argument("--lora-rank",   type=int, default=16)
    p.add_argument("--max-tokens",  type=int, default=128)
    p.add_argument("--output",      type=str, default="./dagger_output")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--smoke",       action="store_true", help="Quick smoke test")
    return p.parse_args()


# ─── Reward helpers (same as train_local.py) ─────────────────────────────────

def format_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for c in completions:
        try:
            data = json.loads(c.strip())
            rewards.append(0.2 if data.get("action") in VALID_ACTIONS else -0.2)
        except Exception:
            rewards.append(-0.2)
    return rewards


def anti_wait_spam_fn(prompts, completions, **kwargs):
    rewards = []
    for c in completions:
        try:
            data = json.loads(c.strip())
            rewards.append(-0.1 if data.get("action") == "WAIT" else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# ─── Core functions ──────────────────────────────────────────────────────────

def collect_round_data(round_idx, model_path, tasks, episodes, seed, schedule):
    """Collect rollouts for a given DAgger round."""
    clairvoyant_w, model_w = schedule[min(round_idx, len(schedule) - 1)]

    print(f"\n  Collecting rollouts: clairvoyant={clairvoyant_w:.0%}, model={model_w:.0%}")

    if model_w == 0 or model_path is None:
        # Pure clairvoyant — no model needed
        records = collect_rollouts(
            policy_fn=_clairvoyant_policy_fn,
            tasks=tasks,
            n_episodes=episodes,
            base_seed=seed,
        )
    else:
        model_policy = _build_model_policy_fn(model_path)
        records = collect_mixed_rollouts(
            policy_fns=[_clairvoyant_policy_fn, model_policy],
            policy_names=["clairvoyant", "model"],
            policy_weights=[clairvoyant_w, model_w],
            tasks=tasks,
            n_episodes=episodes,
            base_seed=seed,
        )

    return records


def load_model(model_id, lora_rank):
    """Load base model + LoRA. For rounds > 1, model_id is the merged checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total:,} total, {trainable:,} trainable ({trainable/total*100:.2f}%)")

    return model, tokenizer


def build_dataset(records, tokenizer, data_dir):
    """Build HF dataset from rollout records."""
    save_dataset(records, data_dir)
    raw_ds = load_as_hf_dataset(f"{data_dir}/grpo_steps.jsonl")

    def format_prompt(example):
        messages = example["prompt_messages"]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return {"prompt": prompt}

    dataset = raw_ds.map(format_prompt)
    keep = ["prompt", "completion", "reward", "task_id", "seed", "step_idx", "prior_actions"]
    dataset = dataset.select_columns(keep)
    print(f"  Dataset: {len(dataset)} rows, "
          f"reward range: {min(dataset['reward']):.3f} – {max(dataset['reward']):.3f}")
    return dataset


def train_round(model, tokenizer, dataset, args, round_dir, grpo_steps):
    """Run one round of GRPO training."""
    use_bf16 = DEVICE == "cuda"
    use_fp16 = DEVICE == "mps"
    gen_batch = max(args.group_size, args.batch_size)

    grpo_config = GRPOConfig(
        output_dir=round_dir,
        max_steps=grpo_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_generations=args.group_size,
        generation_batch_size=gen_batch,
        max_completion_length=args.max_tokens,
        temperature=0.9,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(grpo_steps * 0.1)),
        seed=args.seed,
        logging_steps=5,
        save_steps=max(grpo_steps, 25),  # save at end only
        report_to="none",
        dataloader_pin_memory=False if DEVICE == "mps" else True,
    )

    env_reward_fn = make_env_reward_fn()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[env_reward_fn, format_reward_fn, anti_wait_spam_fn],
        args=grpo_config,
        train_dataset=dataset,
    )

    trainer.train()
    return trainer


def save_merged(model, tokenizer, output_dir):
    """Save LoRA adapters + merged fp16 model."""
    adapter_path = f"{output_dir}/lora_adapters"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    merged_path = f"{output_dir}/merged_fp16"
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"  Saved merged model → {merged_path}")
    except Exception as e:
        print(f"  Merge skipped ({e}). LoRA adapters at {adapter_path}")
        merged_path = adapter_path

    return merged_path


def quick_eval(model_path, tasks, seed, n_episodes=5):
    """Evaluate a merged checkpoint vs baselines."""
    import warnings
    from baselines.naive_greedy import naive_action
    from baselines.clairvoyant_greedy import clairvoyant_action

    model_policy = _build_model_policy_fn(model_path)
    env = PrintFarmEnvironment()

    total_evals = len(tasks) * n_episodes * 3  # 3 policies per episode
    done_evals = 0

    results = {}
    for task_id in tasks:
        scores = {"naive": [], "clairvoyant": [], "model": []}
        for ep in range(n_episodes):
            s = seed + ep

            obs = env.reset(seed=s, task_id=task_id)
            while not obs.done:
                obs = env.step(naive_action(obs))
            scores["naive"].append(obs.reward)
            done_evals += 1

            obs = env.reset(seed=s, task_id=task_id)
            while not obs.done:
                obs = env.step(clairvoyant_action(env))
            scores["clairvoyant"].append(obs.reward)
            done_evals += 1

            print(f"    eval {task_id} ep {ep+1}/{n_episodes}: "
                  f"naive={scores['naive'][-1]:.4f}  clairvoyant={scores['clairvoyant'][-1]:.4f}  "
                  f"running model...", flush=True)

            obs = env.reset(seed=s, task_id=task_id)
            step_count = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                while not obs.done:
                    obs = env.step(model_policy(env))
                    step_count += 1
            scores["model"].append(obs.reward)
            done_evals += 1
            print(f"    eval {task_id} ep {ep+1}/{n_episodes}: "
                  f"model={scores['model'][-1]:.4f} ({step_count} steps)  "
                  f"[{done_evals}/{total_evals}]", flush=True)

        means = {k: sum(v) / len(v) for k, v in scores.items()}
        gap = (means["model"] - means["naive"]) / max(means["clairvoyant"] - means["naive"], 1e-6) * 100
        results[task_id] = {**means, "gap": gap}

        print(f"  {task_id}: naive={means['naive']:.4f}  clairvoyant={means['clairvoyant']:.4f}  "
              f"model={means['model']:.4f}  gap={gap:.1f}%")

    return results


# ─── Main DAgger loop ────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.smoke:
        args.tasks = ["task_1"]
        args.episodes = 2
        args.grpo_steps = 3
        args.rounds = 2
        args.group_size = 2
        args.batch_size = 1
        args.grad_accum = 1
        print("🔧 SMOKE TEST — 2 rounds, 2 episodes, 3 GRPO steps each\n")

    schedule = DEFAULT_SCHEDULE
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    current_model_id = args.model
    merged_path = None
    all_results = []

    for rnd in range(args.rounds):
        round_start = time.time()
        round_dir = str(output_base / f"round_{rnd + 1}")
        data_dir = str(output_base / f"round_{rnd + 1}" / "data")

        print(f"\n{'=' * 60}")
        print(f"  DAgger Round {rnd + 1}/{args.rounds}")
        print(f"{'=' * 60}")

        # --- Step 1: Collect rollouts ---
        print(f"\n  Step 1: Collecting rollouts")
        records = collect_round_data(
            round_idx=rnd,
            model_path=merged_path,
            tasks=args.tasks,
            episodes=args.episodes,
            seed=args.seed + rnd * 1000,  # different seeds each round
            schedule=schedule,
        )
        print(f"  Collected {len(records)} step records")

        # --- Step 2: Load model + LoRA ---
        print(f"\n  Step 2: Loading model ({current_model_id})")
        # After round 1, load from the previously merged checkpoint
        load_from = merged_path if merged_path else current_model_id
        model, tokenizer = load_model(load_from, args.lora_rank)

        # --- Step 3: Build dataset ---
        print(f"\n  Step 3: Building dataset")
        dataset = build_dataset(records, tokenizer, data_dir)

        # --- Step 4: Train ---
        print(f"\n  Step 4: GRPO training ({args.grpo_steps} steps)")
        train_round(model, tokenizer, dataset, args, round_dir, args.grpo_steps)
        print(f"  Training complete")

        # --- Step 5: Save merged model ---
        print(f"\n  Step 5: Saving model")
        merged_path = save_merged(model, tokenizer, round_dir)

        # Free GPU memory before eval
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()

        # --- Step 6: Evaluate ---
        print(f"\n  Step 6: Evaluation ({args.tasks})")
        results = quick_eval(merged_path, args.tasks, args.seed)
        all_results.append({"round": rnd + 1, **results})

        elapsed = time.time() - round_start
        print(f"\n  Round {rnd + 1} completed in {elapsed / 60:.1f} min")

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  DAgger Training Summary")
    print(f"{'=' * 60}")
    for r in all_results:
        rnd = r.pop("round")
        print(f"\n  Round {rnd}:")
        for task_id, scores in r.items():
            print(f"    {task_id}: model={scores['model']:.4f}  gap={scores['gap']:.1f}%")

    # Save final results
    with open(output_base / "dagger_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  Best model → {merged_path}")
    print(f"  Results    → {output_base / 'dagger_results.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
