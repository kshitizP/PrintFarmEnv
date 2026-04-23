"""
train_local.py — GRPO training on Apple Silicon (M4 Pro / MPS backend)
======================================================================

Replaces the Colab notebook for local training. Uses standard
HuggingFace transformers + PEFT + TRL (no Unsloth, no bitsandbytes).

Qwen2.5-3B in fp16 ≈ 6 GB + LoRA grads + AdamW states ≈ 18-20 GB peak.
Well within 48 GB unified memory on M4 Pro.

Usage:
    # Quick smoke test (2 episodes, 5 GRPO steps)
    PYTHONPATH=. python scripts/train_local.py --smoke

    # Full training
    PYTHONPATH=. python scripts/train_local.py \
        --tasks task_1 --episodes 20 --max-steps 50

    # Resume from SFT checkpoint
    PYTHONPATH=. python scripts/train_local.py \
        --model ./sft_output --tasks task_1 task_2 task_3 --max-steps 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmActionEnum
from scripts.grpo_rollout import (
    collect_rollouts,
    save_dataset,
    load_as_hf_dataset,
    make_env_reward_fn,
    obs_to_prompt_messages,
    parse_action,
    _clairvoyant_policy_fn,
)

# ─── Device selection ────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
print(f"Device: {DEVICE}")
if DEVICE == "mps":
    # MPS fallback for ops not yet implemented on MPS
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
ROLLOUT_DIR    = "./data/grpo_local"
OUTPUT_DIR     = "./grpo_output_local"
SEED           = 42

def parse_args():
    p = argparse.ArgumentParser(description="Local GRPO training (Apple Silicon)")
    p.add_argument("--model",       type=str, default=DEFAULT_MODEL, help="HF model id or local path")
    p.add_argument("--tasks",       nargs="+", default=["task_1"])
    p.add_argument("--episodes",    type=int, default=20)
    p.add_argument("--max-steps",   type=int, default=50)
    p.add_argument("--group-size",  type=int, default=4,  help="GRPO group size (G)")
    p.add_argument("--batch-size",  type=int, default=2,  help="Per-device batch size")
    p.add_argument("--grad-accum",  type=int, default=4,  help="Gradient accumulation steps")
    p.add_argument("--lr",          type=float, default=5e-6)
    p.add_argument("--lora-rank",   type=int, default=16)
    p.add_argument("--max-tokens",  type=int, default=128)
    p.add_argument("--output",      type=str, default=OUTPUT_DIR)
    p.add_argument("--seed",        type=int, default=SEED)
    p.add_argument("--smoke",       action="store_true", help="Quick 5-step smoke test")
    p.add_argument("--skip-rollouts", action="store_true", help="Reuse existing rollout data")
    return p.parse_args()

# ─── Phase 1: Collect rollouts ───────────────────────────────────────────────

def collect_training_data(tasks, episodes, seed, skip=False):
    out_path = Path(ROLLOUT_DIR) / "grpo_steps.jsonl"
    if skip and out_path.exists():
        print(f"Reusing existing rollout data: {out_path}")
        return

    print(f"\n{'='*60}")
    print(f"Phase 1: Collecting rollouts ({episodes} episodes × {len(tasks)} tasks)")
    print(f"{'='*60}")

    records = collect_rollouts(
        policy_fn=_clairvoyant_policy_fn,
        tasks=tasks,
        n_episodes=episodes,
        base_seed=seed,
        verbose=False,
    )
    save_dataset(records, ROLLOUT_DIR)
    print(f"Collected {len(records)} step records → {ROLLOUT_DIR}/")

# ─── Phase 2: Load model + LoRA ─────────────────────────────────────────────

def load_model_and_tokenizer(model_id, lora_rank):
    print(f"\n{'='*60}")
    print(f"Phase 2: Loading model ({model_id})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fp16 for MPS/CUDA, fp32 fallback for CPU
    dtype = torch.float16 if DEVICE in ("mps", "cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        # Don't use device_map="auto" with MPS — load to CPU then move
    )

    # Apply LoRA
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
    print(f"Total params   : {total:,}")
    print(f"Trainable (LoRA): {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"Model dtype    : {dtype}")
    print(f"Memory estimate: ~{total * 2 / 1e9:.1f} GB (fp16 weights)")

    return model, tokenizer

# ─── Phase 3: Build dataset ─────────────────────────────────────────────────

def build_dataset(tokenizer):
    print(f"\n{'='*60}")
    print(f"Phase 3: Building HF dataset")
    print(f"{'='*60}")

    raw_ds = load_as_hf_dataset(f"{ROLLOUT_DIR}/grpo_steps.jsonl")

    def format_prompt(example):
        messages = example["prompt_messages"]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt}

    dataset = raw_ds.map(format_prompt)
    keep = ["prompt", "completion", "reward", "task_id", "seed", "step_idx", "prior_actions"]
    dataset = dataset.select_columns(keep)

    print(f"Dataset: {len(dataset)} rows")
    print(f"Reward range: {min(dataset['reward']):.3f} – {max(dataset['reward']):.3f}")
    return dataset

# ─── Phase 4: Reward functions ───────────────────────────────────────────────

VALID_ACTIONS = {a.value for a in FarmActionEnum}

def format_reward_fn(prompts, completions, **kwargs):
    """Valid JSON FarmAction → +0.2, else -0.2."""
    rewards = []
    for c in completions:
        try:
            data = json.loads(c.strip())
            rewards.append(0.2 if data.get("action") in VALID_ACTIONS else -0.2)
        except Exception:
            rewards.append(-0.2)
    return rewards

def anti_wait_spam_fn(prompts, completions, **kwargs):
    """Penalty for WAIT spam — prevents local optimum of always waiting."""
    rewards = []
    for c in completions:
        try:
            data = json.loads(c.strip())
            rewards.append(-0.1 if data.get("action") == "WAIT" else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

# ─── Phase 5: Train ─────────────────────────────────────────────────────────

def train(model, tokenizer, dataset, args):
    print(f"\n{'='*60}")
    print(f"Phase 5: GRPO Training")
    print(f"  Max steps   : {args.max_steps}")
    print(f"  Group size  : {args.group_size}")
    print(f"  Batch size  : {args.batch_size} × {args.grad_accum} grad accum")
    print(f"  LR          : {args.lr}")
    print(f"  Device      : {DEVICE}")
    print(f"{'='*60}")

    # MPS-compatible config — no bf16 (MPS doesn't support it), use fp16
    use_bf16 = DEVICE == "cuda"
    use_fp16 = DEVICE == "mps"

    # generation_batch_size must be divisible by num_generations
    gen_batch = max(args.group_size, args.batch_size)

    grpo_config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        # GRPO-specific
        num_generations=args.group_size,
        generation_batch_size=gen_batch,
        max_completion_length=args.max_tokens,
        temperature=0.9,
        # Precision — MPS needs fp16, CUDA can do bf16
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",  # adamw_8bit needs bitsandbytes (CUDA-only)
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(args.max_steps * 0.1)),
        seed=args.seed,
        # Logging
        logging_steps=5,
        save_steps=25,
        report_to="none",
        # MPS compat: disable features that can cause issues
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

    print("\nStarting training...")
    trainer.train()
    print("Training complete!")

    return trainer

# ─── Phase 6: Save ───────────────────────────────────────────────────────────

def save_model(model, tokenizer, output_dir):
    print(f"\n{'='*60}")
    print(f"Phase 6: Saving model → {output_dir}")
    print(f"{'='*60}")

    adapter_path = f"{output_dir}/lora_adapters"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapters saved → {adapter_path}")

    # Merge and save fp16 for easy inference
    merged_path = f"{output_dir}/merged_fp16"
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"Merged fp16 model → {merged_path}")
    except Exception as e:
        print(f"Merge skipped ({e}). LoRA adapters saved separately.")

# ─── Phase 7: Quick eval ─────────────────────────────────────────────────────

def quick_eval(model, tokenizer, tasks, seed, max_tokens):
    print(f"\n{'='*60}")
    print(f"Phase 7: Quick evaluation")
    print(f"{'='*60}")

    from baselines.naive_greedy import naive_action
    from baselines.clairvoyant_greedy import clairvoyant_action

    model.eval()

    for task_id in tasks:
        env = PrintFarmEnvironment()
        scores = {"naive": [], "clairvoyant": [], "model": []}

        for ep in range(5):
            # Naive
            obs = env.reset(seed=seed + ep, task_id=task_id)
            while not obs.done:
                obs = env.step(naive_action(obs))
            scores["naive"].append(obs.reward)

            # Clairvoyant
            obs = env.reset(seed=seed + ep, task_id=task_id)
            while not obs.done:
                obs = env.step(clairvoyant_action(env))
            scores["clairvoyant"].append(obs.reward)

            # Trained model
            obs = env.reset(seed=seed + ep, task_id=task_id)
            while not obs.done:
                messages = obs_to_prompt_messages(obs)
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.2,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                decoded = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                obs = env.step(parse_action(decoded))
            scores["model"].append(obs.reward)

        print(f"\n{task_id} (5 episodes):")
        for name in ["naive", "clairvoyant", "model"]:
            mean = sum(scores[name]) / len(scores[name])
            print(f"  {name:<15}: {mean:.4f}")

        naive_mean = sum(scores["naive"]) / len(scores["naive"])
        clair_mean = sum(scores["clairvoyant"]) / len(scores["clairvoyant"])
        model_mean = sum(scores["model"]) / len(scores["model"])
        gap = (model_mean - naive_mean) / max(clair_mean - naive_mean, 1e-6) * 100
        print(f"  gap captured  : {gap:.1f}%")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.smoke:
        args.episodes = 2
        args.max_steps = 5
        args.group_size = 2
        args.batch_size = 1
        args.grad_accum = 1
        print("🔧 SMOKE TEST MODE — minimal config to verify the loop runs")

    # Phase 1: Collect rollouts
    collect_training_data(args.tasks, args.episodes, args.seed, skip=args.skip_rollouts)

    # Phase 2: Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.lora_rank)

    # Phase 3: Build dataset
    dataset = build_dataset(tokenizer)

    # Phase 5: Train
    trainer = train(model, tokenizer, dataset, args)

    # Phase 6: Save
    save_model(model, tokenizer, args.output)

    # Phase 7: Eval
    quick_eval(model, tokenizer, args.tasks, args.seed, args.max_tokens)

    print(f"\n{'='*60}")
    print(f"Done! Output → {args.output}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
