"""
GRPO Training Script for PrintFarmEnv.

Designed for Unsloth + TRL GRPOTrainer on Colab T4 (Gemma 3 1B).
Can be run locally for smoke tests or on HF for full training.

Usage:
    # Smoke test (5 steps)
    python -m submission.training.train_grpo --smoke

    # Full overnight run
    python -m submission.training.train_grpo --max_steps 200 --output ./grpo_runs/overnight

    # With Gemma 3 4B (for HF credits)
    python -m submission.training.train_grpo --model google/gemma-3-4b-it --max_steps 500
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure submission is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args():
    p = argparse.ArgumentParser(description="GRPO training for PrintFarmEnv")
    p.add_argument("--model", default="google/gemma-3-1b-it",
                   help="Base model (default: gemma-3-1b-it)")
    p.add_argument("--max_steps", type=int, default=200,
                   help="Max training steps")
    p.add_argument("--n_prompts", type=int, default=100,
                   help="Number of decision prompts to generate")
    p.add_argument("--n_generations", type=int, default=8,
                   help="Completions per prompt for GRPO")
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--output", default="./grpo_runs/latest",
                   help="Output directory for checkpoints")
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 3 steps, 2 prompts")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable W&B logging")
    p.add_argument("--tasks", nargs="+",
                   default=["task_1", "task_2", "task_3", "task_4"],
                   help="Tasks to use for training prompts")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        args.max_steps = 3
        args.n_prompts = 4
        args.n_generations = 2
        args.save_steps = 2
        print("=== SMOKE TEST MODE ===")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {output_dir / 'config.json'}")

    # Step 1: Generate decision prompts
    print(f"\n{'='*60}")
    print(f"Step 1: Generating {args.n_prompts} decision prompts...")
    print(f"{'='*60}")
    from submission.training.rollout import generate_decision_prompts
    prompts = generate_decision_prompts(
        n_prompts=args.n_prompts,
        tasks=args.tasks,
        seed=args.seed,
    )
    print(f"Generated {len(prompts)} prompts across tasks: {args.tasks}")

    # Save prompts for reproducibility
    prompt_records = []
    for p in prompts:
        prompt_records.append({
            "task_id": p["task_id"],
            "seed": p["seed"],
            "prompt_length": len(p["prompt"]),
            "has_notes": bool(p["decision_obs"].operator_notes),
            "has_messages": bool(p["decision_obs"].customer_messages),
            "has_anomalies": bool(p["decision_obs"].anomaly_flags),
        })
    with open(output_dir / "prompts_meta.json", "w") as f:
        json.dump(prompt_records, f, indent=2)

    # Step 2: Try to load model with Unsloth, fall back to transformers
    print(f"\n{'='*60}")
    print(f"Step 2: Loading model {args.model}...")
    print(f"{'='*60}")

    use_unsloth = False
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=3500,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_rank,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        use_unsloth = True
        peft_cfg = None  # Unsloth handles LoRA internally
        print("Loaded with Unsloth (4-bit)")
    except ImportError:
        print("Unsloth not available, using transformers + PEFT...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig
        import torch

        # Determine device
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_map = "mps"
            torch_dtype = torch.float32  # Gemma on MPS needs float32
        else:
            device_map = "cpu"
            torch_dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        peft_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"Loaded with transformers (dtype={torch_dtype}, device={device_map})")

    # Print model info
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} (trainable: {trainable:,}, {100*trainable/total:.2f}%)")
    if peft_cfg:
        print(f"LoRA config: rank={peft_cfg.r}, will be applied by GRPOTrainer")

    # Step 3: Build the reward function for TRL
    print(f"\n{'='*60}")
    print(f"Step 3: Setting up GRPO reward function...")
    print(f"{'='*60}")

    from submission.training.rollout import evaluate_completion

    # Build a dataset — TRL GRPOTrainer expects a "prompt" column.
    # Extra columns (task_id, seed) are passed to reward_fn as **kwargs.
    dataset_records = []
    for p in prompts:
        dataset_records.append({
            "prompt": p["messages"],  # Conversational format for chat models
            "task_id": p["task_id"],
            "seed": p["seed"],
        })

    # Lookup table: (task_id, seed) -> prompt_info
    prompt_lookup = {}
    for p in prompts:
        prompt_lookup[(p["task_id"], p["seed"])] = p

    def reward_fn(prompts=None, completions=None, completion_ids=None, **kwargs):
        """Reward function compatible with TRL 1.2.0 GRPOTrainer.

        TRL calls: reward_func(prompts=..., completions=..., completion_ids=..., **extra_columns)
        - completions: list of strings (decoded text) or list of message dicts
        - extra columns from dataset passed in kwargs as lists
        """
        task_ids = kwargs.get("task_id", [])
        seeds = kwargs.get("seed", [])
        rewards = []
        for i, completion in enumerate(completions):
            # Extract text from completion
            if isinstance(completion, list):  # Conversational: list of message dicts
                text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, str):
                text = completion
            else:
                text = str(completion)

            # Find the matching prompt info by (task_id, seed)
            tid = task_ids[i] if i < len(task_ids) else None
            sid = seeds[i] if i < len(seeds) else None
            info = prompt_lookup.get((tid, sid))

            if info is None:
                info = prompts_list[0]  # fallback

            components = evaluate_completion(text, info)
            rewards.append(components["total"])

        return rewards

    prompts_list = prompts  # Capture for closure

    # Step 4: Configure and run GRPO training
    print(f"\n{'='*60}")
    print(f"Step 4: Training GRPO ({args.max_steps} steps)...")
    print(f"{'='*60}")

    try:
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset

        dataset = Dataset.from_list(dataset_records)

        report_to = "none" if args.no_wandb else "wandb"

        grpo_config = GRPOConfig(
            learning_rate=args.learning_rate,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=args.n_generations,
            max_completion_length=args.max_completion_length,
            num_train_epochs=1,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            logging_steps=1,
            report_to=report_to,
            output_dir=str(output_dir),
            seed=args.seed,
            bf16=False,  # Gemma on MPS needs float32
            fp16=False,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            reward_funcs=reward_fn,
            processing_class=tokenizer,
            peft_config=peft_cfg,
        )

        print("Starting GRPO training...")
        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time
        print(f"\nTraining complete! Elapsed: {elapsed/60:.1f} minutes")

        # Save final adapter
        model.save_pretrained(str(output_dir / "final_adapter"))
        tokenizer.save_pretrained(str(output_dir / "final_adapter"))
        print(f"Saved adapter to {output_dir / 'final_adapter'}")

    except ImportError as e:
        print(f"\nTRL/datasets not available: {e}")
        print("Running manual GRPO loop instead...")
        _manual_grpo_loop(model, tokenizer, prompts, args, output_dir)

    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        # Save whatever we have
        try:
            model.save_pretrained(str(output_dir / "crash_adapter"))
            tokenizer.save_pretrained(str(output_dir / "crash_adapter"))
            print(f"Saved crash checkpoint to {output_dir / 'crash_adapter'}")
        except Exception:
            pass
        sys.exit(1)

    print(f"\n{'='*60}")
    print("DONE — Training pipeline complete")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


def _manual_grpo_loop(model, tokenizer, prompts, args, output_dir):
    """Fallback manual training loop when TRL is not available."""
    print("Manual GRPO loop is a placeholder — install TRL for real training.")
    # Log what would happen
    results = []
    from submission.training.rollout import evaluate_completion
    for i, p in enumerate(prompts[:min(5, len(prompts))]):
        # Generate with model
        chat_text = tokenizer.apply_chat_template(
            p["messages"], tokenize=False, add_generation_prompt=True,
        )
        components = evaluate_completion('{"action_type":"WAIT"}', p)
        results.append({"prompt_idx": i, "rewards": components})
        print(f"  Prompt {i}: total_reward={components['total']:.3f}")

    with open(output_dir / "manual_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
