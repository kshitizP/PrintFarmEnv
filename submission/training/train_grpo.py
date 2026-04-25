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

import torch

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
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--max_seq_length", type=int, default=2048)
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
    is_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
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

            # Log per-component rewards to W&B if available
            if not args.no_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "reward/format": components.get("r_format", 0.0),
                            "reward/economic": components.get("r_economic", 0.0),
                            "reward/fault_precision": components.get("r_fault_precision", 0.0),
                            "reward/message_handling": components.get("r_message_handling", 0.0),
                            "reward/unnecessary_action": components.get("r_unnecessary_action", 0.0),
                            "reward/novel_fault": components.get("r_novel_fault", 0.0),
                            "reward/total": components["total"],
                        })
                except Exception:
                    pass  # W&B logging is best-effort

        return rewards

    prompts_list = prompts  # Capture for closure

    # --- Live Monitoring Callback ---
    from collections import Counter
    from transformers import TrainerCallback
    from submission.shared.parse_action import parse_action as _parse_action, _ACTION_TAG_RE

    class GRPOMonitorCallback(TrainerCallback):
        """Logs real-time diagnostics every `eval_every` steps.

        Tracks: action distribution, tag compliance, echo rate,
        reward trend, and sample completions. Writes to monitor.jsonl
        for live tailing with: tail -f grpo_runs/.../monitor.jsonl
        """

        def __init__(self, eval_prompts, tokenizer, output_dir, eval_every=10):
            self.eval_prompts = eval_prompts[:8]  # 8 fixed prompts for consistency
            self.tokenizer = tokenizer
            self.monitor_file = Path(output_dir) / "monitor.jsonl"
            self.eval_every = eval_every
            self.reward_history = []  # (step, mean_reward)

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Capture reward from TRL's built-in logging."""
            if logs and "reward" in logs:
                self.reward_history.append((state.global_step, logs["reward"]))

        def on_step_end(self, args, state, control, model=None, **kwargs):
            step = state.global_step
            if step % self.eval_every != 0 and step != 1:
                return

            if model is None:
                return

            model.eval()
            results = []
            action_counts = Counter()
            tag_ok = 0
            parse_ok = 0
            echo_count = 0

            for p in self.eval_prompts:
                prompt_str = self.tokenizer.apply_chat_template(
                    p["messages"], tokenize=False, add_generation_prompt=True,
                )
                inputs = self.tokenizer(prompt_str, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = model.generate(
                        **inputs, max_new_tokens=100,
                        temperature=0.3, do_sample=True,
                    )

                response = self.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                has_tag = bool(_ACTION_TAG_RE.search(response))
                parsed = _parse_action(response)
                obs_text = p.get("observation_text", "")

                if has_tag:
                    tag_ok += 1
                if parsed is not None:
                    parse_ok += 1
                    action_counts[parsed.action_type] += 1

                # Echo detection
                if obs_text and len(obs_text) > 20:
                    out_tokens = set(response.lower().split())
                    obs_tokens = set(obs_text.lower().split())
                    if out_tokens and len(out_tokens & obs_tokens) / len(out_tokens) > 0.5:
                        echo_count += 1

                tag_closed = "</action>" in response
                results.append({
                    "action": parsed.action_type if parsed else "NONE",
                    "has_tag": has_tag,
                    "tag_closed": tag_closed,
                    "snippet": response[:200],
                    "len": len(response),
                })

            n = len(self.eval_prompts)
            tag_pct = tag_ok / n * 100
            parse_pct = parse_ok / n * 100
            echo_pct = echo_count / n * 100
            unique_actions = len(action_counts)

            # Reward trend
            recent_rewards = [r for s, r in self.reward_history if s > step - 20]
            reward_avg = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0

            # Build action distribution string
            dist_str = " ".join(f"{a}={c}" for a, c in action_counts.most_common(5))

            # Health verdict
            issues = []
            if echo_pct > 20:
                issues.append(f"ECHO={echo_pct:.0f}%")
            if parse_pct < 60:
                issues.append(f"FORMAT_FAIL={100-parse_pct:.0f}%")
            if unique_actions <= 1 and n > 2:
                issues.append("ACTION_COLLAPSE")
            if reward_avg < -0.15:
                issues.append(f"REWARD_LOW={reward_avg:.3f}")

            health = "HEALTHY" if not issues else "WARNING: " + ", ".join(issues)

            # Print live
            print(f"\n{'─'*60}")
            print(f"MONITOR step={step} | reward_avg={reward_avg:+.3f} | "
                  f"tag={tag_pct:.0f}% parse={parse_pct:.0f}% echo={echo_pct:.0f}%")
            print(f"  actions: {dist_str}  ({unique_actions} unique)")
            print(f"  status: {health}")
            closed_count = sum(1 for r in results if r["tag_closed"])
            avg_len = sum(r["len"] for r in results) / max(len(results), 1)
            print(f"  closed_tag: {closed_count}/{n} | avg_chars: {avg_len:.0f}")
            print(f"  sample: {results[0]['snippet']}")
            print(f"{'─'*60}")

            # Write to JSONL for offline analysis
            record = {
                "step": step,
                "reward_avg": round(reward_avg, 4),
                "tag_pct": round(tag_pct, 1),
                "parse_pct": round(parse_pct, 1),
                "echo_pct": round(echo_pct, 1),
                "unique_actions": unique_actions,
                "action_dist": dict(action_counts),
                "health": health,
                "sample": results[0]["snippet"],
            }
            with open(self.monitor_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            model.train()

    monitor_callback = GRPOMonitorCallback(
        eval_prompts=prompts,
        tokenizer=tokenizer,
        output_dir=output_dir,
        eval_every=10 if args.max_steps > 20 else 1,
    )

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
            per_device_train_batch_size=args.n_generations,
            gradient_accumulation_steps=4,
            num_generations=args.n_generations,
            generation_batch_size=args.n_generations,
            max_completion_length=args.max_completion_length,
            num_train_epochs=1,
            max_steps=args.max_steps,
            save_steps=args.save_steps,
            logging_steps=1,
            report_to=report_to,
            output_dir=str(output_dir),
            seed=args.seed,
            bf16=False,
            fp16=False,
            warmup_steps=max(1, int(args.max_steps * 0.05)),
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            dataloader_pin_memory=not is_mps,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            reward_funcs=reward_fn,
            processing_class=tokenizer,
            peft_config=peft_cfg,
            callbacks=[monitor_callback],
        )

        print("Starting GRPO training...")
        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time
        print(f"\nTraining complete! Elapsed: {elapsed/60:.1f} minutes")

        # Save final adapter (use trainer.model which has PEFT wrapper)
        trained_model = trainer.model
        trained_model.save_pretrained(str(output_dir / "final_adapter"))
        tokenizer.save_pretrained(str(output_dir / "final_adapter"))
        print(f"Saved adapter to {output_dir / 'final_adapter'}")

        # Merge LoRA into base weights
        try:
            merged = trained_model.merge_and_unload()
            merged_path = str(output_dir / "merged")
            merged.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)
            print(f"Saved merged model to {merged_path}")
        except Exception as e:
            print(f"Merge skipped ({e}). LoRA adapters at {output_dir / 'final_adapter'}")

        # Free memory
        del model, trained_model, trainer
        if is_mps:
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

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
