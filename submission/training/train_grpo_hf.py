"""
train_grpo_hf.py — GRPO training optimised for HF Jobs (Unsloth + CUDA).

Differences from train_grpo.py (local/MPS):
  - Unsloth required — 4-bit QLoRA, gradient_checkpointing="unsloth"
  - bf16 training throughout
  - --init_adapter accepts a HF Hub repo ID (e.g. sikkaBolega/printfarm-sft-adapter)
    in addition to local paths — adapter is merged into base before GRPO LoRA is added
  - n_generations=8 default (L4 has 24 GB; use --n_generations 4 for T4)
  - Pushes final adapter + merged model to HF Hub on completion
  - W&B always enabled (pass WANDB_API_KEY as a Job secret)
  - No MPS / CPU fallback paths

Hardware targets:
  T4-small (16 GB): --n_generations 4  --max_completion_length 128  --max_steps 200
  L4       (24 GB): --n_generations 8  --max_completion_length 256  --max_steps 500

Usage (HF Jobs — see jobs/run_grpo_l4.sh):
    hf jobs run --flavor l4x1 --secrets HF_TOKEN WANDB_API_KEY \\
      -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \\
      pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \\
      bash -c "pip install unsloth trl datasets wandb -q && \\
               cd /code && python -m submission.training.train_grpo_hf \\
               --init_adapter sikkaBolega/printfarm-sft-adapter \\
               --max_steps 500 --n_generations 8"
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
for _c in [_here.parent, _here.parent.parent]:
    if (_c / "submission" / "__init__.py").exists():
        sys.path.insert(0, str(_c))
        break

import torch


def parse_args():
    p = argparse.ArgumentParser(description="GRPO training (HF Jobs / Unsloth)")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--init_adapter", default=None,
                   help="SFT adapter — HF Hub repo ID or local path (strongly recommended)")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_prompts", type=int, default=100)
    p.add_argument("--n_generations", type=int, default=8,
                   help="Completions per prompt (L4: 8 / T4: 4)")
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--max_completion_length", type=int, default=256,
                   help="L4: 256 / T4: 128 to stay within 16 GB")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tasks", nargs="+",
                   default=["task_1", "task_2", "task_3", "task_4"])
    p.add_argument("--hub_adapter_repo", default="sikkaBolega/printfarm-grpo-adapter",
                   help="HF Hub repo for the final LoRA adapter")
    p.add_argument("--hub_merged_repo", default="sikkaBolega/printfarm-grpo-merged",
                   help="HF Hub repo for the merged 16-bit model")
    p.add_argument("--out", default="/outputs/grpo")
    p.add_argument("--no_push", action="store_true",
                   help="Skip hub push (for dry-run testing)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── 1. Generate decision prompts ──────────────────────────────────────────
    print(f"\n{'='*60}\nStep 1: Generating {args.n_prompts} GRPO prompts\n{'='*60}")
    from submission.training.rollout import generate_decision_prompts
    prompts = generate_decision_prompts(
        n_prompts=args.n_prompts, tasks=args.tasks, seed=args.seed,
    )
    print(f"Generated {len(prompts)} prompts")

    # ── 2. Load base model + merge SFT adapter ────────────────────────────────
    print(f"\n{'='*60}\nStep 2: Loading {args.model} with Unsloth\n{'='*60}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    if args.init_adapter:
        print(f"Merging SFT adapter from {args.init_adapter} into base...")
        from peft import PeftModel
        sft = PeftModel.from_pretrained(model, args.init_adapter)
        model = sft.merge_and_unload()
        print("SFT adapter merged — GRPO will train a fresh LoRA on top")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,}  trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # ── 3. Reward function + dataset ──────────────────────────────────────────
    print(f"\n{'='*60}\nStep 3: Wiring reward function\n{'='*60}")
    from submission.training.rollout import evaluate_completion

    prompt_lookup = {(p["task_id"], p["seed"]): p for p in prompts}
    dataset_records = [
        {"prompt": p["messages"], "task_id": p["task_id"], "seed": p["seed"]}
        for p in prompts
    ]

    def reward_fn(prompts=None, completions=None, **kwargs):
        task_ids = kwargs.get("task_id", [])
        seeds = kwargs.get("seed", [])
        rewards = []
        for i, completion in enumerate(completions):
            text = (completion[-1]["content"] if isinstance(completion, list)
                    else str(completion))
            info = prompt_lookup.get((task_ids[i] if i < len(task_ids) else None,
                                      seeds[i] if i < len(seeds) else None),
                                     prompts[0])
            rewards.append(evaluate_completion(text, info)["total"])
        return rewards

    # ── 4. Monitor callback ───────────────────────────────────────────────────
    from transformers import TrainerCallback
    from submission.shared.parse_action import parse_action as _parse, _ACTION_TAG_RE
    import wandb

    class MonitorCallback(TrainerCallback):
        def __init__(self, eval_prompts, tokenizer, out_dir, every=10):
            self.eval_prompts = eval_prompts[:8]
            self.tokenizer = tokenizer
            self.log_file = Path(out_dir) / "monitor.jsonl"
            self.every = every
            self.reward_hist = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "reward" in logs:
                self.reward_hist.append((state.global_step, logs["reward"]))

        def on_step_end(self, args, state, control, model=None, **kwargs):
            step = state.global_step
            if step % self.every != 0 and step != 1:
                return
            if model is None:
                return

            model.eval()
            action_counts = Counter()
            tag_ok = parse_ok = 0
            comp_accum = defaultdict(list)
            sig_rewards = defaultdict(list)
            lens = []

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
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                resp = self.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                )
                lens.append(len(resp))
                if _ACTION_TAG_RE.search(resp):
                    tag_ok += 1
                parsed = _parse(resp)
                if parsed:
                    parse_ok += 1
                    action_counts[parsed.action_type] += 1
                try:
                    comps = evaluate_completion(resp, p)
                    sig_rewards[p.get("target_signal", "unknown")].append(comps["total"])
                    for k in ("r_format", "r_economic", "r_fault_precision",
                              "r_message_handling", "r_unnecessary_action", "r_novel_fault"):
                        comp_accum[k].append(comps.get(k, 0.0))
                except Exception:
                    pass

            n = len(self.eval_prompts)
            tag_pct = tag_ok / n * 100
            parse_pct = parse_ok / n * 100
            recent = [r for s, r in self.reward_hist if s > step - 20]
            reward_avg = sum(recent) / len(recent) if recent else 0.0
            per_signal = {s: round(sum(v) / len(v), 4) for s, v in sig_rewards.items() if v}
            per_comp = {k: round(sum(v) / len(v), 4) for k, v in comp_accum.items() if v}
            p50 = sorted(lens)[len(lens) // 2] if lens else 0
            p95 = sorted(lens)[min(int(len(lens) * 0.95), len(lens) - 1)] if lens else 0

            issues = []
            if parse_pct < 40:
                issues.append(f"FORMAT_FAIL")
            if len(action_counts) <= 1 and n > 2:
                issues.append("ACTION_COLLAPSE")
            if reward_avg < -0.15:
                issues.append(f"REWARD_LOW={reward_avg:.3f}")
            health = "HEALTHY" if not issues else "WARNING:" + ",".join(issues)

            print(f"\n── step {step} | reward_avg={reward_avg:+.3f} | "
                  f"tag={tag_pct:.0f}% parse={parse_pct:.0f}% | {health}")
            print(f"   actions: {dict(action_counts.most_common(5))} | len p50={p50} p95={p95}")
            if per_signal:
                print(f"   signals: {per_signal}")
            if per_comp:
                print(f"   components: {per_comp}")

            record = {"step": step, "reward_avg": round(reward_avg, 4),
                      "tag_pct": round(tag_pct, 1), "parse_pct": round(parse_pct, 1),
                      "unique_actions": len(action_counts),
                      "action_dist": dict(action_counts),
                      "completion_len_p50": p50, "completion_len_p95": p95,
                      "per_signal_reward": per_signal,
                      "reward_components": per_comp, "health": health}
            with open(self.log_file, "a") as fh:
                fh.write(json.dumps(record) + "\n")

            if wandb.run:
                wandb.log({"monitor/reward_avg": reward_avg,
                           "monitor/tag_pct": tag_pct,
                           "monitor/parse_pct": parse_pct,
                           "monitor/unique_actions": len(action_counts),
                           "monitor/len_p50": p50, "monitor/len_p95": p95,
                           **{f"monitor/signal/{k}": v for k, v in per_signal.items()},
                           **{f"monitor/{k}": v for k, v in per_comp.items()}},
                          step=step)

            model.train()

    # ── 5. GRPO training ──────────────────────────────────────────────────────
    print(f"\n{'='*60}\nStep 4: GRPO training ({args.max_steps} steps)\n{'='*60}")
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    grpo_config = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.n_generations,
        gradient_accumulation_steps=2,
        num_generations=args.n_generations,
        generation_batch_size=args.n_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        report_to="wandb",
        seed=args.seed,
        bf16=True,
        fp16=False,
        warmup_steps=max(1, int(args.max_steps * 0.05)),
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        dataloader_pin_memory=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=Dataset.from_list(dataset_records),
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        callbacks=[MonitorCallback(prompts, tokenizer, out_dir,
                                   every=10 if args.max_steps > 50 else 1)],
    )

    start = time.time()
    trainer.train()
    print(f"GRPO complete in {(time.time() - start) / 60:.1f} min")

    # ── 6. Save adapter ───────────────────────────────────────────────────────
    adapter_path = out_dir / "final_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to {adapter_path}")

    if not args.no_push:
        # Push LoRA adapter
        print(f"\nPushing adapter to {args.hub_adapter_repo} ...")
        trainer.model.push_to_hub(args.hub_adapter_repo, private=False)
        tokenizer.push_to_hub(args.hub_adapter_repo, private=False)
        print(f"Adapter: https://huggingface.co/{args.hub_adapter_repo}")

        # Push merged 16-bit model (Unsloth safe merge path)
        print(f"\nMerging + pushing to {args.hub_merged_repo} ...")
        model.push_to_hub_merged(
            args.hub_merged_repo, tokenizer,
            save_method="merged_16bit", private=False,
        )
        print(f"Merged model: https://huggingface.co/{args.hub_merged_repo}")

    print(f"\n{'='*60}\nDONE — output: {out_dir}\n{'='*60}")


if __name__ == "__main__":
    main()
