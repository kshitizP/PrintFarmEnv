"""
train_grpo_v2.py — "Satisfaction Run" GRPO trainer.

Key improvements over train_grpo_hf.py:
  1. Loads pre-generated forced-balance prompts from grpo_train.jsonl
     (built by build_grpo_dataset.py — 112 per action, 9 actions = 1008 total)
  2. Monitor uses grpo_eval.jsonl: 90-prompt STRATIFIED hold-out set
     (10 per action, never seen during training) — not train[:8]
  3. n_generations=8 default, temperature=0.8 for wider exploration
  4. Monitor logs per-action reward breakdown (all 9 actions visible)

Usage (HF Jobs):
    hf jobs run --flavor l4x1 --secrets HF_TOKEN WANDB_API_KEY \\
      -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \\
      pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \\
      bash -c "pip install unsloth trl datasets wandb -q && \\
               cd /code && python -m submission.training.train_grpo_v2 \\
               --init_model sikkaBolega/printfarm-sft-v3-merged \\
               --max_steps 600 --n_generations 8"
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
    p = argparse.ArgumentParser(description="GRPO v2 — satisfaction run")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--init_model", default=None,
                   help="Pre-merged SFT v3 model (HF Hub or local path)")
    p.add_argument("--train_data", default=None,
                   help="Path to grpo_train.jsonl (default: submission/data/grpo_train.jsonl)")
    p.add_argument("--eval_data", default=None,
                   help="Path to grpo_eval.jsonl (default: submission/data/grpo_eval.jsonl)")
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--n_generations", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hub_adapter_repo", default="sikkaBolega/printfarm-grpo-v2-adapter")
    p.add_argument("--hub_merged_repo", default="sikkaBolega/printfarm-grpo-v2-merged")
    p.add_argument("--out", default="/outputs/grpo_v2")
    p.add_argument("--no_push", action="store_true")
    p.add_argument("--monitor_every", type=int, default=25,
                   help="Eval on hold-out set every N steps (default 25 — 90 prompts takes ~4 min on L4)")
    return p.parse_args()


def _load_jsonl(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── 1. Load pre-generated prompts ────────────────────────────────────────
    print(f"\n{'='*60}\nStep 1: Loading pre-generated GRPO prompts\n{'='*60}")

    # Resolve data paths
    code_root = Path(__file__).resolve().parent.parent.parent
    data_dir = code_root / "submission" / "data"

    train_path = Path(args.train_data) if args.train_data else data_dir / "grpo_train.jsonl"
    eval_path  = Path(args.eval_data)  if args.eval_data  else data_dir / "grpo_eval.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train prompts not found at {train_path}. "
            "Run: python3 -m submission.training.build_grpo_dataset"
        )
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Eval prompts not found at {eval_path}. "
            "Run: python3 -m submission.training.build_grpo_dataset"
        )

    train_prompts = _load_jsonl(train_path)
    eval_prompts  = _load_jsonl(eval_path)

    print(f"Train prompts : {len(train_prompts)}")
    print(f"Eval prompts  : {len(eval_prompts)} (hold-out, stratified by action)")

    # Log action distribution of train set
    train_action_dist = Counter(p.get("inferred_action", "?") for p in train_prompts)
    print("Train distribution:", dict(train_action_dist.most_common()))

    eval_action_dist = Counter(p.get("inferred_action", "?") for p in eval_prompts)
    print("Eval distribution :", dict(eval_action_dist.most_common()))

    # Build prompt lookup keyed by (task_id, seed) for reward replay
    prompt_lookup = {(p["task_id"], p["seed"]): p for p in train_prompts + eval_prompts}

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nStep 2: Loading model\n{'='*60}")
    from unsloth import FastLanguageModel
    from peft import get_peft_model, LoraConfig, TaskType

    base_model_id = args.init_model or args.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"Mixed precision: {'bf16' if bf16_ok else 'fp16'}")

    # Standard PEFT LoRA (not Unsloth fast_lora — avoids dtype bug in GRPO rollouts)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── 3. Reward function ────────────────────────────────────────────────────
    print(f"\n{'='*60}\nStep 3: Wiring reward function\n{'='*60}")
    from submission.training.rollout import evaluate_completion

    dataset_records = [
        {"prompt": p["messages"], "task_id": p["task_id"], "seed": p["seed"]}
        for p in train_prompts
    ]

    def reward_fn(prompts=None, completions=None, **kwargs):
        task_ids = kwargs.get("task_id", [])
        seeds    = kwargs.get("seed", [])
        rewards  = []
        for i, completion in enumerate(completions):
            text = (completion[-1]["content"] if isinstance(completion, list)
                    else str(completion))
            key  = (task_ids[i] if i < len(task_ids) else None,
                    seeds[i]    if i < len(seeds)    else None)
            info = prompt_lookup.get(key, train_prompts[0])
            rewards.append(evaluate_completion(text, info)["total"])
        return rewards

    # ── 4. Monitor callback (uses HOLD-OUT eval set, never train) ─────────────
    from transformers import TrainerCallback
    from submission.shared.parse_action import parse_action as _parse, _ACTION_TAG_RE
    import os, wandb

    use_wandb = bool(os.environ.get("WANDB_API_KEY"))

    class MonitorCallback(TrainerCallback):
        """Evaluates on the 90-prompt stratified hold-out set every `every` steps.

        Critical difference from v1: eval_prompts comes from grpo_eval.jsonl,
        NOT from train_prompts[:8]. Reward movement → genuine generalisation.
        """

        def __init__(self, eval_prompts, tokenizer, out_dir, every=10):
            # Use all 90 eval prompts (10 per action)
            self.eval_prompts = eval_prompts
            self.tokenizer = tokenizer
            self.log_file  = Path(out_dir) / "monitor.jsonl"
            self.every     = every
            self.reward_hist: list[tuple[int, float]] = []

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
            action_counts:     Counter             = Counter()
            per_action_reward: dict[str, list]     = defaultdict(list)
            sig_rewards:       dict[str, list]     = defaultdict(list)
            comp_accum:        dict[str, list]     = defaultdict(list)
            tag_ok = parse_ok = 0
            lens: list[int] = []

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

                inferred = p.get("inferred_action", "?")
                try:
                    comps = evaluate_completion(resp, p)
                    r = comps["total"]
                    per_action_reward[inferred].append(r)
                    sig_rewards[p.get("target_signal", "?")].append(r)
                    for k in ("r_format", "r_economic", "r_fault_precision",
                              "r_message_handling", "r_unnecessary_action", "r_novel_fault"):
                        comp_accum[k].append(comps.get(k, 0.0))
                except Exception:
                    pass

            n = len(self.eval_prompts)
            tag_pct   = tag_ok / n * 100
            parse_pct = parse_ok / n * 100
            recent    = [r for s, r in self.reward_hist if s > step - 20]
            reward_avg = sum(recent) / len(recent) if recent else 0.0

            per_action_avg = {a: round(sum(v) / len(v), 3)
                              for a, v in sorted(per_action_reward.items()) if v}
            per_signal = {s: round(sum(v) / len(v), 4) for s, v in sig_rewards.items() if v}
            per_comp   = {k: round(sum(v) / len(v), 4) for k, v in comp_accum.items() if v}
            p50 = sorted(lens)[len(lens) // 2] if lens else 0
            p95 = sorted(lens)[min(int(len(lens) * 0.95), len(lens) - 1)] if lens else 0

            issues = []
            if parse_pct < 40:
                issues.append("FORMAT_FAIL")
            if len(action_counts) <= 1 and n > 5:
                issues.append("ACTION_COLLAPSE")
            if reward_avg < -0.15:
                issues.append(f"REWARD_LOW={reward_avg:.3f}")
            health = "HEALTHY" if not issues else "WARNING:" + ",".join(issues)

            print(f"\n── step {step} | reward_avg={reward_avg:+.3f} | "
                  f"tag={tag_pct:.0f}% parse={parse_pct:.0f}% | {health}")
            print(f"   predicted actions : {dict(action_counts.most_common())}")
            print(f"   per-action reward : {per_action_avg}")
            if per_signal:
                print(f"   signals           : {per_signal}")
            if per_comp:
                print(f"   components        : {per_comp}")

            record = {
                "step": step, "reward_avg": round(reward_avg, 4),
                "tag_pct": round(tag_pct, 1), "parse_pct": round(parse_pct, 1),
                "unique_actions": len(action_counts),
                "action_dist": dict(action_counts),
                "per_action_reward": per_action_avg,
                "completion_len_p50": p50, "completion_len_p95": p95,
                "per_signal_reward": per_signal,
                "reward_components": per_comp, "health": health,
            }
            with open(self.log_file, "a") as fh:
                fh.write(json.dumps(record) + "\n")

            if use_wandb and wandb.run:
                wandb.log({
                    "monitor/reward_avg": reward_avg,
                    "monitor/tag_pct": tag_pct,
                    "monitor/parse_pct": parse_pct,
                    "monitor/unique_actions": len(action_counts),
                    **{f"monitor/action/{a}": r for a, r in per_action_avg.items()},
                    **{f"monitor/signal/{k}": v for k, v in per_signal.items()},
                    **{f"monitor/{k}": v for k, v in per_comp.items()},
                }, step=step)

            model.train()

    # ── 5. GRPO training ──────────────────────────────────────────────────────
    print(f"\n{'='*60}\nStep 4: GRPO training ({args.max_steps} steps)\n{'='*60}")
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    print(f"W&B: {'ENABLED' if use_wandb else 'DISABLED'}")

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
        report_to="wandb" if use_wandb else "none",
        seed=args.seed,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        warmup_steps=max(1, int(args.max_steps * 0.05)),
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        dataloader_pin_memory=True,
    )

    monitor_every = args.monitor_every
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=Dataset.from_list(dataset_records),
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        callbacks=[MonitorCallback(eval_prompts, tokenizer, out_dir, every=monitor_every)],
    )

    start = time.time()
    trainer.train()
    print(f"GRPO complete in {(time.time() - start) / 60:.1f} min")

    # ── 6. Save adapter ───────────────────────────────────────────────────────
    print(f"\n{'='*60}\nStep 5: Saving adapter\n{'='*60}")
    adapter_path = out_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved → {adapter_path}")

    if not args.no_push:
        print(f"Pushing adapter → {args.hub_adapter_repo}")
        model.push_to_hub(args.hub_adapter_repo)
        tokenizer.push_to_hub(args.hub_adapter_repo)

    # ── 7. Clean merge (fp16 base + adapter) ──────────────────────────────────
    if not args.no_push and args.hub_merged_repo:
        print(f"\n{'='*60}\nStep 6: Clean merge → {args.hub_merged_repo}\n{'='*60}")
        import gc
        from transformers import AutoModelForCausalLM as _AMCL
        from peft import PeftModel as _PeftModel

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        merged_path = out_dir / "merged"
        merged_path.mkdir(exist_ok=True)
        print("  Loading original base in fp16 for clean merge...")
        _base = _AMCL.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cpu")
        _peft = _PeftModel.from_pretrained(_base, str(adapter_path))
        merged = _peft.merge_and_unload()
        merged.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))

        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=str(merged_path),
            repo_id=args.hub_merged_repo,
            repo_type="model",
        )
        print(f"Merged model pushed → {args.hub_merged_repo}")


if __name__ == "__main__":
    main()
