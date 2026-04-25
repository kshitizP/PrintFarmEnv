"""
train_sft_hf.py — SFT warm-start optimised for HF Jobs (Unsloth + CUDA).

Differences from train_sft.py (local/MPS):
  - Unsloth required — 4-bit QLoRA, gradient_checkpointing="unsloth"
  - bf16 training (T4/L4 support it; float32 is MPS-only)
  - Larger batch: per_device=4, grad_accum=2 (vs 2/4 locally)
  - Pushes final adapter to HF Hub on completion
  - W&B always enabled (pass WANDB_API_KEY as a Job secret)

Usage (HF Jobs — see jobs/run_sft.sh):
    hf jobs run --flavor t4-small --secrets HF_TOKEN WANDB_API_KEY \\
      -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \\
      pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \\
      bash -c "pip install unsloth trl datasets wandb -q && \\
               cd /code && python -m submission.training.train_sft_hf"

Usage (direct, after installing deps):
    python -m submission.training.train_sft_hf \\
        --data submission/data/sft_warm.jsonl \\
        --hub_repo sikkaBolega/printfarm-sft-adapter
"""

import argparse
import json
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
for _c in [_here.parent, _here.parent.parent]:
    if (_c / "submission" / "__init__.py").exists():
        sys.path.insert(0, str(_c))
        break


def parse_args():
    p = argparse.ArgumentParser(description="SFT warm-start (HF Jobs / Unsloth)")
    p.add_argument("--data", default="submission/data/sft_warm.jsonl")
    p.add_argument("--model", default="google/gemma-3-1b-it")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Per-device batch (T4: 4, L4: 8)")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--hub_repo", default="sikkaBolega/printfarm-sft-adapter",
                   help="HF Hub repo to push the final adapter")
    p.add_argument("--out", default="/outputs/sft",
                   help="Local output dir (inside the Job container)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_push", action="store_true",
                   help="Skip hub push (for dry-run testing)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print(f"\n{'='*60}\nStep 1: Loading SFT dataset from {args.data}\n{'='*60}")
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    from datasets import Dataset
    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print(f"Loaded {len(dataset)} examples")

    # ── 2. Load model — Unsloth 4-bit QLoRA ──────────────────────────────────
    print(f"\n{'='*60}\nStep 2: Loading {args.model} with Unsloth\n{'='*60}")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto: bfloat16 on Ampere+, float16 on older
    )
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

    # ── 3. SFT training ────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nStep 3: SFT for {args.epochs} epochs\n{'='*60}")
    from trl import SFTTrainer, SFTConfig

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            logging_steps=5,
            save_strategy="no",
            report_to="wandb",
            seed=args.seed,
            max_length=args.max_seq_length,
            bf16=True,
            fp16=False,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            completion_only_loss=True,
        ),
    )

    start = time.time()
    trainer.train()
    print(f"SFT complete in {(time.time() - start) / 60:.1f} min")

    # ── 4. Save adapter ───────────────────────────────────────────────────────
    adapter_path = out_dir / "final_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to {adapter_path}")

    # ── 5. Push to HF Hub ─────────────────────────────────────────────────────
    if not args.no_push:
        print(f"\nPushing adapter to {args.hub_repo} ...")
        trainer.model.push_to_hub(args.hub_repo, private=False)
        tokenizer.push_to_hub(args.hub_repo, private=False)
        print(f"Done — https://huggingface.co/{args.hub_repo}")
        print(f"\nNext: run GRPO with --init_adapter {args.hub_repo}")


if __name__ == "__main__":
    main()
