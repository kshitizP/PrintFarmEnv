"""
train_sft.py — Supervised fine-tuning warm-start for the GRPO policy.

Trains a LoRA adapter on the dataset built by `build_sft_dataset.py`,
teaching the model the Safety > Throughput > Customer-Service hierarchy
BEFORE GRPO so the RL phase has a sensible starting policy.

Usage:
    # build the dataset first
    python -m submission.training.build_sft_dataset \
        --out submission/data/sft_warm.jsonl --n 120

    # train the warm-start adapter
    python -m submission.training.train_sft \
        --data submission/data/sft_warm.jsonl \
        --model google/gemma-3-1b-it \
        --epochs 4 \
        --out grpo_runs/sft_warm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure submission is importable
_here = Path(__file__).resolve().parent
for _candidate in [_here.parent, _here.parent.parent]:
    if (_candidate / "submission" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break


def parse_args():
    p = argparse.ArgumentParser(description="SFT warm-start training")
    p.add_argument("--data", default="submission/data/sft_warm.jsonl",
                   help="Path to JSONL dataset")
    p.add_argument("--model", default="google/gemma-3-1b-it")
    p.add_argument("--out", default="grpo_runs/sft_warm",
                   help="Output directory for adapter")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {out_dir / 'config.json'}")

    # Load dataset
    print(f"\n{'='*60}")
    print(f"Step 1: Loading SFT dataset from {args.data}")
    print(f"{'='*60}")
    from datasets import Dataset

    rows = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rows.append({"messages": ex["messages"]})
    print(f"Loaded {len(rows)} examples")

    dataset = Dataset.from_list(rows)

    # Load model
    print(f"\n{'='*60}")
    print(f"Step 2: Loading model {args.model}")
    print(f"{'='*60}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig

    is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
    elif is_mps:
        device_map = "mps"
        torch_dtype = torch.float32
    else:
        device_map = "cpu"
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    print(f"Loaded with transformers (dtype={torch_dtype}, device={device_map})")

    peft_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Run SFT
    print(f"\n{'='*60}")
    print(f"Step 3: Running SFT for {args.epochs} epochs")
    print(f"{'='*60}")
    from trl import SFTTrainer, SFTConfig

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=2,
        save_strategy="no",  # we save final adapter manually below
        report_to="none",
        seed=args.seed,
        max_length=args.max_seq_length,
        bf16=False,
        fp16=False,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=not is_mps,
        gradient_checkpointing=False,
        completion_only_loss=True,  # mask the prompt; only learn the action
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\nSFT training complete in {elapsed/60:.1f} min")

    # Save adapter
    adapter_path = out_dir / "final_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to {adapter_path}")

    # Free memory
    del model, trainer
    if is_mps:
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"DONE — SFT warm-start complete")
    print(f"Adapter: {adapter_path}")
    print(f"Next: python -m submission.training.verify_sft --adapter {adapter_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
