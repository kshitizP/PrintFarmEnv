"""
merge_sft_hf.py — Load SFT LoRA adapter via Unsloth and push a merged 16-bit model.

Run this ONCE after SFT training to produce a full merged model suitable for
loading directly in train_grpo_hf.py (--init_model flag).

Why: Unsloth's fast_lora GRPO kernels break when a model goes through
PeftModel.merge_and_unload() inside the GRPO script (dtype mismatch in matmul_lora).
The fix is to pre-merge using Unsloth's own push_to_hub_merged, then load the
merged model in GRPO with the normal 4-bit path.

Usage (HF Jobs — see jobs/merge_sft.sh):
    hf jobs run --flavor t4-small --secrets HF_TOKEN \\
      -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \\
      pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \\
      bash -c "pip install unsloth -q && \\
               cd /code && python -m submission.training.merge_sft_hf"
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Merge SFT LoRA adapter and push to Hub")
    p.add_argument("--adapter_repo", default="sikkaBolega/printfarm-sft-adapter",
                   help="HF Hub repo ID of the SFT LoRA adapter")
    p.add_argument("--merged_repo", default="sikkaBolega/printfarm-sft-merged",
                   help="HF Hub repo ID for the merged 16-bit model")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--no_push", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading SFT adapter from {args.adapter_repo} via Unsloth...")
    from unsloth import FastLanguageModel

    # Unsloth reads adapter_config.json → base_model_name_or_path automatically
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_repo,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    print("Adapter loaded.")

    if not args.no_push:
        print(f"Merging + pushing 16-bit model to {args.merged_repo} ...")
        model.push_to_hub_merged(
            args.merged_repo, tokenizer,
            save_method="merged_16bit", private=False,
        )
        print(f"Done — https://huggingface.co/{args.merged_repo}")
        print(f"\nNow run GRPO with: --init_model {args.merged_repo}")
    else:
        print("--no_push set, skipping hub upload.")


if __name__ == "__main__":
    main()
