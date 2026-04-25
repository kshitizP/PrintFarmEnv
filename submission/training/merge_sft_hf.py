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
    p.add_argument("--out", default="/tmp/sft-merged",
                   help="Writable local staging dir (push_to_hub_merged needs writable fs)")
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

    # push_to_hub_merged internally calls os.makedirs(save_directory) before uploading.
    # /code is mounted read-only in HF Jobs, so we must stage to a writable path first,
    # then upload with huggingface_hub.upload_folder.
    import os
    os.makedirs(args.out, exist_ok=True)
    print(f"Saving merged 16-bit model to {args.out} ...")
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    print(f"Saved to {args.out}")

    if not args.no_push:
        print(f"Uploading to hub: {args.merged_repo} ...")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.merged_repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=args.out,
            repo_id=args.merged_repo,
            repo_type="model",
            commit_message="Add merged 16-bit SFT model",
        )
        print(f"Done — https://huggingface.co/{args.merged_repo}")
        print(f"\nNow run GRPO with: --init_model {args.merged_repo}")
    else:
        print("--no_push set, skipping hub upload.")


if __name__ == "__main__":
    main()
