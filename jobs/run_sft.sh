#!/usr/bin/env bash
# SFT warm-start on T4-small (~30 min, ~$0.20)
# Produces: sikkaBolega/printfarm-sft-adapter on HF Hub
#
# Prerequisites:
#   hf auth login (done)
#   WANDB_API_KEY in your shell or passed as --secrets
#
# Usage:
#   bash jobs/run_sft.sh
#   bash jobs/run_sft.sh --detach   # fire-and-forget

set -euo pipefail

hf jobs run \
  --flavor t4-small \
  --timeout 1h \
  --secrets HF_TOKEN \
  --secrets WANDB_API_KEY \
  -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \
  bash -c "
    set -e
    pip install unsloth trl peft datasets wandb --quiet
    cd /code
    python -m submission.training.train_sft_hf \
      --data submission/data/sft_warm.jsonl \
      --model Qwen/Qwen2.5-3B-Instruct \
      --epochs 4 \
      --batch_size 4 \
      --grad_accum 2 \
      --hub_repo sikkaBolega/printfarm-sft-adapter \
      --out /outputs/sft
  " \
  "$@"
