#!/usr/bin/env bash
# Satisfaction Run — Step 1: SFT v3 (T4-small, ~15 min)
# Uses sft_v3.jsonl: 180 examples, perfectly flat 20/action
# Pushes adapter to sikkaBolega/printfarm-sft-v3-adapter
# Merge separately with run_satisfaction_merge.sh

set -euo pipefail

hf jobs run \
  --flavor t4-small \
  --timeout 1h \
  --secrets HF_TOKEN \
  -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \
  bash -c "
    set -e
    pip install unsloth trl peft datasets wandb --quiet
    cd /code
    python -m submission.training.train_sft_hf \
      --data submission/data/sft_v3.jsonl \
      --model Qwen/Qwen2.5-3B-Instruct \
      --epochs 4 \
      --batch_size 4 \
      --grad_accum 2 \
      --hub_repo sikkaBolega/printfarm-sft-v3-adapter \
      --out /outputs/sft_v3
  " \
  "$@"
