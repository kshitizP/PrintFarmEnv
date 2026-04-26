#!/usr/bin/env bash
# Satisfaction Run — Step 2: Merge SFT v3 adapter → fp16
# Run after run_satisfaction_sft.sh completes.

set -euo pipefail

hf jobs run \
  --flavor t4-small \
  --timeout 1h \
  --secrets HF_TOKEN \
  -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \
  bash -c "
    set -e
    pip install unsloth peft transformers huggingface_hub --quiet
    cd /code
    python -m submission.training.merge_sft_hf \
      --adapter_repo sikkaBolega/printfarm-sft-v3-adapter \
      --merged_repo sikkaBolega/printfarm-sft-v3-merged
  " \
  "$@"
