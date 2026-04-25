#!/usr/bin/env bash
# Merge SFT LoRA adapter into full 16-bit model and push to Hub (~10 min, ~$0.05)
# Run this ONCE after run_sft.sh before running run_grpo_*.sh
#
# Produces: sikkaBolega/printfarm-sft-merged on HF Hub
# Then GRPO uses: --init_model sikkaBolega/printfarm-sft-merged
#
# Usage:
#   bash jobs/merge_sft.sh
#   bash jobs/merge_sft.sh --detach

set -euo pipefail

hf jobs run \
  --flavor t4-small \
  --timeout 30m \
  --secrets HF_TOKEN \
  -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \
  bash -c "
    set -e
    pip install unsloth --quiet
    cd /code
    python -m submission.training.merge_sft_hf \
      --adapter_repo sikkaBolega/printfarm-sft-adapter \
      --merged_repo sikkaBolega/printfarm-sft-merged \
      --max_seq_length 2048
  " \
  "$@"
