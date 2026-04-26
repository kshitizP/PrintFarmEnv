#!/usr/bin/env bash
# GRPO round 2 run on A100-large (500 steps, n_gen=16, ~$7.50)
# Upgrade vs L4:
#   flavor:          l4x1  →  a100-large  (80 GB VRAM vs 24 GB)
#   n_generations:   8     →  16          (2× rollout diversity)
#   n_prompts:       100   →  200         (wider scenario coverage)
#   temperature:     0.7   →  0.9         (more exploration past SFT prior)
#
# Usage:
#   bash jobs/run_grpo_a100.sh
#   bash jobs/run_grpo_a100.sh --detach

set -euo pipefail

hf jobs run \
  --flavor a100-large \
  --timeout 5h \
  --secrets HF_TOKEN \
  -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \
  bash -c "
    set -e
    pip install unsloth trl peft datasets wandb --quiet
    cd /code
    python -m submission.training.train_grpo_hf \
      --model Qwen/Qwen2.5-3B-Instruct \
      --init_model sikkaBolega/printfarm-sft-merged \
      --max_steps 500 \
      --n_prompts 100 \
      --n_generations 8 \
      --max_completion_length 256 \
      --max_seq_length 2048 \
      --temperature 0.9 \
      --save_steps 50 \
      --hub_adapter_repo sikkaBolega/printfarm-grpo-adapter \
      --hub_merged_repo sikkaBolega/printfarm-grpo-merged \
      --out /outputs/grpo_a100
  " \
  "$@"
