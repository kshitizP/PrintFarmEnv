#!/usr/bin/env bash
# GRPO proper run on L4 (500 steps, n_gen=8, ~3 hr, ~$2.40)
# Run this AFTER run_grpo_t4.sh validates the learning loop.
#
# Upgrade vs T4 run:
#   n_generations:       4  →  8   (2× more rollouts per prompt)
#   max_completion_len: 128 → 256  (room for longer reasoning)
#   max_steps:          200 → 500  (more gradient updates)
#
# Usage:
#   bash jobs/run_grpo_l4.sh
#   bash jobs/run_grpo_l4.sh --detach

set -euo pipefail

hf jobs run \
  --flavor l4x1 \
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
      --n_generations 8 \
      --max_completion_length 256 \
      --max_seq_length 2048 \
      --temperature 0.7 \
      --save_steps 50 \
      --hub_adapter_repo sikkaBolega/printfarm-grpo-adapter \
      --hub_merged_repo sikkaBolega/printfarm-grpo-merged \
      --out /outputs/grpo_l4
  " \
  "$@"
