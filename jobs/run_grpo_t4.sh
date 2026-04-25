#!/usr/bin/env bash
# GRPO validation run on T4-small (200 steps, n_gen=4, ~1.5 hr, ~$0.60)
# Run this AFTER run_sft.sh completes and pushes the adapter.
#
# Decision gate: check monitor.jsonl at step ~30 and ~75:
#   - step 30: tag_pct < 40% → abort, fix prompt template
#   - step 75: reward_avg not rising AND echo issues → abort, review rewards
#   - step 200: reward_avg > -0.01 and unique_actions >= 4 → proceed to L4
#
# Usage:
#   bash jobs/run_grpo_t4.sh
#   bash jobs/run_grpo_t4.sh --detach

set -euo pipefail

hf jobs run \
  --flavor t4-small \
  --timeout 3h \
  --secrets HF_TOKEN \
  --secrets WANDB_API_KEY \
  -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \
  bash -c "
    set -e
    pip install unsloth trl peft datasets wandb --quiet
    cd /code
    python -m submission.training.train_grpo_hf \
      --model Qwen/Qwen2.5-3B-Instruct \
      --init_adapter sikkaBolega/printfarm-sft-adapter \
      --max_steps 200 \
      --n_generations 4 \
      --max_completion_length 128 \
      --max_seq_length 2048 \
      --temperature 0.7 \
      --save_steps 50 \
      --hub_adapter_repo sikkaBolega/printfarm-grpo-adapter \
      --hub_merged_repo sikkaBolega/printfarm-grpo-merged \
      --out /outputs/grpo_t4
  " \
  "$@"
