#!/usr/bin/env bash
# Satisfaction Run — Step 3: GRPO v2 (L4, 600 steps, ~4 hrs)
#
# Key differences from run_grpo_l4.sh:
#   - Uses train_grpo_v2.py (fixed eval set, no train-leakage in monitor)
#   - Loads pre-generated grpo_train.jsonl (forced action balance)
#   - Monitor uses grpo_eval.jsonl (90-prompt hold-out, 10/action)
#   - temperature=0.8 (more exploration)
#   - Pushes to v2 repos (separate from round-2 artifacts)
#
# Prerequisites:
#   1. run_satisfaction_sft.sh completed → sikkaBolega/printfarm-sft-v3-adapter
#   2. run_satisfaction_merge.sh completed → sikkaBolega/printfarm-sft-v3-merged
#   3. build_grpo_dataset.py run locally → grpo_train.jsonl, grpo_eval.jsonl in submission/data/
#      (these are committed to the Space, so they arrive via the -v mount)

set -euo pipefail

hf jobs run \
  --flavor l4x1 \
  --timeout 6h \
  --secrets HF_TOKEN WANDB_API_KEY \
  -v hf://spaces/sikkaBolega/printfarm-env:/code:ro \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel \
  bash -c "
    set -e
    pip install unsloth trl peft datasets wandb huggingface_hub --quiet
    cd /code
    python -m submission.training.train_grpo_v2 \
      --model Qwen/Qwen2.5-3B-Instruct \
      --init_model sikkaBolega/printfarm-sft-v3-merged \
      --max_steps 600 \
      --n_generations 8 \
      --max_completion_length 256 \
      --max_seq_length 2048 \
      --temperature 0.8 \
      --learning_rate 5e-6 \
      --save_steps 50 \
      --monitor_every 25 \
      --hub_adapter_repo sikkaBolega/printfarm-grpo-v2-adapter \
      --hub_merged_repo sikkaBolega/printfarm-grpo-v2-merged \
      --out /outputs/grpo_v2
  " \
  "$@"
