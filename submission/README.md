---
title: PrintFarmEnv
emoji: 🖨️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# PrintFarm Dispatcher — OpenEnv Submission

> **OpenEnv Hackathon 2025** · Reinforcement learning for LLM-driven 3D print-farm operations

## Quick Links

| Resource | Link |
|---|---|
| **HF Space (live env + REST API)** | https://huggingface.co/spaces/sikkaBolega/printfarm-env |
| **Colab Training Notebook** | [notebooks/PrintFarmEnv_Training.ipynb](notebooks/PrintFarmEnv_Training.ipynb) |
| **GRPO Adapter (submission model)** | https://huggingface.co/sikkaBolega/printfarm-grpo-adapter |
| **SFT Adapter** | https://huggingface.co/sikkaBolega/printfarm-sft-adapter |
| **Merged Model** | https://huggingface.co/sikkaBolega/printfarm-grpo-merged |

---

## 1. Problem

Large 3D print farms (e.g. [Slant3D](https://www.slant3d.com/), [Bitfab](https://bitfab.io/)) run dozens to hundreds of FDM printers in parallel. A human dispatcher monitors machine states, allocates jobs, responds to failures, and handles operator and customer communications — under time pressure, with incomplete information.

**Structured-only environments don't capture this.** Existing gym-style manufacturing simulators reduce every printer state to a clean enum and every fault to a labelled boolean. Real dispatchers must:

- Read freeform operator shift notes ("bearing sounds rough but only on layer 40+")
- Triage ambiguous customer messages ("my order is urgent" — but is it late yet?)
- Respond to novel compound faults that don't match any named failure mode

A rules engine can handle structured signals perfectly. But text signals require **reading comprehension and context-sensitive judgment** — exactly what an LLM can learn via RL that a rules engine cannot.

---

## 2. Environment

### What we built

`PrintFarmEnvironment` is a discrete-event simulator of a 6-printer, 5-operator 3D print farm. Each step advances time by ~10 minutes and surfaces a `FarmObservation` with:

| Field | Description |
|---|---|
| `printers` | State, material, fatigue, spool level, active job |
| `active_queue` | Jobs with priority, deadline, required material |
| `operators` | Shift schedule, fatigue, specialization |
| `operator_notes` | **Freeform text** from shift operators (unstructured) |
| `customer_messages` | **Freeform text** from customers (unstructured) |
| `anomaly_flags` | **Novel compound-fault** descriptions (unstructured) |

The agent chooses one `FarmAction` per step from 9 scorable action types: `ASSIGN_JOB`, `CANCEL_JOB`, `PAUSE_JOB`, `RESUME_JOB`, `RUN_DIAGNOSTIC`, `DISPATCH_TICKET`, `REQUEST_SPOOL_SWAP`, `REQUEST_MAINTENANCE`, `WAIT`.

### Three unstructured signal augmentations

**Operator notes** (~40% of steps) — randomly drawn from a corpus of 20 templated notes covering spool state, mechanical symptoms, layer adhesion issues, and shift handover remarks.

**Customer messages** (~20% of steps) — drawn from 15 templates covering urgency escalation, cancellation requests, and deadline ambiguity. Correct action depends on whether the relevant job is late vs. whether the customer's claim is warranted.

**Novel anomaly flags** (~15% of steps when printers are PRINTING) — procedurally generated compound faults ("MECHANICAL_VIBRATION + LAYER_SHIFT at 67% progress on P3"). These have no matching named failure mode in the rules engine.

### Decision-point sub-skill RL

Training LLMs on full 40-step episodes is expensive. Instead, we use a **decision-point wrapper** (`DecisionPointEnv`):

1. Roll forward with the rules agent until an unstructured signal appears.
2. Pause and present the observation to the LLM.
3. Execute the LLM's action; measure reward as **delta vs. a rules-only counterfactual** on the same seed.

This focuses all gradient signal on the moments where the LLM adds unique value.

### Six-component Composite Reward

| Component | Signal | What it measures |
|---|---|---|
| `r_format` | Always | `<action>…</action>` tag present and parseable |
| `r_economic` | Always | Outcome improvement vs. rules counterfactual |
| `r_fault_precision` | If fault evidence visible | Correct action for mechanical/thermal signal |
| `r_message_handling` | If customer message present | Appropriate escalation given job deadline |
| `r_unnecessary_action` | Always | −0.4 penalty for intervening when WAIT was optimal |
| `r_novel_fault` | If anomaly flag visible | Diagnostic response to compound fault |

Evidence-gated rewards fire only when the supporting signal is visible — rewarding **reasoning from evidence**, not lucky guesses.

### OpenEnv compliance

- REST API via FastAPI (`/env/reset`, `/env/step`) — [`server/app.py`](server/app.py)
- Manifest at [`openenv.yaml`](openenv.yaml)
- Dockerfile: `CMD uvicorn submission.server.app:app --host 0.0.0.0 --port 7860`
- **HF Space:** https://huggingface.co/spaces/sikkaBolega/printfarm-env

---

## 3. Training Pipeline

### Architecture

```
Qwen/Qwen2.5-3B-Instruct
    ↓  SFT warm-start (Unsloth QLoRA, 4 epochs, 125 examples)
sikkaBolega/printfarm-sft-adapter
    ↓  Unsloth merge → fp16
sikkaBolega/printfarm-sft-merged
    ↓  GRPO (TRL GRPOTrainer + standard PEFT LoRA, 500 steps, L4 GPU)
sikkaBolega/printfarm-grpo-adapter  ← submission model
```

**Key design choices:**
- Standard PEFT LoRA (not Unsloth `get_peft_model`) for GRPO — avoids Unsloth 2026.4.x `matmul_lora` dtype bug in GRPO rollouts
- Clean merge: reload original base in fp16 + adapter, not `merge_and_unload()` on 4-bit model
- Balanced SFT data: 125 examples across all 9 scorable actions (round 2 fix for action collapse)

### SFT Dataset (Round 2 — Balanced)

| Action | Count | Source |
|---|---|---|
| `RUN_DIAGNOSTIC` | 20 | Capped from oracle rollouts |
| `WAIT` | 20 | Capped from oracle rollouts |
| `ASSIGN_JOB` | 15 | Oracle rollouts |
| `REQUEST_SPOOL_SWAP` | 10 | Oracle rollouts |
| `CANCEL_JOB` | 12 | Hand-authored synthetic |
| `PAUSE_JOB` | 12 | Hand-authored synthetic |
| `RESUME_JOB` | 12 | Hand-authored synthetic |
| `DISPATCH_TICKET` | 12 | Hand-authored synthetic |
| `REQUEST_MAINTENANCE` | 12 | Hand-authored synthetic |
| **Total** | **125** | |

### Running the Training

The full pipeline runs on HuggingFace Jobs. A re-runnable demo is in the Colab notebook.

```bash
# Step 1 — SFT (~12 min, T4 via HF Jobs)
python -m submission.training.train_sft_hf \
  --data data/sft_warm.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --epochs 4 --batch_size 4 \
  --hub_repo sikkaBolega/printfarm-sft-adapter

# Step 2 — Merge SFT adapter to fp16
python -m submission.training.merge_sft_hf \
  --adapter_repo sikkaBolega/printfarm-sft-adapter \
  --merged_repo sikkaBolega/printfarm-sft-merged

# Step 3 — GRPO (500 steps, ~6 hrs, L4)
python -m submission.training.train_grpo_hf \
  --model Qwen/Qwen2.5-3B-Instruct \
  --init_model sikkaBolega/printfarm-sft-merged \
  --max_steps 500 --n_generations 8 \
  --hub_adapter_repo sikkaBolega/printfarm-grpo-adapter \
  --hub_merged_repo sikkaBolega/printfarm-grpo-merged
```

**Colab notebook** (re-runnable, ~20 min on T4): [notebooks/PrintFarmEnv_Training.ipynb](notebooks/PrintFarmEnv_Training.ipynb)
- Mini SFT (1 epoch) + Mini GRPO (30 steps) — proves the pipeline end-to-end
- Inference demo loading the full trained model from HF Hub

---

## 4. Results

### Training Metrics (Round 2 GRPO — L4, 500 steps)

Live monitor from the training run (logged every 10 steps, 8-prompt eval):

| Step | reward_avg | tag | parse | Notes |
|------|-----------|-----|-------|-------|
| 1 | +0.000 | 100% | 88% | Warmup |
| 10 | +0.140 | 100% | 100% | LR warming up |
| 20 | +0.133 | 100% | 100% | |
| 30 | +0.103 | 100% | 100% | |
| 40 | +0.061 | 100% | 100% | Mid-warmup dip |
| 50 | +0.011 | 100% | 100% | |
| 60 | +0.037 | 100% | 100% | Recovery begins |
| 70 | +0.117 | 100% | 100% | |
| 80 | +0.110 | 100% | 100% | Holding above baseline |

All checkpoints: `HEALTHY`. Parse rate 100% from step 10 onward.

**Action diversity** (new actions appearing vs. round 1 collapse):
- Step 1: WAIT, ASSIGN_JOB, RUN_DIAGNOSTIC, **DISPATCH_TICKET**, **REQUEST_MAINTENANCE**
- Step 50: REQUEST_MAINTENANCE
- Step 60: DISPATCH_TICKET
- Round 1 (0 balanced data): only ASSIGN_JOB for 10/11 eval episodes

### Reward Plots

![Reward curve](plots/reward_curve.png)

![Component decomposition](plots/component_decomposition.png)

![Action distribution](plots/action_distribution.png)

![Before/after generations](plots/before_after_generations.png)

### Baseline Comparison

| Model | Avg Reward (11 seeds) | Action distribution |
|---|---|---|
| dp_random | −0.127 | Uniform |
| dp_rules | −0.018 | Rule-based |
| dp_wait | −0.018 | Always WAIT |
| Round 1 GRPO | −0.000 | 10/11 ASSIGN_JOB (collapse) |
| **Round 2 GRPO** | **in progress** | WAIT, RUN_DIAGNOSTIC, ASSIGN_JOB, DISPATCH_TICKET, REQUEST_MAINTENANCE |

### Live Environment

| Endpoint | URL |
|---|---|
| Space | https://huggingface.co/spaces/sikkaBolega/printfarm-env |
| Reset | `POST https://sikkaBolega-printfarm-env.hf.space/env/reset` |
| Step | `POST https://sikkaBolega-printfarm-env.hf.space/env/step` |
| Docs | https://sikkaBolega-printfarm-env.hf.space/docs |

---

## 5. Why It Matters

**The structured-only gap is real.** Every production dispatching system we surveyed treats free-text as out-of-scope — it's routed to humans via support tickets. But text signals are often the *earliest* fault indicators: operators write shift notes before a mechanical failure registers in telemetry. An LLM dispatcher that can read those notes can act a full shift earlier than a rules engine.

**The decision-point sub-skill pattern generalizes.** This env demonstrates a training primitive applicable to any operations domain with mixed structured/unstructured signals: warehouse logistics, hospital bed management, network operations centers. The key insight is to isolate the decision moments where an LLM adds unique value and train only there — not on every timestep.

**The action collapse lesson.** Round 1 collapse (model defaulting to ASSIGN_JOB) proved the environment is correctly punishing — the RL algorithm found the SFT prior's safe harbor. Round 2 fixed this with balanced SFT data across all 9 actions, and new actions (DISPATCH_TICKET, REQUEST_MAINTENANCE) are appearing in training probes by step 60.

---

## 6. Quick Start

```bash
pip install -e .
uvicorn submission.server.app:app --host 0.0.0.0 --port 7860
# → POST http://localhost:7860/env/reset
# → POST http://localhost:7860/env/step
```

```bash
# Run eval against trained adapter
python -m submission.eval.eval_model \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter sikkaBolega/printfarm-grpo-adapter \
  --n_episodes 11

# Run baselines
python -m submission.eval.run_baselines
```

---

## 7. Repository Layout

```
./
├── env/
│   ├── env.py                   # PrintFarmEnvironment (discrete-event sim)
│   ├── decision_point.py        # DecisionPointEnv + round-robin signal targeting
│   ├── models.py                # Pydantic models (FarmObservation, FarmAction, ...)
│   ├── customer_messages.py     # Unstructured signal generator
│   ├── operator_notes.py        # Unstructured signal generator
│   └── novel_faults.py          # Compound-fault generator
├── rewards/
│   ├── composite.py             # Aggregates all 6 reward components
│   ├── format_reward.py         # <action> tag parsing
│   ├── economic_reward.py       # Outcome delta vs. counterfactual
│   ├── fault_precision.py       # Evidence-gated fault response
│   ├── message_handling.py      # Customer message triage
│   ├── unnecessary_action.py    # Over-intervention penalty
│   └── novel_fault.py           # Novel anomaly response
├── training/
│   ├── train_sft_hf.py          # SFT warm-start (Unsloth QLoRA, HF Jobs)
│   ├── train_grpo_hf.py         # GRPO training (TRL + standard PEFT LoRA)
│   ├── merge_sft_hf.py          # Merge SFT adapter → fp16 model
│   ├── build_sft_dataset.py     # Oracle rollout dataset builder
│   ├── build_sft_dataset_v2.py  # Round 2: balanced 9-action dataset
│   └── rollout.py               # Decision-prompt generation
├── shared/
│   ├── prompt.py                # SYSTEM_PROMPT (single source of truth)
│   ├── serialize.py             # FarmObservation → JSON
│   ├── obs_formatter.py         # JSON → human-readable text
│   └── parse_action.py          # <action> tag parser
├── eval/
│   ├── eval_model.py            # LLM checkpoint evaluator (11 fixed seeds)
│   ├── run_baselines.py         # Baseline evaluator
│   └── baselines.json           # Pre-computed baseline results
├── data/
│   └── sft_warm.jsonl           # 125-example balanced SFT dataset
├── notebooks/
│   └── PrintFarmEnv_Training.ipynb  # Colab training notebook (re-runnable)
├── plots/                       # Training evidence plots
├── server/
│   └── app.py                   # FastAPI (OpenEnv REST API)
├── tests/
│   └── test_smoke.py            # Smoke tests
├── openenv.yaml                 # OpenEnv manifest
├── Dockerfile
└── requirements.txt
```


---

## Authors

Built for the OpenEnv Hackathon 2025.
