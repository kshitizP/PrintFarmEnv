# PrintFarm Dispatcher — OpenEnv Submission

> **OpenEnv Hackathon 2025** · Reinforcement learning for LLM-driven 3D print-farm operations

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

The agent chooses one `FarmAction` per step from ten action types: `ASSIGN_JOB`, `CANCEL_JOB`, `PAUSE_JOB`, `RESUME_JOB`, `RUN_DIAGNOSTIC`, `DISPATCH_TICKET`, `REQUEST_SPOOL_SWAP`, `REQUEST_MAINTENANCE`, `OVERRIDE_OPERATOR`, `WAIT`.

### Three unstructured signal augmentations

**Operator notes** (~40% of steps) — randomly drawn from a corpus of 20 templated notes covering spool state, mechanical symptoms, layer adhesion issues, and shift handover remarks. The agent must parse urgency from prose.

**Customer messages** (~20% of steps) — drawn from 15 templates covering urgency escalation, cancellation requests, and deadline ambiguity. Correct action depends on whether the relevant job is late (computable from queue state) vs. whether the customer's claim is warranted.

**Novel anomaly flags** (~15% of steps when printers are PRINTING) — procedurally generated compound faults ("MECHANICAL_VIBRATION + LAYER_SHIFT at 67% progress on P3"). These have no matching named failure mode in the rules engine. Correct response is typically `RUN_DIAGNOSTIC` or `REQUEST_MAINTENANCE`.

### Decision-point sub-skill RL

Training LLMs on full 40-step episodes is expensive. Instead, we use a **decision-point wrapper** (`DecisionPointEnv`):

1. Roll forward with the rules agent until an unstructured signal appears.
2. Pause and present the observation to the LLM.
3. Execute the LLM's action; measure reward as **delta vs. a rules-only counterfactual** on the same seed.

This focuses all gradient signal on the moments where the LLM can add unique value. Prompts are generated with **round-robin signal targeting** (notes → messages → anomalies → structured, cycling) so all four signal types get equal training coverage.

### Six reward components

| Component | Signal | What it measures |
|---|---|---|
| `r_format` | Always | `<action>…</action>` tag present and parseable (+0 / -0.1) |
| `r_economic` | Always | Outcome improvement vs. rules counterfactual (throughput, waste) |
| `r_fault_precision` | If fault evidence visible | Correct action for mechanical/thermal signal in observation |
| `r_message_handling` | If customer message present | Appropriate escalation given job deadline and message urgency |
| `r_unnecessary_action` | Always | Penalty for intervening when rules + WAIT was already optimal |
| `r_novel_fault` | If anomaly flag visible | Diagnostic response to compound fault with no named classification |

Evidence-gated rewards (`r_fault_precision`, `r_novel_fault`) fire only when the supporting signal is visible in the observation — rewarding **reasoning from evidence**, not lucky guesses.

### OpenEnv compliance

- REST API via FastAPI (`/env/reset`, `/env/step`) — see [`server/app.py`](server/app.py)
- Manifest at [`openenv.yaml`](openenv.yaml)
- Docker image: `CMD uvicorn submission.server.app:app --host 0.0.0.0 --port 7860`

---

## 3. Results

### Baselines (pre-training)

| Policy | Avg reward | Notes |
|---|---|---|
| Full-ep rules | **+0.104** | Best structured baseline; can't read text |
| Full-ep random | +0.025 | Random valid actions |
| Full-ep wait | +0.038 | Always WAIT |
| **DP rules** | **-0.018** | At unstructured decision points, rules ≈ WAIT |
| DP random | -0.127 | Random AgentActions, format penalty |
| DP wait | -0.018 | Format penalty only |

The critical result: `dp_rules ≡ dp_wait` (both -0.018, difference is noise). At unstructured decision points, the rules engine has **no signal** — it returns WAIT by default. The entire -0.018 is the `r_format` penalty (-0.1) scaled by the fraction of episodes where the format reward applies.

A trained GRPO model that:
1. Learns to output `<action>` tags reliably (→ `r_format = 0`)
2. Reads text signals and acts on them (→ positive `r_message_handling`, `r_novel_fault`)

...should exceed `dp_rules` by > 0.1 reward units — a measurable, interpretable lift attributable entirely to reading comprehension.

### Training reward curve

> *Generated after Phase C (real GRPO run). Run `python submission/scripts/make_plots.py --run grpo_runs/overnight_v1` to regenerate.*

![Reward curve](plots/reward_curve.png)

### Reward component decomposition

![Component decomposition](plots/component_decomposition.png)

### Action distribution (before vs. after training)

![Action distribution](plots/action_distribution.png)

### Sample generations: before vs. after

![Before/after generations](plots/before_after_generations.png)

### W&B run

> Link to be added after Phase C training run.

---

## 4. Why It Matters

**The structured-only gap is real.** Every production dispatching system we surveyed treats free-text as out-of-scope — it's routed to humans via support tickets. But text signals are often the *earliest* fault indicators: operators write shift notes before a mechanical failure registers in telemetry. An LLM dispatcher that can read those notes can act a full shift earlier than a rules engine.

**The decision-point sub-skill pattern generalizes.** This env demonstrates a training primitive applicable to any operations domain with mixed structured/unstructured signals: warehouse logistics, hospital bed management, network operations centers. The key insight is to isolate the decision moments where an LLM adds unique value and train only there — not on every timestep of a long episode.

**The ablation is honest.** We explicitly show that rules performance does *not* degrade on the augmented env (it stays at +0.104 on full episodes, -0.018 at unstructured DPs) — the unstructured signals are additive difficulty, not a broken environment. The LLM has to earn its reward by reading.

---

## Quick Start

### Docker

```bash
docker build -t printfarm-env .
docker run -p 7860:7860 printfarm-env
# → POST http://localhost:7860/env/reset
# → POST http://localhost:7860/env/step
```

### Local (dev)

```bash
pip install -e .
uvicorn submission.server.app:app --host 0.0.0.0 --port 7860
```

### Tests

```bash
pytest submission/tests/test_smoke.py -v   # 12 smoke tests, ~30s
```

### Baselines

```bash
python -m submission.eval.run_baselines
# → submission/eval/baselines.json
```

### GRPO training (Colab T4)

```bash
# smoke test first (CPU, ~2min)
python -m submission.training.train_grpo --smoke

# real run
python -m submission.training.train_grpo \
  --model google/gemma-3-1b-it \
  --max_steps 200 \
  --n_generations 4 \
  --save_steps 25 \
  --out grpo_runs/overnight_v1
```

### Plots

```bash
python submission/scripts/make_plots.py \
  --run grpo_runs/overnight_v1 \
  --out submission/plots
```

---

## Repository Layout

```
submission/
├── env/
│   ├── env.py                  # PrintFarmEnvironment (discrete-event sim)
│   ├── decision_point.py       # DecisionPointEnv + round-robin signal targeting
│   ├── models.py               # Pydantic models (FarmObservation, FarmAction, ...)
│   ├── customer_messages.py    # Unstructured signal generator
│   ├── operator_notes.py       # Unstructured signal generator
│   └── novel_faults.py         # Compound-fault generator
├── rewards/
│   ├── composite.py            # Aggregates all 6 reward components
│   ├── format_reward.py        # <action> tag parsing
│   ├── economic_reward.py      # Outcome delta vs. counterfactual
│   ├── fault_precision.py      # Evidence-gated fault response
│   ├── message_handling.py     # Customer message triage
│   ├── unnecessary_action.py   # Over-intervention penalty
│   └── novel_fault.py          # Novel anomaly response
├── training/
│   ├── train_grpo.py           # Main GRPO training script (TRL + Unsloth)
│   └── rollout.py              # Decision-prompt generation with round-robin signals
├── shared/
│   ├── prompt.py               # SYSTEM_PROMPT
│   ├── serialize.py            # FarmObservation → text
│   └── parse_action.py         # <action> tag parser → AgentAction
├── server/
│   └── app.py                  # FastAPI app (OpenEnv REST API)
├── eval/
│   ├── run_baselines.py        # Baseline evaluator
│   ├── eval_model.py           # LLM checkpoint evaluator
│   └── baselines.json          # Pre-computed baseline results
├── scripts/
│   └── make_plots.py           # Generate submission plots from training artifacts
├── tests/
│   └── test_smoke.py           # 12-test smoke suite
├── openenv.yaml                # OpenEnv manifest
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Authors

Built for the OpenEnv Hackathon 2025.
