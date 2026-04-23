# Reward-Hacking Audit

> "Do not optimize a reward you have not tried to break yourself first."
> — Hackathon FAQ #57

This document records every adversarial policy the team ran against the
reward function before training, the component that caught each exploit,
and the invariant test that enforces it.

All tests live in [`tests/test_reward_hacking.py`](../tests/test_reward_hacking.py) and are part of the CI suite.

---

## Reward components (multi-layer, FAQ #13 guidance)

| Component | Source | Sign | Notes |
|---|---|---|---|
| **Action delta** | `economics.py → _dispatch_action` | ± | Revenue / cost of each dispatcher decision |
| **Labor cost** | `economics.py → op_labor_total` | − | Subtracted every step for each active operator |
| **Physics delta** | `env.py → _tick_physics` | ± | Revenue accrual while printing; scrap on failure |
| **SLA penalty** | `economics.py → sla_penalty_this_step` | − | −$50 fixed + −$5/step past deadline (capped at 80% job price) |
| **WAIT cost** | `economics.ACTION_REJECT_COST` | − | −$0.10 per WAIT action |
| **Catastrophic failure** | `economics.CATASTROPHIC_COST` | − | −$250 on unmitigated fatigue collapse |
| **Diagnostic base cost** | `economics.DIAGNOSTIC_BASE_COST` | − | −$0.50 per RUN_DIAGNOSTIC |
| **Diagnostic bonus** | `economics.DIAGNOSTIC_BONUS` | + | +$2.00 when diagnostic finds a real fault |
| **Override cost** | `economics.OVERRIDE_COST` | − | −$0.10 per OVERRIDE_OPERATOR |
| **Scrap + clawback** | `economics.scrap_cost` | − | On CANCEL_JOB: material cost + revenue clawback |

The **grader score** (`obs.reward`) additionally normalises episode P&L
against the calibrated naive floor and clairvoyant ceiling, so the
training signal is robust to task length differences.

---

## Adversarial attempts & how each was caught

### 1. WAIT spam
**Attempt:** always emit `{"action":"WAIT"}` to avoid risk and collect passive revenue.

| Catch mechanism | Detail |
|---|---|
| `WAIT_COST` | −$0.10/step accumulates immediately |
| SLA penalties | Jobs in queue miss deadlines → −$50 fixed + −$5/step each |
| No revenue | Printers sit IDLE; zero printing revenue |

**Result:** episode reward well below honest naive-greedy (test `test_wait_spam_stays_below_honest_baseline`).

---

### 2. Diagnostic bomb
**Attempt:** run `RUN_DIAGNOSTIC` on every printer every step, even healthy ones.

| Catch mechanism | Detail |
|---|---|
| `DIAGNOSTIC_BASE_COST` | −$0.50 per call on a printer with no fault → net −$0.50 |
| Bonus only on real faults | `DIAGNOSTIC_BONUS` (+$2.00) fires only when a fault is found |

**Result:** cumulative step reward goes negative within a few steps (test `test_diagnostic_bomb_is_unprofitable`).

---

### 3. Sensor ignorer
**Attempt:** assign jobs blindly, never respond to ERROR or PAUSED_RUNOUT states.

| Catch mechanism | Detail |
|---|---|
| `CATASTROPHIC_COST` | −$250 when fatigue reaches 10 with no maintenance queued |
| SLA penalties | Runout-paused jobs miss deadlines |
| Physics damage | Unresolved ERRORs block revenue and may cascade |

**Result:** scores below honest play once catastrophic failure fires (test `test_sensor_ignorer_scores_below_honest`).

---

### 4. Cancel everything
**Attempt:** cancel every job immediately to eliminate SLA and catastrophic risk.

| Catch mechanism | Detail |
|---|---|
| Scrap cost | Material filament weight × cost rate charged on cancel |
| Revenue clawback | `accrued_revenue` subtracted from P&L on cancel |
| No completions | Zero completion bonuses; grader score stays at floor |

**Result:** cumulative step reward ≤ 0 and grader score below naive (test `test_cancel_all_is_worse_than_honest`).

---

### 5. Operator queue flood
**Attempt:** dispatch maintenance tickets to every operator every step to... do what exactly? (Testing whether labor cost is real.)

| Catch mechanism | Detail |
|---|---|
| `op_labor_total` | Operator labor is subtracted from `step_reward` every step |
| Redundant tickets rejected | `ACTION_REJECT_COST` on tickets the operator refuses |
| Printers held in MAINTENANCE_QUEUED | Blocks revenue during unnecessary service |

**Result:** scores at or below honest (test `test_operator_flood_is_unprofitable`).

---

### 6. Pause-resume cycling
**Attempt:** rapidly toggle every printing job between PAUSED and PRINTING to exploit any revenue-leak timing gap.

| Catch mechanism | Detail |
|---|---|
| Revenue accrual | Only accrues on `PrinterState.PRINTING` steps in `_tick_physics` |
| PAUSED steps earn nothing | Nozzle stays hot but no revenue increments |

**Result:** no extra revenue extracted; scores match or trail honest play (test `test_pause_resume_cycle_no_extra_revenue`).

---

### 7. Override operator spam
**Attempt:** immediately override every queued ticket to avoid labor cost.

| Catch mechanism | Detail |
|---|---|
| `OVERRIDE_COST` | −$0.10 per override |
| Printers never serviced | Faults persist; catastrophic failure risk grows |
| Grader penalises unresolved faults | Task graders track unresolved fault rate |

**Result:** scores at or below honest (test `test_override_spam_stays_below_honest`).

---

## Summary table

| Exploit | Components that blocked it | Test |
|---|---|---|
| WAIT spam | `WAIT_COST`, SLA penalties | `test_wait_spam_stays_below_honest_baseline` |
| Diagnostic bomb | `DIAGNOSTIC_BASE_COST`, conditional bonus | `test_diagnostic_bomb_is_unprofitable` |
| Sensor ignorer | `CATASTROPHIC_COST`, SLA penalties | `test_sensor_ignorer_scores_below_honest` |
| Cancel all | Scrap cost, revenue clawback | `test_cancel_all_is_worse_than_honest` |
| Operator flood | Labor deduction, `ACTION_REJECT_COST` | `test_operator_flood_is_unprofitable` |
| Pause-resume cycling | Revenue only on PRINTING steps | `test_pause_resume_cycle_no_extra_revenue` |
| Override spam | `OVERRIDE_COST`, unserviced printers | `test_override_spam_stays_below_honest` |

---

## How this helps during GRPO training

The same components that blocked these hand-crafted adversarial policies
are what the optimizer will encounter during GRPO rollouts. The multi-layer
design (Help Guide §7: "use multiple independent reward functions") means
a model that learns to exploit one component will immediately encounter
pushback from another.

During training, watch for:
- **rising `reward/format_reward_fn` with flat `reward/env_reward_fn`** → model learned JSON format but not good decisions
- **`reward/anti_wait_spam_fn` going more negative** → WAIT-spam emerging under RL pressure
- **generation sampling showing repetitive actions** → reward hacking in progress; roll back or add shaping

See Help Guide §15 and FAQ #43 for monitoring guidance.
