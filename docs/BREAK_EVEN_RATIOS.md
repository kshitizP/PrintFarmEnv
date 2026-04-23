# PrintFarmEnv — Break-Even Ratios

Extracted from SIMULATOR_SPEC.md §5. These are the economic design invariants that
justify the dollar constants chosen in `economics.py`.

## Dollar constants summary

| Item | Value |
|---|---|
| Filament PLA | $0.020/g |
| Filament PETG | $0.025/g |
| Filament ABS | $0.030/g |
| Electricity (PRINTING/MAINTENANCE) | `watts × 2.5e-6` $/step |
| Electricity (WARMING_UP) | `1.5 × watts × 2.5e-6` $/step |
| Electricity (PAUSED agent) | `1.0 × watts × 2.5e-6` $/step |
| Electricity (PAUSED_RUNOUT) | `0.5 × watts × 2.5e-6` $/step |
| Electricity (IDLE/OFFLINE) | $0 |
| Labor — junior | $0.30/ticket-step ($18/hr) |
| Labor — senior | $0.47/ticket-step ($28/hr) |
| Labor — lead | $0.67/ticket-step ($40/hr) |
| Amortization | `amortization_per_hour / 60` $/step (PRINTING only) |
| SLA miss (fixed) | −$50 (once per job, first overdue step) |
| SLA late fee | −$5/step (while overdue) |
| SLA cap | 80% of `price_usd` |
| WAIT cost | −$0.10/step |
| Invalid action | −$0.20 |
| Unnecessary RUN_DIAGNOSTIC | −$0.50 |
| RUN_DIAGNOSTIC (fault found) | +$2.00 (net +$1.50) |
| Catastrophic failure | −$250 + revenue clawback |
| OVERRIDE_OPERATOR | −$0.10 |

## Break-even ratios

| Question | Formula | Threshold | Design intent |
|---|---|---|---|
| WAITs vs. avoiding a $250 crash | 250 / 0.10 | **2,500 WAIT steps** | WAIT is nearly free vs. catastrophic risk; agent should not be penalised for caution |
| Unnecessary vs. caught diagnostics | 2.00 / 0.50 | **4:1 tolerance** | Agent should diagnose when P(fault) > 20%; matches practical threshold |
| Late completion vs. cancellation | −(50 + 5k) capped at 0.8×price vs. −scrap | **Late always dominates** | SLA cap guarantees partial revenue; agent should push through, not cancel |
| Crash vs. premium job revenue | −$250 vs. ~$75 (priority-3 job) | **~3 jobs** | Safety is cheap — agent should sacrifice ~3 jobs' revenue to prevent one crash |
| Diagnostic bonus vs. catastrophe avoided | +$2 signal vs. −$250 real cost | **125:1** | Bonus is a pedagogy signal; the real incentive is downstream loss avoidance |

## Normalised score formula

```
score = clamp((profit - naive_floor) / (clairvoyant_ceiling - naive_floor) - action_penalty, 0.001, 0.999)
```

Where:
- `naive_floor` ≈ naive-greedy mean profit − buffer
- `clairvoyant_ceiling` ≈ clairvoyant-greedy mean profit + buffer
- `action_penalty` = 0.005×failed_actions + 0.002×wasted_waits + 0.003×unnecessary_diags

## Calibrated floor/ceiling values (2026-04-22, 20 episodes each, seed=42)

| Task | Naive mean | Clairvoyant mean | Floor | Ceiling |
|---|---|---|---|---|
| task_0_1 | +$32.8 | +$72.2 | $0 | $100 |
| task_0_2 | −$46.0 | +$4.9 | −$100 | $30 |
| task_0_3 | +$111.0 | +$109.3 | $70 | $130 |
| task_1 | +$123.2 | +$175.3 | $50 | $220 |
| task_2 | −$166.0 | +$26.6 | −$220 | $60 |
| task_3 | −$142.0 | +$5.9 | −$200 | $50 |
| task_4 | −$283.4 | +$269.4 | −$360 | $310 |
| task_5 | −$553.0 | +$448.2 | −$700 | $530 |
