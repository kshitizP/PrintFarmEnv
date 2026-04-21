# PrintFarmEnv — Simulator Specification

**Status:** 🟢 v1.0 — ready to scaffold against
**Version:** 1.0

> **Usage rule:** Tables > prose. 🟡 = judgment call worth re-reading before committing to code. This doc is authoritative — if code and spec disagree, code is wrong.

---

## Section 0 — Glossary & Conventions

| Term | Definition |
|---|---|
| Dispatcher | The AI agent being trained |
| Operator | Simulated human NPC (env-controlled) |
| Oversight | LLM that audits Dispatcher actions (not trained) |
| Step | One tick of env time. **1 step = 1 real minute.** |
| Episode | A full task run from reset to done |
| Ticket | A work order the Dispatcher creates for an Operator |
| Telemetry | What the Dispatcher sees — may be corrupted by failure modes |
| Ground truth | The env's internal state — never directly visible to Dispatcher |
| Profile | A `PrinterProfile` config that parameterises reliability, failure rates, power draw, costs for a printer model |
| Fault | An actively-injected failure-mode instance (at most one per mode per printer at any moment) |

**Time convention.** All durations are in **env steps** unless explicitly labelled `_g` (grams), `_usd` (dollars), `_min` (real minutes), or `_kwh` (kilowatt-hours).

**Single-action convention.** The Dispatcher emits **exactly one action per step**. 🟡 Chose this over batching because (a) matches Round 1, (b) maps cleanly to the LLM chat loop, (c) gives the Oversight Agent a per-step audit trail.

**RNG convention.** One `random.Random` seeded at `reset(seed)`. All stochastic sampling — failure injection, operator success rolls, reliability rolls, latency jitter, anomaly-report generation — draws from this single stream, so episodes are bit-reproducible from seed.

---

## Section 1 — Action Semantics

### Action space (8 actions)

| Action | Preconditions | Effects | Cost ($) | Reject → |
|---|---|---|---|---|
| `ASSIGN_JOB(printer_id, job_id)` | Printer is `IDLE`; job is `PENDING`; `printer.current_material == job.material_required`; `printer.spool_weight_g >= job.weight_required_g` | Printer → `WARMING_UP` (`warmup_remaining=1`); job → `PRINTING`; `printer.current_job_id = job_id` | $0 at dispatch (filament + electricity + amortization accrue while printing) | -$0.20 with `metadata.reason` |
| `CANCEL_JOB(job_id)` | Job is `PENDING`, `PRINTING`, or `PAUSED` | Job → `CANCELLED`; if printing/paused, printer → `IDLE`; **filament consumed so far is billed as scrap**; accrued revenue is clawed back | Scrap: `filament_used_g × filament_$/g` | -$0.20 |
| `RUN_DIAGNOSTIC(printer_id)` | Printer exists | Reveals ground-truth telemetry for **this step only** (overrides sensor corruption in the observation for that printer). Does **not** clear the underlying fault. For persistent faults, escalate via `diagnostic_physical` operator ticket | -$0.50 base; if a real fault is active on that printer: +$2.00 credit (net +$1.50) | None |
| `PAUSE_JOB(printer_id)` | Printer is `PRINTING` | Printer → `PAUSED` (agent-induced, distinct from `PAUSED_RUNOUT`); job → `PAUSED`; nozzle stays hot (electricity continues); no revenue accrual, no fatigue increment | $0 at pause; heating electricity accrues while paused | -$0.20 |
| `RESUME_JOB(printer_id, job_id)` | Either (a) printer is `PAUSED` (agent-paused), OR (b) printer is `IDLE` with `current_job_id` set and the linked job in `PAUSED` (i.e., it just returned from `PAUSED_RUNOUT` via a completed swap). Material matches; `spool_weight_g > 0` | Printer → `PRINTING` (no warmup). If resuming from case (b), set `reliability_penalty_active = True` for the next PRINTING tick. Job → `PRINTING`; `progress_steps` preserved | $0 | -$0.20 |
| `DISPATCH_TICKET(operator_id, ticket_type, target_printer_id, payload)` | Operator on-shift AND `len(operator.queue) < operator.queue_capacity` | Ticket appended to operator's queue; `created_step = t`. Auto-reassigns to lowest-queue qualified on-shift operator if the named operator goes off-shift before pickup | $0 at dispatch; labor billed on pickup | -$0.20 (off-shift, queue full, invalid operator_id) |
| `OVERRIDE_OPERATOR(operator_id, ticket_id, reason)` | Ticket is in operator's queue AND `operator.current_ticket != ticket_id` | Removes ticket from queue; writes reason to Oversight log | -$0.10 | -$0.20 (ticket already in progress) |
| `WAIT` | Always valid | No state change | -$0.10 | Never |

**Why `PAUSE_JOB` exists.** Proactive pause before runout lets the operator swap filament with the nozzle still hot — clean resume, no quality risk. Abrupt `PAUSED_RUNOUT` lets the nozzle cool, so resuming from runout carries a one-time reliability penalty (§2). The agent should watch `spool_weight_g` vs. `weight_required_g - weight_used` and pause proactively when a swap is likely.

Operator ticket types (passed as `ticket_type`): `spool_swap`, `filament_reload_from_stock`, `maintenance_basic`, `maintenance_full_rebuild`, `diagnostic_physical`, `unjam_printer`. Full table in §4.

**Helper sugar.** `REQUEST_SPOOL_SWAP(printer_id, material)` and `REQUEST_MAINTENANCE(printer_id, type)` resolve to `DISPATCH_TICKET` with the env picking the target operator: **on-shift roster, skill ≥ required, smallest queue wins; ties broken by skill descending, then operator_id ascending**. If no operator qualifies, the action is rejected.

### Action validation order

1. Parse action JSON → reject if malformed (`metadata.reason = "malformed_action"`)
2. Check preconditions → reject with `metadata.reason = "<specific>"`
3. Apply effects
4. Emit observation (see §7)

### Resolved open questions

- **`CANCEL_JOB` scrap cost:** Yes. `(weight_required × progress/print_time_steps) × filament_$/g` is billed. Real extruded filament can't be reclaimed.
- **Multiple actions per step:** No. Exactly one.
- **Ticket against non-existent operator/printer:** Invalid action, -$0.20.

---

## Section 2 — State Machine

### Printer states

```
IDLE ─► WARMING_UP ─► PRINTING ─┬─► IDLE              (print succeeded)
                                 ├─► ERROR            (stochastic reliability failure)
                                 ├─► PAUSED            (agent-induced via PAUSE_JOB)
                                 ├─► PAUSED_RUNOUT    (spool hit 0, nozzle cools)
                                 └─► OFFLINE          (fatigue = 10, catastrophic)

PAUSED ──(RESUME_JOB)──► PRINTING                    (clean resume, nozzle hot)
PAUSED_RUNOUT ──(operator spool_swap)──► IDLE ──(RESUME_JOB, reliability penalty)──► PRINTING
IDLE ─► MAINTENANCE_QUEUED ─► MAINTENANCE ─► IDLE
ERROR ──(operator unjam)──► IDLE
OFFLINE ──(offline_remaining == 0)──► IDLE
```

**New in Round 2:**

- `PAUSED` — agent-induced pause via `PAUSE_JOB`. Nozzle kept hot (electricity accrues); no revenue, no fatigue, no amortization. Used for proactive swap coordination: agent pauses before `spool_weight_g` hits 0, dispatches a swap ticket, operator does the swap cleanly, agent resumes.
- `PAUSED_RUNOUT` — env-induced pause when spool hits 0. Nozzle cools over ≥5 steps. `RESUME_JOB` after a `spool_swap` ticket completes carries a **one-time reliability penalty**: the first PRINTING tick rolls at `reliability × 0.85` (15% extra failure chance from reheating / layer-adhesion risk). This creates the incentive to pause proactively.
- `MAINTENANCE_QUEUED` — 🟡 first-class state. Entered when a maintenance ticket is dispatched to an `IDLE` printer. Stays in this state until the operator picks up AND the `consecutive_idle_steps >= 3` cooldown gate is satisfied.

### Transition table

| Current state | Trigger | New state | Reward Δ ($) | Notes |
|---|---|---|---|---|
| IDLE | `ASSIGN_JOB` (valid) | WARMING_UP | $0 | `warmup_remaining = 1`, `current_job_id = job_id` |
| IDLE | env tick | IDLE | $0 | `consecutive_idle_steps += 1` |
| WARMING_UP | env tick, `warmup_remaining > 1` | WARMING_UP | -(heating_electricity) | ≈ -$(1.5 × W × 2.5e-6) ≈ -$0.0007 at 180 W |
| WARMING_UP | env tick, `warmup_remaining == 1` | PRINTING (if job) / IDLE (if no job) | $0 | — |
| PRINTING | env tick, reliability roll passes | PRINTING | +step_revenue − filament_burn − electricity − amortization | `step_revenue = price / print_time_steps` |
| PRINTING | env tick, reliability roll fails (P = 1 − reliability) | ERROR | -scrap_cost − clawback(accrued_revenue) | Job → `PENDING`, `progress_steps = 0`, printer waits for `unjam_printer` ticket |
| PRINTING | `PAUSE_JOB` | PAUSED | -(heating_electricity for step t) | Job → `PAUSED`, `progress_steps` preserved, nozzle stays hot |
| PRINTING | `spool_weight_g` hits 0 mid-step | PAUSED_RUNOUT | $0 | Job → `PAUSED`, `progress_steps` preserved, nozzle begins cooling |
| PRINTING | `fatigue_level` reaches 10 | OFFLINE | -$250 − clawback | `offline_remaining = 10`, job → `FAILED` |
| PAUSED | env tick | PAUSED | -(heating_electricity) | Indefinite; agent decides when to `RESUME_JOB` or `CANCEL_JOB` |
| PAUSED | operator completes `spool_swap` (if dispatched during pause) | PAUSED | -op_labor | Spool swapped in place; agent can `RESUME_JOB` |
| PAUSED | `RESUME_JOB` | PRINTING | $0 | Clean resume, no penalty |
| PAUSED_RUNOUT | operator completes `spool_swap` | IDLE | -op_labor | Printer ready; next `RESUME_JOB` carries reliability penalty |
| IDLE (post-PAUSED_RUNOUT swap) | `RESUME_JOB` | PRINTING | $0 | **First PRINTING tick rolls at `reliability × 0.85`**; reverts to normal on tick 2+ |
| ERROR | operator completes `unjam_printer` | IDLE | -op_labor | Job stays `PENDING` and can be reassigned |
| IDLE | `DISPATCH_TICKET(maintenance_basic\|full)` | MAINTENANCE_QUEUED | $0 | `outstanding_ticket_id` set |
| MAINTENANCE_QUEUED | operator picks up AND `consecutive_idle_steps ≥ 3` | MAINTENANCE | -op_labor (starts billing) | Cooldown gate satisfied |
| MAINTENANCE_QUEUED | operator picks up AND cooldown unmet | MAINTENANCE_QUEUED | -op_labor (operator idle-waits) | 🟡 Operator is billed while waiting on cooldown. Dispatcher's incentive: dispatch maintenance only when the printer is already cold |
| MAINTENANCE | env tick, `remaining > 0` | MAINTENANCE | $0 | `warmup_remaining -= 1` (reused counter) |
| MAINTENANCE | env tick, `remaining == 0` (basic, 3 steps) | IDLE | $0 | `fatigue_level = 0`, `reliability = min(base + 0.05, 1.0)`, `maintenance_due_in = 50`, `bed_level_drift_counter = 0` |
| MAINTENANCE | env tick, `remaining == 0` (full, 8 steps) | IDLE | $0 | All of the above + `reliability = profile.reliability_base`, all active faults cleared |
| OFFLINE | env tick, `offline_remaining > 0` | OFFLINE | $0 | — |
| OFFLINE | env tick, `offline_remaining == 0` | IDLE | $0 | `fatigue_level = 0` |
| any active | `CANCEL_JOB` on current job | IDLE | -scrap_cost − clawback | Printer freed immediately |

**Revenue accrual.** Job price accrues evenly: `step_revenue = price_usd / print_time_steps` credited each successful `PRINTING` tick. On `FAILED` / `CANCELLED`, accrued revenue is **clawed back** (the customer isn't paying for scrap). On `COMPLETED`, the final tick closes out the full price.

### Resolved open questions

- **`PAUSED_RUNOUT` + `MAINTENANCE_QUEUED` simultaneously:** No. Mutually exclusive by construction — `MAINTENANCE_QUEUED` requires entering from `IDLE`.
- **Swap ticket picked up but printer went `OFFLINE`:** Operator completes as a no-op with `status = REJECTED, reason = "printer_offline"`. Labor is still billed (the operator walked over). The scheduled new spool is returned to inventory.
- **When does fatigue increment:** Once per `PRINTING` tick only. Never during warmup or maintenance.

---

## Section 3 — Failure Mode Specification

### Arrival rates and durations

Rates are **per-printer, per-step probabilities** (approximating Poisson for small p). Starting budget: ~`4e-3` per mode per printer per step across all modes, scaled per mode by how often it shows up in real OctoPrint issue reports. 🟡 Re-calibrate after first end-to-end run so expected faults per episode ∈ [1, 3] on a 5-printer, 60-step task.

| Mode ID | Trigger | Symptom (corrupted telemetry) | Ground truth | Duration (steps) | Clears on | Interactions |
|---|---|---|---|---|---|---|
| `thermistor_open` | Bernoulli(1e-3) per step | `hotend_temp: 0` persistent | Temp unchanged | Exp(μ=8) | `diagnostic_physical` ticket OR duration expiry | Blocks naive `ASSIGN_JOB`; tempts spurious maintenance tickets |
| `thermistor_short` | Bernoulli(5e-4) | `hotend_temp: 400+` persistent | Temp unchanged | Exp(μ=8) | `diagnostic_physical` OR expiry | Tempts mistaken `CANCEL_JOB` on a healthy print |
| `filament_sensor_false_runout` | Bernoulli(2e-3), only if `spool_weight_g > 200` | `state: PAUSED_RUNOUT` (corrupted); the actual job stays `PRINTING` in ground truth | Printing normally | Exp(μ=5) | Operator reviewing the ticket at the printer rejects it (they see real spool weight) OR expiry | Wastes operator time on a phantom swap |
| `filament_sensor_missed_runout` | Bernoulli(3e-4) | `state: PRINTING`, spool weight frozen at last-known > 0 | Spool physically empty | Until stall detection (ground-truth: extrusion yields no progress for 2 steps → job FAILS) | The physical stall unambiguously | Catastrophic: -$250 + scrap if not caught via cross-check |
| `webcam_freeze` | Bernoulli(3e-3) | `webcam_hash` identical for ≥ 3 steps | — | Exp(μ=10) | `RUN_DIAGNOSTIC` (fetches live frame) OR expiry | Masks visual evidence that would catch other failures |
| `klipper_mcu_disconnect` | Bernoulli(5e-4) | All fields frozen; `telemetry_ts` stale (stops advancing) | — | Exp(μ=6) | Self-clear (most common) OR `diagnostic_physical` | Easiest mode to catch — timestamp stops advancing |
| `progress_drift` | Bernoulli(1e-3), only on jobs with `print_time_steps > 8` | `progress_steps` stuck at value X for ≥ 3 steps while `state: PRINTING`; temps nominal | Printing fine (slightly slower than reported) | Exp(μ=6) | `RUN_DIAGNOSTIC` OR the job actually completes | Confusable with genuine stall; tempts spurious `CANCEL_JOB` |
| `fan_rpm_ghost` | Bernoulli(2e-3) | `fan_rpm: 0` | Fan spinning | Until printer returns to IDLE | — | **Low-severity trap** — correct policy is to ignore. Penalises over-reactive dispatchers |
| `bed_level_drift` | Deterministic: `bed_drift_counter += 0.0002` per PRINTING step per printer; triggers when `>= 1.0` | Early-step failures (reliability roll fails within first 2 steps of PRINTING, repeatedly, on the same printer) | — | Persists until `maintenance_basic` ticket completes | `maintenance_basic` OR `maintenance_full_rebuild` | Only detectable as a pattern across multiple failed prints — rewards bookkeeping |

Arrival rates are further modulated by `profile.failure_rate_multipliers[mode]`; task configs can override via `stochastic_failure_rates`.

### Cross-failure interactions

- All 9 modes are **independent** per `(printer, mode)`. New arrivals of an already-active mode on the same printer are dropped.
- Different modes **can co-occur on the same printer** (e.g., `webcam_freeze` + `thermistor_open`). This is realistic and forces the Dispatcher to reason about multiple corruptions at once.
- Only `filament_sensor_missed_runout` causes a ground-truth state transition (print fails on physical stall). Every other mode is purely telemetry corruption.

### Operator visibility into failures

- Operators see ground truth when physically at a printer (resolving any ticket).
- Operators do NOT pre-emptively report failures — only on ticket completion.
- 🟡 On ticket completion, an operator emits a `REPORT_ANOMALY` observation with **P = 0.60** (probability they notice something out of place at all). Conditional on emitting, accuracy:

| Condition | P(correctly describes ground truth) |
|---|---|
| Real mechanical fault (`bed_level_drift`, printer needs unjam) | 0.85 |
| Real electrical/sensor fault (`thermistor_*`, `webcam_freeze`, `klipper_mcu_disconnect`, `*_runout`) | 0.40 |
| No fault present | 0.95 (correctly reports "looks fine") |

### Resolved open questions

- **Failure rates scale with fatigue?** No. Fatigue governs mechanical catastrophic risk (→ OFFLINE at 10); sensor failure rates are fixed per profile.
- **Independent across printers?** Yes. No cross-printer correlation in Round 2.
- **Does `RUN_DIAGNOSTIC` clear the fault?** No — only reveals. Clearing requires a physical operator ticket or duration expiry. This keeps `RUN_DIAGNOSTIC` (cheap, agent-only) meaningfully distinct from `diagnostic_physical` (labor-billed operator ticket).

---

## Section 4 — Operator NPC Policy

**Line budget: ≤ 180 LOC in `operators.py`.** Increased from 150 to accommodate memory logic.

### Operator attributes

| Field | Type | Values | Notes |
|---|---|---|---|
| `operator_id` | str | — | Unique in roster |
| `skill_level` | enum | `{junior, senior, lead}` | |
| `shift_window` | `(start_step, end_step)` | within episode | Inclusive start, exclusive end |
| `queue_capacity` | int | 3 / 5 / 7 | junior / senior / lead |
| `current_fatigue` | float | `[0, 1]` | Starts 0 at shift start; grows linearly to 0.8 by shift end |
| `base_latency_steps` | int | 7 / 5 / 4 | junior / senior / lead |
| `labor_rate_usd_per_step` | float | 0.30 / 0.47 / 0.67 | $18/$28/$40 per hour ÷ 60 min/hr |
| `current_ticket` | Ticket? | — | At most one |
| `busy_until` | int | — | `env.current_step + latency` after pickup |
| `queue` | list[Ticket] | FIFO | Capacity-gated |
| `printer_memory` | dict[printer_id → list[VisitRecord]] | Shift-scoped | See below. Reset at `shift_window.start` |

### Operator memory

On every completed ticket, the operator appends a `VisitRecord` to `printer_memory[target_printer_id]`:

```python
VisitRecord = {
    "step": int,
    "ticket_type": str,
    "observed_fault": str | None,   # Ground-truth fault mode observed, if any
    "status": "SUCCESS" | "FAILED" | "REJECTED",
}
```

Memory resets at `shift_window.start` (fresh day, fresh memory) and is discarded at `shift_window.end`.

**Effects of memory on operator behaviour:**

1. **Improved anomaly reporting accuracy on repeat visits.** For the 2nd+ visit to the same printer within a shift, add **+0.10** to `P(correct report)` (capped at 0.95). Rationale: the operator knows the printer's baseline — they can tell what's "off" faster.
2. **Pattern-based recommendations.** If `printer_memory[pid]` contains ≥ 2 VisitRecords with the same `observed_fault` (mechanical class: `bed_level_drift` or repeated `unjam_printer` completions), the operator's `REPORT_ANOMALY` attaches a recommendation string, e.g. `"recommend maintenance_full_rebuild — 3rd jam this shift"`. The Dispatcher receives this as a structured hint.

### Tick loop (pseudocode)

```python
def operator_tick(operator, env, rng):
    if not operator.is_on_shift(env.current_step):
        return  # off-shift — queue persists but no pickups

    # Finalise a ticket whose timer has expired
    if operator.current_ticket is not None and operator.busy_until <= env.current_step:
        _finalise_ticket(operator, env, rng)  # apply effects, bill labor, roll success, maybe emit REPORT_ANOMALY
        operator.current_ticket = None

    if operator.current_ticket is not None:
        return  # still mid-ticket

    if not operator.queue:
        return

    next_ticket = _pick_next(operator.queue)  # FIFO; urgent-maintenance can priority-bump

    if _required_skill(next_ticket.type) > operator.skill_level:
        _escalate(operator, next_ticket, env)  # move to lowest-queue senior/lead on shift
        return

    # Latency: base × (1 + fatigue) + Gaussian jitter, floored at 1
    latency = max(1, int(operator.base_latency_steps * (1 + operator.current_fatigue) + rng.gauss(0, 1)))
    operator.busy_until = env.current_step + latency
    operator.current_ticket = next_ticket
```

### Success rate

```
P(ticket_succeeds) = base_skill_p × (1 − 0.5 × current_fatigue)
    base_skill_p: junior=0.85, senior=0.95, lead=0.98
```

Skill gating is enforced at pickup via `_escalate`, so once a ticket is picked up, the skill requirement is already satisfied and is not a multiplier on success.

**On failure:** `status = FAILED`. Labor is still billed. The underlying problem persists (jammed printer stays jammed, maintenance didn't actually happen, sensor fault remains). Dispatcher must dispatch a follow-up ticket.

### Ticket types

| Type | Min skill | Duration | Labor ($) | Effect on completion (success path) |
|---|---|---|---|---|
| `spool_swap` | junior | 2 | rate × 2 | Swap to requested material (1000g spool, -50g purge), printer → IDLE |
| `filament_reload_from_stock` | junior | 3 | rate × 3 | Refill existing-material spool to 1000g |
| `maintenance_basic` | senior | 3 | rate × 3 | fatigue→0, reliability += 0.05 (cap 1.0), `maintenance_due_in=50`, clear `bed_level_drift` |
| `maintenance_full_rebuild` | lead | 8 | rate × 8 | All of basic + reliability → `profile.reliability_base`, clear ALL active faults |
| `diagnostic_physical` | junior | 1 | rate × 1 | Reveals ground truth to operator (enables `REPORT_ANOMALY`); clears `thermistor_*` and `klipper_mcu_disconnect` faults |
| `unjam_printer` | senior | 4 | rate × 4 | Printer `ERROR` → `IDLE` (job is already back to `PENDING`) |

### Escalation

A junior picking up a ticket whose `required_skill > junior` triggers `_escalate`: the ticket is moved to the **lowest-queue on-shift senior/lead**, tie-broken by skill descending. Junior is billed `labor_rate × 1` for the walk-over time and becomes free immediately. If no eligible escalation target exists, the ticket stays in the junior's queue (effectively stuck until someone higher-skilled comes on shift).

### Anomaly reporting

Attached to the ticket-completion event in the observation. See §3 for P(correct) by fault class.

### Ticket reassignment on operator unavailability

🟡 If a ticket sits in an operator's queue (not yet picked up) when that operator's shift ends, the env auto-reassigns it to the **lowest-queue qualified on-shift operator**. If no qualified operator is on shift, the ticket stays in the original queue until one is (effectively carried into the next shift). The agent sees ticket status (`PENDING → IN_PROGRESS [operator_id] → COMPLETED/REJECTED/FAILED`) but doesn't manage reassignment manually. The 1:1 claim constraint is enforced internally and is not part of the Dispatcher's decision surface.

**Rationale:** in a real farm, operators are instructed to finish or hand off their tickets, and management ensures shift coverage. This is an organisational concern, not a dispatcher skill.

### Resolved open questions

- **Memory across tickets:** Yes, shift-scoped. See `printer_memory` above.
- **1:1 ticket claim:** Enforced internally. Auto-reassigns on operator unavailability (shift end). Not agent-facing.
- **Target printer state changed mid-pickup:** Operator's timer still expires on schedule; if the printer's state is now incompatible, completes with `status = REJECTED, reason = "<specific>"`. Labor is still billed.

---

## Section 5 — Economic Layer

### Dollar constants

| Item | Value | Notes |
|---|---|---|
| Filament — PLA | $0.020 / g | $20/kg |
| Filament — PETG | $0.025 / g | $25/kg |
| Filament — ABS | $0.030 / g | $30/kg |
| Electricity during PRINTING/MAINTENANCE | `profile.avg_power_watts × 2.5e-6` $/step | $0.15/kWh × W × (1/60 h) |
| Electricity during WARMING_UP | `1.5 × profile.avg_power_watts × 2.5e-6` $/step | Heating draws more than steady-state |
| Electricity during PAUSED (agent-induced) | `1.0 × profile.avg_power_watts × 2.5e-6` $/step | Nozzle held hot |
| Electricity during PAUSED_RUNOUT | `0.5 × profile.avg_power_watts × 2.5e-6` $/step | Nozzle cooling; fan-off curve |
| Electricity during IDLE / OFFLINE | $0 | Rounding to zero |
| Operator labor — junior | $0.30 / step | $18/hour |
| Operator labor — senior | $0.47 / step | $28/hour |
| Operator labor — lead | $0.67 / step | $40/hour |
| Printer amortization | `profile.amortization_per_hour / 60` $/step | Applied every PRINTING step |
| SLA miss (fixed) | -$50 | On the first step past deadline, once per job |
| SLA late fee | -$5 / step late | Accrues until the job completes or is cancelled |
| 🟡 SLA penalty cap | 80% of `job.price_usd` | Net SLA penalty never exceeds this. Ensures late completion always beats cancellation. |
| Rush premium | 1.5 × base price | Baked into `price_usd` at task-config time for priority=3 jobs |
| Scrap cost | `filament_used_g × filament_$/g` | On FAILED or CANCELLED mid-print |
| `WAIT` | -$0.10 | Per step |
| Invalid action | -$0.20 | Syntax or precondition failure |
| Unnecessary `RUN_DIAGNOSTIC` | -$0.50 | No active fault on target printer |
| `RUN_DIAGNOSTIC` catches real fault | +$2.00 | Net +$1.50 on catch (base cost always applied) |
| Catastrophic failure (fatigue=10) | -$250 | Parts replacement + 10-step downtime |
| `OVERRIDE_OPERATOR` | -$0.10 | Admin cost, regardless of outcome |

### Job pricing ranges (task-config guidance)

- Priority 1 (low): $15–$25, typically `deadline_steps = null`
- Priority 2 (normal): $25–$50, deadline within episode
- Priority 3 (urgent): base × 1.5 → $37–$75 at listed price, tight deadline

### Break-even ratios (reviewed)

| Question | Formula | Implicit threshold | Verdict |
|---|---|---|---|
| How many WAITs justify avoiding a $250 crash? | 250 / 0.10 | 2,500 WAIT steps | Fine — episodes are 60–90 steps, so WAIT is ~free vs. catastrophic risk |
| Unnecessary diagnostics per real catch? | 2.00 / 0.50 | 4:1 | Agent diagnoses when P(fault) > 20%. Matches our intent |
| Late completion vs. cancellation (capped SLA) | -(50 + 5·k) capped at 0.8·price vs. -scrap | Capped penalty + any accrued revenue means late completion strictly dominates | Correct — we want agents to push through late, not cancel |
| Crash vs. 1 premium job completed | -$250 vs. ~$75 | Agent avoids a crash even if it costs ~3 premium jobs' revenue | Matches "safety is cheap" bias we want |
| `RUN_DIAGNOSTIC` bonus vs. avoided scrap | +$2 nominal vs. real $250 catastrophe avoided | The +$2 is a pedagogy signal; the real value is downstream avoided loss | Keep small on purpose |

### Resolved open questions

- **Cumulative vs. per-step reward:** Both. `step()` returns per-step dollar delta; episode summary includes `total_net_profit_usd`.
- **Optimal ceiling for normalization:** Clairvoyant-greedy. Normalised score = `(actual − naive_greedy) / (clairvoyant_greedy − naive_greedy)`.

---

## Section 6 — Task Scenarios

Every task uses the same `TaskConfig` schema. All tasks set `seed = hash(task_id)` at reset for deterministic eval.

### Template

```yaml
task_id: task_N
name: <short name>
episode_length_steps: int
printer_count: int
printer_profiles: [list of profile IDs]   # length == printer_count
printer_overrides:                        # optional per-printer starting state
  - {printer_id, current_material, spool_weight_g, fatigue_level, ...}
operator_roster:
  - {operator_id, skill_level, shift_window}
job_queue:
  - {job_id, material, weight_g, print_time_steps, priority, deadline_steps, price_usd}
initial_inventory_g: {PLA, PETG, ABS}
scheduled_failures:                       # deterministic — for reproducible eval
  - {printer_id, mode, trigger_step, duration}
scheduled_operator_reports:               # for Task 3 "ground-truth disagreement" setup
  - {step, operator_id, printer_id, report}
stochastic_failure_rates:                 # overrides profile defaults
  <mode>: <rate | "profile_default" | "zero">
success_criteria:
  min_net_profit_usd: float
  max_catastrophic_failures: 0
  max_operator_queue_overflow_steps: int
  ...
```

### Task 1 — Human Latency Coordination

**Theme:** Multi-Agent. **Blocking task.**

```yaml
task_id: task_1
name: "Human Latency Coordination"
episode_length_steps: 60
printer_count: 8
printer_profiles: [bambu_x1c, bambu_x1c, prusa_mk4, prusa_mk4, creality_k1, creality_k1, voron_24, voron_24]
printer_overrides:
  # Two printers start with the wrong material to force swap-ticket pressure
  - {printer_id: 5, current_material: ABS,  spool_weight_g: 800}
  - {printer_id: 7, current_material: PETG, spool_weight_g: 600}
operator_roster:
  - {operator_id: op_j1, skill_level: junior, shift_window: [0, 60]}
  - {operator_id: op_j2, skill_level: junior, shift_window: [0, 60]}
  - {operator_id: op_s1, skill_level: senior, shift_window: [0, 60]}
job_queue:
  - {job_id: j1, material: PLA,  weight_g: 150, print_time_steps: 12, priority: 2, deadline_steps: 25, price_usd: 35}
  - {job_id: j2, material: PLA,  weight_g: 200, print_time_steps: 15, priority: 2, deadline_steps: 35, price_usd: 40}
  - {job_id: j3, material: PETG, weight_g: 180, print_time_steps: 14, priority: 3, deadline_steps: 30, price_usd: 75}
  - {job_id: j4, material: ABS,  weight_g: 250, print_time_steps: 18, priority: 2, deadline_steps: 50, price_usd: 55}
  - {job_id: j5, material: PLA,  weight_g: 120, print_time_steps: 10, priority: 1, deadline_steps: null, price_usd: 20}
  - {job_id: j6, material: PETG, weight_g: 160, print_time_steps: 13, priority: 3, deadline_steps: 55, price_usd: 65}
initial_inventory_g: {PLA: 1500, PETG: 1000, ABS: 1000}
scheduled_failures: []
stochastic_failure_rates: profile_default
success_criteria:
  min_net_profit_usd: 150
  max_catastrophic_failures: 0
  max_operator_queue_overflow_steps: 0
```

**Challenge.** Two simultaneous swap needs and a maintenance-ish load naturally push juniors over their queue cap of 3. The Dispatcher must batch PLA jobs on matching printers first, route the ABS job through the senior, and avoid spamming juniors with parallel swap tickets they can't absorb.

### Task 2 — Sensor Trust Calibration

**Theme:** Zero-Trust. **Blocking task.**

```yaml
task_id: task_2
name: "Sensor Trust Calibration"
episode_length_steps: 60
printer_count: 5
printer_profiles: [bambu_x1c, prusa_mk4, prusa_mk4, creality_k1, voron_24]
operator_roster:
  - {operator_id: op_j1, skill_level: junior, shift_window: [0, 60]}
  - {operator_id: op_s1, skill_level: senior, shift_window: [0, 60]}
job_queue:
  - {job_id: j1, material: PLA,  weight_g: 180, print_time_steps: 14, priority: 2, deadline_steps: 30, price_usd: 40}
  - {job_id: j2, material: PETG, weight_g: 220, print_time_steps: 18, priority: 3, deadline_steps: 40, price_usd: 70}
  - {job_id: j3, material: PLA,  weight_g: 150, print_time_steps: 12, priority: 2, deadline_steps: 45, price_usd: 35}
  - {job_id: j4, material: ABS,  weight_g: 250, print_time_steps: 20, priority: 2, deadline_steps: 55, price_usd: 55}
initial_inventory_g: {PLA: 2000, PETG: 1000, ABS: 1000}
scheduled_failures:
  - {printer_id: 1, mode: thermistor_open, trigger_step: 8,  duration: 10}
  - {printer_id: 2, mode: filament_sensor_false_runout, trigger_step: 18, duration: 5}
  - {printer_id: 3, mode: webcam_freeze,  trigger_step: 25, duration: 12}
stochastic_failure_rates:
  thermistor_open: zero
  thermistor_short: zero
  filament_sensor_false_runout: zero
  webcam_freeze: zero
  fan_rpm_ghost: zero
  klipper_mcu_disconnect: zero
  progress_drift: zero
  filament_sensor_missed_runout: zero
  bed_level_drift: zero
success_criteria:
  min_net_profit_usd: 100
  max_catastrophic_failures: 0
  max_unnecessary_diagnostic_cost_usd: 2.00   # ≤ 4 unnecessary RUN_DIAGNOSTICs
```

**Challenge.** All failures scheduled (reproducible for eval). Diagnosing every printer every step burns money; ignoring telemetry crashes jobs. Correct play: temporal-consistency check → diagnose when telemetry transitions look physically impossible.

### Task 3 — Disagreement Resolution

**Theme:** Multi-Agent + Zero-Trust. **Blocking task.**

```yaml
task_id: task_3
name: "Disagreement Resolution"
episode_length_steps: 60
printer_count: 4
printer_profiles: [bambu_x1c, prusa_mk4, creality_k1, voron_24]
operator_roster:
  - {operator_id: op_j1, skill_level: junior, shift_window: [0, 60]}
  - {operator_id: op_s1, skill_level: senior, shift_window: [0, 60]}
job_queue:
  - {job_id: j1, material: PLA,  weight_g: 180, print_time_steps: 14, priority: 2, deadline_steps: 30, price_usd: 40}
  - {job_id: j2, material: PETG, weight_g: 250, print_time_steps: 20, priority: 3, deadline_steps: 42, price_usd: 80}
  - {job_id: j3, material: ABS,  weight_g: 200, print_time_steps: 16, priority: 2, deadline_steps: 50, price_usd: 50}
initial_inventory_g: {PLA: 1500, PETG: 1000, ABS: 1000}
scheduled_failures:
  # Printer 1: telemetry will lie (says broken), operator will correctly say "fine"
  - {printer_id: 1, mode: thermistor_open,  trigger_step: 6,  duration: 20}
  # Printer 3: real slowdown, telemetry looks normal enough to miss, operator will notice
  - {printer_id: 3, mode: progress_drift,   trigger_step: 20, duration: 15}
scheduled_operator_reports:
  - {step: 10, operator_id: op_j1, printer_id: 1, report: "printing fine"}   # correct; telemetry is lying
  - {step: 24, operator_id: op_s1, printer_id: 3, report: "seems slow"}      # correct; drift is real
stochastic_failure_rates: zero_all
success_criteria:
  min_net_profit_usd: 80
  max_catastrophic_failures: 0
```

**Challenge.** One case where the operator is right and telemetry is wrong; another where the operator spots a subtle drift the Dispatcher would miss. No single rule works — the Dispatcher must reason about each disagreement on its merits.

### Task 4 — Long-Horizon Maintenance Planning (stretch)

**Theme:** Long-Horizon planning. **Stretch task.**

```yaml
task_id: task_4
name: "Long-Horizon Maintenance Planning"
episode_length_steps: 90
printer_count: 6
printer_profiles: [bambu_x1c, bambu_x1c, prusa_mk4, prusa_mk4, creality_k1, voron_24]
printer_overrides:
  # Two printers are near maintenance threshold at episode start
  - {printer_id: 1, maintenance_due_in: 12}
  - {printer_id: 4, maintenance_due_in: 18}
  # Rest default to maintenance_due_in = 50
operator_roster:
  - {operator_id: op_j1, skill_level: junior, shift_window: [0, 90]}
  - {operator_id: op_j2, skill_level: junior, shift_window: [0, 90]}
  - {operator_id: op_s1, skill_level: senior, shift_window: [0, 90]}
  - {operator_id: op_l1, skill_level: lead,   shift_window: [0, 90]}
job_queue:
  - {job_id: j1, material: PLA,  weight_g: 200, print_time_steps: 16, priority: 2, deadline_steps: 25, price_usd: 40}
  - {job_id: j2, material: PETG, weight_g: 300, print_time_steps: 24, priority: 3, deadline_steps: 55, price_usd: 85}
  - {job_id: j3, material: PLA,  weight_g: 220, print_time_steps: 18, priority: 2, deadline_steps: 60, price_usd: 45}
  - {job_id: j4, material: ABS,  weight_g: 280, print_time_steps: 22, priority: 3, deadline_steps: 80, price_usd: 75}
  - {job_id: j5, material: PETG, weight_g: 180, print_time_steps: 14, priority: 2, deadline_steps: 75, price_usd: 50}
  - {job_id: j6, material: PLA,  weight_g: 240, print_time_steps: 19, priority: 2, deadline_steps: 85, price_usd: 48}
initial_inventory_g: {PLA: 1500, PETG: 1200, ABS: 1000}
scheduled_failures:
  - {printer_id: 3, mode: bed_level_drift, trigger_step: 40, duration: 50}
stochastic_failure_rates: profile_default
success_criteria:
  min_net_profit_usd: 150
  max_catastrophic_failures: 0
  min_avg_reliability_end_of_episode: 0.80
```

**Challenge.** Two printers are already near maintenance threshold; a third will develop `bed_level_drift` mid-episode (detectable via repeat early-print failures on the same printer, plus senior/lead operator's pattern-recommendation when they visit twice). Dispatcher must schedule `maintenance_basic` at natural idle windows without (a) letting any printer hit catastrophic failure from neglected maintenance, (b) stalling the job pipeline by pulling too many printers offline for maintenance simultaneously.

**Skills stressed:** watching `maintenance_due_in` as a horizon signal, reading operator pattern-recommendations from §4 memory, sequencing maintenance against job deadlines.

### Task 5 — Economic Stress Test (stretch / holdout)

**Theme:** World Modeling + P&L. **Stretch / held-out for eval.** Do NOT include Task 5 trajectories in SFT/DPO training data — it's the generalization test.

```yaml
task_id: task_5
name: "Economic Stress Test"
episode_length_steps: 90
printer_count: 6
printer_profiles: [bambu_x1c, bambu_x1c, prusa_mk4, creality_k1, creality_k1, voron_24]
operator_roster:
  - {operator_id: op_j1, skill_level: junior, shift_window: [0, 90]}
  - {operator_id: op_j2, skill_level: junior, shift_window: [0, 90]}
  - {operator_id: op_s1, skill_level: senior, shift_window: [0, 90]}
  - {operator_id: op_l1, skill_level: lead,   shift_window: [0, 90]}
job_queue:
  # 4 urgent (priority 3), 8 normal (priority 2), 3 low (priority 1) — 15 jobs
  - {job_id: u1, material: PLA,  weight_g: 200, print_time_steps: 15, priority: 3, deadline_steps: 25, price_usd: 90}
  - {job_id: u2, material: PETG, weight_g: 250, print_time_steps: 20, priority: 3, deadline_steps: 40, price_usd: 100}
  - {job_id: u3, material: ABS,  weight_g: 220, print_time_steps: 18, priority: 3, deadline_steps: 55, price_usd: 85}
  - {job_id: u4, material: PLA,  weight_g: 180, print_time_steps: 14, priority: 3, deadline_steps: 70, price_usd: 75}
  - {job_id: n1, material: PLA,  weight_g: 150, print_time_steps: 12, priority: 2, deadline_steps: 40, price_usd: 35}
  - {job_id: n2, material: PLA,  weight_g: 180, print_time_steps: 14, priority: 2, deadline_steps: 45, price_usd: 40}
  - {job_id: n3, material: PETG, weight_g: 160, print_time_steps: 13, priority: 2, deadline_steps: 50, price_usd: 42}
  - {job_id: n4, material: PETG, weight_g: 200, print_time_steps: 16, priority: 2, deadline_steps: 60, price_usd: 48}
  - {job_id: n5, material: ABS,  weight_g: 180, print_time_steps: 14, priority: 2, deadline_steps: 65, price_usd: 45}
  - {job_id: n6, material: ABS,  weight_g: 220, print_time_steps: 18, priority: 2, deadline_steps: 75, price_usd: 50}
  - {job_id: n7, material: PLA,  weight_g: 150, print_time_steps: 12, priority: 2, deadline_steps: 80, price_usd: 38}
  - {job_id: n8, material: PETG, weight_g: 130, print_time_steps: 10, priority: 2, deadline_steps: 85, price_usd: 32}
  - {job_id: l1, material: PLA,  weight_g: 120, print_time_steps: 10, priority: 1, deadline_steps: null, price_usd: 18}
  - {job_id: l2, material: PLA,  weight_g: 100, print_time_steps: 8,  priority: 1, deadline_steps: null, price_usd: 15}
  - {job_id: l3, material: PETG, weight_g: 110, print_time_steps: 9,  priority: 1, deadline_steps: null, price_usd: 20}
initial_inventory_g: {PLA: 1200, PETG: 1000, ABS: 800}   # deliberately short: can't serve everyone
scheduled_failures:
  - {printer_id: 3, mode: bed_level_drift, trigger_step: 30, duration: 60}
stochastic_failure_rates: profile_default
success_criteria:
  min_net_profit_usd: 400
  max_catastrophic_failures: 0
```

**Challenge.** ~13 printer-hours of demand vs. 9 printer-hours of capacity (6 printers × 90 steps, but with swaps/maintenance/warmup overhead). Correct play: sacrifice low-priority jobs (no SLA penalty), detect the bed_level_drift pattern on printer 3 via repeated early-step failures, dispatch `maintenance_basic` once the pattern is clear.

### Resolved open questions

- **Deterministic seeds for eval:** Yes. `seed = hash(task_id)`.
- **Hidden test task:** Task 5. Held out from SFT/DPO training data.

---

## Section 7 — Pipeline Order (Per-Step Execution)

**This section is authoritative for execution order.** If another section conflicts with this, this wins.

### Step sequence (per `step()` call)

```
 1. Receive Dispatcher action for step t (single action)
 2. Validate action
    └── on reject: emit -$0.20 penalty with metadata.reason; jump to step 9
 3. Apply Dispatcher action effects to ground-truth state
    ├── ASSIGN_JOB / CANCEL_JOB / RESUME_JOB : direct state change
    ├── RUN_DIAGNOSTIC                       : mark printer.revealed_this_step = True
    ├── DISPATCH_TICKET / REQUEST_*          : append to operator queue
    └── OVERRIDE_OPERATOR                    : remove ticket from queue
 4. Tick all operators (in operator_id order)
    a. If current_ticket has timer expired → finalise (effects, bill labor, success roll, maybe REPORT_ANOMALY)
    b. If free & on-shift & queue non-empty → pick next ticket, set busy_until
 5. Tick all printers (in printer_id order)
    a. Advance warmup_remaining / maintenance counter / offline_remaining
    b. If PRINTING: reliability roll (apply ×0.85 penalty on the first tick after PAUSED_RUNOUT→RESUME), fatigue +=1, consume filament, accrue step_revenue, amortization, electricity
    c. If PAUSED / PAUSED_RUNOUT: accrue heating electricity at the appropriate rate (§5); no fatigue, no filament, no revenue
    d. Natural transitions: spool_weight==0 → PAUSED_RUNOUT; fatigue==10 → OFFLINE (catastrophic, clawback, -$250); progress complete → COMPLETED
    e. Increment bed_level_drift counters for currently-PRINTING printers
 6. Sample world events (per printer × failure mode)
    a. Inject scheduled_failures matching trigger_step == t
    b. For each mode with rate > 0: Bernoulli roll; if arrival and no active instance, inject
    c. Advance durations of existing faults; expire at duration end
 7. Compute ground-truth telemetry snapshot
 8. Apply sensor corruption layer → Dispatcher-visible observation
    └── If printer.revealed_this_step: substitute ground-truth for that printer's slice
 9. Compute reward components for step t and sum (see §5)
    ├── Action cost (step 2/3)
    ├── Filament + electricity + amortization (step 5b)
    ├── Revenue accrual / clawback (step 5b / 5c)
    ├── Operator labor (step 4)
    ├── SLA penalties (for each active job: check deadline vs. current_step, apply fixed + per-step)
    ├── Catastrophic events (step 5c)
    └── Diagnostic bonus (step 3 if RUN_DIAGNOSTIC and a real fault was active)
10. Assemble FarmObservation (corrupted telemetry + operator queues + inventory + reward + metadata + ticket-completion events)
11. Check termination: current_step >= max_steps OR all jobs in {COMPLETED, FAILED, CANCELLED}
12. Write Oversight log line (ground-truth state + corrupted view + action + reward components)
13. Increment time_step; return observation
```

### Critical invariants about the pipeline

- **Ground truth is computed BEFORE sensor corruption.** Never the reverse.
- **Reward uses ground-truth events.** If `webcam_freeze` masks a fatigue=10 → OFFLINE transition, the agent still eats the -$250; the agent just doesn't see the transition in the next observation (until the freeze clears).
- **Oversight log captures both views**, so audit precision/recall is measurable against ground truth.
- **Action effects apply before world events**, i.e., step 3 before step 6. If the agent `ASSIGN_JOB`s on step t and a sensor fault happens to trigger this same step, the assignment succeeds; the fault affects the agent's observation starting t+1.
- **Operators tick before printers.** An operator completing a spool swap on step t means the printer's step-5 update sees the new spool.

### Resolved open questions

- **Multiple actions per step:** Disallowed. See §0.
- **World events vs. action effects order:** Action first, then world. Confirmed.

---

## Section 8 — Invariants (Assertion Checklist)

Enforced via Python `assert` in dev; disabled in release. Each should be a one-line test in `tests/test_invariants.py`, run after every `step()` during CI.

### Physical invariants

- `inventory[material] >= 0` for all materials at all times
- For every printer: `spool_weight_g >= 0`
- `fatigue_level ∈ [0, 10]` (clamped)
- `reliability ∈ [0.0, 1.0]`
- Every printer is in exactly one `PrinterState`
- `Σ len(operator.queue) ≤ Σ operator.queue_capacity` across all operators
- `warmup_remaining`, `offline_remaining`, `maintenance_due_in` all ≥ 0
- `consecutive_idle_steps == 0` for any state other than IDLE

### Semantic invariants

- A printer in `MAINTENANCE` or `MAINTENANCE_QUEUED` has `current_job_id is None`
- A job in state `PRINTING` has exactly one printer whose `current_job_id` matches it
- `operator.busy_until > current_step` ⟺ `operator.current_ticket is not None`
- No two operators claim the same `ticket_id`
- A `MAINTENANCE_QUEUED` printer has exactly one outstanding maintenance ticket
- A `PAUSED_RUNOUT` printer's associated job is in state `PAUSED`
- A `PAUSED` printer has `spool_weight_g > 0` and an associated job in state `PAUSED`
- `reliability_penalty_active` flag is True only for the single first PRINTING tick following a `PAUSED_RUNOUT → RESUME_JOB`; cleared after that tick regardless of outcome
- No printer has two active instances of the same failure mode
- Operator `printer_memory` keys are a subset of `{printer_id for printer in env.printers}`; memory is cleared at `shift_window.start`

### Economic invariants

- Sum of per-step rewards over an episode equals the cumulative `total_net_profit_usd` (no hidden bonuses)
- `naive_greedy_baseline_profit ≤ clairvoyant_greedy_baseline_profit` on every task (otherwise the reward function is inverted)
- Any SLA-missed job still contributes at least `0.2 × price_usd` in net to revenue (the 80% cap on the penalty guarantees it)
- `total_labor_billed == Σ_completed_ticket (operator.rate × ticket.duration_steps)` (operators never do free work, including escalations and rejections)

### Observation invariants

- `FarmObservation.reward` for step t equals the scalar returned by `step()`
- Every field in `PrinterObservation` and `FarmObservation` is JSON-serialisable (no numpy types leaking through)
- Corrupted telemetry never reveals that it's corrupted (no `is_hallucinating` flag in the public observation). The only signal the agent has is the `revealed_this_step` flag set by `RUN_DIAGNOSTIC`.

### Training invariants (checked post-hoc, not per-step)

- DPO preference-pair states appear in > 20% of SFT model rollouts (otherwise DPO has no signal to shift)
- Per-step reward magnitude does not exceed ±$500 on any step (catastrophic events capped at -$250; no single-step revenue > $100 in any task)
- Episode-total reward stays within ±$2,000 (safe for standard LR on 7B models in Unsloth)

---

## Appendix A — Change Log

| Version | Date | Changes |
|---|---|---|
| 0.1 | 2026-04-19 | Initial skeleton |
| 1.0 | 2026-04-20 | First freeze: all sections populated, open questions resolved, 🟡 flags on genuine judgment calls |
| 1.1 | 2026-04-21 | Post-review revisions: added `PAUSE_JOB` + `PAUSED` state for proactive swap coordination; `PAUSED_RUNOUT` resume now carries reliability penalty; operators now have shift-scoped `printer_memory` with repeat-visit accuracy bonus + pattern-recommendation; ticket auto-reassignment on shift-end unavailability (1:1 claim no longer agent-facing); Task 4 re-themed from "Shift Handoff" to "Long-Horizon Maintenance Planning"; `RUN_DIAGNOSTIC` escalation path via `diagnostic_physical` made explicit |
