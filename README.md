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

# PrintFarmEnv

A trust-and-coordination layer for 3D print farm software — an AI Dispatcher that orchestrates human operator NPCs, refuses to trust hallucinating sensors, and maximises dollar P&L, with every action mapped 1:1 to OctoPrint / Moonraker APIs.

**Themes:** Multi-Agent Interactions (Theme 1) · Zero-Trust Sensor Handling (Theme 5) · Long-Horizon Planning (Theme 2) · World Modeling (Theme 3.1)

## Motivation

Farms with 10–100+ networked printers must coordinate job scheduling, human technicians, degrading sensors, and tight customer SLAs — simultaneously. Current solutions are either manual or rigid rule-based. PrintFarmEnv forces an AI agent to survive all three challenges at once under a realistic dollar-denominated cost model. A trained agent can be pointed at a real farm's OctoPrint REST API — the simulator is the training ground, not the endgame.

## Action Space

The Dispatcher emits one `FarmAction` per step. Every action maps to a real OctoPrint / Moonraker API call:

| Action | OctoPrint / Moonraker | Description |
|---|---|---|
| `ASSIGN_JOB` | `POST /api/job` | Start a PENDING job on an IDLE printer (material must match, spool sufficient) |
| `CANCEL_JOB` | `POST /api/job` cancel | Cancel a PENDING / PRINTING / PAUSED job — incurs scrap cost + revenue clawback |
| `PAUSE_JOB` | `POST /api/job` pause | Pause a PRINTING printer (nozzle stays hot) |
| `RESUME_JOB` | `POST /api/job` resume | Resume a PAUSED / PAUSED_RUNOUT job (spool swap must have completed first) |
| `RUN_DIAGNOSTIC` | `GET /api/printer` + Moonraker `query_endstops` | Costs −$0.50 but reveals ground-truth telemetry; +$2.00 bonus if a real fault is found |
| `DISPATCH_TICKET` | *(External work-order queue)* | Send a work order to a named operator (`spool_swap`, `maintenance_basic`, `unjam_printer`, etc.) |
| `REQUEST_SPOOL_SWAP` | Sugar for `DISPATCH_TICKET(spool_swap)` | Auto-routes spool swap to best available operator |
| `REQUEST_MAINTENANCE` | Sugar for `DISPATCH_TICKET(maintenance_*)` | Auto-routes maintenance; printer enters `MAINTENANCE_QUEUED` |
| `OVERRIDE_OPERATOR` | Cancel a queued ticket | Cancel a queued (not in-progress) ticket with a narrative reason (written to Oversight log) |
| `WAIT` | *(no-op)* | Costs −$0.10 |

**Action fields:** `printer_id`, `job_id`, `operator_id`, `ticket_type`, `ticket_id`, `material`, `maintenance_type`, `reason` (all optional; required fields depend on action type).

## Observation Space

Each `env.step()` returns a `FarmObservation`:

| Field | Type | Description |
|---|---|---|
| `printers` | list[PrinterObservation] | State of all printers (⚠️ telemetry may be sensor-corrupted) |
| `active_queue` | list[PrintJob] | All jobs and their statuses |
| `operators` | list[OperatorObservation] | Human operator NPC state (shift, fatigue, queue) |
| `inventory` | dict[str, float] | Material → grams in stock |
| `time_step` / `max_steps` | int | Current step / episode length |
| `reward` | float | Normalised grader score [0, 1] |
| `done` | bool | Episode finished? |
| `net_profit_usd` | float | Cumulative dollar P&L |
| `total_labor_billed` | float | Cumulative operator labor cost |
| `ticket_events` | list[dict] | Ticket completions this step |
| `oversight_log` | list[dict] | Last 10 audit log entries |
| `reward_breakdown` | dict[str, float] | Per-step action / labor / physics / SLA deltas |
| `metadata` | dict | Error messages, `step_reward_usd` |

### PrinterObservation

| Field | Type | Description |
|---|---|---|
| `printer_id` | int | Printer ID |
| `profile_id` | str | `bambu_x1c` · `prusa_mk4` · `creality_k1` · `voron_24` |
| `state` | enum | `IDLE` · `WARMING_UP` · `PRINTING` · `PAUSED` · `PAUSED_RUNOUT` · `ERROR` · `MAINTENANCE_QUEUED` · `MAINTENANCE` · `OFFLINE` |
| `current_material` | str? | Loaded filament type |
| `current_job_id` | str? | Currently assigned job |
| `spool_weight_g` | float | Remaining filament (grams) |
| `reliability` | float | Per-step success probability [0, 1] |
| `maintenance_due_in` | int | Steps until maintenance needed |
| `fatigue_level` | float | 0–10; catastrophic at 10 (−$250) |
| `hotend_temp` | float | ⚠️ May be corrupted: 0 = `thermistor_open`, >400 = `thermistor_short` |
| `fan_rpm` | int | ⚠️ 0 may indicate `fan_rpm_ghost` fault |
| `webcam_hash` | str | ⚠️ Repeated hash = `webcam_freeze` |
| `telemetry_ts` | int | ⚠️ Stale = `klipper_mcu_disconnect` |
| `revealed_this_step` | bool | `true` = ground truth (after `RUN_DIAGNOSTIC`) |
| `outstanding_ticket_id` | str? | Active maintenance ticket |

### OperatorObservation

| Field | Type | Description |
|---|---|---|
| `operator_id` | str | Operator name |
| `skill_level` | str | `junior` · `senior` · `lead` |
| `shift_window` | list[int] | `[start_step, end_step]` |
| `is_on_shift` | bool | Currently available? |
| `queue_size` / `queue_capacity` | int | Current vs max tickets |
| `current_fatigue` | float | 0.0–0.8 |
| `pattern_recommendations` | list[str] | Operator memory-based suggestions |

### PrintJob

| Field | Type | Description |
|---|---|---|
| `job_id` | str | Unique identifier |
| `material_required` | str | Required filament type |
| `weight_required_g` | float | Filament needed (grams) |
| `print_time_steps` | int | Steps to complete |
| `priority` | int | 1=low, 2=normal, 3=urgent |
| `deadline_steps` | int? | Must complete by this step |
| `price_usd` | float | Customer-paid revenue if completed |
| `state` | enum | `PENDING` · `PRINTING` · `PAUSED` · `COMPLETED` · `FAILED` · `CANCELLED` |
| `progress_steps` | int | Steps completed so far |
| `total_sla_penalty` | float | Cumulative SLA penalty accrued |

## Zero-Trust Sensor Failure Modes

Telemetry is corrupted by structured, physically-plausible failure modes — not random noise:

| Failure | Symptom | Detection Heuristic |
|---|---|---|
| `thermistor_open` | `hotend_temp: 0` | Temp drop >100°C/step while heater ON |
| `thermistor_short` | `hotend_temp: 300+` | Temp above physical max |
| `filament_sensor_false_runout` | `PAUSED_RUNOUT` on a full spool | Spool weight still high |
| `filament_sensor_missed_runout` | Printer reports PRINTING past end of spool | Extrusion > last known weight |
| `webcam_freeze` | Same `webcam_hash` for N steps | Hash duplicate detection |
| `klipper_mcu_disconnect` | All fields frozen | `telemetry_ts` stale |
| `progress_drift` | Progress stuck at X% | Rate deviates from `print_time_steps` |
| `fan_rpm_ghost` | Fan reports 0 RPM | Disagrees with print success |
| `bed_level_drift` | First-layer fails despite ABL | Repeated early failures on one printer |

Use `RUN_DIAGNOSTIC` to reveal ground truth. Cross-reference operator `REPORT_ANOMALY` — operators are right ~85% on mechanical issues; telemetry is right ~95% on electrical.

## Human Operator NPCs

Operators are **not** agents you train — they are realistic stochastic NPCs:

- **3 skill levels:** `junior` (spool swaps, diagnostics) · `senior` (maintenance, unjam) · `lead` (full rebuilds)
- **Response latency:** 2–8 step realistic distribution before acting on tickets
- **Shift windows:** Only available during their shift
- **Fatigue:** Accumulates over shift; increases latency and error rate
- **Queue capacity:** Finite — don't overwhelm operators
- **Actions:** `COMPLETE_TICKET`, `ESCALATE_TICKET`, `REJECT_TICKET`, `REPORT_ANOMALY`

## Economic Model (Dollar P&L)

Every reward term is denominated in dollars:

| Cost / Revenue | Amount |
|---|---|
| Revenue per printing step | `job.price_usd / print_time_steps` |
| SLA miss (fixed) | −$50 per missed deadline |
| SLA miss (per step late) | −$5/step (capped at 80% of job price) |
| Catastrophic failure (fatigue=10) | −$250 |
| Unnecessary diagnostic | −$0.50 |
| Diagnostic catches real fault | +$2.00 |
| WAIT | −$0.10 |
| Invalid/rejected action | −$0.20 |
| Operator labor | $18/hr (junior) · $28/hr (senior) · $40/hr (lead) |

**Primary signal:** `net_profit_usd` (dense, per-step). **Normalised:** `reward ∈ [0, 1]` via naive-floor / clairvoyant-ceiling calibration.

## Tasks

### Warmup (Round 1 regression)

| Task | Setup | Challenge |
|---|---|---|
| `task_0_1` Traffic Jam | 1 printer, 5 jobs, 25 steps | Material batching vs deadline prioritisation |
| `task_0_2` Spool Runout | 2 printers, 2 jobs, 30 steps | Swap + resume recovery (not cancel) |
| `task_0_3` Thermal Cooldown | 2 printers, 3 jobs, 30 steps | Fatigue management near catastrophic threshold |

### Round 2 (blocking — evaluated by judges)

| Task | Theme | Setup | Challenge |
|---|---|---|---|
| `task_1` Human Latency Coordination | Multi-Agent | 8 printers (4 models), 3 operators, 6 jobs, 60 steps | Operator queue management, ticket routing, latency tolerance |
| `task_2` Sensor Trust Calibration | Zero-Trust | 5 printers, 4 jobs, 60 steps, 3 scheduled faults | Selective `RUN_DIAGNOSTIC` vs wasting operator time on false alarms |
| `task_3` Disagreement Resolution | Multi-Agent + Zero-Trust | 5 printers, 3 operators, 5 jobs, 60 steps | Evaluate each human↔telemetry disagreement on its merits |

### Stretch

| Task | Theme | Setup |
|---|---|---|
| `task_4` Long-Horizon Maintenance | Long-Horizon | 8 printers, 90 steps spanning 2 operator shifts |
| `task_5` Economic Stress Test | World Modeling + P&L | 15 jobs, limited inventory, degrading nozzle |

## Baselines

| Baseline | Mean Score (20 eps, tasks 1–3) | Description |
|---|---|---|
| **Naive-greedy** (floor) | 0.33 / 0.07 / 0.11 | FIFO assignment, ignores sensors & operators |
| **Clairvoyant-greedy** (ceiling) | 0.64 / 0.72 / 0.66 | Ground-truth access, priority-ordered greedy |

Results normalised as: *(agent − naive) / (clairvoyant − naive)*.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run baselines
```bash
python baselines/naive_greedy.py --tasks task_1 task_2 task_3 --episodes 20
python baselines/clairvoyant_greedy.py --tasks task_1 task_2 task_3 --episodes 20
```

### Run LLM inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="your-model-name"
export OPENAI_API_KEY="your-api-key"
python inference.py
```

### Run tests
```bash
python -m pytest tests/ -q           # 64 invariant + reward-hacking tests
bash validate-submission.sh           # Pre-submission validator
```

### Run via Docker
```bash
docker build -t printfarm-env .
docker run -p 7860:7860 printfarm-env
```

### API endpoints (`http://localhost:7860`)
- `POST /reset` — Reset environment (`episode_id`: `task_1` … `task_5`)
- `POST /step` — Execute a `FarmAction`
- `GET /health` — Health check

## Training Pipeline

1. **SFT warm-start** — `scripts/build_sft_dataset.py` builds ChatML dataset from clairvoyant + teacher trajectories
2. **GRPO** — `scripts/grpo_rollout.py` collects rollouts; `notebooks/grpo_dispatcher.ipynb` trains with TRL + Unsloth using `step_reward_usd` as the verifiable reward
3. **Teacher rollouts** — `scripts/generate_teacher_rollouts.py` collects high-quality trajectories from Claude / GPT-4o

## Project Structure

```
printfarm_env/          # Core environment package
  env.py                # Environment engine (~1100 lines)
  models.py             # Pydantic models (actions, observations, jobs, operators)
  tasks.py              # Task definitions + TaskGrader with floor/ceiling calibration
  economics.py          # Dollar cost constants and helpers
  failures.py           # 9 failure modes, injection schedule, sensor corruption
  operators.py          # NPC operator policy (skill, fatigue, memory, escalation)
  profiles.py           # Heterogeneous printer profiles (Bambu, Prusa, Creality, Voron)
  adapters/octoprint.py # OctoPrint API adapter (mock + real)
server/app.py           # FastAPI server (OpenEnv-compliant)
inference.py            # LLM-based Dispatcher agent
baselines/              # Naive-greedy (floor) + clairvoyant-greedy (ceiling)
scripts/                # SFT dataset builder, GRPO rollout, teacher generation
notebooks/              # Colab quickstart + GRPO training notebook
tests/                  # 64 invariant tests + 7 reward-hacking adversarial tests
docs/                   # Simulator spec, implementation plan, reward hacking audit
```

## Repo

- **Source:** https://github.com/kshitizP/PrintFarmEnv
- `GET /state` — Get current state
- `GET /health` — Health check
- `GET /schema` — Action/observation JSON schemas
- `GET /metadata` — Environment name and description

## Repository Structure

- `printfarm_env/` — Core environment: models, physics simulation, task definitions, graders
  - `models.py` — Pydantic models (FarmAction, PrintJob, PrinterObservation, FarmObservation)
  - `env.py` — Environment logic: action handlers, physics tick, state management
  - `tasks.py` — Task definitions and TaskGrader scoring functions
- `server/` — FastAPI server wrapping the environment for HTTP access
- `openenv.yaml` — OpenEnv spec configuration with model/task documentation
- `inference.py` — Baseline agent using OpenAI-compatible API
- `visualize_rewards.py` — Generates `reward_design.png` explaining the scoring system
- `Dockerfile` — Container for Hugging Face Spaces deployment
- `validate-submission.sh` — Pre-submission validation script
