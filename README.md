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

An OpenEnv environment that simulates managing a fleet of networked 3D printers. The agent acts as an automated floor manager — assigning print jobs, swapping filament spools, performing maintenance, recovering from mid-print failures, and managing material inventory under time pressure and uncertainty.

## Motivation

3D print farm management is a genuine, unsolved operational challenge. Farms with 10–100+ printers must continuously match incoming orders to available machines, handle mid-print failures (filament runouts, random hardware jams), prioritise urgent customer orders, and schedule preventive maintenance — all with limited inventory and tight deadlines. Current solutions rely on manual oversight or rigid rule-based schedulers. This environment provides a realistic testbed for evaluating whether AI agents can learn these coordination, planning, and recovery skills under stochastic conditions.

## Action Space

The agent submits one `FarmAction` per time step:

| Field | Type | Description |
|---|---|---|
| `action` | enum | `ASSIGN_JOB`, `SWAP_FILAMENT`, `CANCEL_JOB`, `PERFORM_MAINTENANCE`, `RESUME_JOB`, `WAIT` |
| `printer_id` | int (optional) | Target printer ID |
| `job_id` | string (optional) | Target job from the active queue |
| `material` | string (optional) | Filament type for `SWAP_FILAMENT` (e.g. `"PLA"`, `"PETG"`, `"ABS"`) |

**Action details:**

- **ASSIGN_JOB** — Starts a pending job on an idle printer. Requires matching material and sufficient spool weight. Triggers a 2-step warmup before printing begins.
- **SWAP_FILAMENT** — Replaces the spool on an idle, errored, or paused-runout printer with a fresh spool from inventory (950g net after 50g purge). Enters WARMING_UP for 2 steps.
- **CANCEL_JOB** — Cancels a pending, printing, or paused job, freeing the printer.
- **PERFORM_MAINTENANCE** — Services an idle or errored printer (takes 3 steps). Resets `fatigue_level` to 0 and restores reliability to 0.95. **Thermal cooldown**: if the printer has `fatigue_level > 0`, it must have been continuously IDLE for 3 steps (`consecutive_idle_steps ≥ 3`) before maintenance can begin.
- **RESUME_JOB** — Resumes a paused job on an idle printer from where it left off (no warmup required).
- **WAIT** — Do nothing. Penalised by the grader when actionable work exists.

## Observation Space

Each step returns a `FarmObservation`:

| Field | Type | Description |
|---|---|---|
| `printers` | list[PrinterObservation] | State of all printers |
| `active_queue` | list[PrintJob] | All jobs and their statuses |
| `inventory` | dict[str, float] | Material stock in grams |
| `time_step` | int | Current step |
| `max_steps` | int | Episode length |
| `reward` | float | Current grader score (0.0–1.0) |
| `done` | bool | Whether the episode has ended |
| `metadata` | dict | Error messages from last action |

**PrinterObservation:**

| Field | Type | Description |
|---|---|---|
| `printer_id` | int | Printer ID |
| `state` | enum | `IDLE`, `WARMING_UP`, `PRINTING`, `PAUSED_RUNOUT`, `ERROR`, `MAINTENANCE`, `OFFLINE` |
| `current_material` | str or null | Loaded filament type |
| `current_job_id` | str or null | Currently assigned job |
| `spool_weight_g` | float | Remaining filament on spool (grams) |
| `reliability` | float | Per-step success probability while printing (0.0–1.0) |
| `warmup_remaining` | int | Steps until printing begins (or maintenance completes) |
| `maintenance_due_in` | int | Steps until maintenance is needed |
| `fatigue_level` | int | 0–10; catastrophic failure at 10 (printer goes OFFLINE for 10 steps) |
| `offline_remaining` | int | Steps remaining in OFFLINE state |
| `consecutive_idle_steps` | int | Steps printer has been continuously IDLE (3 required before maintenance if fatigued) |

**PrintJob:**

| Field | Type | Description |
|---|---|---|
| `job_id` | str | Unique job identifier |
| `material_required` | str | Required filament type |
| `weight_required_g` | float | Filament needed (grams) |
| `print_time_steps` | int | Steps to complete |
| `priority` | int | 1=low, 2=normal, 3=urgent |
| `deadline_steps` | int or null | Must complete by this step |
| `state` | enum | `PENDING`, `PRINTING`, `PAUSED`, `COMPLETED`, `FAILED`, `CANCELLED` |
| `progress_steps` | int | Steps completed so far |

## Environment Mechanics

- **Fatigue**: Each printing step increments `fatigue_level` by 1. At level 10, the printer suffers catastrophic failure — the current job is marked FAILED and the printer goes OFFLINE for 10 steps. `PERFORM_MAINTENANCE` resets fatigue to 0.
- **Thermal cooldown**: Fatigued printers (`fatigue_level > 0`) must remain IDLE for 3 consecutive steps before maintenance can be performed. The `consecutive_idle_steps` field tracks this — it increments while IDLE and resets to 0 for any other state.
- **Stochastic failures**: Each printing step, a printer can fail with probability `1 − reliability`. Failed jobs are marked FAILED and the printer enters ERROR state.
- **Filament runout**: If a spool runs empty mid-print, the printer enters PAUSED_RUNOUT and the job becomes PAUSED. A filament swap + RESUME_JOB is required to continue. Progress is preserved.
- **Warmup phase**: After assigning a job or swapping filament, the printer spends 2 steps warming up before work begins.
- **Maintenance**: Takes 3 steps. Resets fatigue to 0, restores reliability to 0.95, and resets `maintenance_due_in` to 50.
- **Priorities & deadlines**: Urgent jobs (priority 3) are worth 2× in the score. Meeting deadlines gives full credit; late completion decays at 5% per step (floor 10%).
- **Step penalties**: Every WAIT action costs −0.01 and every failed/invalid action costs −0.02.

## Tasks

### Task 1: Traffic Jam (Easy)
**Budget:** 25 steps | **Jobs:** 5 (3 PLA + 2 PETG) | **Printers:** 1

A single PLA-loaded printer must handle 5 jobs across two materials. All print times are 1 step, making the 2-step filament swap brutally expensive. One PETG job has a tight deadline (step 12). The agent faces a batching paradox: batch all PLA first then swap once (optimal), or chase the urgent PETG job immediately and pay multiple swap penalties.

### Task 2: Spool Runout (Medium)
**Budget:** 30 steps | **Jobs:** 2 (PLA + ABS) | **Printers:** 10

An 800g urgent PLA job on a printer with only 300g of filament. The spool runs out mid-print (~step 4), pausing the job. The agent must swap filament and resume the job to preserve progress. The trap: frontier models often cancel the paused job (losing all progress) instead of performing the correct swap → resume recovery sequence.

### Task 3: Thermal Cooldown (Hard)
**Budget:** 30 steps | **Jobs:** 3 (ABS + PETG) | **Printers:** 10

Printer 1 has `fatigue_level=7` and an urgent 5-step ABS job. Assigning directly would push fatigue to 12, triggering catastrophic failure. The obvious fix — immediate maintenance — is rejected because the thermal cooldown mechanic requires 3 consecutive IDLE steps first. The agent must: WAIT 3 steps (cooldown) → PERFORM_MAINTENANCE (3 steps) → then assign the job. Meanwhile, Printer 2 has PETG loaded for secondary jobs. Models that attempt immediate maintenance thrash on rejected actions, accumulating penalties.

## Scoring

Scores range from 0.001 to 0.999 (clamped per OpenEnv spec). The grader uses priority-weighted job completion with continuous latency decay:

- **Priority weights**: priority 3 = 2×, priority 2 = 1×, priority 1 = 0.5×
- **On-time completion**: full credit (weight × 1.0)
- **Late completion**: 5% decay per step past deadline, floor at 10%
- **In-progress jobs**: partial credit (40–50% × progress fraction × decay)
- **Failed jobs with progress**: small credit (10–15% of weight)
- **Step penalties**: WAIT = −0.01, failed action = −0.02
- **Task 3 bonus**: +0.5 bonus (added to denominator) for meeting all urgent deadlines on time

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the baseline inference script:
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="your-model-name"
export OPENAI_API_KEY="your-api-key"
python inference.py
```

Run locally via Docker:
```bash
docker build -t printfarm-env .
docker run -p 7860:7860 printfarm-env
```

API endpoints at `http://localhost:7860`:
- `POST /reset` — Reset environment (pass `episode_id`: `task_1`, `task_2`, `task_3`)
- `POST /step` — Execute an action
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
