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

An OpenEnv environment that simulates managing a fleet of 10 networked 3D printers. The agent acts as an automated floor manager — assigning print jobs, swapping filament spools, performing maintenance, triaging hardware errors, and managing material inventory under time pressure and uncertainty.

## Motivation

3D print farm management is a genuine, unsolved operational challenge. Farms with 10–100+ printers must continuously match incoming orders to available machines, handle mid-print failures (filament runouts, random hardware jams), prioritise urgent customer orders, and schedule preventive maintenance — all with limited inventory and tight deadlines. Current solutions rely on manual oversight or rigid rule-based schedulers. This environment provides a realistic testbed for evaluating whether AI agents can learn these coordination, planning, and recovery skills under stochastic conditions.

## Action Space

The agent submits one `FarmAction` per time step:

| Field | Type | Description |
|---|---|---|
| `action` | enum | `ASSIGN_JOB`, `SWAP_FILAMENT`, `CANCEL_JOB`, `PERFORM_MAINTENANCE`, `WAIT` |
| `printer_id` | int (optional) | Target printer ID (1–10) |
| `job_id` | string (optional) | Target job from the active queue |
| `material` | string (optional) | Filament type for `SWAP_FILAMENT` (e.g. `"PLA"`, `"PETG"`, `"ABS"`) |

**Action details:**

- **ASSIGN_JOB** — Starts a pending job on an idle printer. Requires matching material and sufficient spool weight. Triggers a 1-step warmup (2 steps if `purge_needed`).
- **SWAP_FILAMENT** — Replaces the spool on an idle/errored printer with a fresh 1000g spool from inventory. Sets `purge_needed = true`. Clears ERROR state.
- **CANCEL_JOB** — Cancels a pending or in-progress job, freeing the printer.
- **PERFORM_MAINTENANCE** — Services an idle/errored printer (takes 3 steps). Resets `maintenance_due_in` to 50 and adds +5% reliability.
- **WAIT** — Do nothing. Penalised by the grader.

## Observation Space

Each step returns a `FarmObservation`:

| Field | Type | Description |
|---|---|---|
| `printers` | list[PrinterObservation] | State of all 10 printers |
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
| `printer_id` | int | Printer ID (1–10) |
| `state` | enum | `IDLE`, `WARMING_UP`, `PRINTING`, `ERROR`, `MAINTENANCE` |
| `current_material` | str or null | Loaded filament type |
| `spool_weight_g` | float | Remaining filament on spool |
| `reliability` | float | Per-step success probability while printing (0.0–1.0) |
| `warmup_remaining` | int | Steps until printing begins (or maintenance completes) |
| `maintenance_due_in` | int | Steps until maintenance is overdue (degrades reliability at 0) |
| `purge_needed` | bool | True after filament swap; adds +1 warmup step |
| `current_job_id` | str or null | Currently assigned job |

**PrintJob:**

| Field | Type | Description |
|---|---|---|
| `job_id` | str | Unique job identifier |
| `material_required` | str | Required filament type |
| `weight_required_g` | float | Filament needed |
| `print_time_steps` | int | Steps to complete |
| `priority` | int | 1=low, 2=normal, 3=urgent |
| `deadline_steps` | int or null | Must complete by this step |
| `state` | enum | `PENDING`, `PRINTING`, `COMPLETED`, `FAILED`, `CANCELLED` |
| `progress_steps` | int | Steps completed so far |

## Environment Mechanics

- **Stochastic failures**: Each printing step, a printer can fail with probability `1 - reliability`. Failed jobs revert to `PENDING` (progress reset) and can be re-assigned to another printer.
- **Filament runout**: If a spool runs empty mid-print, the job `FAILED` permanently and the printer enters `ERROR`.
- **Warmup phase**: After assigning a job, the printer spends 1 step warming up (2 if `purge_needed`) before printing begins.
- **Maintenance**: Printers degrade reliability when `maintenance_due_in` reaches 0. `PERFORM_MAINTENANCE` takes 3 steps but resets the counter and boosts reliability.
- **Purge**: After every filament swap, the nozzle must be purged (adds 1 extra warmup step).
- **Priorities & deadlines**: Urgent jobs (priority 3) are worth 2x in the score. Meeting deadlines gives full credit; late completion is penalised.
- **Step penalty**: Every `WAIT` action costs -0.01 and every failed action costs -0.02.

## Tasks

### Task 1: Night Shift Scheduling (Easy)
**Budget:** 20 steps | **Jobs:** 5 PLA jobs with mixed priorities

Assign 5 PLA jobs across 10 printers. Three printers are loaded and reliable, two are loaded but unreliable (reliability=0.80), and five are empty. One job is urgent with a tight deadline. The agent must prioritise the urgent job on a reliable printer and manage filament swaps for remaining jobs.

### Task 2: Material Juggle (Medium)
**Budget:** 30 steps | **Jobs:** 3 jobs across 2 materials (PETG + ABS)

Three jobs requiring two different materials. Printer 1 has PETG but only 200g — not enough for the urgent 500g job, so the agent must swap filament onto another printer first. Printer 2 has ABS ready to go. One printer is unreliable (reliability=0.80). The urgent PETG job has a deadline. The agent must recognise the insufficient spool, swap proactively, and run all jobs in parallel. The 30-step budget allows recovery from one stochastic failure.

### Task 3: Chaos Shift (Hard)
**Budget:** 35 steps | **Jobs:** 4 jobs, mixed materials, 2 urgent with deadlines

Four jobs with competing priorities across printers with varying reliability. One printer starts in `ERROR`. Printer 2 needs maintenance within 5 steps (will degrade if ignored). The unreliable printer 3 (reliability=0.70) will likely fail mid-print. Two urgent jobs have tight deadlines. The agent must triage the broken printer, schedule maintenance proactively, route critical jobs to reliable printers, re-assign after stochastic failures, and manage limited inventory. Even an optimal strategy scores 0.0–0.95 depending on RNG — this is intentional and mirrors real-world uncertainty.

## Scoring

Scores range from 0.0 to 1.0. The grader uses priority-weighted job completion:
- **Priority 3** jobs are worth 2x, **priority 2** worth 1x, **priority 1** worth 0.5x
- Meeting deadlines gives full credit; late completion is penalised (40–60% credit)
- In-progress jobs earn partial credit proportional to progress
- Failed jobs with partial progress earn small credit (10–15%)
- Step penalties for wasted actions and invalid commands
- Task 3 awards a bonus for meeting all urgent deadlines

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the baseline inference script:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"
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

## Baseline Scores

| Model | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | Total | Time |
|---|---|---|---|---|---|
| GPT-4 | 0.990 | 0.850 | **0.920** | **2.760** | 17.6min |
| GPT-5.2 | 0.990 | 0.820 | **0.800** | **2.610** | 1.1min |
| GPT-5.4 | 0.980 | **0.940** | 0.567 | **2.487** | 1.1min |
| GPT-4.1 | 0.990 | 0.840 | 0.557 | 2.387 | 1.6min |
| GPT-5.1 | 0.990 | 0.900 | 0.473 | 2.363 | 1.2min |
| GPT-4o | **1.000** | 0.737 | 0.400 | 2.137 | 1.6min |
| GPT-3.5-turbo | 0.260 | 0.497 | 0.000 | 0.757 | 1.1min |
| WAIT-only (no API key) | 0.000 | 0.000 | 0.000 | 0.000 | — |

Task 3 scores vary across runs due to stochastic printer failures (seeded RNG). All models complete well under the 20-minute inference limit.

## Repository Structure

- `printfarm_env/` — Core environment: models, physics simulation, task definitions, graders
- `server/` — FastAPI server wrapping the environment for HTTP access
- `openenv.yaml` — OpenEnv spec configuration
- `inference.py` — Baseline agent using OpenAI-compatible API
- `Dockerfile` — Container for Hugging Face Spaces deployment
