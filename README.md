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

An OpenEnv environment that simulates managing a fleet of 10 networked 3D printers. The agent acts as an automated floor manager — assigning print jobs, swapping filament spools, triaging hardware errors, and managing material inventory — mirroring the real operational decisions made daily in commercial print farms and makerspaces.

## Motivation

3D print farm management is a genuine, unsolved operational challenge. Farms with 10–100+ printers must continuously match incoming orders to available machines, handle mid-print failures (filament runouts, hardware jams), and minimize downtime. Current solutions rely on manual oversight or rigid rule-based schedulers. This environment provides a realistic testbed for evaluating whether AI agents can learn these coordination and recovery skills.

## Action Space

The agent submits one action per time step as a `FarmAction` with the following fields:

| Field | Type | Description |
|---|---|---|
| `action` | enum | One of: `ASSIGN_JOB`, `SWAP_FILAMENT`, `CANCEL_JOB`, `WAIT` |
| `printer_id` | int (optional) | Target printer ID (1–10) |
| `job_id` | string (optional) | Target job ID from the active queue |
| `material` | string (optional) | Filament type for `SWAP_FILAMENT` (e.g. `"PLA"`, `"PETG"`, `"ABS"`) |

**Action details:**
- **ASSIGN_JOB**: Assigns a pending job to an idle printer. Requires matching material and sufficient spool weight.
- **SWAP_FILAMENT**: Replaces the filament spool on an idle/errored printer from inventory. Takes a full 1000g spool.
- **CANCEL_JOB**: Cancels a pending or in-progress job, freeing the printer.
- **WAIT**: Do nothing this step (penalized by the grader to discourage idle behavior).

## Observation Space

Each step returns a `FarmObservation` containing:

| Field | Type | Description |
|---|---|---|
| `printers` | list[PrinterObservation] | State of all 10 printers |
| `active_queue` | list[PrintJob] | All jobs and their current status |
| `inventory` | dict[str, float] | Material inventory in grams (e.g. `{"PLA": 5000.0}`) |
| `reward` | float | Current grader score (0.0–1.0) |
| `done` | bool | Whether the episode has ended |
| `metadata` | dict | Error messages and action feedback |

**PrinterObservation fields:** `printer_id`, `state` (IDLE/PRINTING/ERROR), `current_material`, `current_job_id`, `spool_weight_g`

**PrintJob fields:** `job_id`, `material_required`, `weight_required_g`, `state` (PENDING/PRINTING/COMPLETED/FAILED/CANCELLED), `print_time_steps`, `progress_steps`

## Tasks

### Task 1: Night Shift (Easy)
**Objective:** Assign 5 PLA print jobs across 10 printers (3 have loaded spools, others need filament swaps).

The agent must recognize which printers are ready and assign jobs without material mismatches. Straightforward scheduling — no failures occur.

**Scoring:** Fractional progress across all 5 jobs. Full credit (1.0) when all jobs complete. Any failed or cancelled job scores 0.0.

### Task 2: Spool Runout (Medium)
**Objective:** Complete a heavy 800g PETG print, but the only loaded printer has just 200g remaining.

The agent must swap the nearly-empty spool for a full one from inventory before starting the print. Naively assigning the job causes a mid-print filament runout and failure.

**Scoring:** 1.0 for successful completion with proactive filament swap. 0.3 if the job fails mid-print (partial progress credit). Continuous signal while printing.

### Task 3: Hardware Triage (Hard)
**Objective:** Complete a 500g ABS print that triggers a hardware error at step 5, requiring the agent to recover.

The agent must detect the error, re-assign or re-route the job to another printer (possibly after a filament swap), and complete it with minimal total downtime.

**Scoring:** Based on completion and downtime efficiency. Maximum 1.0 for fast recovery, floor of 0.2 if completed with high downtime. Zero if job never completes.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
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

The API will be available at `http://localhost:7860` with endpoints:
- `POST /reset` — Reset the environment
- `POST /step` — Execute an action
- `GET /state` — Get current state
- `GET /health` — Health check
- `GET /schema` — Action/observation schemas

## Baseline Scores

Without an API key (WAIT-only fallback agent):
- Task 1: 0.0 (no jobs assigned)
- Task 2: 0.0 (no action taken)
- Task 3: 0.0 (no action taken)

Expected scores with a capable LLM agent:
- Task 1 (Easy): ~0.8–1.0
- Task 2 (Medium): ~0.7–1.0
- Task 3 (Hard): ~0.3–0.8

## Repository Structure

- `printfarm_env/` — Core environment: models, physics simulation, task definitions, graders
- `server/` — FastAPI server wrapping the environment for HTTP access
- `openenv.yaml` — OpenEnv spec configuration
- `inference.py` — Baseline agent using OpenAI-compatible API
- `Dockerfile` — Container definition for Hugging Face Spaces deployment
