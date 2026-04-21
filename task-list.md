# Round 2 Implementation Tasks

## Phase 1: models.py — Add Missing Types

- `[x]` Add `TicketState` enum
- `[x]` Add `Ticket` dataclass
- `[x]` Add `OperatorObservation` Pydantic model
- `[x]` Update `FarmActionEnum` with full Round 2 action set
- `[x]` Expand `FarmAction` with all new fields
- `[x]` Add `PAUSED` and `MAINTENANCE_QUEUED` to `PrinterState`
- `[x]` Expand `PrinterObservation` with economic + telemetry fields
- `[x]` Expand `FarmObservation` with operators, tickets, economic fields, oversight log

## Phase 2: env.py — Round 2 Engine

- `[x]` Integrate `PrinterProfile` / load profiles at init
- `[x]` Add internal `PrinterInternal` state (ground truth)
- `[x]` Rewrite `reset()` to load tasks and bootstrap operators/printers
- `[x]` Implement new action handlers: `RUN_DIAGNOSTIC`, `PAUSE_JOB`, `DISPATCH_TICKET`, `REQUEST_SPOOL_SWAP`, `REQUEST_MAINTENANCE`, `OVERRIDE_OPERATOR`
- `[x]` Integrate `operator_tick()` (tick operators before printers)
- `[x]` Integrate `failures.py` — scheduled + stochastic injection
- `[x]` Apply `apply_corruption()` to build corrupted observation
- `[x]` Replace old reward with `economics.py` dollar-denominated reward
- `[x]` Implement per-step reward accumulation (SLA, labor, electricity, profit)
- `[x]` Add Oversight log writes per step

## Phase 3: tasks.py — Round 2 Task Definitions

- `[x]` Rewrite `task_1` — Human Latency Coordination (per SIMULATOR_SPEC §6)
- `[x]` Rewrite `task_2` — Sensor Trust Calibration
- `[x]` Rewrite `task_3` — Disagreement Resolution
- `[x]` Add `task_4` — Long-Horizon Maintenance Planning (stretch)
- `[x]` Add `task_5` — Economic Stress Test (stretch/holdout)
- `[x]` Update `TaskGrader` to use dollar-based scoring

## Phase 4: Cleanup

- `[x]` Archive `experimental_tasks.py` → `_experimental_tasks_archived.py`
- `[x]` Smoke test passed (all 6 tasks load; sensor corruption verified; RUN_DIAGNOSTIC bonus confirmed)
