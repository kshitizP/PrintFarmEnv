"""
build_sft_dataset_v2.py — Rebuild balanced SFT dataset for round 2.

Strategy:
  1. Cap existing examples: RUN_DIAGNOSTIC ≤ 20, WAIT ≤ 20, keep all ASSIGN_JOB (15),
     REQUEST_SPOOL_SWAP (10). Subtotal: ~65 real examples.
  2. Add 5-6 unique hand-authored observations × 1-2 copies for each of 5 missing actions:
     CANCEL_JOB, PAUSE_JOB, RESUME_JOB, DISPATCH_TICKET, REQUEST_MAINTENANCE. Subtotal: ~50.
  3. Total: ~115. Assert ≥10 per scorable action, 0 OVERRIDE_OPERATOR.

Usage:
    python -m submission.training.build_sft_dataset_v2 \
        --in_jsonl submission/data/sft_warm.jsonl \
        --out submission/data/sft_warm.jsonl
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

_here = Path(__file__).resolve().parent
for _candidate in [_here.parent, _here.parent.parent]:
    if (_candidate / "submission" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from submission.shared.prompt import SYSTEM_PROMPT


# ── Capping logic ──────────────────────────────────────────────────────────────

CAPS = {
    "RUN_DIAGNOSTIC": 20,
    "WAIT": 20,
    "ASSIGN_JOB": 15,
    "REQUEST_SPOOL_SWAP": 10,
}


def _action_type(row: dict) -> str:
    for m in row["messages"]:
        if m["role"] == "assistant":
            t = re.search(r'"action_type":"([^"]+)"', m["content"])
            if t:
                return t.group(1)
    return "UNKNOWN"


def cap_existing(in_path: str) -> list:
    counts: Counter = Counter()
    kept = []
    with open(in_path) as f:
        for line in f:
            row = json.loads(line)
            at = _action_type(row)
            cap = CAPS.get(at, 0)
            if cap == 0:
                continue
            if counts[at] < cap:
                # Normalize system prompt to current version (drops OVERRIDE_OPERATOR)
                for m in row["messages"]:
                    if m["role"] == "system":
                        m["content"] = SYSTEM_PROMPT
                kept.append(row)
                counts[at] += 1
    print(f"Kept from existing: {dict(counts)}  total={len(kept)}")
    return kept


# ── Synthetic example factory ──────────────────────────────────────────────────

def _make_row(obs_text: str, action_dict: dict) -> dict:
    action_str = "<action>" + json.dumps(action_dict, separators=(",", ":")) + "</action>"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current state:\n{obs_text}"},
            {"role": "assistant", "content": action_str},
        ],
        "scenario": "synthetic",
        "oracle_action_type": action_dict["action_type"],
    }


# ── CANCEL_JOB synthetic scenarios ────────────────────────────────────────────
# Customer explicitly asks to cancel a pending job.

_CANCEL_OBS = [
    # 1
    ("""STEP 8/60 | Net profit: $3.20

PRINTERS:
  - P1: IDLE, spool=900g
  - P2: PRINTING, mat=PLA, spool=750g, job=j1
  - P3: IDLE, spool=1000g

JOBS:
  - j1: PRINTING, mat=PLA, progress=12%, $45, deadline_in=40steps
  - j2: PENDING, mat=PLA, weight=200g, time=16steps, $50, deadline_in=30steps
  - j3: PENDING, mat=PETG, weight=180g, time=14steps, $60, deadline_in=35steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=500g, PETG=1000g, PLA=2000g

CUSTOMER MESSAGES:
  - [?] job=j2: j2 — customer says cancel my order, no longer needed""",
     {"action_type": "CANCEL_JOB", "job_id": "j2"}),

    # 2
    ("""STEP 15/90 | Net profit: $8.10

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: IDLE, spool=1000g
  - P3: PRINTING, mat=PETG, spool=600g, job=j5
  - P4: IDLE, spool=800g

JOBS:
  - j4: PENDING, mat=PLA, weight=220g, time=18steps, $55, deadline_in=28steps
  - j5: PRINTING, mat=PETG, progress=5%, $80, deadline_in=45steps, priority=HIGH
  - j6: PENDING, mat=ABS, weight=160g, time=13steps, $40, deadline_in=50steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=1/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=1000g, PETG=800g, PLA=1500g

CUSTOMER MESSAGES:
  - [?] job=j4: j4 — customer says nevermind, please drop this print""",
     {"action_type": "CANCEL_JOB", "job_id": "j4"}),

    # 3
    ("""STEP 5/60 | Net profit: $1.50

PRINTERS:
  - P1: PRINTING, mat=ABS, spool=700g, job=j1
  - P2: IDLE, spool=1000g
  - P3: IDLE, spool=1000g

JOBS:
  - j1: PRINTING, mat=ABS, progress=3%, $70, deadline_in=38steps
  - j2: PENDING, mat=PLA, weight=150g, time=12steps, $30, deadline_in=20steps
  - j7: PENDING, mat=PETG, weight=200g, time=16steps, $65, deadline_in=42steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s2: senior, OFF-SHIFT, queue=0/5

INVENTORY: ABS=900g, PETG=1000g, PLA=2000g

CUSTOMER MESSAGES:
  - [?] job=j7: j7 — customer says abort the print, uploaded wrong file""",
     {"action_type": "CANCEL_JOB", "job_id": "j7"}),

    # 4
    ("""STEP 12/75 | Net profit: $6.40

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: IDLE, spool=1000g
  - P3: PRINTING, mat=PLA, spool=850g, job=j3
  - P4: IDLE, spool=600g

JOBS:
  - j3: PRINTING, mat=PLA, progress=18%, $55, deadline_in=55steps
  - j8: PENDING, mat=PETG, weight=250g, time=20steps, $90, deadline_in=32steps, priority=HIGH
  - j9: PENDING, mat=PLA, weight=170g, time=13steps, $35, deadline_in=48steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=500g, PETG=1000g, PLA=1800g

CUSTOMER MESSAGES:
  - [?] job=j9: j9 — customer says please cancel, switching to a different vendor""",
     {"action_type": "CANCEL_JOB", "job_id": "j9"}),

    # 5
    ("""STEP 20/90 | Net profit: $12.00

PRINTERS:
  - P1: PRINTING, mat=PLA, spool=750g, job=j2
  - P2: IDLE, spool=1000g
  - P3: PRINTING, mat=PETG, spool=600g, job=j4
  - P4: IDLE, spool=900g

JOBS:
  - j2: PRINTING, mat=PLA, progress=22%, $50, deadline_in=40steps
  - j4: PRINTING, mat=PETG, progress=10%, $80, deadline_in=60steps
  - j10: PENDING, mat=ABS, weight=200g, time=16steps, $60, deadline_in=25steps
  - j11: PENDING, mat=PLA, weight=130g, time=10steps, $25, deadline_in=38steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3

INVENTORY: ABS=800g, PETG=900g, PLA=2000g

CUSTOMER MESSAGES:
  - [?] job=j10: j10 — customer says need to cancel, requirements changed""",
     {"action_type": "CANCEL_JOB", "job_id": "j10"}),

    # 6
    ("""STEP 3/60 | Net profit: $0.30

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: IDLE, spool=1000g

JOBS:
  - j1: PENDING, mat=PLA, weight=200g, time=16steps, $45, deadline_in=22steps
  - j2: PENDING, mat=PETG, weight=180g, time=14steps, $55, deadline_in=30steps
  - j3: PENDING, mat=ABS, weight=250g, time=20steps, $70, deadline_in=45steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=1000g, PETG=1000g, PLA=2000g

CUSTOMER MESSAGES:
  - [?] job=j1: j1 — customer says cancel this please, bought the item elsewhere""",
     {"action_type": "CANCEL_JOB", "job_id": "j1"}),
]


# ── PAUSE_JOB synthetic scenarios ─────────────────────────────────────────────
# Novel physical anomaly during printing → pause to prevent damage.

_PAUSE_OBS = [
    # 1 — oscillating Z wobble
    ("""STEP 22/90 | Net profit: $14.50

PRINTERS:
  - P1: IDLE, spool=900g
  - P2: IDLE, spool=1000g
  - P3: PRINTING, mat=PLA, spool=700g, job=j3, fatigue=3
  - P4: IDLE, spool=800g

JOBS:
  - j3: PRINTING, mat=PLA, progress=28%, $55, deadline_in=42steps
  - j5: PENDING, mat=PETG, weight=200g, time=16steps, $65, deadline_in=35steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=800g, PETG=1000g, PLA=1500g

OPERATOR NOTES:
  - P3: oscillating Z wobble visible, layer shifting in progress — may damage print""",
     {"action_type": "PAUSE_JOB", "printer_id": 3}),

    # 2 — grinding noise
    ("""STEP 30/90 | Net profit: $20.00

PRINTERS:
  - P1: PRINTING, mat=PETG, spool=650g, job=j1, fatigue=2
  - P2: PRINTING, mat=ABS, spool=500g, job=j2
  - P3: IDLE, spool=1000g
  - P4: IDLE, spool=1000g

JOBS:
  - j1: PRINTING, mat=PETG, progress=35%, $75, deadline_in=50steps, priority=HIGH
  - j2: PRINTING, mat=ABS, progress=20%, $60, deadline_in=60steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=1/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=1000g, PETG=900g, PLA=2000g

OPERATOR NOTES:
  - P2: grinding noise from extruder gears, unusual resistance when pushing filament""",
     {"action_type": "PAUSE_JOB", "printer_id": 2}),

    # 3 — burning smell
    ("""STEP 18/60 | Net profit: $11.20

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: IDLE, spool=800g
  - P3: IDLE, spool=1000g
  - P4: PRINTING, mat=ABS, spool=600g, job=j6, fatigue=1

JOBS:
  - j6: PRINTING, mat=ABS, progress=15%, $80, deadline_in=35steps, priority=HIGH
  - j7: PENDING, mat=PLA, weight=180g, time=14steps, $40, deadline_in=28steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3

INVENTORY: ABS=900g, PETG=1000g, PLA=1500g

OPERATOR NOTES:
  - P4: unusual burning smell from hotend area, not normal ABS odor""",
     {"action_type": "PAUSE_JOB", "printer_id": 4}),

    # 4 — belt skip / irregular layers
    ("""STEP 40/90 | Net profit: $28.00

PRINTERS:
  - P1: PRINTING, mat=PLA, spool=800g, job=j4
  - P2: IDLE, spool=1000g
  - P3: PRINTING, mat=PETG, spool=700g, job=j5, fatigue=4

JOBS:
  - j4: PRINTING, mat=PLA, progress=44%, $55, deadline_in=48steps
  - j5: PRINTING, mat=PETG, progress=38%, $90, deadline_in=52steps, priority=HIGH
  - j8: PENDING, mat=ABS, weight=220g, time=18steps, $65, deadline_in=30steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=700g, PETG=800g, PLA=2000g

OPERATOR NOTES:
  - P3: layer lines irregular on side walls, suspected belt skip mid-print""",
     {"action_type": "PAUSE_JOB", "printer_id": 3}),

    # 5 — abnormal print head vibration
    ("""STEP 25/75 | Net profit: $16.00

PRINTERS:
  - P1: PRINTING, mat=PLA, spool=850g, job=j2, fatigue=5
  - P2: IDLE, spool=1000g
  - P3: IDLE, spool=900g

JOBS:
  - j2: PRINTING, mat=PLA, progress=30%, $50, deadline_in=45steps
  - j3: PENDING, mat=PETG, weight=200g, time=16steps, $70, deadline_in=40steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=1000g, PETG=1000g, PLA=1800g

OPERATOR NOTES:
  - P1: print head vibrating abnormally, carriage wobbling more than usual""",
     {"action_type": "PAUSE_JOB", "printer_id": 1}),

    # 6 — intermittent ticking, misaligned layers
    ("""STEP 35/90 | Net profit: $22.50

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: PRINTING, mat=PETG, spool=550g, job=j7
  - P3: IDLE, spool=1000g
  - P4: PRINTING, mat=PLA, spool=800g, job=j8, fatigue=3
  - P5: IDLE, spool=900g
  - P6: PRINTING, mat=ABS, spool=600g, job=j9, fatigue=2

JOBS:
  - j7: PRINTING, mat=PETG, progress=40%, $85, deadline_in=55steps, priority=HIGH
  - j8: PRINTING, mat=PLA, progress=32%, $45, deadline_in=48steps
  - j9: PRINTING, mat=ABS, progress=25%, $65, deadline_in=50steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=900g, PETG=700g, PLA=1500g

OPERATOR NOTES:
  - P6: intermittent ticking noise from Z stepper, layer surfaces look misaligned""",
     {"action_type": "PAUSE_JOB", "printer_id": 6}),
]


# ── RESUME_JOB synthetic scenarios ────────────────────────────────────────────
# Post-pause: operator resolved the issue, printer PAUSED, ready to resume.

_RESUME_OBS = [
    # 1 — P2 jam cleared
    ("""STEP 28/90 | Net profit: $18.00

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: PAUSED, mat=PLA, spool=700g, job=j3
  - P3: PRINTING, mat=PETG, spool=650g, job=j4

JOBS:
  - j3: PAUSED, mat=PLA, progress=25%, $50, deadline_in=50steps
  - j4: PRINTING, mat=PETG, progress=18%, $75, deadline_in=60steps, priority=HIGH
  - j5: PENDING, mat=ABS, weight=200g, time=16steps, $60, deadline_in=35steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=1000g, PETG=900g, PLA=2000g

OPERATOR NOTES:
  - P2: jam cleared, filament reloaded and primed, ready to resume""",
     {"action_type": "RESUME_JOB", "printer_id": 2, "job_id": "j3"}),

    # 2 — P4 bed re-leveled
    ("""STEP 45/90 | Net profit: $32.00

PRINTERS:
  - P1: PRINTING, mat=PETG, spool=600g, job=j1
  - P2: IDLE, spool=1000g
  - P3: IDLE, spool=1000g
  - P4: PAUSED, mat=ABS, spool=500g, job=j7

JOBS:
  - j1: PRINTING, mat=PETG, progress=50%, $80, deadline_in=45steps, priority=HIGH
  - j7: PAUSED, mat=ABS, progress=30%, $70, deadline_in=40steps
  - j9: PENDING, mat=PLA, weight=150g, time=12steps, $35, deadline_in=25steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=800g, PETG=700g, PLA=1500g

OPERATOR NOTES:
  - P4: bed re-leveled and mesh updated, first layer test passed, ready to resume""",
     {"action_type": "RESUME_JOB", "printer_id": 4, "job_id": "j7"}),

    # 3 — P3 Z axis fixed
    ("""STEP 38/90 | Net profit: $26.00

PRINTERS:
  - P1: IDLE, spool=800g
  - P2: PRINTING, mat=PLA, spool=750g, job=j2
  - P3: PAUSED, mat=PETG, spool=600g, job=j5
  - P4: IDLE, spool=1000g

JOBS:
  - j2: PRINTING, mat=PLA, progress=42%, $55, deadline_in=38steps
  - j5: PAUSED, mat=PETG, progress=35%, $90, deadline_in=50steps, priority=HIGH
  - j6: PENDING, mat=ABS, weight=220g, time=18steps, $65, deadline_in=30steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=900g, PETG=800g, PLA=1800g

OPERATOR NOTES:
  - P3: Z wobble resolved, tightened eccentric nuts and belt, confirmed stable""",
     {"action_type": "RESUME_JOB", "printer_id": 3, "job_id": "j5"}),

    # 4 — P1 extruder gears replaced
    ("""STEP 55/90 | Net profit: $40.00

PRINTERS:
  - P1: PAUSED, mat=PLA, spool=850g, job=j2
  - P2: PRINTING, mat=PETG, spool=500g, job=j6
  - P3: IDLE, spool=1000g

JOBS:
  - j2: PAUSED, mat=PLA, progress=28%, $50, deadline_in=32steps
  - j6: PRINTING, mat=PETG, progress=55%, $75, deadline_in=35steps, priority=HIGH
  - j10: PENDING, mat=ABS, weight=200g, time=16steps, $55, deadline_in=28steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=700g, PETG=600g, PLA=2000g

OPERATOR NOTES:
  - P1: grinding fixed, extruder gears replaced, extrusion test normal, ready to resume""",
     {"action_type": "RESUME_JOB", "printer_id": 1, "job_id": "j2"}),

    # 5 — P5 sensor recalibrated
    ("""STEP 20/75 | Net profit: $13.00

PRINTERS:
  - P1: PRINTING, mat=PLA, spool=800g, job=j1
  - P2: IDLE, spool=1000g
  - P3: IDLE, spool=1000g
  - P4: IDLE, spool=900g
  - P5: PAUSED, mat=ABS, spool=650g, job=j9

JOBS:
  - j1: PRINTING, mat=PLA, progress=22%, $45, deadline_in=50steps
  - j9: PAUSED, mat=ABS, progress=18%, $65, deadline_in=40steps
  - j11: PENDING, mat=PETG, weight=180g, time=14steps, $55, deadline_in=30steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=1/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=800g, PETG=1000g, PLA=1500g

OPERATOR NOTES:
  - P5: filament sensor recalibrated, false-positive runout trigger resolved, cleared to resume""",
     {"action_type": "RESUME_JOB", "printer_id": 5, "job_id": "j9"}),

    # 6 — P6 inspected, no damage found
    ("""STEP 62/90 | Net profit: $45.00

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: PRINTING, mat=PLA, spool=700g, job=j3
  - P3: PRINTING, mat=PETG, spool=600g, job=j5
  - P4: IDLE, spool=900g
  - P5: IDLE, spool=800g
  - P6: PAUSED, mat=ABS, spool=550g, job=j11

JOBS:
  - j3: PRINTING, mat=PLA, progress=68%, $50, deadline_in=22steps
  - j5: PRINTING, mat=PETG, progress=60%, $80, deadline_in=28steps, priority=HIGH
  - j11: PAUSED, mat=ABS, progress=42%, $70, deadline_in=25steps
  - j12: PENDING, mat=PLA, weight=150g, time=12steps, $30, deadline_in=18steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=700g, PETG=800g, PLA=1800g

OPERATOR NOTES:
  - P6: full inspection complete, no structural damage, ticking noise was loose fan screw, cleared for restart""",
     {"action_type": "RESUME_JOB", "printer_id": 6, "job_id": "j11"}),
]


# ── DISPATCH_TICKET synthetic scenarios ───────────────────────────────────────
# Escalation needed: diagnostic ran but fault persists, or safety concern.

_DISPATCH_OBS = [
    # 1 — recurring fault, second diagnostic needed escalation
    ("""STEP 35/90 | Net profit: $22.00

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: IDLE, spool=900g
  - P3: IDLE, spool=1000g, reliability=0.72
  - P4: PRINTING, mat=PETG, spool=600g, job=j4

JOBS:
  - j4: PRINTING, mat=PETG, progress=38%, $75, deadline_in=52steps, priority=HIGH
  - j5: PENDING, mat=PLA, weight=200g, time=16steps, $50, deadline_in=30steps
  - j6: PENDING, mat=ABS, weight=250g, time=20steps, $65, deadline_in=45steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=1/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=900g, PETG=800g, PLA=1500g

OPERATOR NOTES:
  - P3: diagnostic ran at step 28 but fault_code THERM_ERR still present, requires specialist""",
     {"action_type": "DISPATCH_TICKET", "printer_id": 3, "operator_id": "op_s1", "ticket_type": "repair"}),

    # 2 — smoke observed, safety escalation
    ("""STEP 18/60 | Net profit: $10.00

PRINTERS:
  - P1: PRINTING, mat=PLA, spool=800g, job=j1
  - P2: IDLE, spool=1000g
  - P3: IDLE, spool=1000g
  - P4: IDLE, spool=700g, reliability=0.80
  - P5: PRINTING, mat=ABS, spool=500g, job=j3

JOBS:
  - j1: PRINTING, mat=PLA, progress=18%, $50, deadline_in=40steps
  - j3: PRINTING, mat=ABS, progress=12%, $70, deadline_in=38steps, priority=HIGH
  - j7: PENDING, mat=PETG, weight=180g, time=14steps, $55, deadline_in=28steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=800g, PETG=1000g, PLA=1800g

OPERATOR NOTES:
  - P4: faint smoke trace observed from hotend, diagnostic ran but hotend_temp normal — safety check required""",
     {"action_type": "DISPATCH_TICKET", "printer_id": 4, "operator_id": "op_s1", "ticket_type": "safety_inspection"}),

    # 3 — customer complaint + anomaly, escalate
    ("""STEP 50/90 | Net profit: $36.00

PRINTERS:
  - P1: PRINTING, mat=PLA, spool=700g, job=j2
  - P2: IDLE, spool=1000g
  - P3: IDLE, spool=800g, reliability=0.75
  - P4: PRINTING, mat=PETG, spool=550g, job=j6

JOBS:
  - j2: PRINTING, mat=PLA, progress=55%, $55, deadline_in=35steps
  - j6: PRINTING, mat=PETG, progress=48%, $90, deadline_in=42steps, priority=HIGH
  - j8: PENDING, mat=ABS, weight=200g, time=16steps, $60, deadline_in=22steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=700g, PETG=600g, PLA=1800g

OPERATOR NOTES:
  - P3: repeated jam events (3x this session), diagnostic inconclusive each time — escalate for hands-on repair""",
     {"action_type": "DISPATCH_TICKET", "printer_id": 3, "operator_id": "op_s1", "ticket_type": "repair"}),

    # 4 — hotend failure pattern
    ("""STEP 42/75 | Net profit: $30.00

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: PRINTING, mat=ABS, spool=600g, job=j3
  - P3: IDLE, spool=900g
  - P4: IDLE, spool=1000g
  - P5: IDLE, spool=800g, reliability=0.68

JOBS:
  - j3: PRINTING, mat=ABS, progress=45%, $75, deadline_in=30steps, priority=HIGH
  - j4: PENDING, mat=PLA, weight=200g, time=16steps, $50, deadline_in=25steps
  - j5: PENDING, mat=PETG, weight=180g, time=14steps, $60, deadline_in=38steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=800g, PETG=1000g, PLA=1500g

OPERATOR NOTES:
  - P5: hotend clog recurring every 20 steps, two diagnostics done, nozzle likely worn — needs replacement""",
     {"action_type": "DISPATCH_TICKET", "printer_id": 5, "operator_id": "op_s1", "ticket_type": "repair"}),

    # 5 — layer adhesion defects requiring specialist
    ("""STEP 28/60 | Net profit: $18.50

PRINTERS:
  - P1: PRINTING, mat=PLA, spool=850g, job=j1
  - P2: IDLE, spool=1000g
  - P3: IDLE, spool=700g
  - P4: IDLE, spool=900g, reliability=0.78

JOBS:
  - j1: PRINTING, mat=PLA, progress=32%, $55, deadline_in=28steps
  - j2: PENDING, mat=ABS, weight=230g, time=18steps, $70, deadline_in=22steps
  - j3: PENDING, mat=PETG, weight=160g, time=13steps, $50, deadline_in=32steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=900g, PETG=1000g, PLA=1800g

OPERATOR NOTES:
  - P4: structural layer adhesion defects on last three jobs, temp sensor suspected faulty, diagnostic unclear""",
     {"action_type": "DISPATCH_TICKET", "printer_id": 4, "operator_id": "op_s1", "ticket_type": "diagnostic"}),

    # 6 — wiring concern
    ("""STEP 15/75 | Net profit: $9.50

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: PRINTING, mat=PETG, spool=700g, job=j5
  - P3: IDLE, spool=1000g
  - P4: IDLE, spool=900g
  - P5: PRINTING, mat=PLA, spool=800g, job=j6
  - P6: IDLE, spool=600g, reliability=0.70

JOBS:
  - j5: PRINTING, mat=PETG, progress=15%, $80, deadline_in=60steps, priority=HIGH
  - j6: PRINTING, mat=PLA, progress=10%, $45, deadline_in=55steps
  - j7: PENDING, mat=ABS, weight=200g, time=16steps, $55, deadline_in=30steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=800g, PETG=900g, PLA=1500g

OPERATOR NOTES:
  - P6: visible wire insulation fraying near heated bed connector, safety hazard — do not run until inspected""",
     {"action_type": "DISPATCH_TICKET", "printer_id": 6, "operator_id": "op_s1", "ticket_type": "safety_inspection"}),
]


# ── REQUEST_MAINTENANCE synthetic scenarios ───────────────────────────────────
# IDLE printer with high fatigue (≥ 7), no active fault.

_MAINTENANCE_OBS = [
    # 1 — P3 fatigue=7
    ("""STEP 10/90 | Net profit: $5.50

PRINTERS:
  - P1: IDLE, spool=1000g
  - P2: PRINTING, mat=PETG, spool=700g, job=j2
  - P3: IDLE, spool=900g, fatigue=7
  - P4: IDLE, spool=1000g

JOBS:
  - j2: PRINTING, mat=PETG, progress=10%, $80, deadline_in=70steps, priority=HIGH
  - j4: PENDING, mat=PLA, weight=200g, time=16steps, $50, deadline_in=50steps
  - j5: PENDING, mat=ABS, weight=230g, time=18steps, $65, deadline_in=60steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=1000g, PETG=900g, PLA=2000g""",
     {"action_type": "REQUEST_MAINTENANCE", "printer_id": 3, "maintenance_type": "maintenance_basic"}),

    # 2 — P1 fatigue=8
    ("""STEP 25/90 | Net profit: $16.00

PRINTERS:
  - P1: IDLE, spool=800g, fatigue=8
  - P2: IDLE, spool=1000g
  - P3: PRINTING, mat=ABS, spool=600g, job=j3
  - P4: PRINTING, mat=PLA, spool=750g, job=j4

JOBS:
  - j3: PRINTING, mat=ABS, progress=28%, $70, deadline_in=55steps
  - j4: PRINTING, mat=PLA, progress=22%, $50, deadline_in=58steps
  - j6: PENDING, mat=PETG, weight=180g, time=14steps, $60, deadline_in=40steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=900g, PETG=1000g, PLA=1500g""",
     {"action_type": "REQUEST_MAINTENANCE", "printer_id": 1, "maintenance_type": "maintenance_basic"}),

    # 3 — P5 fatigue=9
    ("""STEP 40/90 | Net profit: $28.00

PRINTERS:
  - P1: PRINTING, mat=PLA, spool=700g, job=j1
  - P2: PRINTING, mat=PETG, spool=600g, job=j3
  - P3: IDLE, spool=1000g
  - P4: IDLE, spool=900g
  - P5: IDLE, spool=800g, fatigue=9

JOBS:
  - j1: PRINTING, mat=PLA, progress=44%, $55, deadline_in=48steps
  - j3: PRINTING, mat=PETG, progress=38%, $80, deadline_in=50steps, priority=HIGH
  - j7: PENDING, mat=ABS, weight=200g, time=16steps, $60, deadline_in=35steps
  - j8: PENDING, mat=PLA, weight=150g, time=12steps, $35, deadline_in=42steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=800g, PETG=700g, PLA=2000g""",
     {"action_type": "REQUEST_MAINTENANCE", "printer_id": 5, "maintenance_type": "maintenance_basic"}),

    # 4 — P2 fatigue=7
    ("""STEP 55/90 | Net profit: $40.00

PRINTERS:
  - P1: PRINTING, mat=PETG, spool=500g, job=j2
  - P2: IDLE, spool=900g, fatigue=7
  - P3: PRINTING, mat=ABS, spool=600g, job=j4
  - P4: IDLE, spool=1000g

JOBS:
  - j2: PRINTING, mat=PETG, progress=60%, $85, deadline_in=30steps, priority=HIGH
  - j4: PRINTING, mat=ABS, progress=50%, $70, deadline_in=35steps
  - j9: PENDING, mat=PLA, weight=180g, time=14steps, $45, deadline_in=28steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=700g, PETG=600g, PLA=1800g""",
     {"action_type": "REQUEST_MAINTENANCE", "printer_id": 2, "maintenance_type": "maintenance_basic"}),

    # 5 — P4 fatigue=8
    ("""STEP 20/60 | Net profit: $12.00

PRINTERS:
  - P1: PRINTING, mat=PLA, spool=800g, job=j1
  - P2: IDLE, spool=1000g
  - P3: IDLE, spool=1000g
  - P4: IDLE, spool=700g, fatigue=8
  - P5: IDLE, spool=900g

JOBS:
  - j1: PRINTING, mat=PLA, progress=20%, $55, deadline_in=38steps
  - j3: PENDING, mat=PETG, weight=200g, time=16steps, $70, deadline_in=32steps, priority=HIGH
  - j5: PENDING, mat=ABS, weight=180g, time=14steps, $50, deadline_in=40steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=1000g, PETG=900g, PLA=1500g""",
     {"action_type": "REQUEST_MAINTENANCE", "printer_id": 4, "maintenance_type": "maintenance_basic"}),

    # 6 — P6 fatigue=7
    ("""STEP 32/75 | Net profit: $22.00

PRINTERS:
  - P1: PRINTING, mat=PETG, spool=600g, job=j2
  - P2: IDLE, spool=1000g
  - P3: PRINTING, mat=PLA, spool=750g, job=j4
  - P4: IDLE, spool=800g
  - P5: IDLE, spool=1000g
  - P6: IDLE, spool=900g, fatigue=7

JOBS:
  - j2: PRINTING, mat=PETG, progress=35%, $80, deadline_in=40steps, priority=HIGH
  - j4: PRINTING, mat=PLA, progress=30%, $50, deadline_in=42steps
  - j6: PENDING, mat=ABS, weight=220g, time=18steps, $65, deadline_in=30steps
  - j7: PENDING, mat=PLA, weight=150g, time=12steps, $35, deadline_in=36steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: ABS=900g, PETG=700g, PLA=2000g""",
     {"action_type": "REQUEST_MAINTENANCE", "printer_id": 6, "maintenance_type": "maintenance_basic"}),
]


# ── Assemble synthetic examples with duplication ──────────────────────────────

def build_synthetic(obs_list: list, copies: int = 2) -> list:
    rows = []
    for obs_text, action_dict in obs_list:
        for _ in range(copies):
            rows.append(_make_row(obs_text.strip(), action_dict))
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_jsonl", default="submission/data/sft_warm.jsonl")
    p.add_argument("--out", default="submission/data/sft_warm.jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # 1. Cap existing examples
    real_rows = cap_existing(args.in_jsonl)

    # 2. Build synthetic (2 copies each → 12 per action)
    synthetic_rows = (
        build_synthetic(_CANCEL_OBS, copies=2)
        + build_synthetic(_PAUSE_OBS, copies=2)
        + build_synthetic(_RESUME_OBS, copies=2)
        + build_synthetic(_DISPATCH_OBS, copies=2)
        + build_synthetic(_MAINTENANCE_OBS, copies=2)
    )
    print(f"Synthetic rows: {len(synthetic_rows)}")

    # 3. Combine and shuffle
    all_rows = real_rows + synthetic_rows
    random.seed(args.seed)
    random.shuffle(all_rows)

    # 4. Verify distribution
    counts = Counter(_action_type(r) for r in all_rows)
    print(f"\nFinal action distribution ({len(all_rows)} total):")
    for k, v in counts.most_common():
        print(f"  {k:30s}: {v:3d}")

    assert "OVERRIDE_OPERATOR" not in counts, "OVERRIDE_OPERATOR found — abort"
    scorable = [
        "RUN_DIAGNOSTIC", "WAIT", "ASSIGN_JOB", "REQUEST_SPOOL_SWAP",
        "CANCEL_JOB", "PAUSE_JOB", "RESUME_JOB",
        "DISPATCH_TICKET", "REQUEST_MAINTENANCE",
    ]
    for action in scorable:
        assert counts[action] >= 10, f"{action} has only {counts[action]} examples (need ≥10)"
    print("\nAll distribution assertions passed.")

    # 5. Write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(all_rows)} examples → {out_path}")


if __name__ == "__main__":
    main()
