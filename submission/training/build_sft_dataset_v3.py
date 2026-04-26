"""
Build SFT dataset v3: 20 examples per action, 180 total.

Design rules:
- Unambiguous signals: if PAUSE_JOB, the fault is blatant
- Intra-class variance: 20 distinct phrasings, setups, printer IDs, prices
- No duplicates within a class
- Zero OVERRIDE_OPERATOR

Run: python3 -m submission.training.build_sft_dataset_v3
Writes: submission/data/sft_v3.jsonl
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from submission.shared.prompt import SYSTEM_PROMPT

OUT = ROOT / "submission" / "data" / "sft_v3.jsonl"


def _row(obs_text: str, action_json: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current state:\n{obs_text}"},
            {"role": "assistant", "content": f"<action>{action_json}</action>"},
        ]
    }


# ---------------------------------------------------------------------------
# CANCEL_JOB — customer explicitly requesting cancellation
# ---------------------------------------------------------------------------
CANCEL_PHRASES = [
    "please cancel my order, changed my mind",
    "can you cancel j{jid}? Found a cheaper supplier",
    "abort j{jid} — client pulled the contract",
    "stop j{jid}, we don't need it anymore",
    "cancel order j{jid} immediately, wrong design uploaded",
    "please cancel j{jid}, we're pausing the project indefinitely",
    "cancel j{jid} — our order was a mistake",
    "kill the print please, customer changed dimensions",
    "j{jid}: cancellation request, overordered by accident",
    "need j{jid} cancelled ASAP, deadline moved to next quarter",
    "please pull j{jid}, we'll resubmit with different specs",
    "cancel j{jid}, found it was already in stock",
    "please stop j{jid}, wrong material was selected",
    "halt j{jid} — we have a design revision incoming",
    "j{jid} cancel — client decided not to proceed",
    "need to cancel j{jid}, internal cost cut",
    "abort print j{jid} if possible, project shelved",
    "j{jid}: nevermind on this one, cancel it",
    "customer withdrawal — please stop j{jid}",
    "j{jid} cancellation request from client, confirmed via email",
]

CANCEL_SETUPS = [
    # (pid, mat, spool, jid, job_state, progress_or_weight, price, deadline_in, urgency)
    (1, "PLA",  900, "j1", "PENDING",  "weight=200g, time=15steps", 40,  20, "LOW"),
    (2, "PETG", 800, "j2", "PENDING",  "weight=250g, time=20steps", 80,  35, "?"),
    (3, "ABS",  700, "j3", "PRINTING", "progress=5%",               60,  40, "HIGH"),
    (1, "PLA",  950, "j4", "PENDING",  "weight=180g, time=12steps", 35,  18, "LOW"),
    (4, "TPU",  600, "j5", "PRINTING", "progress=3%",               90,  50, "?"),
    (2, "PETG", 850, "j6", "PENDING",  "weight=300g, time=22steps", 70,  30, "LOW"),
    (5, "PLA",  1000,"j7", "PENDING",  "weight=150g, time=10steps", 30,  15, "?"),
    (3, "ABS",  750, "j1", "PRINTING", "progress=7%",               55,  45, "LOW"),
    (1, "PLA",  900, "j2", "PENDING",  "weight=220g, time=17steps", 45,  22, "?"),
    (6, "PETG", 950, "j3", "PRINTING", "progress=4%",               75,  38, "HIGH"),
    (2, "PLA",  800, "j4", "PENDING",  "weight=190g, time=14steps", 38,  25, "LOW"),
    (4, "ABS",  700, "j5", "PRINTING", "progress=6%",               65,  42, "?"),
    (1, "PETG", 850, "j6", "PENDING",  "weight=280g, time=19steps", 72,  32, "LOW"),
    (3, "TPU",  600, "j7", "PRINTING", "progress=2%",               95,  55, "?"),
    (5, "PLA",  1000,"j1", "PENDING",  "weight=160g, time=11steps", 32,  17, "HIGH"),
    (2, "ABS",  750, "j2", "PENDING",  "weight=210g, time=16steps", 48,  28, "LOW"),
    (6, "PETG", 900, "j3", "PRINTING", "progress=8%",               82,  48, "?"),
    (1, "PLA",  850, "j4", "PENDING",  "weight=175g, time=13steps", 36,  20, "LOW"),
    (3, "ABS",  700, "j5", "PENDING",  "weight=230g, time=18steps", 58,  35, "?"),
    (4, "PLA",  950, "j6", "PRINTING", "progress=9%",               44,  26, "HIGH"),
]


def build_cancel():
    rows = []
    for i, (pid, mat, spool, jid, jstate, job_detail, price, deadline, urgency) in enumerate(CANCEL_SETUPS):
        phrase = CANCEL_PHRASES[i].replace("{jid}", jid)
        step = 5 + i % 10
        obs = f"""STEP {step}/60 | Net profit: ${step * 0.8 - 0.2:.2f}

PRINTERS:
  - P{pid}: {'PRINTING' if jstate == 'PRINTING' else 'IDLE'}, mat={mat}, spool={spool}g, job={jid}

JOBS:
  - {jid}: {jstate}, mat={mat}, {job_detail}, ${price}, deadline_in={deadline}steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: PLA=1500g, PETG=1200g, ABS=900g, TPU=600g

CUSTOMER MESSAGES:
  - [{urgency}] job={jid}: {phrase}"""
        rows.append(_row(obs, f'{{"action_type":"CANCEL_JOB","job_id":"{jid}"}}'))
    return rows


# ---------------------------------------------------------------------------
# PAUSE_JOB — blatant novel fault during printing
# ---------------------------------------------------------------------------
PAUSE_FAULTS = [
    "P{pid}: grinding noise from extruder, getting louder each layer",
    "P{pid}: burning smell coming from motor area, smoke visible",
    "P{pid}: extruder jamming intermittently, head skipping violently",
    "P{pid}: loud banging sound at start of each layer — something loose",
    "P{pid}: entire print head shaking, severe layer misalignment",
    "P{pid}: filament grinding to dust, extruder motor stalling",
    "P{pid}: crackling sound from hotend, possible electrical issue",
    "P{pid}: z-axis wobbling severely, layers completely misaligned",
    "P{pid}: visible sparks near extruder assembly — stop immediately",
    "P{pid}: smoke from under the print head, unplug recommended",
    "P{pid}: loud rhythmic clunking — belt slipping or broken tooth",
    "P{pid}: extruder clicking and skipping badly, print failing",
    "P{pid}: catastrophic layer shift at 40% — layers offset by 5mm",
    "P{pid}: stepper motor overheating, too hot to touch",
    "P{pid}: print head scraping across print — nozzle crash imminent",
    "P{pid}: vibration so severe the whole printer is moving on bench",
    "P{pid}: extruder clogged hard — motor straining audibly",
    "P{pid}: bed suddenly unlevel by 3mm — print adhesion lost",
    "P{pid}: unusual electrical buzzing from control board area",
    "P{pid}: coolant hose disconnected — hotend temp spiking uncontrolled",
]

PAUSE_SETUPS = [
    (1, "PLA",  800, "j1", 25, 55, 35),
    (2, "PETG", 650, "j2", 40, 80, 22),
    (3, "ABS",  700, "j3", 18, 60, 45),
    (4, "TPU",  500, "j4", 55, 95, 10),
    (5, "PLA",  900, "j5", 30, 45, 38),
    (1, "PETG", 750, "j6", 62, 70, 15),
    (2, "ABS",  600, "j7", 15, 50, 50),
    (3, "PLA",  850, "j1", 48, 65, 18),
    (4, "PETG", 700, "j2", 72, 85, 8),
    (5, "ABS",  550, "j3", 35, 55, 30),
    (6, "PLA",  950, "j4", 20, 40, 42),
    (1, "TPU",  450, "j5", 58, 90, 12),
    (2, "PLA",  800, "j6", 42, 75, 20),
    (3, "PETG", 650, "j7", 68, 80, 9),
    (4, "ABS",  700, "j1", 25, 60, 38),
    (5, "PLA",  900, "j2", 33, 50, 28),
    (6, "PETG", 750, "j3", 50, 70, 16),
    (1, "ABS",  600, "j4", 15, 45, 48),
    (2, "PLA",  850, "j5", 78, 88, 7),
    (3, "TPU",  500, "j6", 44, 65, 22),
]


def build_pause():
    rows = []
    for i, (pid, mat, spool, jid, progress, price, deadline) in enumerate(PAUSE_SETUPS):
        fault = PAUSE_FAULTS[i].replace("{pid}", str(pid))
        step = 10 + i * 2
        obs = f"""STEP {step}/60 | Net profit: ${step * 1.2:.2f}

PRINTERS:
  - P{pid}: PRINTING, mat={mat}, spool={spool}g, job={jid}

JOBS:
  - {jid}: PRINTING, mat={mat}, progress={progress}%, ${price}, deadline_in={deadline}steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: PLA=1500g, PETG=1200g, ABS=900g, TPU=600g

OPERATOR NOTES:
  - {fault}"""
        rows.append(_row(obs, f'{{"action_type":"PAUSE_JOB","printer_id":{pid}}}'))
    return rows


# ---------------------------------------------------------------------------
# RESUME_JOB — operator confirmed fix, printer PAUSED
# ---------------------------------------------------------------------------
RESUME_NOTES = [
    "P{pid}: cleared the extruder jam, spool reloaded, good to go",
    "P{pid}: replaced clogged nozzle, bed re-leveled, ready to resume",
    "P{pid}: filament runout resolved, new spool loaded, spool swap done",
    "P{pid}: motor issue fixed, belt retensioned, tested OK",
    "P{pid}: z-axis wobble fixed — eccentric nut tightened, smooth now",
    "P{pid}: jammed filament cleared, cold pull done, path clean",
    "P{pid}: bed adhesion issue resolved, new PEI sheet applied",
    "P{pid}: hotend thermal runaway cleared, thermistor reseated",
    "P{pid}: print head cleaned, lubed, and tested — all clear",
    "P{pid}: spool swap complete, correct material loaded and confirmed",
    "P{pid}: electrical short resolved, connection retightened",
    "P{pid}: layer shift root cause fixed — pulley set screw tightened",
    "P{pid}: firmware fault cleared by power cycle, printer responsive",
    "P{pid}: filament sensor recalibrated, false runout won't repeat",
    "P{pid}: cooling fan replaced, temp stable, safe to resume",
    "P{pid}: vibration dampeners fitted, bench secure — printer calm now",
    "P{pid}: extruder drive gear replaced, grip confirmed good",
    "P{pid}: ABL probe cleaned and calibrated, first layer looking good",
    "P{pid}: broken bowden tube swapped out, no more grinding noise",
    "P{pid}: manual unclog done, filament flowing cleanly",
]

RESUME_SETUPS = [
    (1, "PLA",  800, "j1", 30, 55, 30),
    (2, "PETG", 700, "j2", 45, 80, 20),
    (3, "ABS",  650, "j3", 20, 60, 45),
    (4, "TPU",  500, "j4", 60, 95, 12),
    (5, "PLA",  900, "j5", 35, 45, 38),
    (1, "PETG", 750, "j6", 55, 70, 18),
    (2, "ABS",  600, "j7", 15, 50, 50),
    (3, "PLA",  850, "j1", 50, 65, 22),
    (4, "PETG", 700, "j2", 70, 85, 10),
    (5, "ABS",  550, "j3", 40, 55, 28),
    (6, "PLA",  950, "j4", 25, 40, 42),
    (1, "TPU",  450, "j5", 65, 90, 15),
    (2, "PLA",  800, "j6", 48, 75, 25),
    (3, "PETG", 650, "j7", 72, 80, 8),
    (4, "ABS",  700, "j1", 30, 60, 35),
    (5, "PLA",  900, "j2", 38, 50, 30),
    (6, "PETG", 750, "j3", 55, 70, 18),
    (1, "ABS",  600, "j4", 18, 45, 48),
    (2, "PLA",  850, "j5", 80, 88, 6),
    (3, "TPU",  500, "j6", 44, 65, 22),
]


def build_resume():
    rows = []
    for i, (pid, mat, spool, jid, progress, price, deadline) in enumerate(RESUME_SETUPS):
        note = RESUME_NOTES[i].replace("{pid}", str(pid))
        step = 15 + i * 2
        obs = f"""STEP {step}/60 | Net profit: ${step * 1.0:.2f}

PRINTERS:
  - P{pid}: PAUSED, mat={mat}, spool={spool}g, job={jid}

JOBS:
  - {jid}: PAUSED, mat={mat}, progress={progress}%, ${price}, deadline_in={deadline}steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: PLA=1500g, PETG=1200g, ABS=900g, TPU=600g

OPERATOR NOTES:
  - {note}"""
        rows.append(_row(obs, f'{{"action_type":"RESUME_JOB","printer_id":{pid},"job_id":"{jid}"}}'))
    return rows


# ---------------------------------------------------------------------------
# DISPATCH_TICKET — escalation needed, operator required
# ---------------------------------------------------------------------------
DISPATCH_SCENARIOS = [
    # (pid, fault_desc, ticket_type, op_id, mat, spool, jid, state, progress, price, deadline)
    (1, "MECHANICAL_VIBRATION + LAYER_SHIFT at 67% — novel compound fault, investigate", "diagnostic_physical", "op_s1", "PLA", 700, "j1", "PRINTING", 67, 65, 20),
    (2, "THERMAL_RUNAWAY + EXTRUDER_SKIP at 45% — safety risk, senior needed", "diagnostic_physical", "op_s1", "PETG", 600, "j2", "PRINTING", 45, 80, 15),
    (3, "recurring layer delamination on P3 every 3 prints — pattern suggests hardware fault", "maintenance_basic", "op_s1", "ABS", 750, "j3", "PRINTING", 30, 55, 35),
    (4, "FAN_FAILURE + TEMP_SPIKE at 55% — needs physical inspection immediately", "diagnostic_physical", "op_l1", "TPU", 500, "j4", "PRINTING", 55, 90, 10),
    (5, "BED_ADHESION_FAILURE + NOZZLE_CLOG at 22% — compound fault, manual intervention", "unjam_printer", "op_s1", "PLA", 800, "j5", "PRINTING", 22, 45, 40),
    (1, "EXTRUDER_GRINDING + FILAMENT_SENSOR_FAULT at 78% — two simultaneous faults", "diagnostic_physical", "op_s1", "PETG", 650, "j6", "PRINTING", 78, 70, 8),
    (2, "camera hash frozen for 5 steps + temp reading implausible — possible MCU disconnect", "diagnostic_physical", "op_j1", "ABS", 700, "j7", "PRINTING", 40, 60, 25),
    (3, "OSCILLATING_Z + BELT_SLIP at 33% on P3 — vibration-related compound fault", "maintenance_basic", "op_s1", "PLA", 900, "j1", "PRINTING", 33, 50, 38),
    (4, "electrical buzzing from control board area — safety check required before continuing", "diagnostic_physical", "op_l1", "PETG", 750, "j2", "PRINTING", 18, 85, 12),
    (5, "P5 third jam this shift — repeated unjam not sticking, senior evaluation needed", "maintenance_basic", "op_s1", "ABS", 600, "j3", "PAUSED", 25, 55, 30),
    (6, "PROGRESS_DRIFT + WEBCAM_FREEZE at 50% — telemetry unreliable, physical check needed", "diagnostic_physical", "op_j1", "PLA", 850, "j4", "PRINTING", 50, 40, 42),
    (1, "HOTEND_TEMP_ERRATIC + LAYER_ADHESION_FAIL at 60% — compound thermal fault", "diagnostic_physical", "op_s1", "TPU", 450, "j5", "PRINTING", 60, 95, 10),
    (2, "bed level drift detected across 4 prints on P2 — ABL not compensating", "maintenance_basic", "op_s1", "PLA", 800, "j6", "PRINTING", 38, 48, 28),
    (3, "smoke odor reported by operator near P3 — safety stop, senior inspection required", "diagnostic_physical", "op_l1", "PETG", 650, "j7", "PRINTING", 70, 75, 9),
    (4, "FILAMENT_SENSOR_FALSE_RUNOUT recurring every 10 steps — sensor replacement needed", "maintenance_basic", "op_s1", "ABS", 700, "j1", "PAUSED", 15, 60, 40),
    (5, "P5 extruder clicking and skipping — gear worn, not a simple jam", "maintenance_basic", "op_s1", "PLA", 900, "j2", "PRINTING", 48, 50, 22),
    (6, "temperature sensor reading -40C then 999C alternating — likely loose connection", "diagnostic_physical", "op_s1", "PETG", 750, "j3", "PRINTING", 55, 70, 15),
    (1, "KLIPPER_MCU_DISCONNECT + PRINT_CONTINUING at 35% — telemetry fully unreliable", "diagnostic_physical", "op_j1", "ABS", 600, "j4", "PRINTING", 35, 45, 45),
    (2, "novel vibration pattern at layer transitions only — not matching any known fault", "diagnostic_physical", "op_s1", "PLA", 850, "j5", "PRINTING", 80, 88, 6),
    (3, "customer escalation + anomaly flag on same printer — combined urgency", "diagnostic_physical", "op_s1", "TPU", 500, "j6", "PRINTING", 42, 65, 20),
]


def build_dispatch():
    rows = []
    for i, (pid, fault, ticket, op, mat, spool, jid, jstate, progress, price, deadline) in enumerate(DISPATCH_SCENARIOS):
        step = 12 + i * 2
        state_line = f"PRINTING, mat={mat}, spool={spool}g, job={jid}" if jstate == "PRINTING" else f"PAUSED, mat={mat}, spool={spool}g, job={jid}"
        job_detail = f"progress={progress}%" if jstate in ("PRINTING", "PAUSED") else f"weight=200g, time=15steps"
        obs = f"""STEP {step}/60 | Net profit: ${step * 0.9:.2f}

PRINTERS:
  - P{pid}: {state_line}

JOBS:
  - {jid}: {jstate}, mat={mat}, {job_detail}, ${price}, deadline_in={deadline}steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5
  - op_l1: lead, ON-SHIFT, queue=0/2

INVENTORY: PLA=1500g, PETG=1200g, ABS=900g, TPU=600g

ANOMALY FLAGS:
  - {fault}"""
        rows.append(_row(obs, f'{{"action_type":"DISPATCH_TICKET","printer_id":{pid},"operator_id":"{op}","ticket_type":"{ticket}"}}'))
    return rows


# ---------------------------------------------------------------------------
# REQUEST_MAINTENANCE — high fatigue IDLE printer
# ---------------------------------------------------------------------------
MAINT_SETUPS = [
    (1, 8.2, "PLA",  900, 2),
    (2, 7.5, "PETG", 800, 3),
    (3, 9.0, "ABS",  700, 1),
    (4, 7.8, "PLA",  950, 2),
    (5, 8.5, "TPU",  600, 2),
    (6, 7.2, "PLA", 1000, 3),
    (1, 9.3, "PETG", 750, 1),
    (2, 8.0, "ABS",  650, 2),
    (3, 7.6, "PLA",  850, 3),
    (4, 9.1, "PETG", 700, 2),
    (5, 8.3, "ABS",  550, 2),
    (6, 7.9, "TPU",  450, 3),
    (1, 8.7, "PLA",  900, 1),
    (2, 7.4, "PETG", 800, 2),
    (3, 9.2, "ABS",  700, 2),
    (4, 8.1, "PLA",  950, 3),
    (5, 7.7, "TPU",  600, 2),
    (6, 8.6, "PLA", 1000, 1),
    (1, 9.4, "PETG", 750, 2),
    (2, 8.4, "ABS",  650, 3),
]


def build_maintenance():
    rows = []
    for i, (pid, fatigue, mat, spool, maint_due) in enumerate(MAINT_SETUPS):
        step = 20 + i * 2
        obs = f"""STEP {step}/60 | Net profit: ${step * 1.1:.2f}

PRINTERS:
  - P{pid}: IDLE, mat={mat}, spool={spool}g, fatigue={fatigue:.1f}, maint_due_in={maint_due}

JOBS:
  - j1: PENDING, mat=PLA, weight=200g, time=15steps, $45, deadline_in=35steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: PLA=1500g, PETG=1200g, ABS=900g, TPU=600g"""
        rows.append(_row(obs, f'{{"action_type":"REQUEST_MAINTENANCE","printer_id":{pid},"maintenance_type":"maintenance_basic"}}'))
    return rows


# ---------------------------------------------------------------------------
# REQUEST_SPOOL_SWAP — low spool or PAUSED_RUNOUT
# ---------------------------------------------------------------------------
SPOOL_SETUPS = [
    (1, "PLA",  120, "j1", "PRINTING", 35, 55, 28, "PRINTING"),
    (2, "PETG",  90, "j2", "PRINTING", 60, 80, 15, "PRINTING"),
    (3, "ABS",  150, "j3", "PAUSED_RUNOUT", 45, 60, 30, "PAUSED"),
    (4, "PLA",   80, "j4", "PRINTING", 20, 45, 40, "PRINTING"),
    (5, "TPU",  100, "j5", "PAUSED_RUNOUT", 70, 90, 10, "PAUSED"),
    (6, "PETG", 130, "j6", "PRINTING", 50, 70, 20, "PRINTING"),
    (1, "ABS",   70, "j7", "PRINTING", 30, 55, 35, "PRINTING"),
    (2, "PLA",  110, "j1", "PAUSED_RUNOUT", 55, 65, 18, "PAUSED"),
    (3, "PETG", 140, "j2", "PRINTING", 25, 75, 25, "PRINTING"),
    (4, "ABS",   85, "j3", "PRINTING", 65, 60, 12, "PRINTING"),
    (5, "PLA",   95, "j4", "PAUSED_RUNOUT", 40, 45, 38, "PAUSED"),
    (6, "TPU",  160, "j5", "PRINTING", 78, 95, 7,  "PRINTING"),
    (1, "PLA",  125, "j6", "PRINTING", 18, 40, 42, "PRINTING"),
    (2, "PETG", 105, "j7", "PAUSED_RUNOUT", 62, 80, 14, "PAUSED"),
    (3, "ABS",   75, "j1", "PRINTING", 33, 55, 30, "PRINTING"),
    (4, "PLA",  145, "j2", "PRINTING", 48, 50, 22, "PRINTING"),
    (5, "PETG", 115, "j3", "PAUSED_RUNOUT", 72, 70, 9, "PAUSED"),
    (6, "ABS",   88, "j4", "PRINTING", 22, 45, 45, "PRINTING"),
    (1, "TPU",  135, "j5", "PRINTING", 55, 88, 12, "PRINTING"),
    (2, "PLA",   98, "j6", "PAUSED_RUNOUT", 38, 48, 28, "PAUSED"),
]


def build_spool_swap():
    rows = []
    for i, (pid, mat, spool, jid, pstate, progress, price, deadline, jstate) in enumerate(SPOOL_SETUPS):
        step = 8 + i * 2
        obs = f"""STEP {step}/60 | Net profit: ${step * 0.9:.2f}

PRINTERS:
  - P{pid}: {pstate}, mat={mat}, spool={spool}g, job={jid}

JOBS:
  - {jid}: {jstate}, mat={mat}, progress={progress}%, ${price}, deadline_in={deadline}steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: PLA=1500g, PETG=1200g, ABS=900g, TPU=600g"""
        rows.append(_row(obs, f'{{"action_type":"REQUEST_SPOOL_SWAP","printer_id":{pid},"material":"{mat}"}}'))
    return rows


# ---------------------------------------------------------------------------
# ASSIGN_JOB — IDLE printer + PENDING job, matching material, sufficient spool
# ---------------------------------------------------------------------------
ASSIGN_SETUPS = [
    (1, "PLA",  900, "j1", "PLA", 200, 15, 45, 25, 3),
    (2, "PETG", 800, "j2", "PETG",250, 20, 80, 35, 2),
    (3, "ABS",  700, "j3", "ABS", 180, 14, 55, 20, 3),
    (4, "PLA",  950, "j4", "PLA", 160, 12, 35, 18, 2),
    (5, "TPU",  600, "j5", "TPU", 300, 22, 90, 40, 3),
    (6, "PETG", 850, "j6", "PETG",220, 17, 70, 30, 2),
    (1, "ABS",  750, "j7", "ABS", 190, 15, 60, 22, 3),
    (2, "PLA",  900, "j1", "PLA", 175, 13, 40, 16, 2),
    (3, "PETG", 800, "j2", "PETG",260, 21, 82, 38, 3),
    (4, "ABS",  700, "j3", "ABS", 210, 16, 58, 24, 2),
    (5, "PLA",  950, "j4", "PLA", 145, 11, 30, 14, 3),
    (6, "TPU",  600, "j5", "TPU", 280, 20, 88, 36, 2),
    (1, "PETG", 850, "j6", "PETG",240, 18, 72, 28, 3),
    (2, "ABS",  750, "j7", "ABS", 200, 15, 62, 22, 2),
    (3, "PLA",  900, "j1", "PLA", 185, 14, 42, 19, 3),
    (4, "PETG", 800, "j2", "PETG",270, 22, 84, 40, 2),
    (5, "ABS",  700, "j3", "ABS", 195, 15, 56, 21, 3),
    (6, "PLA",  950, "j4", "PLA", 155, 12, 32, 15, 2),
    (1, "TPU",  600, "j5", "TPU", 290, 21, 92, 38, 3),
    (2, "PETG", 850, "j6", "PETG",230, 18, 74, 32, 2),
]


def build_assign():
    rows = []
    for i, (pid, mat, spool, jid, jmat, weight, time_s, price, deadline, priority) in enumerate(ASSIGN_SETUPS):
        step = 3 + i
        obs = f"""STEP {step}/60 | Net profit: ${step * 0.5:.2f}

PRINTERS:
  - P{pid}: IDLE, mat={mat}, spool={spool}g

JOBS:
  - {jid}: PENDING, mat={jmat}, weight={weight}g, time={time_s}steps, ${price}, deadline_in={deadline}steps, priority={'HIGH' if priority == 3 else 'NORMAL'}

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: PLA=1500g, PETG=1200g, ABS=900g, TPU=600g"""
        rows.append(_row(obs, f'{{"action_type":"ASSIGN_JOB","printer_id":{pid},"job_id":"{jid}"}}'))
    return rows


# ---------------------------------------------------------------------------
# WAIT — healthy farm, all printing, jobs on track, no signals
# ---------------------------------------------------------------------------
WAIT_FARMS = [
    ("P1: PRINTING, mat=PLA, spool=750g, job=j1\n  - P2: PRINTING, mat=PETG, spool=600g, job=j2",
     "j1: PRINTING, mat=PLA, progress=40%, $55, deadline_in=35steps\n  - j2: PRINTING, mat=PETG, progress=28%, $80, deadline_in=50steps, priority=HIGH"),
    ("P1: PRINTING, mat=ABS, spool=700g, job=j3\n  - P3: PRINTING, mat=PLA, spool=850g, job=j4",
     "j3: PRINTING, mat=ABS, progress=55%, $60, deadline_in=22steps\n  - j4: PRINTING, mat=PLA, progress=32%, $45, deadline_in=40steps"),
    ("P2: PRINTING, mat=PETG, spool=650g, job=j1\n  - P4: PRINTING, mat=ABS, spool=600g, job=j2",
     "j1: PRINTING, mat=PETG, progress=48%, $75, deadline_in=28steps\n  - j2: PRINTING, mat=ABS, progress=22%, $50, deadline_in=45steps"),
    ("P1: PRINTING, mat=PLA, spool=800g, job=j5\n  - P2: PRINTING, mat=TPU, spool=450g, job=j6",
     "j5: PRINTING, mat=PLA, progress=62%, $45, deadline_in=18steps\n  - j6: PRINTING, mat=TPU, progress=35%, $90, deadline_in=30steps, priority=HIGH"),
    ("P3: PRINTING, mat=PETG, spool=720g, job=j2\n  - P5: PRINTING, mat=ABS, spool=680g, job=j3",
     "j2: PRINTING, mat=PETG, progress=70%, $80, deadline_in=12steps\n  - j3: PRINTING, mat=ABS, progress=42%, $55, deadline_in=38steps"),
    ("P1: PRINTING, mat=PLA, spool=880g, job=j1\n  - P2: PRINTING, mat=PETG, spool=550g, job=j4",
     "j1: PRINTING, mat=PLA, progress=25%, $50, deadline_in=42steps\n  - j4: PRINTING, mat=PETG, progress=58%, $70, deadline_in=20steps"),
    ("P4: PRINTING, mat=ABS, spool=630g, job=j7\n  - P6: PRINTING, mat=PLA, spool=920g, job=j1",
     "j7: PRINTING, mat=ABS, progress=35%, $65, deadline_in=30steps\n  - j1: PRINTING, mat=PLA, progress=18%, $42, deadline_in=48steps"),
    ("P1: PRINTING, mat=PLA, spool=760g, job=j3\n  - P3: PRINTING, mat=PETG, spool=680g, job=j5",
     "j3: PRINTING, mat=PLA, progress=52%, $48, deadline_in=25steps\n  - j5: PRINTING, mat=PETG, progress=30%, $78, deadline_in=40steps"),
    ("P2: PRINTING, mat=ABS, spool=700g, job=j2\n  - P5: PRINTING, mat=PLA, spool=840g, job=j4",
     "j2: PRINTING, mat=ABS, progress=65%, $58, deadline_in=16steps\n  - j4: PRINTING, mat=PLA, progress=40%, $44, deadline_in=35steps"),
    ("P1: PRINTING, mat=TPU, spool=480g, job=j6\n  - P4: PRINTING, mat=PLA, spool=780g, job=j1",
     "j6: PRINTING, mat=TPU, progress=44%, $88, deadline_in=22steps\n  - j1: PRINTING, mat=PLA, progress=22%, $46, deadline_in=44steps"),
    ("P2: PRINTING, mat=PETG, spool=610g, job=j7\n  - P6: PRINTING, mat=ABS, spool=660g, job=j3",
     "j7: PRINTING, mat=PETG, progress=72%, $82, deadline_in=10steps\n  - j3: PRINTING, mat=ABS, progress=48%, $52, deadline_in=30steps"),
    ("P1: PRINTING, mat=PLA, spool=820g, job=j2\n  - P3: PRINTING, mat=TPU, spool=420g, job=j5",
     "j2: PRINTING, mat=PLA, progress=38%, $46, deadline_in=38steps\n  - j5: PRINTING, mat=TPU, progress=55%, $92, deadline_in=16steps"),
    ("P4: PRINTING, mat=PETG, spool=700g, job=j4\n  - P5: PRINTING, mat=PLA, spool=870g, job=j6",
     "j4: PRINTING, mat=PETG, progress=28%, $76, deadline_in=42steps\n  - j6: PRINTING, mat=PLA, progress=62%, $44, deadline_in=18steps"),
    ("P2: PRINTING, mat=ABS, spool=640g, job=j1\n  - P6: PRINTING, mat=PETG, spool=580g, job=j3",
     "j1: PRINTING, mat=ABS, progress=50%, $62, deadline_in=24steps\n  - j3: PRINTING, mat=PETG, progress=35%, $72, deadline_in=36steps"),
    ("P1: PRINTING, mat=PLA, spool=900g, job=j7\n  - P3: PRINTING, mat=ABS, spool=720g, job=j2",
     "j7: PRINTING, mat=PLA, progress=15%, $40, deadline_in=50steps\n  - j2: PRINTING, mat=ABS, progress=68%, $58, deadline_in=14steps"),
    ("P4: PRINTING, mat=TPU, spool=460g, job=j5\n  - P5: PRINTING, mat=PETG, spool=630g, job=j4",
     "j5: PRINTING, mat=TPU, progress=80%, $95, deadline_in=8steps\n  - j4: PRINTING, mat=PETG, progress=25%, $68, deadline_in=40steps"),
    ("P1: PRINTING, mat=PLA, spool=740g, job=j3\n  - P2: PRINTING, mat=ABS, spool=690g, job=j6",
     "j3: PRINTING, mat=PLA, progress=45%, $48, deadline_in=28steps\n  - j6: PRINTING, mat=ABS, progress=58%, $56, deadline_in=20steps"),
    ("P3: PRINTING, mat=PETG, spool=660g, job=j1\n  - P6: PRINTING, mat=PLA, spool=810g, job=j7",
     "j1: PRINTING, mat=PETG, progress=32%, $74, deadline_in=38steps\n  - j7: PRINTING, mat=PLA, progress=48%, $42, deadline_in=26steps"),
    ("P2: PRINTING, mat=ABS, spool=670g, job=j4\n  - P4: PRINTING, mat=TPU, spool=440g, job=j2",
     "j4: PRINTING, mat=ABS, progress=60%, $60, deadline_in=18steps\n  - j2: PRINTING, mat=TPU, progress=22%, $86, deadline_in=44steps"),
    ("P5: PRINTING, mat=PLA, spool=860g, job=j5\n  - P6: PRINTING, mat=PETG, spool=600g, job=j3",
     "j5: PRINTING, mat=PLA, progress=75%, $44, deadline_in=12steps\n  - j3: PRINTING, mat=PETG, progress=38%, $70, deadline_in=32steps"),
]


def build_wait():
    rows = []
    for i, (printer_lines, job_lines) in enumerate(WAIT_FARMS):
        step = 20 + i * 2
        obs = f"""STEP {step}/60 | Net profit: ${step * 1.5:.2f}

PRINTERS:
  - {printer_lines}

JOBS:
  - {job_lines}

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: PLA=1500g, PETG=1200g, ABS=900g"""
        rows.append(_row(obs, '{"action_type":"WAIT"}'))
    return rows


# ---------------------------------------------------------------------------
# RUN_DIAGNOSTIC — ambiguous sensor reading, not clearly mechanical
# ---------------------------------------------------------------------------
DIAG_SCENARIOS = [
    # (pid, mat, spool, jid, progress, price, deadline, suspicious_reading)
    (1, "PLA",  750, "j1", 30, 55, 28, "hotend_temp showing 0C while print continues — thermistor issue?"),
    (2, "PETG", 680, "j2", 45, 80, 20, "webcam hash repeated for last 3 steps — freeze or normal?"),
    (3, "ABS",  700, "j3", 20, 60, 40, "telemetry_ts stale by 8 steps — klipper hiccup or disconnect?"),
    (4, "PLA",  820, "j4", 62, 45, 15, "hotend_temp reading 420C — thermistor short or actual overheat?"),
    (5, "PETG", 630, "j5", 38, 75, 25, "fan_rpm reported as 0 but print quality looks normal — sensor glitch?"),
    (6, "ABS",  710, "j6", 55, 58, 18, "progress stuck at 55% for 3 consecutive steps — drift or real stall?"),
    (1, "PLA",  780, "j7", 28, 42, 35, "operator notes 'slight vibration on P1' but no anomaly flag — investigate?"),
    (2, "PETG", 650, "j1", 70, 82, 10, "reliability dropped to 0.72 this step — unusual, check history"),
    (3, "ABS",  720, "j2", 42, 56, 30, "hotend_temp fluctuating ±50C each step — intermittent sensor fault"),
    (4, "TPU",  480, "j3", 15, 92, 45, "webcam hash changed only twice in 10 steps — possible freeze"),
    (5, "PLA",  840, "j4", 58, 47, 22, "telemetry_ts advancing irregularly — some steps have same timestamp"),
    (6, "PETG", 660, "j5", 35, 76, 28, "spool weight decreased much faster than expected for current job weight"),
    (1, "ABS",  700, "j6", 72, 62, 12, "bed_temp reading 200C during PLA print — sensor wrong or runaway?"),
    (2, "PLA",  800, "j7", 22, 44, 40, "operator: 'P2 seems slower today' — no anomaly flag, verify"),
    (3, "PETG", 640, "j1", 48, 78, 20, "fan_rpm oscillating between 0 and 3000 — unstable reading"),
    (4, "ABS",  690, "j2", 65, 58, 14, "hotend_temp steady at 0C for 2 steps then normal — intermittent fault"),
    (5, "PLA",  860, "j3", 30, 46, 38, "webcam_hash unchanged for 4 steps — need to verify print is progressing"),
    (6, "PETG", 670, "j4", 52, 72, 18, "reliability 0.68 this step — unusually low, anomaly or new fault mode?"),
    (1, "ABS",  710, "j5", 38, 54, 26, "telemetry_ts 15 steps behind wall clock — MCU may be disconnected"),
    (2, "TPU",  460, "j6", 80, 90, 8,  "progress jumped from 72% to 80% in one step — progress_drift possible"),
]


def build_diagnostic():
    rows = []
    for i, (pid, mat, spool, jid, progress, price, deadline, reading) in enumerate(DIAG_SCENARIOS):
        step = 10 + i * 2
        obs = f"""STEP {step}/60 | Net profit: ${step * 0.8:.2f}

PRINTERS:
  - P{pid}: PRINTING, mat={mat}, spool={spool}g, job={jid}

JOBS:
  - {jid}: PRINTING, mat={mat}, progress={progress}%, ${price}, deadline_in={deadline}steps

OPERATORS:
  - op_j1: junior, ON-SHIFT, queue=0/3
  - op_s1: senior, ON-SHIFT, queue=0/5

INVENTORY: PLA=1500g, PETG=1200g, ABS=900g, TPU=600g

OPERATOR NOTES:
  - {reading}"""
        rows.append(_row(obs, f'{{"action_type":"RUN_DIAGNOSTIC","printer_id":{pid}}}'))
    return rows


# ---------------------------------------------------------------------------
# Assemble, shuffle, write
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    all_rows = (
        build_cancel()       # 20
        + build_pause()      # 20
        + build_resume()     # 20
        + build_dispatch()   # 20
        + build_maintenance()# 20
        + build_spool_swap() # 20
        + build_assign()     # 20
        + build_wait()       # 20
        + build_diagnostic() # 20
    )
    assert len(all_rows) == 180, f"Expected 180 rows, got {len(all_rows)}"

    # Verify distribution
    import re
    from collections import Counter
    counts = Counter()
    for row in all_rows:
        for m in row["messages"]:
            if m["role"] == "assistant":
                t = re.search(r'"action_type":"([^"]+)"', m["content"])
                if t:
                    counts[t.group(1)] += 1

    print("Action distribution:")
    for action, n in sorted(counts.items()):
        print(f"  {action}: {n}")

    assert "OVERRIDE_OPERATOR" not in counts, "OVERRIDE_OPERATOR must not appear"
    for action, n in counts.items():
        assert n == 20, f"{action} has {n} examples, expected 20"

    random.shuffle(all_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(all_rows)} examples to {OUT}")


if __name__ == "__main__":
    main()
