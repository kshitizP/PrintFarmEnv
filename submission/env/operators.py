"""Operator NPC policy — shift-scoped memory, ticket dispatch, anomaly reports."""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import PrinterState, Ticket, TicketState, OperatorObservation

SKILL_PARAMS = {
    "junior": {"latency": 7, "capacity": 3, "rate": 0.30, "base_p": 0.85},
    "senior": {"latency": 5, "capacity": 5, "rate": 0.47, "base_p": 0.95},
    "lead":   {"latency": 4, "capacity": 7, "rate": 0.67, "base_p": 0.98},
}
SKILL_RANK = {"junior": 0, "senior": 1, "lead": 2}
TICKET_SKILL_REQ = {
    "spool_swap": "junior", "filament_reload_from_stock": "junior",
    "maintenance_basic": "senior", "maintenance_full_rebuild": "lead",
    "diagnostic_physical": "junior", "unjam_printer": "senior",
}
TICKET_DURATIONS = {
    "spool_swap": 2, "filament_reload_from_stock": 3,
    "maintenance_basic": 3, "maintenance_full_rebuild": 8,
    "diagnostic_physical": 1, "unjam_printer": 4,
}


@dataclass
class VisitRecord:
    step: int
    ticket_type: str
    observed_fault: Optional[str]
    status: str  # SUCCESS | FAILED | REJECTED


@dataclass
class OperatorInternal:
    operator_id: str
    skill_level: str
    shift_window: List[int]
    queue_capacity: int
    base_latency_steps: int
    labor_rate: float
    base_success_prob: float
    queue: List[Ticket] = field(default_factory=list)
    current_ticket: Optional[Ticket] = None
    busy_until: int = 0
    current_fatigue: float = 0.0
    printer_memory: Dict[int, List[VisitRecord]] = field(default_factory=dict)
    cooldown_labor_owed: float = 0.0  # billed per-step while waiting for maintenance cooldown


def create_operator(operator_id: str, skill_level: str, shift_window: List[int]) -> OperatorInternal:
    p = SKILL_PARAMS[skill_level]
    return OperatorInternal(
        operator_id=operator_id, skill_level=skill_level, shift_window=shift_window,
        queue_capacity=p["capacity"], base_latency_steps=p["latency"],
        labor_rate=p["rate"], base_success_prob=p["base_p"],
    )


def is_on_shift(op: OperatorInternal, step: int) -> bool:
    return op.shift_window[0] <= step < op.shift_window[1]


def queue_total(op: OperatorInternal) -> int:
    return len(op.queue) + (1 if op.current_ticket else 0)


def operator_tick(
    op: OperatorInternal, current_step: int,
    printers_by_id: Dict, jobs_by_id: Dict,
    all_operators: List["OperatorInternal"], rng: random.Random,
) -> tuple[float, Optional[Ticket], Optional[Dict[str, Any]]]:
    """One tick. Returns (labor_cost_this_step, completed_ticket_or_None, anomaly_or_None)."""
    if not is_on_shift(op, current_step):
        _reassign_queue(op, all_operators, current_step)
        return 0.0, None, None

    _update_fatigue(op, current_step)

    labor = 0.0
    completed_ticket = None
    anomaly = None

    # Handle maintenance cooldown waiting
    if (op.current_ticket is not None
            and op.busy_until <= current_step
            and op.current_ticket.ticket_type in ("maintenance_basic", "maintenance_full_rebuild")):
        pid = op.current_ticket.target_printer_id
        printer = printers_by_id.get(pid)
        if printer is not None and printer.consecutive_idle_steps < 3:
            op.busy_until = current_step + 1  # extend 1 step
            op.cooldown_labor_owed += op.labor_rate
            return op.labor_rate, None, None  # bill waiting step

    if op.current_ticket is not None and op.busy_until <= current_step:
        labor, completed_ticket, anomaly = _finalise(
            op, op.current_ticket, current_step, printers_by_id, jobs_by_id, rng
        )
        op.current_ticket = None

    if op.current_ticket is None and op.queue:
        nxt = _pick_next(op.queue)
        req_rank = SKILL_RANK.get(TICKET_SKILL_REQ.get(nxt.ticket_type, "junior"), 0)
        if SKILL_RANK.get(op.skill_level, 0) < req_rank:
            walk_cost = op.labor_rate * 1
            labor += walk_cost
            if not _escalate(nxt, op, all_operators, current_step):
                op.queue.insert(0, nxt)  # no qualified target; put back
        else:
            op.queue.remove(nxt)
            latency = max(1, int(op.base_latency_steps * (1 + op.current_fatigue) + rng.gauss(0, 1)))
            op.busy_until = current_step + latency
            op.current_ticket = nxt
            nxt.state = TicketState.IN_PROGRESS
            nxt.operator_id = op.operator_id

    return labor, completed_ticket, anomaly


def _update_fatigue(op: OperatorInternal, step: int) -> None:
    span = op.shift_window[1] - op.shift_window[0]
    if span > 0:
        elapsed = step - op.shift_window[0]
        op.current_fatigue = min(0.8, 0.8 * elapsed / span)


def _pick_next(queue: List[Ticket]) -> Ticket:
    for t in queue:
        if t.ticket_type == "maintenance_full_rebuild":
            return t
    return queue[0]


def _escalate(ticket: Ticket, op: OperatorInternal,
              all_operators: List[OperatorInternal], step: int) -> bool:
    req_rank = SKILL_RANK.get(TICKET_SKILL_REQ.get(ticket.ticket_type, "junior"), 0)
    candidates = [
        o for o in all_operators
        if o.operator_id != op.operator_id
        and SKILL_RANK.get(o.skill_level, 0) >= req_rank
        and is_on_shift(o, step)
        and queue_total(o) < o.queue_capacity
    ]
    if not candidates:
        return False
    target = min(candidates, key=lambda o: (queue_total(o), -SKILL_RANK.get(o.skill_level, 0)))
    op.queue.remove(ticket)
    target.queue.append(ticket)
    ticket.operator_id = target.operator_id
    return True


def _reassign_queue(op: OperatorInternal, all_ops: List[OperatorInternal], step: int) -> None:
    for ticket in list(op.queue):
        req_rank = SKILL_RANK.get(TICKET_SKILL_REQ.get(ticket.ticket_type, "junior"), 0)
        candidates = [
            o for o in all_ops
            if o.operator_id != op.operator_id
            and SKILL_RANK.get(o.skill_level, 0) >= req_rank
            and is_on_shift(o, step)
        ]
        if candidates:
            target = min(candidates, key=lambda o: (queue_total(o), -SKILL_RANK.get(o.skill_level, 0)))
            target.queue.append(ticket)
            ticket.operator_id = target.operator_id
            op.queue.remove(ticket)


def _finalise(
    op: OperatorInternal, ticket: Ticket, step: int,
    printers_by_id: Dict, jobs_by_id: Dict, rng: random.Random,
) -> tuple[float, Ticket, Optional[Dict]]:
    duration = TICKET_DURATIONS.get(ticket.ticket_type, 1)
    labor = op.labor_rate * duration + op.cooldown_labor_owed
    op.cooldown_labor_owed = 0.0

    printer = printers_by_id.get(ticket.target_printer_id)
    if printer is None or _is_rejected(ticket, printer):
        ticket.state = TicketState.REJECTED
        ticket.rejection_reason = "printer_offline" if (printer and printer.state == PrinterState.OFFLINE) else "incompatible_state"
        _record_visit(op, ticket.target_printer_id, step, ticket.ticket_type, None, "REJECTED")
        return labor, ticket, None

    success_p = op.base_success_prob * (1 - 0.5 * op.current_fatigue)
    success = rng.random() < success_p

    if success:
        observed_fault = _apply_effects(ticket, printer, jobs_by_id)
        ticket.state = TicketState.COMPLETED
        _record_visit(op, ticket.target_printer_id, step, ticket.ticket_type, observed_fault, "SUCCESS")
        anomaly = _maybe_anomaly(op, ticket.target_printer_id, step, observed_fault, rng)
    else:
        ticket.state = TicketState.FAILED
        _record_visit(op, ticket.target_printer_id, step, ticket.ticket_type, None, "FAILED")
        anomaly = None

    return labor, ticket, anomaly


def _is_rejected(ticket: Ticket, printer) -> bool:
    if printer.state == PrinterState.OFFLINE:
        return True
    if ticket.ticket_type == "unjam_printer" and printer.state != PrinterState.ERROR:
        return True
    if ticket.ticket_type in ("maintenance_basic", "maintenance_full_rebuild"):
        return printer.state not in (PrinterState.MAINTENANCE_QUEUED, PrinterState.MAINTENANCE)
    return False


def _apply_effects(ticket: Ticket, printer, jobs_by_id: Dict) -> Optional[str]:
    t = ticket.ticket_type
    observed_fault: Optional[str] = None

    if t == "spool_swap":
        material = ticket.payload.get("material", printer.current_material or "PLA")
        printer.spool_weight_g = 950.0  # 1000g - 50g purge
        printer.current_material = material
        if printer.state == PrinterState.PAUSED_RUNOUT:
            printer.state = PrinterState.IDLE
        # PAUSED stays PAUSED; agent must RESUME_JOB

    elif t == "filament_reload_from_stock":
        printer.spool_weight_g = 1000.0

    elif t in ("maintenance_basic", "maintenance_full_rebuild"):
        full = (t == "maintenance_full_rebuild")
        printer.state = PrinterState.MAINTENANCE
        printer.warmup_remaining = 8 if full else 3
        printer.maintenance_type = t
        printer.outstanding_ticket_id = None

    elif t == "diagnostic_physical":
        from .failures import FAILURE_MODES, clear_fault
        for mode in list(printer.active_faults.keys()):
            if "diagnostic_physical" in FAILURE_MODES[mode]["clears_on"]:
                clear_fault(printer.active_faults, mode)
        if printer.active_faults:
            observed_fault = next(iter(printer.active_faults))

    elif t == "unjam_printer":
        printer.state = PrinterState.IDLE
        printer.current_job_id = None

    return observed_fault


def _record_visit(op: OperatorInternal, pid: int, step: int,
                  ttype: str, fault: Optional[str], status: str) -> None:
    op.printer_memory.setdefault(pid, []).append(
        VisitRecord(step=step, ticket_type=ttype, observed_fault=fault, status=status)
    )


def _maybe_anomaly(op: OperatorInternal, pid: int, step: int,
                   observed_fault: Optional[str], rng: random.Random) -> Optional[Dict]:
    if rng.random() > 0.60:
        return None

    visits = op.printer_memory.get(pid, [])
    repeat_bonus = 0.10 if len(visits) >= 2 else 0.0

    if observed_fault is None:
        base_p = 0.95
    elif observed_fault in ("bed_level_drift",):
        base_p = 0.85
    else:
        base_p = 0.40

    accuracy = min(0.95, base_p + repeat_bonus)
    correct = rng.random() < accuracy

    recommendation = None
    fault_counts: Dict[str, int] = {}
    for r in visits:
        if r.observed_fault:
            fault_counts[r.observed_fault] = fault_counts.get(r.observed_fault, 0) + 1
    for fault, cnt in fault_counts.items():
        if cnt >= 2 and fault == "bed_level_drift":
            recommendation = f"recommend maintenance_full_rebuild — {cnt + 1} occurrences this shift"

    return {
        "type": "REPORT_ANOMALY",
        "operator_id": op.operator_id,
        "printer_id": pid,
        "step": step,
        "observed_fault": observed_fault if correct else None,
        "recommendation": recommendation,
    }


def to_observation(op: OperatorInternal, step: int) -> OperatorObservation:
    recs = op.printer_memory
    recommendations: List[str] = []
    for pid, visits in recs.items():
        fault_counts: Dict[str, int] = {}
        for r in visits:
            if r.observed_fault:
                fault_counts[r.observed_fault] = fault_counts.get(r.observed_fault, 0) + 1
        for fault, cnt in fault_counts.items():
            if cnt >= 2 and fault in ("bed_level_drift",):
                recommendations.append(
                    f"P{pid}: recommend maintenance_full_rebuild — {cnt} occurrences this shift"
                )
    return OperatorObservation(
        operator_id=op.operator_id, skill_level=op.skill_level,
        shift_window=op.shift_window, queue_capacity=op.queue_capacity,
        current_fatigue=op.current_fatigue, is_on_shift=is_on_shift(op, step),
        queue_size=len(op.queue),
        current_ticket_id=op.current_ticket.ticket_id if op.current_ticket else None,
        busy_until=op.busy_until,
        printer_visit_counts={pid: len(v) for pid, v in recs.items()},
        pattern_recommendations=recommendations,
    )
