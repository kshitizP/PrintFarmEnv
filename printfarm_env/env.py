"""
PrintFarmEnv — Round 2 Environment Engine

Architecture: Dispatcher / Operator / Oversight (the AI agent is the Dispatcher).

Key design decisions (from SIMULATOR_SPEC.md §0):
  - 1 step = 1 real minute
  - Dispatcher emits exactly ONE action per step
  - Single random.Random seeded at reset() — all stochasticity comes from this
  - Ground truth computed BEFORE sensor corruption
  - Operators tick BEFORE printers (spool swap visible to printer physics)
  - Action effects applied BEFORE world events (failure injection)
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .economics import (
    amortization_cost, electricity_cost, filament_cost, filament_price,
    scrap_cost, sla_penalty_this_step, step_revenue as _step_revenue,
    ACTION_REJECT_COST, CATASTROPHIC_COST, DIAGNOSTIC_BASE_COST,
    DIAGNOSTIC_BONUS, OVERRIDE_COST, WAIT_COST,
)
from .failures import (
    FAILURE_MODES, FaultInstance,
    advance_faults, apply_corruption, clear_fault, clear_fault_class,
    get_corrupted_progress, inject_scheduled, try_inject,
)
from .models import (
    FarmAction, FarmActionEnum, FarmObservation,
    JobState, OperatorObservation, PrintJob,
    PrinterObservation, PrinterState,
    Ticket, TicketState,
)
from .operators import (
    OperatorInternal, create_operator, is_on_shift,
    operator_tick, to_observation,
)
from .profiles import PROFILES, PrinterProfile
from .tasks import TaskConfig, TaskGrader, get_task_config

# Resolve OpenEnv base class
try:
    from openenv.core import Environment as _BaseEnvironment
except ImportError:
    try:
        from openenv_core import Environment as _BaseEnvironment
    except ImportError:
        class _BaseEnvironment:
            def __init__(self, **kwargs):
                pass


# ---------------------------------------------------------------------------
#  Internal printer state (ground truth, never directly serialised to agent)
# ---------------------------------------------------------------------------

@dataclass
class PrinterInternal:
    printer_id:          int
    profile:             PrinterProfile
    state:               PrinterState    = PrinterState.IDLE
    current_material:    Optional[str]   = None
    spool_weight_g:      float           = 1000.0
    current_job_id:      Optional[str]   = None

    reliability:         float           = 0.95
    fatigue_level:       float           = 0.0
    maintenance_due_in:  int             = 50
    warmup_remaining:    int             = 0
    offline_remaining:   int             = 0
    error_remaining:     int             = 0   # steps until ERROR self-recovery
    consecutive_idle_steps: int          = 0
    maintenance_type:    Optional[str]   = None
    outstanding_ticket_id: Optional[str] = None

    # Telemetry (ground truth; corrupted copy goes to observation)
    hotend_temp:         float = 200.0
    fan_rpm:             int   = 3000
    webcam_hash:         str   = ""
    telemetry_ts:        int   = 0
    bed_drift_counter:   float = 0.0

    # Physics flags
    reliability_penalty_active: bool = False
    revealed_this_step:         bool = False

    # Sensor fault registry: {mode_id: FaultInstance}
    active_faults: Dict[str, FaultInstance] = field(default_factory=dict)

    def __post_init__(self):
        self.reliability  = self.profile.reliability_base
        self.webcam_hash  = f"cam_{self.printer_id}_ok"


# ---------------------------------------------------------------------------
#  PrintFarmEnvironment
# ---------------------------------------------------------------------------

class PrintFarmEnvironment(_BaseEnvironment):

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        # Initialised properly in reset()
        self.current_task_id:  str          = "task_1"
        self._cfg:             TaskConfig   = get_task_config("task_1")
        self._printers:        List[PrinterInternal]   = []
        self._operators:       List[OperatorInternal]  = []
        self._jobs:            Dict[str, PrintJob]     = {}
        self._inventory:       Dict[str, float]        = {}
        self._tickets:         Dict[str, Ticket]       = {}
        self._oversight_log:   List[Dict[str, Any]]    = []
        self.time_step:        int          = 0
        self.max_steps:        int          = 60
        self.grader:           TaskGrader   = TaskGrader("task_1")
        self._rng:             random.Random = random.Random(42)
        self._net_profit:      float        = 0.0
        self._total_labor:     float        = 0.0
        self._ticket_counter:  int          = 0
        self._obs:             FarmObservation = FarmObservation()

    # ------------------------------------------------------------------
    #  OpenEnv reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:       Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> FarmObservation:
        task_id = kwargs.get("task_id", episode_id or "task_1")
        self.current_task_id = task_id
        self._cfg            = get_task_config(task_id)
        self.max_steps       = self._cfg.episode_length_steps
        self.grader          = TaskGrader(task_id)
        self._rng            = random.Random(seed if seed is not None else abs(hash(task_id)))

        self._oversight_log = []
        self._net_profit    = 0.0
        self._total_labor   = 0.0
        self._ticket_counter = 0
        self._tickets        = {}
        self.time_step       = 0

        # Build printer internal state
        self._printers = self._build_printers()

        # Build operators
        self._operators = [
            create_operator(
                o["operator_id"], o["skill_level"],
                list(o["shift_window"]),
            )
            for o in self._cfg.operator_roster
        ]

        # Build job registry
        self._jobs = {}
        for jspec in self._cfg.job_queue:
            job = PrintJob(
                job_id            = jspec["job_id"],
                material_required = jspec["material"],
                weight_required_g = float(jspec["weight_g"]),
                print_time_steps  = int(jspec["print_time_steps"]),
                priority          = int(jspec.get("priority", 2)),
                deadline_steps    = jspec.get("deadline_steps"),
                price_usd         = float(jspec.get("price_usd", 30.0)),
            )
            self._jobs[job.job_id] = job

        self._inventory = {k: float(v) for k, v in self._cfg.initial_inventory_g.items()}

        self._obs = self._build_observation()
        return self._obs

    # ------------------------------------------------------------------
    #  OpenEnv state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> FarmObservation:
        return self._obs

    # ------------------------------------------------------------------
    #  OpenEnv step
    # ------------------------------------------------------------------

    def step(
        self,
        action:    FarmAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> FarmObservation:

        # Reset per-step flags from the previous step
        for p in self._printers:
            p.revealed_this_step = False

        step_reward      = 0.0
        reward_breakdown: Dict[str, float] = {}
        action_handled   = False
        action_extra:    Dict[str, Any] = {}
        ticket_events:   List[Dict[str, Any]] = []

        # ── 1. Validate & apply Dispatcher action ────────────────────────
        if action is not None:
            delta, action_handled, action_extra = self._dispatch_action(action)
            step_reward += delta
            reward_breakdown["action"]  = delta

        # ── 4. Tick operators (before printers) ──────────────────────────
        printers_by_id = {p.printer_id: p for p in self._printers}
        jobs_by_id     = self._jobs

        op_labor_total = 0.0
        for op in self._operators:
            labor, completed_ticket, anomaly = operator_tick(
                op, self.time_step, printers_by_id, jobs_by_id,
                self._operators, self._rng,
            )
            op_labor_total += labor
            if completed_ticket is not None:
                self._handle_ticket_completion(completed_ticket, printers_by_id)
                ticket_events.append({
                    "ticket_id":   completed_ticket.ticket_id,
                    "ticket_type": completed_ticket.ticket_type,
                    "operator_id": completed_ticket.operator_id,
                    "printer_id":  completed_ticket.target_printer_id,
                    "status":      completed_ticket.state.value,
                    "step":        self.time_step,
                })
            if anomaly is not None:
                ticket_events.append(anomaly)

        step_reward         -= op_labor_total
        self._total_labor   += op_labor_total
        reward_breakdown["labor"] = -op_labor_total

        # ── 5. Tick printers ─────────────────────────────────────────────
        physics_delta = self._tick_physics()
        step_reward  += physics_delta
        reward_breakdown["physics"] = physics_delta

        # ── 6. Inject scheduled & stochastic failures ────────────────────
        self._inject_failures()

        # ── 9. SLA penalties for all active jobs ─────────────────────────
        sla_delta = self._check_sla_penalties()
        step_reward += sla_delta
        reward_breakdown["sla"] = sla_delta

        # ── 12. Oversight log ─────────────────────────────────────────────
        self._oversight_log.append({
            "step":      self.time_step,
            "action":    action.action.value if action else "NONE",
            "action_extra": action_extra,
            "reward":    step_reward,
            "breakdown": reward_breakdown,
        })

        # ── 13. Advance time ──────────────────────────────────────────────
        self.time_step += 1
        for p in self._printers:
            p.telemetry_ts = self.time_step

        self._net_profit += step_reward

        # Update grader stats
        current_obs = self._build_observation()
        current_obs.ticket_events  = ticket_events
        current_obs.oversight_log  = list(self._oversight_log[-10:])  # last 10 lines
        current_obs.reward_breakdown = reward_breakdown

        self.grader.step_update(
            action, action_handled,
            current_obs, self.time_step,
            extra=action_extra,
        )

        # ── Episode termination check ─────────────────────────────────────
        done = self.time_step >= self.max_steps
        all_resolved = all(
            j.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
            for j in self._jobs.values()
        )
        if all_resolved:
            done = True

        current_obs.reward   = self.grader.get_score(current_obs)
        current_obs.done     = done
        current_obs.metadata = {"step_reward_usd": step_reward, **action_extra}

        self._obs = current_obs
        return self._obs

    # ==================================================================
    #  Action dispatch — maps FarmActionEnum → handlers
    # ==================================================================

    def _dispatch_action(
        self, action: FarmAction
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Route action to handler. Returns (reward_delta, handled, extra_info)."""

        act = action.action

        if act == FarmActionEnum.ASSIGN_JOB:
            return self._handle_assign(action)
        elif act == FarmActionEnum.CANCEL_JOB:
            return self._handle_cancel(action)
        elif act == FarmActionEnum.PAUSE_JOB:
            return self._handle_pause(action)
        elif act == FarmActionEnum.RESUME_JOB:
            return self._handle_resume(action)
        elif act == FarmActionEnum.RUN_DIAGNOSTIC:
            return self._handle_run_diagnostic(action)
        elif act == FarmActionEnum.DISPATCH_TICKET:
            return self._handle_dispatch_ticket(action)
        elif act == FarmActionEnum.REQUEST_SPOOL_SWAP:
            return self._handle_request_spool_swap(action)
        elif act == FarmActionEnum.REQUEST_MAINTENANCE:
            return self._handle_request_maintenance(action)
        elif act == FarmActionEnum.OVERRIDE_OPERATOR:
            return self._handle_override_operator(action)
        elif act == FarmActionEnum.WAIT:
            return -WAIT_COST, False, {"note": "WAIT"}

        return -ACTION_REJECT_COST, False, {"error": "unknown_action"}

    # ------------------------------------------------------------------
    #  Individual action handlers
    # ------------------------------------------------------------------

    def _handle_assign(self, action: FarmAction) -> Tuple[float, bool, Dict]:
        if action.printer_id is None or not action.job_id:
            return -ACTION_REJECT_COST, False, {"error": "ASSIGN_JOB requires printer_id and job_id"}

        p = self._get_printer(action.printer_id)
        j = self._jobs.get(action.job_id)

        if p is None or j is None:
            return -ACTION_REJECT_COST, False, {"error": "invalid printer_id or job_id"}
        if p.state != PrinterState.IDLE:
            return -ACTION_REJECT_COST, False, {"error": f"printer {p.printer_id} not IDLE", "reason": "not_idle"}
        if j.state != JobState.PENDING:
            return -ACTION_REJECT_COST, False, {"error": f"job {j.job_id} not PENDING", "reason": "job_not_pending"}
        if p.current_material != j.material_required:
            return -ACTION_REJECT_COST, False, {
                "error": f"material mismatch: printer={p.current_material} job={j.material_required}",
                "reason": "material_mismatch",
            }
        if p.spool_weight_g < j.weight_required_g:
            return -ACTION_REJECT_COST, False, {
                "error": f"insufficient spool: {p.spool_weight_g:.0f}g < {j.weight_required_g:.0f}g",
                "reason": "insufficient_spool",
            }

        p.state          = PrinterState.WARMING_UP
        p.warmup_remaining = 1
        p.current_job_id = j.job_id
        j.state          = JobState.PRINTING

        return 0.0, True, {"assigned": f"{j.job_id} → printer {p.printer_id}"}

    def _handle_cancel(self, action: FarmAction) -> Tuple[float, bool, Dict]:
        if not action.job_id:
            return -ACTION_REJECT_COST, False, {"error": "CANCEL_JOB requires job_id"}

        j = self._jobs.get(action.job_id)
        if j is None:
            return -ACTION_REJECT_COST, False, {"error": "invalid job_id"}
        if j.state not in (JobState.PENDING, JobState.PRINTING, JobState.PAUSED):
            return -ACTION_REJECT_COST, False, {"error": f"job {j.job_id} cannot be cancelled (state={j.state})"}

        cost = 0.0
        if j.state in (JobState.PRINTING, JobState.PAUSED):
            for p in self._printers:
                if p.current_job_id == j.job_id:
                    cost += scrap_cost(j.material_required, j.weight_required_g,
                                       j.progress_steps, j.print_time_steps)
                    cost += j.accrued_revenue   # clawback
                    p.state          = PrinterState.IDLE
                    p.current_job_id = None
                    p.warmup_remaining = 0
                    break

        j.state = JobState.CANCELLED
        return -(cost), True, {"cancelled": j.job_id, "scrap_cost": cost}

    def _handle_pause(self, action: FarmAction) -> Tuple[float, bool, Dict]:
        if action.printer_id is None:
            return -ACTION_REJECT_COST, False, {"error": "PAUSE_JOB requires printer_id"}

        p = self._get_printer(action.printer_id)
        if p is None:
            return -ACTION_REJECT_COST, False, {"error": "invalid printer_id"}
        if p.state != PrinterState.PRINTING:
            return -ACTION_REJECT_COST, False, {"error": f"printer {p.printer_id} not PRINTING"}

        j = self._jobs.get(p.current_job_id) if p.current_job_id else None
        if j:
            j.state = JobState.PAUSED

        p.state = PrinterState.PAUSED
        return 0.0, True, {"paused": p.printer_id}

    def _handle_resume(self, action: FarmAction) -> Tuple[float, bool, Dict]:
        if action.printer_id is None or not action.job_id:
            return -ACTION_REJECT_COST, False, {"error": "RESUME_JOB requires printer_id and job_id"}

        p = self._get_printer(action.printer_id)
        j = self._jobs.get(action.job_id)
        if p is None or j is None:
            return -ACTION_REJECT_COST, False, {"error": "invalid printer_id or job_id"}

        # Case A: agent-paused printer (PAUSED state)
        if p.state == PrinterState.PAUSED and j.state == JobState.PAUSED:
            p.state   = PrinterState.PRINTING
            p.current_job_id = j.job_id
            j.state   = JobState.PRINTING
            return 0.0, True, {"resumed": j.job_id, "clean": True}

        # Case B: runout-paused (printer is now IDLE after spool_swap completed)
        if (p.state == PrinterState.IDLE and p.current_job_id == j.job_id
                and j.state == JobState.PAUSED):
            p.state = PrinterState.PRINTING
            j.state = JobState.PRINTING
            # First PRINTING tick rolls at reliability × 0.85 (nozzle reheating risk)
            p.reliability_penalty_active = True
            return 0.0, True, {"resumed": j.job_id, "reliability_penalty": True}

        return -ACTION_REJECT_COST, False, {
            "error": f"resume precondition not met: p.state={p.state} j.state={j.state}",
            "reason": "incompatible_state",
        }

    def _handle_run_diagnostic(self, action: FarmAction) -> Tuple[float, bool, Dict]:
        if action.printer_id is None:
            return -ACTION_REJECT_COST, False, {"error": "RUN_DIAGNOSTIC requires printer_id"}

        p = self._get_printer(action.printer_id)
        if p is None:
            return -ACTION_REJECT_COST, False, {"error": "invalid printer_id"}

        p.revealed_this_step = True

        has_active_fault = bool(p.active_faults)
        # webcam_freeze is cleared by RUN_DIAGNOSTIC (live frame fetched)
        if "webcam_freeze" in p.active_faults:
            clear_fault(p.active_faults, "webcam_freeze")
        if "progress_drift" in p.active_faults:
            clear_fault(p.active_faults, "progress_drift")

        if has_active_fault:
            # Reward: -base + bonus (net +$1.50 if fault found)
            return DIAGNOSTIC_BONUS - DIAGNOSTIC_BASE_COST, True, {
                "diagnostic_caught_fault": True,
                "faults": list(p.active_faults.keys()),
            }
        else:
            return -DIAGNOSTIC_BASE_COST, True, {
                "diagnostic_was_unnecessary": True,
                "faults": [],
            }

    def _handle_dispatch_ticket(self, action: FarmAction) -> Tuple[float, bool, Dict]:
        if action.operator_id is None or not action.ticket_type or action.printer_id is None:
            return -ACTION_REJECT_COST, False, {
                "error": "DISPATCH_TICKET requires operator_id, ticket_type, printer_id"
            }

        op = self._get_operator(action.operator_id)
        if op is None:
            return -ACTION_REJECT_COST, False, {"error": f"unknown operator {action.operator_id}"}

        from .operators import is_on_shift, queue_total
        if not is_on_shift(op, self.time_step):
            return -ACTION_REJECT_COST, False, {"error": f"operator {action.operator_id} is off-shift"}
        if queue_total(op) >= op.queue_capacity:
            return -ACTION_REJECT_COST, False, {"error": f"operator {action.operator_id} queue full"}

        ticket = self._make_ticket(
            ticket_type=action.ticket_type,
            target_printer_id=action.printer_id,
            operator_id=action.operator_id,
            payload={"material": action.material} if action.material else {},
        )
        op.queue.append(ticket)

        # MAINTENANCE_QUEUED state gate
        if action.ticket_type in ("maintenance_basic", "maintenance_full_rebuild"):
            p = self._get_printer(action.printer_id)
            if p and p.state == PrinterState.IDLE:
                p.state = PrinterState.MAINTENANCE_QUEUED
                p.outstanding_ticket_id = ticket.ticket_id

        return 0.0, True, {"dispatched": ticket.ticket_id, "to": action.operator_id}

    def _handle_request_spool_swap(self, action: FarmAction) -> Tuple[float, bool, Dict]:
        """Sugar: auto-routes spool_swap to the best available operator."""
        if action.printer_id is None or not action.material:
            return -ACTION_REJECT_COST, False, {"error": "REQUEST_SPOOL_SWAP requires printer_id and material"}

        op = self._auto_route_operator("spool_swap")
        if op is None:
            return -ACTION_REJECT_COST, False, {"error": "no qualified operator available for spool_swap"}

        ticket = self._make_ticket(
            ticket_type="spool_swap",
            target_printer_id=action.printer_id,
            operator_id=op.operator_id,
            payload={"material": action.material},
        )
        op.queue.append(ticket)
        return 0.0, True, {"dispatched": ticket.ticket_id, "to": op.operator_id}

    def _handle_request_maintenance(self, action: FarmAction) -> Tuple[float, bool, Dict]:
        """Sugar: auto-routes maintenance_basic or maintenance_full_rebuild."""
        if action.printer_id is None:
            return -ACTION_REJECT_COST, False, {"error": "REQUEST_MAINTENANCE requires printer_id"}

        mtype = action.maintenance_type or "maintenance_basic"
        op    = self._auto_route_operator(mtype)
        if op is None:
            return -ACTION_REJECT_COST, False, {"error": f"no qualified operator available for {mtype}"}

        ticket = self._make_ticket(
            ticket_type=mtype,
            target_printer_id=action.printer_id,
            operator_id=op.operator_id,
        )
        op.queue.append(ticket)

        p = self._get_printer(action.printer_id)
        if p and p.state == PrinterState.IDLE:
            p.state = PrinterState.MAINTENANCE_QUEUED
            p.outstanding_ticket_id = ticket.ticket_id

        return 0.0, True, {"dispatched": ticket.ticket_id, "to": op.operator_id, "type": mtype}

    def _handle_override_operator(self, action: FarmAction) -> Tuple[float, bool, Dict]:
        if not action.ticket_id:
            return -ACTION_REJECT_COST, False, {"error": "OVERRIDE_OPERATOR requires ticket_id"}

        ticket = self._tickets.get(action.ticket_id)
        if ticket is None:
            return -ACTION_REJECT_COST, False, {"error": f"unknown ticket {action.ticket_id}"}

        for op in self._operators:
            if ticket in op.queue:
                # Can only cancel if not in-progress
                op.queue.remove(ticket)
                ticket.state = TicketState.REJECTED
                ticket.rejection_reason = "dispatcher_override"

                # Write to oversight log
                self._oversight_log.append({
                    "step":     self.time_step,
                    "event":    "OVERRIDE_OPERATOR",
                    "ticket_id": action.ticket_id,
                    "reason":   action.reason or "unspecified",
                })

                # Revert MAINTENANCE_QUEUED if applicable
                p = self._get_printer(ticket.target_printer_id)
                if p and p.state == PrinterState.MAINTENANCE_QUEUED:
                    p.state = PrinterState.IDLE
                    p.outstanding_ticket_id = None

                return -OVERRIDE_COST, True, {"overridden": action.ticket_id}

            # Ticket might be the current ticket (in-progress → cannot override)
            if op.current_ticket and op.current_ticket.ticket_id == action.ticket_id:
                return -ACTION_REJECT_COST, False, {"error": "ticket already in progress"}

        return -ACTION_REJECT_COST, False, {"error": "ticket not found in any queue"}

    # ==================================================================
    #  Printer physics tick
    # ==================================================================

    def _tick_physics(self) -> float:
        """Advance all printers one step. Returns total dollar delta."""
        total_delta = 0.0

        for p in self._printers:

            # -- OFFLINE countdown ----------------------------------------
            if p.state == PrinterState.OFFLINE:
                p.offline_remaining -= 1
                if p.offline_remaining <= 0:
                    p.state           = PrinterState.IDLE
                    p.offline_remaining = 0
                    p.fatigue_level   = 0.0
                continue

            # -- ERROR self-recovery (unjam_printer ticket is faster) -------
            if p.state == PrinterState.ERROR:
                p.error_remaining -= 1
                if p.error_remaining <= 0:
                    p.state         = PrinterState.IDLE
                    p.error_remaining = 0
                continue

            # -- MAINTENANCE countdown -------------------------------------
            if p.state == PrinterState.MAINTENANCE:
                p.warmup_remaining -= 1
                total_delta -= electricity_cost(p.profile.avg_power_watts, "MAINTENANCE")
                if p.warmup_remaining <= 0:
                    full = (p.maintenance_type == "maintenance_full_rebuild")
                    if full:
                        p.reliability = p.profile.reliability_base
                        clear_fault_class(p.active_faults, "maintenance_full_rebuild")
                    else:
                        p.reliability = min(1.0, p.reliability + 0.05)
                        clear_fault_class(p.active_faults, "maintenance_basic")
                    p.state              = PrinterState.IDLE
                    p.fatigue_level      = 0.0
                    p.maintenance_due_in = 50
                    p.bed_drift_counter  = 0.0
                    p.maintenance_type   = None
                continue

            # -- MAINTENANCE_QUEUED (waiting for operator + cooldown) ------
            if p.state == PrinterState.MAINTENANCE_QUEUED:
                p.consecutive_idle_steps += 1
                continue

            # -- WARMING_UP countdown --------------------------------------
            if p.state == PrinterState.WARMING_UP:
                total_delta -= electricity_cost(p.profile.avg_power_watts, "WARMING_UP")
                p.warmup_remaining -= 1
                if p.warmup_remaining <= 0:
                    j = self._jobs.get(p.current_job_id) if p.current_job_id else None
                    if j and j.state == JobState.PRINTING:
                        p.state = PrinterState.PRINTING
                    else:
                        p.state = PrinterState.IDLE
                continue

            # -- PAUSED (agent-induced — nozzle hot) -----------------------
            if p.state == PrinterState.PAUSED:
                total_delta -= electricity_cost(p.profile.avg_power_watts, "PAUSED")
                continue

            # -- PAUSED_RUNOUT (env-induced — nozzle cooling) --------------
            if p.state == PrinterState.PAUSED_RUNOUT:
                total_delta -= electricity_cost(p.profile.avg_power_watts, "PAUSED_RUNOUT")
                continue

            # -- IDLE ------------------------------------------------------
            if p.state == PrinterState.IDLE:
                p.consecutive_idle_steps += 1
                continue

            # -- PRINTING --------------------------------------------------
            if p.state == PrinterState.PRINTING:
                j = self._jobs.get(p.current_job_id)
                if j is None:
                    p.state          = PrinterState.IDLE
                    p.current_job_id = None
                    continue

                # Fatigue accumulation (0.1/step → catastrophic after ~100 printing steps)
                p.fatigue_level += 0.1
                if p.fatigue_level >= 10.0:
                    # Catastrophic failure → OFFLINE
                    clawback = j.accrued_revenue
                    j.state          = JobState.FAILED
                    j.accrued_revenue = 0.0
                    p.state           = PrinterState.OFFLINE
                    p.offline_remaining = 10
                    p.current_job_id  = None
                    total_delta      -= (CATASTROPHIC_COST + clawback)
                    self._oversight_log.append({
                        "step":    self.time_step,
                        "event":   "CATASTROPHIC_FAILURE",
                        "printer": p.printer_id,
                        "job":     j.job_id,
                    })
                    continue

                # Reliability roll
                eff_reliability = p.reliability
                if p.reliability_penalty_active:
                    eff_reliability = p.reliability * 0.85
                    p.reliability_penalty_active = False  # only first tick

                # Per-step failure probability = (1 - reliability) / print_time_steps
                # reliability_base represents per-job success probability, not per-step.
                per_step_fail = (1.0 - eff_reliability) / max(j.print_time_steps, 1)
                if self._rng.random() < per_step_fail:
                    # Stochastic job failure → ERROR (job back to PENDING for retry)
                    clawback = j.accrued_revenue
                    scrap    = scrap_cost(j.material_required, j.weight_required_g,
                                         j.progress_steps, j.print_time_steps)
                    j.state           = JobState.PENDING
                    j.progress_steps  = 0
                    j.accrued_revenue = 0.0
                    p.state           = PrinterState.ERROR
                    p.error_remaining = 5   # self-recovers in 5 steps if not unjammed
                    p.current_job_id  = None
                    total_delta      -= (scrap + clawback)
                    self._oversight_log.append({
                        "step":    self.time_step,
                        "event":   "STOCHASTIC_FAILURE",
                        "printer": p.printer_id,
                        "job":     j.job_id,
                    })
                    continue

                # Consume filament
                burn = j.weight_required_g / max(j.print_time_steps, 1)
                p.spool_weight_g = max(0.0, p.spool_weight_g - burn)

                # Accrue revenue
                rev = _step_revenue(j.price_usd, j.print_time_steps)
                j.accrued_revenue += rev
                total_delta       += rev

                # Costs
                total_delta -= electricity_cost(p.profile.avg_power_watts, "PRINTING")
                total_delta -= amortization_cost(p.profile.amortization_per_hour)

                # Bed drift counter
                p.bed_drift_counter += 0.0002
                # Maintenance counter
                p.maintenance_due_in -= 1
                if p.maintenance_due_in <= 0:
                    p.reliability = max(0.5, p.reliability - 0.03)

                # Progress
                j.progress_steps += 1

                # Filament runout → PAUSED_RUNOUT
                if p.spool_weight_g <= 0:
                    p.state = PrinterState.PAUSED_RUNOUT
                    j.state = JobState.PAUSED
                    continue

                # Job complete
                if j.progress_steps >= j.print_time_steps:
                    j.state           = JobState.COMPLETED
                    p.state           = PrinterState.IDLE
                    p.current_job_id  = None
                    p.consecutive_idle_steps = 0
                    # Final revenue already accrued step-by-step

                continue

        return total_delta

    # ==================================================================
    #  Failure injection
    # ==================================================================

    def _inject_failures(self) -> None:
        """Inject scheduled and stochastic sensor failures for this step."""
        # Scheduled from task config
        for sched in self._cfg.scheduled_failures:
            if sched["trigger_step"] == self.time_step:
                p = self._get_printer(sched["printer_id"])
                if p and sched["mode"] not in p.active_faults:
                    j = self._jobs.get(p.current_job_id) if p.current_job_id else None
                    fault = inject_scheduled(
                        p, sched["mode"], sched["duration"], self.time_step, job=j
                    )
                    p.active_faults[sched["mode"]] = fault

        # Stochastic (per FAILURE_MODES rates, modulated by profile)
        rate_overrides = self._cfg.stochastic_failure_rates
        all_zero  = rate_overrides.get("_all") == "zero"
        all_default = rate_overrides.get("_all") == "profile_default"

        if not all_zero:
            for p in self._printers:
                j = self._jobs.get(p.current_job_id) if p.current_job_id else None
                for mode in FAILURE_MODES:
                    override = None
                    if not all_default:
                        mode_rate = rate_overrides.get(mode)
                        if mode_rate == "zero":
                            continue
                        if isinstance(mode_rate, (int, float)):
                            override = float(mode_rate)

                    fault = try_inject(mode, p, j, self.time_step, self._rng, override)
                    if fault is not None:
                        p.active_faults[mode] = fault

        # Advance duration counters, drop expired faults
        for p in self._printers:
            p.active_faults = advance_faults(p.active_faults)

        # Inject scheduled synthetic operator reports
        for report in self._cfg.scheduled_operator_reports:
            if report["step"] == self.time_step:
                self._oversight_log.append({
                    "step":        self.time_step,
                    "event":       "SCHEDULED_OPERATOR_REPORT",
                    "operator_id": report["operator_id"],
                    "printer_id":  report["printer_id"],
                    "report":      report["report"],
                })

    # ==================================================================
    #  SLA penalty evaluation
    # ==================================================================

    def _check_sla_penalties(self) -> float:
        delta = 0.0
        for j in self._jobs.values():
            if j.state in (JobState.COMPLETED, JobState.CANCELLED):
                continue
            if j.deadline_steps is None:
                continue
            penalty, new_fixed = sla_penalty_this_step(
                j.price_usd, j.deadline_steps, self.time_step, j.sla_fixed_applied
            )
            if penalty > 0:
                # Cap: total SLA penalty ≤ 80% of price
                from .economics import SLA_CAP_FRACTION
                remaining_cap = j.price_usd * SLA_CAP_FRACTION - j.total_sla_penalty
                penalty = min(penalty, max(0.0, remaining_cap))
                j.total_sla_penalty  += penalty
                j.sla_fixed_applied   = new_fixed
                delta                -= penalty
        return delta

    # ==================================================================
    #  Observation builder (applies sensor corruption layer)
    # ==================================================================

    def _build_observation(self) -> FarmObservation:
        printer_obs_list: List[PrinterObservation] = []
        for p in self._printers:
            j = self._jobs.get(p.current_job_id) if p.current_job_id else None
            obs = self._build_printer_obs(p, j)
            printer_obs_list.append(obs)

        operator_obs_list = [
            to_observation(op, self.time_step) for op in self._operators
        ]

        return FarmObservation(
            active_queue  = list(self._jobs.values()),
            printers      = printer_obs_list,
            operators     = operator_obs_list,
            inventory     = dict(self._inventory),
            time_step     = self.time_step,
            max_steps     = self.max_steps,
            net_profit_usd = self._net_profit,
            total_labor_billed = self._total_labor,
        )

    def _build_printer_obs(
        self, p: PrinterInternal, job: Optional[PrintJob]
    ) -> PrinterObservation:
        """Build a (potentially corrupted) PrinterObservation from ground truth."""

        # If RUN_DIAGNOSTIC was used this step, expose ground truth for this printer
        if p.revealed_this_step:
            return PrinterObservation(
                printer_id       = p.printer_id,
                profile_id       = p.profile.profile_id,
                state            = p.state,
                current_material = p.current_material,
                current_job_id   = p.current_job_id,
                spool_weight_g   = p.spool_weight_g,
                reliability      = p.reliability,
                maintenance_due_in = p.maintenance_due_in,
                fatigue_level    = p.fatigue_level,
                warmup_remaining = p.warmup_remaining,
                offline_remaining = p.offline_remaining,
                consecutive_idle_steps = p.consecutive_idle_steps,
                hotend_temp      = p.hotend_temp,
                fan_rpm          = p.fan_rpm,
                webcam_hash      = p.webcam_hash,
                telemetry_ts     = p.telemetry_ts,
                revealed_this_step = True,
                bed_drift_counter = p.bed_drift_counter,
                outstanding_ticket_id = p.outstanding_ticket_id,
            )

        # Apply sensor corruption layer
        overrides = apply_corruption(p, job)

        # progress_drift applies to job progress
        corrupted_progress = get_corrupted_progress(p, job)
        corrupted_job_id   = p.current_job_id

        # Map state override
        state_override = overrides.pop("state", p.state)

        return PrinterObservation(
            printer_id       = p.printer_id,
            profile_id       = p.profile.profile_id,
            state            = state_override,
            current_material = p.current_material,
            current_job_id   = corrupted_job_id,
            spool_weight_g   = overrides.get("spool_weight_g", p.spool_weight_g),
            reliability      = p.reliability,
            maintenance_due_in = p.maintenance_due_in,
            fatigue_level    = p.fatigue_level,
            warmup_remaining = p.warmup_remaining,
            offline_remaining = p.offline_remaining,
            consecutive_idle_steps = p.consecutive_idle_steps,
            hotend_temp      = overrides.get("hotend_temp", p.hotend_temp),
            fan_rpm          = overrides.get("fan_rpm", p.fan_rpm),
            webcam_hash      = overrides.get("webcam_hash", p.webcam_hash),
            telemetry_ts     = overrides.get("telemetry_ts", p.telemetry_ts),
            revealed_this_step = False,
            bed_drift_counter = p.bed_drift_counter,
            outstanding_ticket_id = p.outstanding_ticket_id,
        )

    # ==================================================================
    #  Ticket lifecycle helpers
    # ==================================================================

    def _make_ticket(
        self, ticket_type: str, target_printer_id: int,
        operator_id: str, payload: Optional[Dict] = None,
    ) -> Ticket:
        self._ticket_counter += 1
        tid = f"t{self._ticket_counter:04d}"
        ticket = Ticket(
            ticket_id=tid,
            ticket_type=ticket_type,
            target_printer_id=target_printer_id,
            operator_id=operator_id,
            created_step=self.time_step,
            payload=payload or {},
        )
        self._tickets[tid] = ticket
        return ticket

    def _handle_ticket_completion(
        self, ticket: Ticket, printers_by_id: Dict[int, PrinterInternal]
    ) -> None:
        """Post-completion maintenance state sync (maintenance transition)."""
        p = printers_by_id.get(ticket.target_printer_id)
        if p is None:
            return

        if ticket.state == TicketState.COMPLETED:
            if ticket.ticket_type in ("maintenance_basic", "maintenance_full_rebuild"):
                # operators.py sets p.state = MAINTENANCE directly in _apply_effects.
                # Clear the MAINTENANCE_QUEUED flag if not already done.
                if p.outstanding_ticket_id == ticket.ticket_id:
                    p.outstanding_ticket_id = None

    # ==================================================================
    #  Auto-routing helpers
    # ==================================================================

    def _auto_route_operator(self, ticket_type: str) -> Optional[OperatorInternal]:
        """Find the best available operator for a given ticket type."""
        from .operators import SKILL_RANK, TICKET_SKILL_REQ, queue_total
        req_rank = SKILL_RANK.get(TICKET_SKILL_REQ.get(ticket_type, "junior"), 0)
        candidates = [
            op for op in self._operators
            if SKILL_RANK.get(op.skill_level, 0) >= req_rank
            and is_on_shift(op, self.time_step)
            and queue_total(op) < op.queue_capacity
        ]
        if not candidates:
            return None
        # Lowest queue → highest skill → lowest operator_id
        return min(candidates, key=lambda o: (
            queue_total(o),
            -SKILL_RANK.get(o.skill_level, 0),
            o.operator_id,
        ))

    # ==================================================================
    #  Internal lookup helpers
    # ==================================================================

    def _get_printer(self, pid: int) -> Optional[PrinterInternal]:
        for p in self._printers:
            if p.printer_id == pid:
                return p
        return None

    def _get_operator(self, oid: str) -> Optional[OperatorInternal]:
        for op in self._operators:
            if op.operator_id == oid:
                return op
        return None

    # ==================================================================
    #  Printer builder (from TaskConfig)
    # ==================================================================

    def _build_printers(self) -> List[PrinterInternal]:
        cfg = self._cfg
        overrides_by_id = {o["printer_id"]: o for o in cfg.printer_overrides}
        printers = []

        for i, profile_id in enumerate(cfg.printer_profiles, start=1):
            profile = PROFILES.get(profile_id)
            if profile is None:
                raise ValueError(f"Unknown printer profile: '{profile_id}'")

            ov = overrides_by_id.get(i, {})
            p  = PrinterInternal(
                printer_id       = i,
                profile          = profile,
                current_material = ov.get("current_material"),
                spool_weight_g   = float(ov.get("spool_weight_g", 1000.0)),
                fatigue_level    = int(ov.get("fatigue_level", 0)),
                maintenance_due_in = int(ov.get("maintenance_due_in", 50)),
            )
            # Set initial state from override if provided
            state_str = ov.get("state")
            if state_str:
                p.state = PrinterState(state_str)
            p.warmup_remaining = int(ov.get("warmup_remaining", 0))
            printers.append(p)

        return printers

    # ==================================================================
    #  Render helpers
    # ==================================================================

    def render_dashboard(self) -> None:
        icons = {
            PrinterState.IDLE:               "⚪ IDLE           ",
            PrinterState.WARMING_UP:         "🟡 WARMING UP     ",
            PrinterState.PRINTING:           "🟢 PRINTING       ",
            PrinterState.PAUSED:             "🔵 PAUSED         ",
            PrinterState.PAUSED_RUNOUT:      "🟣 PAUSED_RUNOUT  ",
            PrinterState.ERROR:              "🔴 ERROR          ",
            PrinterState.MAINTENANCE_QUEUED: "🟠 MAINT_QUEUED   ",
            PrinterState.MAINTENANCE:        "🟠 MAINTENANCE    ",
            PrinterState.OFFLINE:            "⚫ OFFLINE        ",
        }
        print(f"\n{'─'*80}")
        print(f" STEP {self.time_step}/{self.max_steps}  |  TASK: {self.current_task_id}"
              f"  |  P&L: ${self._net_profit:+.2f}")
        print(f"{'─'*80}")
        for p in self._printers:
            icon = icons.get(p.state, "? ")
            mat  = f"{p.current_material} ({p.spool_weight_g:.0f}g)" if p.current_material else "Empty"
            job  = f"Job: {p.current_job_id}" if p.current_job_id else "—"
            faults = ",".join(p.active_faults.keys()) if p.active_faults else "—"
            print(f"  [P{p.printer_id:02d} {icon}] {mat:22s} | {job:16s}"
                  f" | rel={p.reliability:.0%} fat={p.fatigue_level}/10"
                  f" | faults=[{faults}]")

        print(f"\n  Inventory: { {k: f'{v:.0f}g' for k, v in self._inventory.items()} }")

        for j in self._jobs.values():
            if j.state not in (JobState.COMPLETED, JobState.CANCELLED):
                dl = f" dl={j.deadline_steps}" if j.deadline_steps else ""
                print(f"  {j.job_id}: {j.state.value:10s}  prio={j.priority}"
                      f"  {j.progress_steps}/{j.print_time_steps}{dl}"
                      f"  sla_pen=${j.total_sla_penalty:.2f}")

        print(f"\n  Operators:")
        for op in self._operators:
            on = "ON " if is_on_shift(op, self.time_step) else "OFF"
            ct = op.current_ticket.ticket_id if op.current_ticket else "—"
            print(f"  [{on}] {op.operator_id} ({op.skill_level:6s})"
                  f" fatigue={op.current_fatigue:.2f}"
                  f" queue={len(op.queue)}/{op.queue_capacity}"
                  f" busy_until={op.busy_until} current_ticket={ct}")
        print(f"{'─'*80}")
