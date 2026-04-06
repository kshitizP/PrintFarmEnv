from typing import Optional
import random

from .models import (
    FarmAction, FarmActionEnum, FarmObservation,
    PrinterState, JobState,
)
from .tasks import load_task, TaskGrader

# Resolve the correct base class
try:
    from openenv.core import Environment as _BaseEnvironment
except ImportError:
    try:
        from openenv_core import Environment as _BaseEnvironment
    except ImportError:
        class _BaseEnvironment:
            def __init__(self, **kwargs):
                pass


class PrintFarmEnvironment(_BaseEnvironment):

    def __init__(self):
        super().__init__()
        self.current_task_id = "task_1"
        self._state: FarmObservation = load_task("task_1")
        self.time_step = 0
        self.max_steps = 20
        self.grader = TaskGrader("task_1")
        self._rng = random.Random(42)

    # ------------------------------------------------------------------
    #  OpenEnv reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None,
              episode_id: Optional[str] = None, **kwargs) -> FarmObservation:
        task_id = kwargs.get("task_id", episode_id or "task_1")
        self.current_task_id = task_id
        self._state = load_task(task_id)
        self.time_step = 0
        self.max_steps = self._state.max_steps
        self.grader = TaskGrader(task_id)
        self._rng = random.Random(seed if seed is not None else 42)
        return self._state

    # ------------------------------------------------------------------
    #  OpenEnv state
    # ------------------------------------------------------------------
    @property
    def state(self) -> FarmObservation:
        return self._state

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _printer(self, pid: int):
        for p in self._state.printers:
            if p.printer_id == pid:
                return p
        return None

    def _job(self, jid: str):
        for j in self._state.active_queue:
            if j.job_id == jid:
                return j
        return None

    def render_dashboard(self):
        icons = {
            PrinterState.IDLE: "⚪ IDLE       ",
            PrinterState.WARMING_UP: "🟡 WARMING UP ",
            PrinterState.PRINTING: "🟢 PRINTING   ",
            PrinterState.ERROR: "🔴 ERROR      ",
            PrinterState.MAINTENANCE: "🟠 MAINTENANCE",
        }
        print(f"\n--- STEP {self.time_step}/{self.max_steps}"
              f" | TASK: {self.current_task_id} ---")
        for p in self._state.printers:
            icon = icons.get(p.state, "?")
            mat = (f"{p.current_material} ({p.spool_weight_g:.0f}g)"
                   if p.current_material else "Empty")
            job = f"Job: {p.current_job_id}" if p.current_job_id else "No Job"
            maint = f"maint_in={p.maintenance_due_in}"
            rel = f"rel={p.reliability:.0%}"
            purge = " PURGE!" if p.purge_needed else ""
            print(f"  [{p.printer_id:02d} {icon}] {mat:20s} | {job:20s}"
                  f" | {rel} {maint}{purge}")

        inv = ", ".join(f"{k}: {v:.0f}g" for k, v in self._state.inventory.items())
        print(f"  Inventory: {inv}")

        for j in self._state.active_queue:
            dl = f" deadline={j.deadline_steps}" if j.deadline_steps else ""
            print(f"  {j.job_id}: {j.state.value:10s} prio={j.priority}"
                  f" progress={j.progress_steps}/{j.print_time_steps}{dl}")
        print("─" * 72)

    # ------------------------------------------------------------------
    #  OpenEnv step
    # ------------------------------------------------------------------
    def step(self, action: FarmAction,
             timeout_s: Optional[float] = None, **kwargs) -> FarmObservation:
        action_handled = False
        info: dict = {"error": None}

        # ---- Process agent action ------------------------------------
        if action:
            act = action.action

            if act == FarmActionEnum.ASSIGN_JOB:
                action_handled, info = self._handle_assign(action)

            elif act == FarmActionEnum.SWAP_FILAMENT:
                action_handled, info = self._handle_swap(action)

            elif act == FarmActionEnum.CANCEL_JOB:
                action_handled, info = self._handle_cancel(action)

            elif act == FarmActionEnum.PERFORM_MAINTENANCE:
                action_handled, info = self._handle_maintenance(action)

            # WAIT is always "handled" (no-op)
            elif act == FarmActionEnum.WAIT:
                action_handled = False  # grader tracks this

        # ---- Physics tick --------------------------------------------
        self._tick_physics()

        self.time_step += 1
        self._state.time_step = self.time_step

        # ---- Grading -------------------------------------------------
        self.grader.step_update(action, action_handled,
                                self._state, self.time_step)
        current_score = self.grader.get_score(self._state)

        # ---- Episode termination -------------------------------------
        done = self.time_step >= self.max_steps
        all_resolved = all(
            j.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
            for j in self._state.active_queue
        )
        if all_resolved:
            done = True

        self._state.reward = current_score
        self._state.done = done
        self._state.metadata = info
        return self._state

    # ==================================================================
    #  Action handlers
    # ==================================================================

    def _handle_assign(self, action: FarmAction):
        info = {"error": None}
        if not action.printer_id or not action.job_id:
            return False, {"error": "ASSIGN_JOB requires printer_id and job_id."}

        p = self._printer(action.printer_id)
        j = self._job(action.job_id)
        if not p or not j:
            return False, {"error": "Invalid printer_id or job_id."}

        if p.state != PrinterState.IDLE:
            return False, {"error": f"Printer {p.printer_id} is not IDLE"
                                    f" (state={p.state.value})."}
        if j.state != JobState.PENDING:
            return False, {"error": f"Job {j.job_id} is not PENDING"
                                    f" (state={j.state.value})."}
        if p.current_material != j.material_required:
            return False, {"error": f"Material mismatch: printer has"
                                    f" {p.current_material},"
                                    f" job needs {j.material_required}."}
        if p.spool_weight_g < j.weight_required_g:
            return False, {"error": f"Insufficient spool: {p.spool_weight_g:.0f}g"
                                    f" < {j.weight_required_g:.0f}g needed."}

        # Start warmup (purge adds 1 extra step)
        warmup = 1 + (1 if p.purge_needed else 0)
        p.state = PrinterState.WARMING_UP
        p.warmup_remaining = warmup
        p.purge_needed = False
        p.current_job_id = j.job_id
        j.state = JobState.PRINTING
        return True, info

    def _handle_swap(self, action: FarmAction):
        if not action.printer_id or not action.material:
            return False, {"error": "SWAP_FILAMENT requires printer_id and material."}

        p = self._printer(action.printer_id)
        if not p:
            return False, {"error": "Invalid printer_id."}
        if p.state not in (PrinterState.IDLE, PrinterState.ERROR):
            return False, {"error": f"Printer {p.printer_id} must be IDLE or ERROR"
                                    f" to swap (state={p.state.value})."}

        # Return current spool to inventory
        if p.current_material and p.spool_weight_g > 0:
            self._state.inventory[p.current_material] = (
                self._state.inventory.get(p.current_material, 0) + p.spool_weight_g
            )

        # Take new spool from inventory (1000g per spool)
        avail = self._state.inventory.get(action.material, 0)
        if avail < 1000.0:
            return False, {"error": f"Insufficient {action.material} in inventory"
                                    f" ({avail:.0f}g < 1000g)."}

        self._state.inventory[action.material] -= 1000.0
        p.current_material = action.material
        p.spool_weight_g = 1000.0
        p.purge_needed = True  # Needs purge before printing
        if p.state == PrinterState.ERROR:
            p.state = PrinterState.IDLE  # Swap clears error
        return True, {"error": None}

    def _handle_cancel(self, action: FarmAction):
        if not action.job_id:
            return False, {"error": "CANCEL_JOB requires job_id."}

        j = self._job(action.job_id)
        if not j:
            return False, {"error": "Invalid job_id."}
        if j.state not in (JobState.PENDING, JobState.PRINTING):
            return False, {"error": f"Job {j.job_id} cannot be cancelled"
                                    f" (state={j.state.value})."}

        # Free the printer if printing/warming up
        if j.state == JobState.PRINTING:
            for p in self._state.printers:
                if p.current_job_id == j.job_id:
                    p.state = PrinterState.IDLE
                    p.current_job_id = None
                    p.warmup_remaining = 0

        j.state = JobState.CANCELLED
        return True, {"error": None}

    def _handle_maintenance(self, action: FarmAction):
        if not action.printer_id:
            return False, {"error": "PERFORM_MAINTENANCE requires printer_id."}

        p = self._printer(action.printer_id)
        if not p:
            return False, {"error": "Invalid printer_id."}
        if p.state not in (PrinterState.IDLE, PrinterState.ERROR):
            return False, {"error": f"Printer {p.printer_id} must be IDLE or ERROR"
                                    f" for maintenance (state={p.state.value})."}

        # Maintenance takes 3 time steps (done via MAINTENANCE state)
        p.state = PrinterState.MAINTENANCE
        p.warmup_remaining = 3  # Re-use warmup counter for maintenance duration
        return True, {"error": None}

    # ==================================================================
    #  Physics tick (called once per step, after action)
    # ==================================================================

    def _tick_physics(self):
        for p in self._state.printers:

            # --- Maintenance countdown --------------------------------
            if p.state == PrinterState.MAINTENANCE:
                p.warmup_remaining -= 1
                if p.warmup_remaining <= 0:
                    p.state = PrinterState.IDLE
                    p.maintenance_due_in = 50  # Reset counter
                    p.reliability = min(1.0, p.reliability + 0.05)
                continue

            # --- Warmup countdown -------------------------------------
            if p.state == PrinterState.WARMING_UP:
                p.warmup_remaining -= 1
                if p.warmup_remaining <= 0:
                    p.state = PrinterState.PRINTING
                continue

            # --- Printing logic ---------------------------------------
            if p.state == PrinterState.PRINTING:
                j = self._job(p.current_job_id)
                if not j:
                    p.state = PrinterState.IDLE
                    p.current_job_id = None
                    continue

                # Stochastic failure check
                if self._rng.random() > p.reliability:
                    p.state = PrinterState.ERROR
                    j.state = JobState.PENDING  # Job can be re-routed
                    j.progress_steps = 0        # Must restart
                    p.current_job_id = None
                    continue

                # Consume filament
                burn = j.weight_required_g / j.print_time_steps
                p.spool_weight_g -= burn
                j.progress_steps += 1

                # Filament runout
                if p.spool_weight_g <= 0:
                    p.spool_weight_g = 0
                    p.state = PrinterState.ERROR
                    j.state = JobState.FAILED
                    p.current_job_id = None
                    continue

                # Job complete
                if j.progress_steps >= j.print_time_steps:
                    j.state = JobState.COMPLETED
                    p.state = PrinterState.IDLE
                    p.current_job_id = None

                # Decrement maintenance counter
                p.maintenance_due_in -= 1
                if p.maintenance_due_in <= 0:
                    # Printer degrades reliability when maintenance overdue
                    p.reliability = max(0.5, p.reliability - 0.03)
