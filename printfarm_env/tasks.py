from .models import (
    FarmObservation, PrintJob, PrinterObservation, PrinterState, JobState,
)


def load_task(task_id: str) -> FarmObservation:
    """Build the initial observation for a given task."""

    if task_id == "task_1":
        # -----------------------------------------------------------------
        # EASY — "Night Shift Scheduling"
        # 5 PLA jobs across 10 printers. Printers 1-3 are loaded and ready.
        # Printers 4-5 have old spools (low reliability) but are loaded.
        # Jobs have mixed priorities (one urgent with a tight deadline).
        # Agent must assign jobs intelligently: urgent first, avoid
        # unreliable printers for the urgent job.
        # max_steps=20 — tight enough that WAITing wastes the budget.
        # -----------------------------------------------------------------
        printers = []
        for i in range(1, 11):
            if i <= 3:
                # Good printers, loaded with PLA
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PLA", spool_weight_g=1000.0,
                    reliability=0.98, maintenance_due_in=40,
                ))
            elif i <= 5:
                # Loaded but unreliable (old/worn)
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PLA", spool_weight_g=600.0,
                    reliability=0.80, maintenance_due_in=10,
                ))
            else:
                # Empty printers — need filament swap
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        queue = [
            PrintJob(job_id="job_1", material_required="PLA",
                     weight_required_g=80.0, print_time_steps=4,
                     priority=3, deadline_steps=10),       # URGENT
            PrintJob(job_id="job_2", material_required="PLA",
                     weight_required_g=100.0, print_time_steps=5,
                     priority=2),
            PrintJob(job_id="job_3", material_required="PLA",
                     weight_required_g=100.0, print_time_steps=5,
                     priority=2),
            PrintJob(job_id="job_4", material_required="PLA",
                     weight_required_g=120.0, print_time_steps=6,
                     priority=1),                          # Low priority
            PrintJob(job_id="job_5", material_required="PLA",
                     weight_required_g=60.0, print_time_steps=3,
                     priority=2, deadline_steps=15),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"PLA": 5000.0, "PETG": 1000.0},
            time_step=0, max_steps=20,
        )

    elif task_id == "task_2":
        # -----------------------------------------------------------------
        # MEDIUM — "Material Juggle"
        # 3 jobs requiring 2 different materials (PETG + ABS).
        # Only 1 printer loaded (PETG, low spool). Inventory is limited.
        # One job is urgent with a deadline. Agent must:
        #   1. Swap filament on the right printers
        #   2. Prioritise the urgent job
        #   3. Manage limited inventory (only 1500g PETG, 1000g ABS)
        # Probabilistic failures on printer 3 (reliability=0.75).
        # max_steps=25
        # -----------------------------------------------------------------
        printers = []
        for i in range(1, 11):
            if i == 1:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PETG", spool_weight_g=200.0,
                    reliability=0.95, maintenance_due_in=30,
                ))
            elif i == 2:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="ABS", spool_weight_g=1000.0,
                    reliability=0.92, maintenance_due_in=35,
                ))
            elif i == 3:
                # Unreliable printer, no filament
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.75, maintenance_due_in=15,
                ))
            else:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        queue = [
            PrintJob(job_id="job_urgent", material_required="PETG",
                     weight_required_g=700.0, print_time_steps=14,
                     priority=3, deadline_steps=20),       # Urgent, tight deadline
            PrintJob(job_id="job_abs", material_required="ABS",
                     weight_required_g=400.0, print_time_steps=10,
                     priority=2),
            PrintJob(job_id="job_small", material_required="PETG",
                     weight_required_g=150.0, print_time_steps=5,
                     priority=1),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"PETG": 1500.0, "ABS": 1000.0},
            time_step=0, max_steps=25,
        )

    elif task_id == "task_3":
        # -----------------------------------------------------------------
        # HARD — "Chaos Shift"
        # 4 jobs (mixed materials, priorities, deadlines) across printers
        # with varying reliability. One printer starts in ERROR.
        # Printer 2 will need maintenance mid-run (maintenance_due_in=5).
        # Limited inventory forces trade-offs.
        # Stochastic failures will hit unreliable printers.
        # Agent must triage errors, re-route jobs, manage maintenance,
        # and meet deadlines under uncertainty.
        # max_steps=30
        # -----------------------------------------------------------------
        printers = []
        for i in range(1, 11):
            if i == 1:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.ERROR,  # Starts broken!
                    current_material="ABS", spool_weight_g=800.0,
                    reliability=0.90, maintenance_due_in=20,
                ))
            elif i == 2:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="ABS", spool_weight_g=1000.0,
                    reliability=0.88, maintenance_due_in=5,  # Needs maintenance soon!
                ))
            elif i == 3:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PETG", spool_weight_g=500.0,
                    reliability=0.70, maintenance_due_in=30,  # Unreliable
                ))
            elif i == 4:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.96, maintenance_due_in=50,
                ))
            else:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        queue = [
            PrintJob(job_id="job_critical", material_required="ABS",
                     weight_required_g=500.0, print_time_steps=10,
                     priority=3, deadline_steps=18),       # Critical, tight
            PrintJob(job_id="job_petg_rush", material_required="PETG",
                     weight_required_g=300.0, print_time_steps=8,
                     priority=3, deadline_steps=22),       # Urgent
            PrintJob(job_id="job_bulk", material_required="ABS",
                     weight_required_g=600.0, print_time_steps=15,
                     priority=1),                          # Low, heavy
            PrintJob(job_id="job_filler", material_required="PETG",
                     weight_required_g=100.0, print_time_steps=4,
                     priority=2, deadline_steps=28),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"ABS": 1200.0, "PETG": 1000.0, "PLA": 500.0},
            time_step=0, max_steps=30,
        )

    else:
        raise ValueError(f"Unknown task {task_id}")


# =========================================================================
#  Grader
# =========================================================================

class TaskGrader:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.wasted_steps = 0
        self.failed_actions = 0
        self.deadline_misses = 0

    # ------------------------------------------------------------------
    #  Per-step bookkeeping
    # ------------------------------------------------------------------
    def step_update(self, action, action_handled: bool,
                    state: FarmObservation, time_step: int):
        if action and action.action.value == "WAIT":
            self.wasted_steps += 1
        if action and not action_handled and action.action.value != "WAIT":
            self.failed_actions += 1

        # Check for deadline misses on completed/failed/still-pending jobs
        for job in state.active_queue:
            if job.deadline_steps and time_step >= job.deadline_steps:
                if job.state not in (JobState.COMPLETED,):
                    # We'll count this once at the end via get_score
                    pass

    # ------------------------------------------------------------------
    #  Score calculation  (0.0 – 1.0)
    # ------------------------------------------------------------------
    def get_score(self, state: FarmObservation) -> float:
        step_penalty = (self.wasted_steps * 0.01) + (self.failed_actions * 0.02)

        if self.task_id == "task_1":
            return self._score_task1(state, step_penalty)
        elif self.task_id == "task_2":
            return self._score_task2(state, step_penalty)
        elif self.task_id == "task_3":
            return self._score_task3(state, step_penalty)
        return 0.0

    # --- Task 1: Night Shift Scheduling ----------------------------------
    def _score_task1(self, state: FarmObservation, penalty: float) -> float:
        jobs = state.active_queue
        if not jobs:
            return 0.0

        # Hard fail: any failed or cancelled job
        if any(j.state in (JobState.FAILED, JobState.CANCELLED) for j in jobs):
            # Still give partial credit for what was completed
            completed = sum(1 for j in jobs if j.state == JobState.COMPLETED)
            return _clamp(completed * 0.1 - penalty)

        total_weight = 0.0
        earned = 0.0
        for job in jobs:
            w = _priority_weight(job.priority)
            total_weight += w
            if job.state == JobState.COMPLETED:
                # Deadline bonus/penalty
                if job.deadline_steps and state.time_step <= job.deadline_steps:
                    earned += w  # Full credit
                elif job.deadline_steps:
                    earned += w * 0.6  # Late
                else:
                    earned += w
            elif job.state == JobState.PRINTING and job.print_time_steps > 0:
                earned += w * 0.4 * (job.progress_steps / job.print_time_steps)

        score = earned / total_weight if total_weight > 0 else 0.0
        return _clamp(score - penalty)

    # --- Task 2: Material Juggle -----------------------------------------
    def _score_task2(self, state: FarmObservation, penalty: float) -> float:
        jobs = state.active_queue
        if not jobs:
            return 0.0

        total_weight = 0.0
        earned = 0.0
        for job in jobs:
            w = _priority_weight(job.priority)
            total_weight += w

            if job.state == JobState.COMPLETED:
                if job.deadline_steps and state.time_step <= job.deadline_steps:
                    earned += w
                elif job.deadline_steps:
                    earned += w * 0.5  # Late completion
                else:
                    earned += w
            elif job.state == JobState.FAILED:
                if job.progress_steps > 0:
                    earned += w * 0.15  # Tried but failed
            elif job.state == JobState.PRINTING and job.print_time_steps > 0:
                earned += w * 0.5 * (job.progress_steps / job.print_time_steps)
            elif job.state == JobState.PENDING:
                pass  # No credit

        score = earned / total_weight if total_weight > 0 else 0.0
        return _clamp(score - penalty)

    # --- Task 3: Chaos Shift ---------------------------------------------
    def _score_task3(self, state: FarmObservation, penalty: float) -> float:
        jobs = state.active_queue
        if not jobs:
            return 0.0

        total_weight = 0.0
        earned = 0.0
        for job in jobs:
            w = _priority_weight(job.priority)
            total_weight += w

            if job.state == JobState.COMPLETED:
                if job.deadline_steps and state.time_step <= job.deadline_steps:
                    earned += w
                elif job.deadline_steps:
                    earned += w * 0.4  # Late under chaos is still partial credit
                else:
                    earned += w
            elif job.state == JobState.FAILED:
                if job.progress_steps > 0:
                    earned += w * 0.1
            elif job.state == JobState.PRINTING and job.print_time_steps > 0:
                earned += w * 0.4 * (job.progress_steps / job.print_time_steps)

        # Bonus for zero deadline misses on urgent jobs
        urgent_met = all(
            state.time_step <= j.deadline_steps
            for j in jobs
            if j.priority == 3 and j.deadline_steps and j.state == JobState.COMPLETED
        )
        if urgent_met and any(j.priority == 3 and j.state == JobState.COMPLETED for j in jobs):
            earned += 0.5  # Significant bonus

        score = earned / (total_weight + 0.5) if total_weight > 0 else 0.0
        return _clamp(score - penalty)


# =========================================================================
#  Helpers
# =========================================================================

def _priority_weight(priority: int) -> float:
    """Higher priority jobs count more toward the score."""
    return {1: 0.5, 2: 1.0, 3: 2.0}.get(priority, 1.0)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
