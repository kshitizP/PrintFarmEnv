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
        # Printer 1 has PETG but low spool (must swap for the big job).
        # Printer 2 has ABS ready to go.
        # Agent must:
        #   1. Recognise P1's spool is too small and swap on another printer
        #   2. Prioritise the urgent job on a reliable printer
        #   3. Run all 3 jobs in parallel across available printers
        # Shorter print times and generous budget allow recovery from
        # one failure, making this genuinely medium difficulty.
        # max_steps=30
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
                    reliability=0.95, maintenance_due_in=40,
                ))
            elif i == 3:
                # Unreliable printer, no filament
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.80, maintenance_due_in=15,
                ))
            else:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        queue = [
            PrintJob(job_id="job_urgent", material_required="PETG",
                     weight_required_g=500.0, print_time_steps=8,
                     priority=3, deadline_steps=18),       # Urgent but achievable
            PrintJob(job_id="job_abs", material_required="ABS",
                     weight_required_g=400.0, print_time_steps=8,
                     priority=2),
            PrintJob(job_id="job_small", material_required="PETG",
                     weight_required_g=150.0, print_time_steps=5,
                     priority=1),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"PETG": 1500.0, "ABS": 1000.0},
            time_step=0, max_steps=30,
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
                     priority=3, deadline_steps=20),       # Critical, tight
            PrintJob(job_id="job_petg_rush", material_required="PETG",
                     weight_required_g=300.0, print_time_steps=8,
                     priority=3, deadline_steps=24),       # Urgent
            PrintJob(job_id="job_bulk", material_required="ABS",
                     weight_required_g=400.0, print_time_steps=10,
                     priority=1),                          # Low, heavy
            PrintJob(job_id="job_filler", material_required="PETG",
                     weight_required_g=100.0, print_time_steps=4,
                     priority=2, deadline_steps=30),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"ABS": 1200.0, "PETG": 1000.0, "PLA": 500.0},
            time_step=0, max_steps=35,
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
        # Track the step at which each job was completed (for deadline eval)
        self.completion_step: dict[str, int] = {}

    # ------------------------------------------------------------------
    #  Per-step bookkeeping
    # ------------------------------------------------------------------
    def step_update(self, action, action_handled: bool,
                    state: FarmObservation, time_step: int):
        # Only penalise WAIT when there is actionable work remaining
        has_actionable = any(
            j.state in (JobState.PENDING,)
            for j in state.active_queue
        )
        if action and action.action.value == "WAIT" and has_actionable:
            self.wasted_steps += 1
        if action and not action_handled and action.action.value != "WAIT":
            self.failed_actions += 1

        # Record the step at which jobs complete (first time only)
        for job in state.active_queue:
            if job.state == JobState.COMPLETED and job.job_id not in self.completion_step:
                self.completion_step[job.job_id] = time_step

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

    def _met_deadline(self, job) -> bool:
        """Check if a completed job met its deadline (using recorded step)."""
        if not job.deadline_steps:
            return True  # No deadline = always on time
        completed_at = self.completion_step.get(job.job_id)
        if completed_at is None:
            return False
        return completed_at <= job.deadline_steps

    # --- Task 1: Night Shift Scheduling ----------------------------------
    def _score_task1(self, state: FarmObservation, penalty: float) -> float:
        jobs = state.active_queue
        if not jobs:
            return 0.0

        # Hard fail: any failed or cancelled job
        if any(j.state in (JobState.FAILED, JobState.CANCELLED) for j in jobs):
            completed = sum(1 for j in jobs if j.state == JobState.COMPLETED)
            return _clamp(completed * 0.1 - penalty)

        total_weight = 0.0
        earned = 0.0
        for job in jobs:
            w = _priority_weight(job.priority)
            total_weight += w
            if job.state == JobState.COMPLETED:
                if self._met_deadline(job):
                    earned += w
                else:
                    earned += w * 0.6  # Late
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
                if self._met_deadline(job):
                    earned += w
                else:
                    earned += w * 0.5
            elif job.state == JobState.FAILED:
                if job.progress_steps > 0:
                    earned += w * 0.15
            elif job.state == JobState.PRINTING and job.print_time_steps > 0:
                earned += w * 0.5 * (job.progress_steps / job.print_time_steps)

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
                if self._met_deadline(job):
                    earned += w
                else:
                    earned += w * 0.4
            elif job.state == JobState.FAILED:
                if job.progress_steps > 0:
                    earned += w * 0.1
            elif job.state == JobState.PRINTING and job.print_time_steps > 0:
                earned += w * 0.4 * (job.progress_steps / job.print_time_steps)

        # Bonus for meeting all urgent deadlines
        urgent_jobs = [j for j in jobs if j.priority == 3 and j.state == JobState.COMPLETED]
        if urgent_jobs and all(self._met_deadline(j) for j in urgent_jobs):
            earned += 0.5

        score = earned / (total_weight + 0.5) if total_weight > 0 else 0.0
        return _clamp(score - penalty)


# =========================================================================
#  Helpers
# =========================================================================

def _priority_weight(priority: int) -> float:
    """Higher priority jobs count more toward the score."""
    return {1: 0.5, 2: 1.0, 3: 2.0}.get(priority, 1.0)


def _clamp(value: float) -> float:
    """Clamp score to the open interval (0, 1) — strictly between 0.0 and 1.0.

    The OpenEnv validator requires scores to be strictly in (0, 1),
    i.e. never exactly 0.0 or exactly 1.0.
    """
    EPSILON = 0.001
    return max(EPSILON, min(1.0 - EPSILON, value))
