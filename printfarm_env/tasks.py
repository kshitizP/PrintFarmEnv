from .models import (
    FarmObservation, PrintJob, PrinterObservation, PrinterState, JobState,
)


def load_task(task_id: str) -> FarmObservation:
    """Build the initial observation for a given task."""

    if task_id == "task_1":
        # -----------------------------------------------------------------
        # EASY — "Traffic Jam" Bottleneck
        #
        # 5 jobs: 3 PLA (high priority), 2 PETG (one with tight deadline).
        # Only one printer exists, loaded with PLA.
        # (The rest of the farm is offline for network upgrades.)
        #
        # All print times are 1 step, so fatigue maxes at 5 (safe).
        # The 2-step SWAP_FILAMENT cost is brutally expensive relative
        # to the 1-step jobs, enforcing a genuine batching paradox.
        #
        # Path A (Batch PLA first):
        #   PLA(1)+PLA(1)+PLA(1)+Swap(2)+PETG(1)+PETG(1) = step 7
        #   Urgent PETG finishes at step 7 → meets deadline 12.
        #
        # Path B (Greedy — urgent PETG first):
        #   Swap(2)+PETG(1)+Swap(2)+PLA(1)+PLA(1)+PLA(1)+Swap(2)+PETG(1) = 11
        #   Meets deadline but burns 6 steps on swaps (3 changeovers).
        #
        # Optimal: batch all PLA first, then swap once to PETG.
        # -----------------------------------------------------------------
        printers = [
            PrinterObservation(
                printer_id=1, state=PrinterState.IDLE,
                current_material="PLA", spool_weight_g=1000.0,
                reliability=0.99, maintenance_due_in=50,
            ),
        ]

        queue = [
            PrintJob(job_id="job_1", material_required="PLA",
                     weight_required_g=80.0, print_time_steps=1,
                     priority=3, deadline_steps=16),
            PrintJob(job_id="job_2", material_required="PETG",
                     weight_required_g=80.0, print_time_steps=1,
                     priority=2, deadline_steps=16),
            PrintJob(job_id="job_3", material_required="PLA",
                     weight_required_g=80.0, print_time_steps=1,
                     priority=3, deadline_steps=16),
            PrintJob(job_id="job_4", material_required="PETG",
                     weight_required_g=80.0, print_time_steps=1,
                     priority=3, deadline_steps=12),
            PrintJob(job_id="job_5", material_required="PLA",
                     weight_required_g=80.0, print_time_steps=1,
                     priority=3, deadline_steps=16),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"PLA": 3000.0, "PETG": 3000.0},
            time_step=0, max_steps=25,
        )

    elif task_id == "task_2":
        # -----------------------------------------------------------------
        # MEDIUM — "Spool Runout" (The Sunk Cost Trap)
        #
        # An 800g urgent PLA job must be printed, but the only PLA printer
        # has just 300g on its spool.  The agent starts the print; at ~step 4
        # the spool runs out and the printer transitions to PAUSED_RUNOUT.
        #
        # The agent must: SWAP_FILAMENT (2-step changeover + 50g purge),
        # then RESUME_JOB to continue from where it left off.
        #
        # Trap: a frontier model may cancel the job (losing 300g of progress
        # and the job itself) instead of swapping + resuming.
        # -----------------------------------------------------------------
        printers = []
        for i in range(1, 11):
            if i == 1:
                # PLA loaded but spool too small for the big job
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PLA", spool_weight_g=300.0,
                    reliability=0.99, maintenance_due_in=50,
                ))
            elif i == 2:
                # ABS printer — can't use for the PLA job without a swap
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="ABS", spool_weight_g=1000.0,
                    reliability=0.95, maintenance_due_in=40,
                ))
            else:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        queue = [
            PrintJob(job_id="job_urgent", material_required="PLA",
                     weight_required_g=800.0, print_time_steps=10,
                     priority=3, deadline_steps=22),
            PrintJob(job_id="job_secondary", material_required="ABS",
                     weight_required_g=300.0, print_time_steps=6,
                     priority=2),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"PLA": 2000.0, "ABS": 1000.0},
            time_step=0, max_steps=30,
        )

    elif task_id == "task_3":
        # -----------------------------------------------------------------
        # HARD — "Thermal Cooldown" Trap
        #
        # Printer 1 has fatigue_level=7 and an urgent 5-step ABS job.
        # 7 + 5 = 12 ≥ 10 → catastrophic failure.
        #
        # The obvious fix is PERFORM_MAINTENANCE, but there's a catch:
        # printers with fatigue > 0 must cool down (remain IDLE for 3
        # consecutive steps) before maintenance can be performed.
        #
        # The agent must:
        #   1. WAIT 3 steps for Printer 1 to cool down
        #   2. PERFORM_MAINTENANCE (3 more steps)
        #   3. Then ASSIGN_JOB
        #   Total overhead: 6 steps before printing can start.
        #
        # Meanwhile, Printer 2 has PETG and can handle secondary jobs
        # immediately. The agent should assign side jobs while waiting.
        #
        # Trap: Models immediately try PERFORM_MAINTENANCE (rejected),
        # then thrash with invalid actions, wasting steps. Some may
        # try to assign the job directly (catastrophic failure).
        # -----------------------------------------------------------------
        printers = []
        for i in range(1, 11):
            if i == 1:
                # ABS loaded, high fatigue — must cool down first
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="ABS", spool_weight_g=1000.0,
                    reliability=0.99, maintenance_due_in=50,
                    fatigue_level=7,
                ))
            elif i == 2:
                # PETG printer for secondary jobs
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PETG", spool_weight_g=800.0,
                    reliability=0.95, maintenance_due_in=40,
                ))
            else:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        queue = [
            PrintJob(job_id="job_critical", material_required="ABS",
                     weight_required_g=300.0, print_time_steps=5,
                     priority=3, deadline_steps=18),
            PrintJob(job_id="job_petg", material_required="PETG",
                     weight_required_g=200.0, print_time_steps=5,
                     priority=2, deadline_steps=16),
            PrintJob(job_id="job_filler", material_required="ABS",
                     weight_required_g=200.0, print_time_steps=4,
                     priority=1),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"ABS": 1500.0, "PETG": 1000.0, "PLA": 500.0},
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
        # Track the step at which each job was completed (for deadline eval)
        self.completion_step: dict[str, int] = {}

    # ------------------------------------------------------------------
    #  Per-step bookkeeping
    # ------------------------------------------------------------------
    def step_update(self, action, action_handled: bool,
                    state: FarmObservation, time_step: int):
        # Penalise WAIT when there is actionable work remaining
        has_actionable = any(
            j.state in (JobState.PENDING, JobState.PAUSED)
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
    #  Continuous Latency Decay
    # ------------------------------------------------------------------
    def _late_multiplier(self, job, current_step: int = 0) -> float:
        """For every timestep a job is late, value drops 5%, min 10%."""
        if not job.deadline_steps:
            return 1.0  # No deadline → no decay

        if job.state == JobState.COMPLETED:
            completed_at = self.completion_step.get(job.job_id)
            if completed_at is None or completed_at <= job.deadline_steps:
                return 1.0
            steps_late = completed_at - job.deadline_steps
        else:
            # For in-progress / pending / paused jobs, use current time
            if current_step <= job.deadline_steps:
                return 1.0
            steps_late = current_step - job.deadline_steps

        return max(0.1, 1.0 - 0.05 * steps_late)

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

    # --- Task 1: Traffic Jam (Bottleneck) --------------------------------
    def _score_task1(self, state: FarmObservation, penalty: float) -> float:
        # Exclude dummy blocking jobs from scoring
        jobs = [j for j in state.active_queue
                if not j.job_id.startswith("dummy_")]
        if not jobs:
            return 0.0

        # Hard fail: any FAILED or CANCELLED job
        if any(j.state in (JobState.FAILED, JobState.CANCELLED) for j in jobs):
            completed = sum(1 for j in jobs if j.state == JobState.COMPLETED)
            return _clamp(completed * 0.1 - penalty)

        total_weight = 0.0
        earned = 0.0
        for job in jobs:
            w = _priority_weight(job.priority)
            total_weight += w

            if job.state == JobState.COMPLETED:
                earned += w * self._late_multiplier(job)
            elif job.state in (JobState.PRINTING, JobState.PAUSED) and job.print_time_steps > 0:
                progress = job.progress_steps / job.print_time_steps
                decay = self._late_multiplier(job, state.time_step)
                earned += w * 0.4 * progress * decay

        score = earned / total_weight if total_weight > 0 else 0.0
        return _clamp(score - penalty)

    # --- Task 2: Spool Runout (Sunk Cost Trap) ---------------------------
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
                earned += w * self._late_multiplier(job)
            elif job.state == JobState.FAILED:
                if job.progress_steps > 0:
                    earned += w * 0.15
            elif job.state in (JobState.PRINTING, JobState.PAUSED) and job.print_time_steps > 0:
                progress = job.progress_steps / job.print_time_steps
                decay = self._late_multiplier(job, state.time_step)
                earned += w * 0.5 * progress * decay

        score = earned / total_weight if total_weight > 0 else 0.0
        return _clamp(score - penalty)

    # --- Task 3: Thermal Cooldown (Fatigue + Cooldown Gate) --------------
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
                earned += w * self._late_multiplier(job)
            elif job.state == JobState.FAILED:
                if job.progress_steps > 0:
                    earned += w * 0.1
            elif job.state in (JobState.PRINTING, JobState.PAUSED) and job.print_time_steps > 0:
                progress = job.progress_steps / job.print_time_steps
                decay = self._late_multiplier(job, state.time_step)
                earned += w * 0.4 * progress * decay

        # Bonus for meeting all urgent deadlines
        urgent_jobs = [j for j in jobs if j.priority == 3 and j.state == JobState.COMPLETED]
        if urgent_jobs and all(self._late_multiplier(j) == 1.0 for j in urgent_jobs):
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
