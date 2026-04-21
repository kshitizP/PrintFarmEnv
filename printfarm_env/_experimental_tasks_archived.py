"""
Experimental task scenarios for difficulty testing.

Run with:  python test_experimental.py

These are candidates to replace or supplement task_1/2/3.
After benchmarking, pick the best 3 with clear difficulty progression.
"""

from .models import (
    FarmObservation, PrintJob, PrinterObservation, PrinterState, JobState,
)


def load_experimental_task(task_id: str) -> FarmObservation:
    """Build the initial observation for an experimental task."""

    if task_id == "exp_4":
        # -----------------------------------------------------------------
        # "Priority Inversion" — MEDIUM candidate
        #
        # Three PLA jobs, only ONE PLA printer ready. A second has ABS.
        #   job_bulk:    8-step, priority 3, deadline 14 (listed FIRST)
        #   job_express: 2-step, priority 3, deadline 5 (very tight)
        #   job_medium:  4-step, priority 2, deadline 15
        #
        # Trap: Greedy models assign job_bulk first (queue order, urgent).
        # 8 steps + 1 warmup = finishes step 9 → blocks printer.
        # job_express misses deadline (step 5) completely.
        # Then job_medium also suffers latency decay.
        #
        # Winning strategy: assign job_express FIRST (done step 3),
        # then job_medium (done step 8), then job_bulk (done step 17,
        # late but only 3 steps → 85% credit). Or swap P2 to PLA to
        # parallelize — but that costs 2 steps + 50g.
        #
        # Tests: deadline-aware scheduling vs greedy urgency
        # -----------------------------------------------------------------
        printers = []
        for i in range(1, 11):
            if i == 1:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PLA", spool_weight_g=1000.0,
                    reliability=0.99, maintenance_due_in=50,
                ))
            elif i == 2:
                # ABS printer — needs swap for PLA jobs
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="ABS", spool_weight_g=1000.0,
                    reliability=0.95, maintenance_due_in=50,
                ))
            else:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        # job_bulk listed FIRST to tempt greedy queue-order processing
        queue = [
            PrintJob(job_id="job_bulk", material_required="PLA",
                     weight_required_g=400.0, print_time_steps=8,
                     priority=3, deadline_steps=14),
            PrintJob(job_id="job_express", material_required="PLA",
                     weight_required_g=50.0, print_time_steps=2,
                     priority=3, deadline_steps=5),
            PrintJob(job_id="job_medium", material_required="PLA",
                     weight_required_g=150.0, print_time_steps=4,
                     priority=2, deadline_steps=15),
            PrintJob(job_id="job_low", material_required="ABS",
                     weight_required_g=200.0, print_time_steps=4,
                     priority=1),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"PLA": 2000.0, "ABS": 1000.0},
            time_step=0, max_steps=22,
        )

    elif task_id == "exp_5":
        # -----------------------------------------------------------------
        # "Cascade Failure" — HARD candidate
        #
        # 2 loaded printers with dangerous fatigue + 1 ready printer:
        #   P1: ABS, fatigue=7, 6-step urgent job → 7+6=13 ≥ 10 BOOM
        #   P2: PLA, fatigue=6, 7-step urgent job → 6+7=13 ≥ 10 BOOM
        #   P3: PETG, fatigue=0, safe for normal jobs
        #
        # BOTH P1 and P2 need maintenance before their jobs are safe.
        # Maintenance takes 3 steps each. Deadlines are tight but
        # reachable if the agent maintains correctly.
        #
        # Trap: models assign immediately (catastrophic failure) or
        # maintain one but forget the other.
        #
        # Winning strategy: Maintain P1 and P2 simultaneously (both go
        # into MAINTENANCE at step 0-1), assign PETG job to P3
        # immediately. After maintenance finishes (step 3), assign
        # urgent jobs. All deadlines are met.
        #
        # Tests: fleet-wide fatigue awareness, parallel maintenance
        # -----------------------------------------------------------------
        printers = []
        for i in range(1, 11):
            if i == 1:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="ABS", spool_weight_g=800.0,
                    reliability=0.95, maintenance_due_in=20,
                    fatigue_level=7,
                ))
            elif i == 2:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PLA", spool_weight_g=900.0,
                    reliability=0.90, maintenance_due_in=15,
                    fatigue_level=6,
                ))
            elif i == 3:
                # PETG ready — safe to use immediately
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PETG", spool_weight_g=800.0,
                    reliability=0.95, maintenance_due_in=50,
                    fatigue_level=0,
                ))
            else:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        queue = [
            PrintJob(job_id="job_abs_rush", material_required="ABS",
                     weight_required_g=300.0, print_time_steps=6,
                     priority=3, deadline_steps=18),
            PrintJob(job_id="job_pla_rush", material_required="PLA",
                     weight_required_g=350.0, print_time_steps=7,
                     priority=3, deadline_steps=20),
            PrintJob(job_id="job_petg_norm", material_required="PETG",
                     weight_required_g=200.0, print_time_steps=4,
                     priority=2, deadline_steps=16),
            PrintJob(job_id="job_filler_abs", material_required="ABS",
                     weight_required_g=150.0, print_time_steps=3,
                     priority=1),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"ABS": 1000.0, "PLA": 1000.0, "PETG": 1000.0},
            time_step=0, max_steps=28,
        )

    elif task_id == "exp_6":
        # -----------------------------------------------------------------
        # "Resource Scarcity" — HARD candidate
        #
        # 4 jobs but only enough inventory for 2.5 of them.
        # The agent must TRIAGE — decide which jobs to complete and which
        # to sacrifice.
        #
        # Inventory: PLA=1200g, ABS=500g (not enough for all)
        #   job_vip:    PLA, 600g, priority 3, deadline 12
        #   job_normal: PLA, 500g, priority 2, deadline 18
        #   job_small:  PLA, 200g, priority 2, no deadline
        #   job_abs:    ABS, 400g, priority 1, no deadline
        #
        # Total PLA needed: 1300g, available: 1200g (spool 1000g + swap=950g)
        # Can't complete all 3 PLA jobs.
        #
        # Trap: Models try to complete everything and run out mid-print on
        # the last job, wasting steps. Or they cancel the VIP to "save"
        # material for more jobs.
        #
        # Winning strategy: Complete job_vip first (600g, urgent), then
        # job_small (200g, quick), skip job_normal (500g, would exhaust PLA).
        # Assign job_abs to printer 2.
        #
        # Tests: resource planning, triage under scarcity
        # -----------------------------------------------------------------
        printers = []
        for i in range(1, 11):
            if i == 1:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PLA", spool_weight_g=950.0,
                    reliability=0.99, maintenance_due_in=50,
                ))
            elif i == 2:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="ABS", spool_weight_g=500.0,
                    reliability=0.95, maintenance_due_in=50,
                ))
            else:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        queue = [
            PrintJob(job_id="job_vip", material_required="PLA",
                     weight_required_g=600.0, print_time_steps=8,
                     priority=3, deadline_steps=12),
            PrintJob(job_id="job_normal", material_required="PLA",
                     weight_required_g=500.0, print_time_steps=7,
                     priority=2, deadline_steps=18),
            PrintJob(job_id="job_small", material_required="PLA",
                     weight_required_g=200.0, print_time_steps=3,
                     priority=2),
            PrintJob(job_id="job_abs", material_required="ABS",
                     weight_required_g=400.0, print_time_steps=5,
                     priority=1),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"PLA": 1200.0, "ABS": 500.0},
            time_step=0, max_steps=25,
        )

    elif task_id == "exp_7":
        # -----------------------------------------------------------------
        # "The Gauntlet" — EXPERT candidate
        #
        # Combines multiple traps in one scenario:
        #   1. Fatigue trap: P1 has fatigue=7, urgent 5-step ABS job
        #      → 7+5=12 ≥ 10 BOOM. Must maintain first (3 steps).
        #   2. Runout trap: P2 has PLA but only 150g for a 600g job.
        #      Will run out at ~step 2. Must swap+resume.
        #   3. Batching trap: 3 materials across 6 jobs, only 2 loaded
        #   4. Very tight deadlines — no room for wasted actions
        #   5. A decoy low-priority job tempts wasting a printer
        #
        # The agent must:
        #   - Maintain P1 before ABS job (3 steps overhead)
        #   - Start PLA job on P2 knowing runout is inevitable
        #   - Swap+resume PLA job (2 step warmup + resume)
        #   - Assign PETG job to P3 immediately
        #   - Sequence ABS jobs after maintenance without blowing deadline
        #   - Ignore or defer the decoy job
        #
        # Tests: multi-constraint reasoning under extreme time pressure
        # -----------------------------------------------------------------
        printers = []
        for i in range(1, 11):
            if i == 1:
                # ABS, high fatigue — must maintain first
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="ABS", spool_weight_g=800.0,
                    reliability=0.95, maintenance_due_in=30,
                    fatigue_level=7,
                ))
            elif i == 2:
                # PLA, very low spool — will run out fast
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PLA", spool_weight_g=150.0,
                    reliability=0.99, maintenance_due_in=50,
                ))
            elif i == 3:
                # PETG ready
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material="PETG", spool_weight_g=900.0,
                    reliability=0.95, maintenance_due_in=40,
                ))
            else:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        queue = [
            PrintJob(job_id="job_abs_crit", material_required="ABS",
                     weight_required_g=300.0, print_time_steps=5,
                     priority=3, deadline_steps=12),
            PrintJob(job_id="job_pla_big", material_required="PLA",
                     weight_required_g=600.0, print_time_steps=8,
                     priority=3, deadline_steps=16),
            PrintJob(job_id="job_petg_1", material_required="PETG",
                     weight_required_g=250.0, print_time_steps=4,
                     priority=2, deadline_steps=10),
            PrintJob(job_id="job_abs_2", material_required="ABS",
                     weight_required_g=200.0, print_time_steps=4,
                     priority=2, deadline_steps=18),
            PrintJob(job_id="job_pla_sm", material_required="PLA",
                     weight_required_g=100.0, print_time_steps=2,
                     priority=1),
            PrintJob(job_id="job_decoy", material_required="PETG",
                     weight_required_g=300.0, print_time_steps=6,
                     priority=1),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"ABS": 1000.0, "PLA": 2000.0, "PETG": 1000.0},
            time_step=0, max_steps=25,
        )

    elif task_id == "exp_8":
        # -----------------------------------------------------------------
        # "Error Recovery" — MEDIUM candidate
        #
        # Two printers start in ERROR state with jobs that were reset to
        # PENDING (progress lost). A third printer is in MAINTENANCE
        # (2 steps remaining).
        #
        # The agent must:
        #   - Recover errored printers via SWAP_FILAMENT or MAINTENANCE
        #   - Re-assign the reset jobs to working printers
        #   - Manage a new urgent job that arrives in the queue
        #
        # Trap: Models try ASSIGN_JOB on ERROR printers (invalid),
        # or WAIT for maintenance to finish instead of acting on other
        # printers. Some models don't realize ERROR printers need
        # intervention — they aren't self-healing.
        #
        # Tests: error state handling, recovery planning
        # -----------------------------------------------------------------
        printers = []
        for i in range(1, 11):
            if i == 1:
                # ERROR state — was printing ABS, failed
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.ERROR,
                    current_material="ABS", spool_weight_g=600.0,
                    reliability=0.80, maintenance_due_in=5,
                    fatigue_level=3,
                ))
            elif i == 2:
                # ERROR state — was printing PLA, failed
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.ERROR,
                    current_material="PLA", spool_weight_g=400.0,
                    reliability=0.85, maintenance_due_in=8,
                    fatigue_level=2,
                ))
            elif i == 3:
                # MAINTENANCE — almost done (2 steps left)
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.MAINTENANCE,
                    current_material="PETG", spool_weight_g=800.0,
                    reliability=0.90, maintenance_due_in=50,
                    warmup_remaining=2,
                ))
            else:
                printers.append(PrinterObservation(
                    printer_id=i, state=PrinterState.IDLE,
                    current_material=None, spool_weight_g=0.0,
                    reliability=0.95, maintenance_due_in=50,
                ))

        queue = [
            # Jobs that were being printed — progress reset to 0 by error
            PrintJob(job_id="job_retry_abs", material_required="ABS",
                     weight_required_g=350.0, print_time_steps=6,
                     priority=2, deadline_steps=18),
            PrintJob(job_id="job_retry_pla", material_required="PLA",
                     weight_required_g=300.0, print_time_steps=5,
                     priority=2, deadline_steps=16),
            # New urgent job
            PrintJob(job_id="job_urgent_petg", material_required="PETG",
                     weight_required_g=250.0, print_time_steps=4,
                     priority=3, deadline_steps=12),
        ]

        return FarmObservation(
            active_queue=queue, printers=printers,
            inventory={"ABS": 1500.0, "PLA": 1500.0, "PETG": 1000.0},
            time_step=0, max_steps=25,
        )

    else:
        raise ValueError(f"Unknown experimental task {task_id}")


# =========================================================================
#  Experimental Graders
# =========================================================================

def _priority_weight(priority: int) -> float:
    return {1: 0.5, 2: 1.0, 3: 2.0}.get(priority, 1.0)


def _clamp(value: float) -> float:
    EPSILON = 0.001
    return max(EPSILON, min(1.0 - EPSILON, value))


class ExperimentalGrader:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.wasted_steps = 0
        self.failed_actions = 0
        self.completion_step: dict[str, int] = {}

    def step_update(self, action, action_handled: bool,
                    state: FarmObservation, time_step: int):
        has_actionable = any(
            j.state in (JobState.PENDING, JobState.PAUSED)
            for j in state.active_queue
        )
        if action and action.action.value == "WAIT" and has_actionable:
            self.wasted_steps += 1
        if action and not action_handled and action.action.value != "WAIT":
            self.failed_actions += 1

        for job in state.active_queue:
            if job.state == JobState.COMPLETED and job.job_id not in self.completion_step:
                self.completion_step[job.job_id] = time_step

    def _late_multiplier(self, job, current_step: int = 0) -> float:
        if not job.deadline_steps:
            return 1.0
        if job.state == JobState.COMPLETED:
            completed_at = self.completion_step.get(job.job_id)
            if completed_at is None or completed_at <= job.deadline_steps:
                return 1.0
            steps_late = completed_at - job.deadline_steps
        else:
            if current_step <= job.deadline_steps:
                return 1.0
            steps_late = current_step - job.deadline_steps
        return max(0.1, 1.0 - 0.05 * steps_late)

    def get_score(self, state: FarmObservation) -> float:
        step_penalty = (self.wasted_steps * 0.01) + (self.failed_actions * 0.02)

        if self.task_id == "exp_4":
            return self._score_priority_inversion(state, step_penalty)
        elif self.task_id == "exp_5":
            return self._score_cascade_failure(state, step_penalty)
        elif self.task_id == "exp_6":
            return self._score_resource_scarcity(state, step_penalty)
        elif self.task_id == "exp_7":
            return self._score_gauntlet(state, step_penalty)
        elif self.task_id == "exp_8":
            return self._score_error_recovery(state, step_penalty)
        return 0.0

    # --- exp_4: Priority Inversion ----------------------------------------
    def _score_priority_inversion(self, state: FarmObservation,
                                  penalty: float) -> float:
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
                earned += 0.0
            elif job.state in (JobState.PRINTING, JobState.PAUSED) and job.print_time_steps > 0:
                progress = job.progress_steps / job.print_time_steps
                decay = self._late_multiplier(job, state.time_step)
                earned += w * 0.3 * progress * decay

        # Bonus: job_express completed on time (the key insight)
        express = next((j for j in jobs if j.job_id == "job_express"), None)
        if express and express.state == JobState.COMPLETED and self._late_multiplier(express) == 1.0:
            earned += 0.6

        score = earned / (total_weight + 0.6) if total_weight > 0 else 0.0
        return _clamp(score - penalty)

    # --- exp_5: Cascade Failure -------------------------------------------
    def _score_cascade_failure(self, state: FarmObservation,
                               penalty: float) -> float:
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
                # Partial credit: maintained the printer but job still failed
                # (better than nothing — gives RL gradient)
                earned += w * 0.05
            elif job.state in (JobState.PRINTING, JobState.PAUSED) and job.print_time_steps > 0:
                progress = job.progress_steps / job.print_time_steps
                decay = self._late_multiplier(job, state.time_step)
                earned += w * 0.4 * progress * decay

        # Bonus: both urgent jobs completed on time
        urgent = [j for j in jobs if j.priority == 3 and j.state == JobState.COMPLETED]
        if len(urgent) == 2 and all(self._late_multiplier(j) == 1.0 for j in urgent):
            earned += 0.6

        score = earned / (total_weight + 0.6) if total_weight > 0 else 0.0
        return _clamp(score - penalty)

    # --- exp_6: Resource Scarcity -----------------------------------------
    def _score_resource_scarcity(self, state: FarmObservation,
                                 penalty: float) -> float:
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
                if job.priority == 3:
                    earned -= w * 0.3  # Big penalty for failing VIP
            elif job.state in (JobState.PRINTING, JobState.PAUSED) and job.print_time_steps > 0:
                progress = job.progress_steps / job.print_time_steps
                decay = self._late_multiplier(job, state.time_step)
                earned += w * 0.4 * progress * decay

        # Partial credit for correct triage: completing VIP + small
        # without wasting resources on impossible completions
        vip = next((j for j in jobs if j.job_id == "job_vip"), None)
        if vip and vip.state == JobState.COMPLETED and self._late_multiplier(vip) == 1.0:
            earned += 0.3  # Triage bonus

        score = earned / (total_weight + 0.3) if total_weight > 0 else 0.0
        return _clamp(score - penalty)

    # --- exp_7: The Gauntlet ----------------------------------------------
    def _score_gauntlet(self, state: FarmObservation,
                        penalty: float) -> float:
        jobs = state.active_queue
        if not jobs:
            return 0.0

        failed_count = sum(1 for j in jobs if j.state == JobState.FAILED)

        total_weight = 0.0
        earned = 0.0
        for job in jobs:
            w = _priority_weight(job.priority)
            total_weight += w

            if job.state == JobState.COMPLETED:
                earned += w * self._late_multiplier(job)
            elif job.state == JobState.FAILED:
                earned += w * 0.05  # Minimal credit, gives gradient
            elif job.state in (JobState.PRINTING, JobState.PAUSED) and job.print_time_steps > 0:
                progress = job.progress_steps / job.print_time_steps
                decay = self._late_multiplier(job, state.time_step)
                earned += w * 0.3 * progress * decay

        # Big bonus: both urgent jobs on time (very hard to achieve)
        urgent = [j for j in jobs if j.priority == 3 and j.state == JobState.COMPLETED]
        if len(urgent) == 2 and all(self._late_multiplier(j) == 1.0 for j in urgent):
            earned += 1.0

        score = earned / (total_weight + 1.0) if total_weight > 0 else 0.0
        return _clamp(score - penalty)

    # --- exp_8: Error Recovery --------------------------------------------
    def _score_error_recovery(self, state: FarmObservation,
                              penalty: float) -> float:
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

        # Bonus for completing the urgent PETG job on time
        urgent_petg = next((j for j in jobs if j.job_id == "job_urgent_petg"), None)
        if urgent_petg and urgent_petg.state == JobState.COMPLETED:
            if self._late_multiplier(urgent_petg) == 1.0:
                earned += 0.4

        score = earned / (total_weight + 0.4) if total_weight > 0 else 0.0
        return _clamp(score - penalty)
