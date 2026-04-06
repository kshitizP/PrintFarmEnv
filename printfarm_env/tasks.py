from .models import FarmObservation, PrintJob, PrinterObservation, PrinterState, JobState

def load_task(task_id: str) -> FarmObservation:
    """Initialize observation space based on task."""
    printers = []
    
    if task_id == "task_1":
        # Task 1: Night Shift (3 idle machines, 5 orders)
        for i in range(1, 11):
            printers.append(PrinterObservation(
                printer_id=i,
                state=PrinterState.IDLE,
                current_material="PLA",
                spool_weight_g=1000.0 if i <= 3 else 0.0 # First 3 have filament
            ))
            
        queue = [
            PrintJob(job_id=f"job_{j}", material_required="PLA", weight_required_g=100.0, print_time_steps=5)
            for j in range(1, 6)
        ]
        
        return FarmObservation(
            active_queue=queue,
            printers=printers,
            inventory={"PLA": 5000.0, "PETG": 1000.0}
        )
        
    elif task_id == "task_2":
        # Task 2: Spool Runout (800g print but 200g on spool)
        for i in range(1, 11):
            printers.append(PrinterObservation(
                printer_id=i,
                state=PrinterState.IDLE,
                current_material="PETG" if i == 1 else None,
                spool_weight_g=200.0 if i == 1 else 0.0
            ))
            
        queue = [
            PrintJob(job_id="job_heavy", material_required="PETG", weight_required_g=800.0, print_time_steps=20)
        ]
        
        return FarmObservation(
            active_queue=queue,
            printers=printers,
            inventory={"PETG": 2000.0}
        )
        
    elif task_id == "task_3":
        # Task 3: Hardware Triage
        for i in range(1, 11):
            printers.append(PrinterObservation(
                printer_id=i,
                state=PrinterState.IDLE,
                current_material="ABS" if i <= 2 else None,
                spool_weight_g=1000.0 if i <= 2 else 0.0
            ))
            
        queue = [
            PrintJob(job_id="job_triage", material_required="ABS", weight_required_g=500.0, print_time_steps=10)
        ]
        
        return FarmObservation(
            active_queue=queue,
            printers=printers,
            inventory={"ABS": 1000.0}
        )
    
    else:
        raise ValueError(f"Unknown task {task_id}")

class TaskGrader:
    def __init__(self, task_id: str):
        self.task_id = task_id

        # Tracking variables
        self.failed_actions = 0
        self.wasted_steps = 0
        self.task2_swapped_before_start = False
        self.task2_started = False
        self.task3_downtime = 0
        self.task3_completed = False

    def step_update(self, current_action, action_handled, state: FarmObservation, time_step: int):
        # Track wasted steps (WAIT or failed actions) across all tasks
        if current_action and current_action.action.value == "WAIT":
            self.wasted_steps += 1
        if current_action and not action_handled and current_action.action.value != "WAIT":
            self.failed_actions += 1

        if self.task_id == "task_2":
            if current_action and current_action.action.value == "SWAP_FILAMENT":
                if not self.task2_started:
                    self.task2_swapped_before_start = True
            if current_action and current_action.action.value == "ASSIGN_JOB":
                self.task2_started = True

        elif self.task_id == "task_3":
            for p in state.printers:
                if p.state == PrinterState.ERROR:
                    self.task3_downtime += 1

            for job in state.active_queue:
                if job.job_id == "job_triage" and job.state == JobState.COMPLETED:
                    self.task3_completed = True

    def get_score(self, state: FarmObservation) -> float:
        # Step penalty: discourage wasted actions (-0.01 per wasted step, -0.02 per failed action)
        step_penalty = (self.wasted_steps * 0.01) + (self.failed_actions * 0.02)

        if self.task_id == "task_1":
            # Penalize failures and cancellations
            if any(j.state == JobState.FAILED for j in state.active_queue):
                return 0.0
            if any(j.state == JobState.CANCELLED for j in state.active_queue):
                return 0.0

            total_jobs = len(state.active_queue)
            # Continuous progress: count completed jobs + fractional progress of in-flight jobs
            progress = 0.0
            for job in state.active_queue:
                if job.state == JobState.COMPLETED:
                    progress += 1.0
                elif job.state == JobState.PRINTING and job.print_time_steps > 0:
                    progress += 0.5 * (job.progress_steps / job.print_time_steps)

            score = progress / total_jobs
            return max(0.0, min(1.0, score - step_penalty))

        elif self.task_id == "task_2":
            heavy_job = next((j for j in state.active_queue if j.job_id == "job_heavy"), None)
            if not heavy_job:
                return 0.0

            if heavy_job.state == JobState.COMPLETED:
                # Bonus for swapping filament proactively before starting
                base = 0.9 if not self.task2_swapped_before_start else 1.0
                return max(0.0, min(1.0, base - step_penalty))

            if heavy_job.state == JobState.FAILED:
                if heavy_job.progress_steps > 0:
                    return 0.3  # partial credit: tried but ran out
                return 0.0

            # In-progress partial credit: reward progress toward completion
            if heavy_job.state == JobState.PRINTING and heavy_job.print_time_steps > 0:
                progress_frac = heavy_job.progress_steps / heavy_job.print_time_steps
                return max(0.0, min(0.7, 0.7 * progress_frac - step_penalty))

            # Job assigned but not yet printing? Small credit for having started
            if self.task2_swapped_before_start:
                return max(0.0, 0.1 - step_penalty)

            return 0.0

        elif self.task_id == "task_3":
            triage_job = next((j for j in state.active_queue if j.job_id == "job_triage"), None)

            if self.task3_completed:
                # Scale based on downtime. Max expected downtime is 10.
                efficiency = 1.0 - (self.task3_downtime / 10.0)
                score = max(0.2, min(1.0, efficiency))
                return max(0.0, score - step_penalty)

            # Partial credit during episode: reward progress even before completion
            if triage_job and triage_job.state == JobState.PRINTING and triage_job.print_time_steps > 0:
                progress_frac = triage_job.progress_steps / triage_job.print_time_steps
                return max(0.0, 0.5 * progress_frac - step_penalty)

            # Job re-queued after error: small credit for being assigned at all
            if triage_job and triage_job.state == JobState.PENDING and self.task3_downtime > 0:
                return max(0.0, 0.05 - step_penalty)

            return 0.0

        return 0.0
