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
        self.task1_collision = False
        self.task2_swapped_before_start = False
        self.task2_started = False
        self.task2_failed_mid = False
        self.task3_downtime = 0
        self.task3_completed = False
        self.initial_time = 0

    def step_update(self, current_action, action_handled, state: FarmObservation, time_step: int):
        if self.task_id == "task_1":
            # Check for material collision
            if current_action and current_action.action.value == "ASSIGN_JOB" and not action_handled:
                # Need to be more precise: the collision should check whether a job was assigned incorrectly.
                # If the action was rejected but maybe because of a mismatch? We'll capture that in physics.
                pass
                
        elif self.task_id == "task_2":
            if current_action and current_action.action.value == "SWAP_FILAMENT":
                if not self.task2_started:
                    self.task2_swapped_before_start = True
            if current_action and current_action.action.value == "ASSIGN_JOB":
                self.task2_started = True

        elif self.task_id == "task_3":
            # Track downtime if printer 1 is in ERROR
            for p in state.printers:
                if p.state == PrinterState.ERROR:
                    self.task3_downtime += 1
            
            # Check completion
            for job in state.active_queue:
                if job.job_id == "job_triage" and job.state == JobState.COMPLETED:
                    self.task3_completed = True

    def get_score(self, state: FarmObservation) -> float:
        if self.task_id == "task_1":
            all_completed = all(job.state == JobState.COMPLETED for job in state.active_queue)
            
            # Also we shouldn't have collisions.
            collision = any(job.state == JobState.FAILED for job in state.active_queue)
            if collision:
                return 0.0
                
            # If agent just cancelled everything to "complete" the task
            cancelled = sum(1 for job in state.active_queue if job.state == JobState.CANCELLED)
            if cancelled > 0:
                return 0.0
                
            if all_completed:
                return 1.0
            
            completed_jobs = sum(1 for job in state.active_queue if job.state == JobState.COMPLETED)
            return completed_jobs / len(state.active_queue)
            
        elif self.task_id == "task_2":
            # Cancelled jobs are failures.
            heavy_job = next((j for j in state.active_queue if j.job_id == "job_heavy"), None)
            if heavy_job and heavy_job.state == JobState.COMPLETED:
                return 1.0
            
            if self.task2_failed_mid:
                return 0.5
            elif heavy_job and heavy_job.state == JobState.FAILED:
                return 0.5
                
            return 0.0
            
        elif self.task_id == "task_3":
            if not self.task3_completed:
                return 0.0
                
            # Scale based on downtime. Max expected downtime is 10.
            score = 1.0 - (self.task3_downtime / 10.0)
            return max(0.2, min(1.0, score)) # Floor at 0.2 if completed
            
        return 0.0
