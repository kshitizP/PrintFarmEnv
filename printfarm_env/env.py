from typing import Tuple, Dict, Any
from .models import FarmAction, FarmObservation, FarmActionEnum, JobState, PrinterState
from .tasks import load_task, TaskGrader
import copy

class BaseEnvironment:
    """Mock base class if openenv-core isn't installed."""
    pass

try:
    from openenv_core import Environment as BaseEnvironment
except ImportError:
    pass

class PrintFarmEnvironment(BaseEnvironment):
    def __init__(self):
        self.current_task_id = None
        self._state: FarmObservation = None
        self.time_step = 0
        self.max_steps = 30
        self.grader = None

    def reset(self, task_id: str) -> FarmObservation:
        self.current_task_id = task_id
        self._state = load_task(task_id)
        self.time_step = 0
        self.grader = TaskGrader(task_id)
        return self._state

    def state(self) -> FarmObservation:
        return self._state

    def _get_printer(self, pid: int):
        for p in self._state.printers:
            if p.printer_id == pid:
                return p
        return None

    def _get_job(self, jid: str):
        for j in self._state.active_queue:
            if j.job_id == jid:
                return j
        return None

    def render_dashboard(self):
        print(f"\n--- TIME STEP: {self.time_step} | TASK: {self.current_task_id} ---")
        for p in self._state.printers:
            icon = "⚪ IDLE    "
            if p.state == PrinterState.PRINTING:
                icon = "🟢 PRINTING"
            elif p.state == PrinterState.ERROR:
                icon = "🔴 ERROR   "
            mat_info = f"{p.current_material} ({p.spool_weight_g:.1f}g)" if p.current_material else "Empty"
            job_info = f"Job: {p.current_job_id}" if p.current_job_id else "No Job"
            print(f"[PRINTER {p.printer_id:02d}: {icon}] Material: {mat_info} | {job_info}")
        
        inventory_str = ", ".join([f"{k}: {v:.1f}g" for k, v in self._state.inventory.items()])
        print(f"Inventory: {inventory_str}")
        print("---------------------------------------------------------------")

    def step(self, action: FarmAction) -> Tuple[FarmObservation, float, bool, Dict[str, Any]]:
        action_handled = False
        info = {"error": None}

        if action and action.action == FarmActionEnum.ASSIGN_JOB:
            if action.printer_id and action.job_id:
                p = self._get_printer(action.printer_id)
                j = self._get_job(action.job_id)
                if p and j:
                    if p.state in [PrinterState.IDLE, PrinterState.ERROR] and j.state == JobState.PENDING:
                        if p.current_material == j.material_required:
                            if p.spool_weight_g > 0:
                                p.state = PrinterState.PRINTING
                                p.current_job_id = j.job_id
                                j.state = JobState.PRINTING
                                action_handled = True
                            else:
                                j.state = JobState.FAILED
                                info["error"] = f"Printer {p.printer_id} has no filament on spool."
                        else:
                            j.state = JobState.FAILED
                            info["error"] = f"Material mismatch! Printer has {p.current_material}."
                else:
                    info["error"] = "Invalid printer_id or job_id."
            else:
                info["error"] = "ASSIGN_JOB requires printer_id and job_id."

        elif action and action.action == FarmActionEnum.SWAP_FILAMENT:
            if action.printer_id and getattr(action, "material", None):
                p = self._get_printer(action.printer_id)
                if p and p.state in [PrinterState.IDLE, PrinterState.ERROR]:
                    # Return current to inventory
                    if p.current_material:
                        self._state.inventory[p.current_material] = self._state.inventory.get(p.current_material, 0) + p.spool_weight_g
                    
                    # Take from inventory (assume spools are 1000g)
                    if self._state.inventory.get(action.material, 0) >= 1000.0:
                        self._state.inventory[action.material] -= 1000.0
                        p.current_material = action.material
                        p.spool_weight_g = 1000.0
                        p.state = PrinterState.IDLE # Clearing error conditionally!
                        action_handled = True
                    else:
                        info["error"] = f"Insufficient {action.material} in inventory."
                else:
                    info["error"] = "Printer busy or missing."

        elif action and action.action == FarmActionEnum.CANCEL_JOB:
            if action.job_id:
                j = self._get_job(action.job_id)
                if j and j.state in [JobState.PENDING, JobState.PRINTING]:
                    if j.state == JobState.PRINTING:
                        for p in self._state.printers:
                            if p.current_job_id == j.job_id:
                                p.state = PrinterState.IDLE
                                p.current_job_id = None
                    j.state = JobState.CANCELLED
                    action_handled = True

        # Physics Tick
        for p in self._state.printers:
            if p.state == PrinterState.PRINTING:
                j = self._get_job(p.current_job_id)
                if j:
                    burn_rate = j.weight_required_g / j.print_time_steps
                    p.spool_weight_g -= burn_rate
                    j.progress_steps += 1
                    
                    if p.spool_weight_g <= 0:
                        p.state = PrinterState.ERROR
                        j.state = JobState.FAILED
                            
                            
                    elif self.current_task_id == "task_3" and j.progress_steps == 5:
                        p.state = PrinterState.ERROR
                        j.state = JobState.PENDING # Job can be restarted/re-routed
                        p.current_job_id = None
                    
                    elif j.progress_steps >= j.print_time_steps:
                        j.state = JobState.COMPLETED
                        p.state = PrinterState.IDLE
                        p.current_job_id = None

        self.time_step += 1
        
        # Grading updates
        if self.grader:
            self.grader.step_update(action, action_handled, self._state, self.time_step)
            current_score = self.grader.get_score(self._state)
        else:
            current_score = 0.0

        done = self.time_step >= self.max_steps
        
        all_resolved = all(j.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED] 
                           for j in self._state.active_queue)
        if all_resolved:
            done = True

        return (self._state, current_score, done, info)
