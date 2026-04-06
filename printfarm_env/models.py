from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel

class FarmActionEnum(str, Enum):
    ASSIGN_JOB = "ASSIGN_JOB"
    SWAP_FILAMENT = "SWAP_FILAMENT"
    CANCEL_JOB = "CANCEL_JOB"
    WAIT = "WAIT"

class FarmAction(BaseModel):
    action: FarmActionEnum
    printer_id: Optional[int] = None
    job_id: Optional[str] = None
    
    # Optional field for SWAP_FILAMENT
    material: Optional[str] = None

class JobState(str, Enum):
    PENDING = "PENDING"
    PRINTING = "PRINTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class PrintJob(BaseModel):
    job_id: str
    material_required: str
    weight_required_g: float
    state: JobState = JobState.PENDING
    print_time_steps: int
    progress_steps: int = 0

class PrinterState(str, Enum):
    IDLE = "IDLE"
    PRINTING = "PRINTING"
    ERROR = "ERROR"

class PrinterObservation(BaseModel):
    printer_id: int
    state: PrinterState
    current_material: Optional[str] = None
    current_job_id: Optional[str] = None
    spool_weight_g: float = 0.0

class FarmObservation(BaseModel):
    active_queue: List[PrintJob]
    printers: List[PrinterObservation]
    inventory: Dict[str, float]
