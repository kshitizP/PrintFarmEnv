from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

# Try importing OpenEnv base classes; fall back to plain Pydantic if unavailable
try:
    from openenv.core import Action as _BaseAction, Observation as _BaseObservation
except ImportError:
    try:
        from openenv_core import Action as _BaseAction, Observation as _BaseObservation
    except ImportError:
        # Minimal stand-ins so the module works without openenv installed
        class _BaseAction(BaseModel):
            model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class _BaseObservation(BaseModel):
            model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
            done: bool = Field(default=False)
            reward: float = Field(default=None)
            metadata: Dict[str, Any] = Field(default_factory=dict)


class FarmActionEnum(str, Enum):
    ASSIGN_JOB = "ASSIGN_JOB"
    SWAP_FILAMENT = "SWAP_FILAMENT"
    CANCEL_JOB = "CANCEL_JOB"
    WAIT = "WAIT"

class FarmAction(_BaseAction):
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

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

class FarmObservation(_BaseObservation):
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

    active_queue: List[PrintJob] = Field(default_factory=list)
    printers: List[PrinterObservation] = Field(default_factory=list)
    inventory: Dict[str, float] = Field(default_factory=dict)
