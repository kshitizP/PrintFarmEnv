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
        class _BaseAction(BaseModel):
            model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class _BaseObservation(BaseModel):
            model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
            done: bool = Field(default=False)
            reward: float = Field(default=None)
            metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Actions
# ---------------------------------------------------------------------------

class FarmActionEnum(str, Enum):
    ASSIGN_JOB = "ASSIGN_JOB"
    SWAP_FILAMENT = "SWAP_FILAMENT"
    CANCEL_JOB = "CANCEL_JOB"
    PERFORM_MAINTENANCE = "PERFORM_MAINTENANCE"
    RESUME_JOB = "RESUME_JOB"
    WAIT = "WAIT"


class FarmAction(_BaseAction):
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

    action: FarmActionEnum
    printer_id: Optional[int] = None
    job_id: Optional[str] = None
    material: Optional[str] = None  # For SWAP_FILAMENT


# ---------------------------------------------------------------------------
#  Job model
# ---------------------------------------------------------------------------

class JobState(str, Enum):
    PENDING = "PENDING"
    PRINTING = "PRINTING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PrintJob(BaseModel):
    job_id: str
    material_required: str
    weight_required_g: float
    print_time_steps: int
    priority: int = Field(default=2, ge=1, le=3)  # 1=low, 2=normal, 3=urgent
    deadline_steps: Optional[int] = None           # Must complete by this step
    state: JobState = JobState.PENDING
    progress_steps: int = 0


# ---------------------------------------------------------------------------
#  Printer model
# ---------------------------------------------------------------------------

class PrinterState(str, Enum):
    IDLE = "IDLE"
    WARMING_UP = "WARMING_UP"
    PRINTING = "PRINTING"
    PAUSED_RUNOUT = "PAUSED_RUNOUT"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"


class PrinterObservation(BaseModel):
    printer_id: int
    state: PrinterState
    current_material: Optional[str] = None
    current_job_id: Optional[str] = None
    spool_weight_g: float = 0.0
    reliability: float = Field(default=0.95, ge=0.0, le=1.0)  # Per-step success rate while printing
    warmup_remaining: int = 0        # Steps left before printing begins
    maintenance_due_in: int = 50     # Steps until maintenance is needed
    fatigue_level: int = 0           # 0-10, catastrophic failure at 10
    offline_remaining: int = 0       # Steps remaining in OFFLINE state
    consecutive_idle_steps: int = 0  # Steps printer has been continuously IDLE


# ---------------------------------------------------------------------------
#  Observation (full farm state)
# ---------------------------------------------------------------------------

class FarmObservation(_BaseObservation):
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

    active_queue: List[PrintJob] = Field(default_factory=list)
    printers: List[PrinterObservation] = Field(default_factory=list)
    inventory: Dict[str, float] = Field(default_factory=dict)
    time_step: int = 0
    max_steps: int = 0
