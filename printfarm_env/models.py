"""
PrintFarmEnv — Round 2 Data Models

All Pydantic models, enums, and dataclasses used across env, operators,
failures, and economics modules.
"""

from dataclasses import dataclass, field
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
            reward: float = Field(default=0.0)
            metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Actions (Round 2 full action set)
# ---------------------------------------------------------------------------

class FarmActionEnum(str, Enum):
    # Job management
    ASSIGN_JOB          = "ASSIGN_JOB"          # Maps to OctoPrint POST /api/job
    CANCEL_JOB          = "CANCEL_JOB"          # Maps to OctoPrint cancel
    PAUSE_JOB           = "PAUSE_JOB"           # Maps to OctoPrint pause
    RESUME_JOB          = "RESUME_JOB"          # Maps to OctoPrint resume
    # Diagnostic
    RUN_DIAGNOSTIC      = "RUN_DIAGNOSTIC"       # Maps to Moonraker query_endstops / temperature_store
    # Operator orchestration
    DISPATCH_TICKET     = "DISPATCH_TICKET"      # Create a work order for a human operator
    REQUEST_SPOOL_SWAP  = "REQUEST_SPOOL_SWAP"   # Sugar: DISPATCH_TICKET(spool_swap) with auto-routing
    REQUEST_MAINTENANCE = "REQUEST_MAINTENANCE"  # Sugar: DISPATCH_TICKET(maintenance_*) with auto-routing
    OVERRIDE_OPERATOR   = "OVERRIDE_OPERATOR"    # Cancel a queued (not in-progress) ticket
    # No-op
    WAIT                = "WAIT"


class FarmAction(_BaseAction):
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

    action: FarmActionEnum

    # Job & printer targeting
    printer_id: Optional[int]  = None
    job_id:     Optional[str]  = None

    # Ticket dispatch fields
    operator_id:   Optional[str] = None
    ticket_type:   Optional[str] = None   # spool_swap | filament_reload_from_stock |
                                           # maintenance_basic | maintenance_full_rebuild |
                                           # diagnostic_physical | unjam_printer
    ticket_id:     Optional[str] = None   # For OVERRIDE_OPERATOR

    # Payload passed through to the operator NPC
    material:          Optional[str] = None   # For spool_swap / REQUEST_SPOOL_SWAP
    maintenance_type:  Optional[str] = None   # For REQUEST_MAINTENANCE
    reason:            Optional[str] = None   # For OVERRIDE_OPERATOR narrative (Oversight log)


# ---------------------------------------------------------------------------
#  Job model
# ---------------------------------------------------------------------------

class JobState(str, Enum):
    PENDING   = "PENDING"
    PRINTING  = "PRINTING"
    PAUSED    = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    CANCELLED = "CANCELLED"


class PrintJob(BaseModel):
    job_id:           str
    material_required: str
    weight_required_g: float
    print_time_steps:  int
    priority:          int           = Field(default=2, ge=1, le=3)  # 1=low, 2=normal, 3=urgent
    deadline_steps:    Optional[int] = None
    price_usd:         float         = Field(default=30.0)
    state:             JobState      = JobState.PENDING
    progress_steps:    int           = 0
    # Tracked by the env to support SLA cap logic
    sla_fixed_applied:  bool  = False
    total_sla_penalty:  float = 0.0
    accrued_revenue:    float = 0.0   # Revenue accrued so far; clawed back on fail/cancel


# ---------------------------------------------------------------------------
#  Printer state machine
# ---------------------------------------------------------------------------

class PrinterState(str, Enum):
    IDLE               = "IDLE"
    WARMING_UP         = "WARMING_UP"
    PRINTING           = "PRINTING"
    PAUSED             = "PAUSED"          # Agent-induced pause (nozzle hot)
    PAUSED_RUNOUT      = "PAUSED_RUNOUT"   # Env-induced pause (spool empty, nozzle cooling)
    ERROR              = "ERROR"
    MAINTENANCE_QUEUED = "MAINTENANCE_QUEUED"
    MAINTENANCE        = "MAINTENANCE"
    OFFLINE            = "OFFLINE"


# ---------------------------------------------------------------------------
#  Ticket (work order for a human operator)
# ---------------------------------------------------------------------------

class TicketState(str, Enum):
    PENDING     = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED   = "COMPLETED"
    FAILED      = "FAILED"
    REJECTED    = "REJECTED"


@dataclass
class Ticket:
    ticket_id:        str
    ticket_type:      str          # spool_swap | maintenance_basic | etc.
    target_printer_id: int
    operator_id:      str          # Who it's currently assigned to
    created_step:     int
    payload:          Dict[str, Any] = field(default_factory=dict)
    state:            TicketState    = TicketState.PENDING
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
#  Operator observation (what the Dispatcher can see about each operator)
# ---------------------------------------------------------------------------

class OperatorObservation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    operator_id:         str
    skill_level:         str
    shift_window:        List[int]               # [start_step, end_step]
    queue_capacity:      int
    current_fatigue:     float
    is_on_shift:         bool
    queue_size:          int
    current_ticket_id:   Optional[str] = None
    busy_until:          int           = 0
    # Summary of pattern-based recommendations from printer_memory
    printer_visit_counts:    Dict[int, int]  = Field(default_factory=dict)
    pattern_recommendations: List[str]       = Field(default_factory=list)


# ---------------------------------------------------------------------------
#  Printer observation (what the Dispatcher sees — may be corrupted)
# ---------------------------------------------------------------------------

class PrinterObservation(BaseModel):
    printer_id:           int
    profile_id:           str            = "unknown"
    state:                PrinterState
    current_material:     Optional[str]  = None
    current_job_id:       Optional[str]  = None

    # Filament / spool
    spool_weight_g:       float          = 0.0

    # Reliability & wear
    reliability:          float          = Field(default=0.95, ge=0.0, le=1.0)
    maintenance_due_in:   int            = 50
    fatigue_level:        int            = 0   # 0–10; catastrophic at 10

    # Countdown timers
    warmup_remaining:     int            = 0
    offline_remaining:    int            = 0
    consecutive_idle_steps: int          = 0

    # --- Telemetry fields (may be corrupted by failure modes) ---
    hotend_temp:          float          = 200.0
    fan_rpm:              int            = 3000
    webcam_hash:          str            = ""
    telemetry_ts:         int            = 0    # Copy of time_step; stale when MCU disconnects

    # Set True only on the step where RUN_DIAGNOSTIC was called
    # (Dispatcher-facing signal that telemetry is ground truth this step)
    revealed_this_step:   bool           = False

    # Bed drift counter (deterministic trigger; never directly exposed)
    bed_drift_counter:    float          = 0.0

    # Reliability penalty active on first PRINTING tick after PAUSED_RUNOUT resume
    reliability_penalty_active: bool     = False

    # Outstanding maintenance ticket (set when MAINTENANCE_QUEUED)
    outstanding_ticket_id: Optional[str] = None


# ---------------------------------------------------------------------------
#  Full farm observation returned from step()
# ---------------------------------------------------------------------------

class FarmObservation(_BaseObservation):
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

    # Core state
    active_queue: List[PrintJob]            = Field(default_factory=list)
    printers:     List[PrinterObservation]  = Field(default_factory=list)
    operators:    List[OperatorObservation] = Field(default_factory=list)
    inventory:    Dict[str, float]          = Field(default_factory=dict)
    time_step:    int                       = 0
    max_steps:    int                       = 0

    # Economic summary (cumulative, updated each step)
    net_profit_usd:      float = 0.0
    total_labor_billed:  float = 0.0

    # Ticket event log: one entry per ticket completion event this step
    ticket_events: List[Dict[str, Any]] = Field(default_factory=list)

    # Oversight audit log: one line per step (ground-truth vs corrupted view)
    oversight_log: List[Dict[str, Any]] = Field(default_factory=list)

    # Step-level reward breakdown (for interpretability)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
