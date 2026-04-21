"""
PrintFarmEnv — Round 2 Task Definitions and Graders

Tasks 0.x  — Round 1 warmup scenarios (regression/smoke tests)
Tasks 1–3  — Blocking Round 2 tasks (used for SFT/DPO training data)
Tasks 4–5  — Stretch tasks (task_5 held out from training data)

All tasks use deterministic seeds via seed = hash(task_id).
"""

from typing import Any, Dict, List, Optional

from .models import (
    FarmObservation, FarmAction, FarmActionEnum,
    PrintJob, PrinterObservation, PrinterState, JobState,
)


# ---------------------------------------------------------------------------
#  TaskConfig — typed representation of a task scenario
# ---------------------------------------------------------------------------

class TaskConfig:
    """Carries all parameters needed to initialise a task episode."""

    def __init__(
        self,
        task_id: str,
        name: str,
        episode_length_steps: int,
        printer_count: int,
        printer_profiles: List[str],
        printer_overrides: Optional[List[Dict[str, Any]]] = None,
        operator_roster: Optional[List[Dict[str, Any]]] = None,
        job_queue: Optional[List[Dict[str, Any]]] = None,
        initial_inventory_g: Optional[Dict[str, float]] = None,
        scheduled_failures: Optional[List[Dict[str, Any]]] = None,
        scheduled_operator_reports: Optional[List[Dict[str, Any]]] = None,
        stochastic_failure_rates: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[Dict[str, Any]] = None,
    ):
        self.task_id                  = task_id
        self.name                     = name
        self.episode_length_steps     = episode_length_steps
        self.printer_count            = printer_count
        self.printer_profiles         = printer_profiles
        self.printer_overrides        = printer_overrides or []
        self.operator_roster          = operator_roster or []
        self.job_queue                = job_queue or []
        self.initial_inventory_g      = initial_inventory_g or {}
        self.scheduled_failures       = scheduled_failures or []
        self.scheduled_operator_reports = scheduled_operator_reports or []
        self.stochastic_failure_rates = stochastic_failure_rates or {}
        self.success_criteria         = success_criteria or {}


# ---------------------------------------------------------------------------
#  Task definitions
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, TaskConfig] = {}


def _register(cfg: TaskConfig) -> TaskConfig:
    TASK_CONFIGS[cfg.task_id] = cfg
    return cfg


# --- Task 0.1: Traffic Jam (Round 1 warmup) ---------------------------------
_register(TaskConfig(
    task_id="task_0_1",
    name="Traffic Jam",
    episode_length_steps=25,
    printer_count=1,
    printer_profiles=["bambu_x1c"],
    printer_overrides=[
        {"printer_id": 1, "current_material": "PLA", "spool_weight_g": 1000.0,
         "reliability": 0.99, "maintenance_due_in": 50},
    ],
    operator_roster=[
        {"operator_id": "op_j1", "skill_level": "junior", "shift_window": [0, 25]},
    ],
    job_queue=[
        {"job_id": "j1", "material": "PLA",  "weight_g": 80.0, "print_time_steps": 1, "priority": 3, "deadline_steps": 16, "price_usd": 25},
        {"job_id": "j2", "material": "PETG", "weight_g": 80.0, "print_time_steps": 1, "priority": 2, "deadline_steps": 16, "price_usd": 20},
        {"job_id": "j3", "material": "PLA",  "weight_g": 80.0, "print_time_steps": 1, "priority": 3, "deadline_steps": 16, "price_usd": 25},
        {"job_id": "j4", "material": "PETG", "weight_g": 80.0, "print_time_steps": 1, "priority": 3, "deadline_steps": 12, "price_usd": 30},
        {"job_id": "j5", "material": "PLA",  "weight_g": 80.0, "print_time_steps": 1, "priority": 3, "deadline_steps": 16, "price_usd": 25},
    ],
    initial_inventory_g={"PLA": 3000.0, "PETG": 3000.0},
    scheduled_failures=[],
    stochastic_failure_rates={"_all": "zero"},
    success_criteria={"min_net_profit_usd": 50},
))


# --- Task 0.2: Spool Runout (Round 1 warmup) --------------------------------
_register(TaskConfig(
    task_id="task_0_2",
    name="Spool Runout",
    episode_length_steps=30,
    printer_count=2,
    printer_profiles=["bambu_x1c", "prusa_mk4"],
    printer_overrides=[
        {"printer_id": 1, "current_material": "PLA", "spool_weight_g": 300.0,
         "reliability": 0.99, "maintenance_due_in": 50},
        {"printer_id": 2, "current_material": "ABS", "spool_weight_g": 1000.0,
         "reliability": 0.95, "maintenance_due_in": 40},
    ],
    operator_roster=[
        {"operator_id": "op_j1", "skill_level": "junior", "shift_window": [0, 30]},
    ],
    job_queue=[
        {"job_id": "job_urgent", "material": "PLA",  "weight_g": 800.0, "print_time_steps": 10, "priority": 3, "deadline_steps": 22, "price_usd": 50},
        {"job_id": "job_secondary", "material": "ABS", "weight_g": 300.0, "print_time_steps": 6,  "priority": 2, "deadline_steps": None, "price_usd": 20},
    ],
    initial_inventory_g={"PLA": 2000.0, "ABS": 1000.0},
    scheduled_failures=[],
    stochastic_failure_rates={"_all": "zero"},
    success_criteria={"min_net_profit_usd": 0},
))


# --- Task 0.3: Thermal Cooldown (Round 1 warmup) ----------------------------
_register(TaskConfig(
    task_id="task_0_3",
    name="Thermal Cooldown",
    episode_length_steps=30,
    printer_count=2,
    printer_profiles=["voron_24", "prusa_mk4"],
    printer_overrides=[
        {"printer_id": 1, "current_material": "ABS", "spool_weight_g": 1000.0,
         "reliability": 0.99, "maintenance_due_in": 50, "fatigue_level": 7},
        {"printer_id": 2, "current_material": "PETG", "spool_weight_g": 800.0,
         "reliability": 0.95, "maintenance_due_in": 40},
    ],
    operator_roster=[
        {"operator_id": "op_j1", "skill_level": "junior",  "shift_window": [0, 30]},
        {"operator_id": "op_s1", "skill_level": "senior",  "shift_window": [0, 30]},
    ],
    job_queue=[
        {"job_id": "job_critical", "material": "ABS",  "weight_g": 300.0, "print_time_steps": 5, "priority": 3, "deadline_steps": 18, "price_usd": 60},
        {"job_id": "job_petg",     "material": "PETG", "weight_g": 200.0, "print_time_steps": 5, "priority": 2, "deadline_steps": 16, "price_usd": 35},
        {"job_id": "job_filler",   "material": "ABS",  "weight_g": 200.0, "print_time_steps": 4, "priority": 1, "deadline_steps": None, "price_usd": 18},
    ],
    initial_inventory_g={"ABS": 1500.0, "PETG": 1000.0, "PLA": 500.0},
    scheduled_failures=[],
    stochastic_failure_rates={"_all": "zero"},
    success_criteria={"min_net_profit_usd": 0, "max_catastrophic_failures": 0},
))


# --- Task 1: Human Latency Coordination -------------------------------------
# Theme: Multi-Agent (Theme 1). Blocking task.
# 8 printers, 3 operators (2 juniors + 1 senior), 6 jobs, 60 steps.
# 2 printers start with wrong material; forces swap-ticket pressure.
_register(TaskConfig(
    task_id="task_1",
    name="Human Latency Coordination",
    episode_length_steps=60,
    printer_count=8,
    printer_profiles=[
        "bambu_x1c", "bambu_x1c",
        "prusa_mk4", "prusa_mk4",
        "creality_k1", "creality_k1",
        "voron_24", "voron_24",
    ],
    printer_overrides=[
        # Printers 5 & 7 have wrong materials to force swap-ticket decisions
        {"printer_id": 5, "current_material": "ABS",  "spool_weight_g": 800.0},
        {"printer_id": 7, "current_material": "PETG", "spool_weight_g": 600.0},
    ],
    operator_roster=[
        {"operator_id": "op_j1", "skill_level": "junior", "shift_window": [0, 60]},
        {"operator_id": "op_j2", "skill_level": "junior", "shift_window": [0, 60]},
        {"operator_id": "op_s1", "skill_level": "senior", "shift_window": [0, 60]},
    ],
    job_queue=[
        {"job_id": "j1", "material": "PLA",  "weight_g": 150, "print_time_steps": 12, "priority": 2, "deadline_steps": 25, "price_usd": 35},
        {"job_id": "j2", "material": "PLA",  "weight_g": 200, "print_time_steps": 15, "priority": 2, "deadline_steps": 35, "price_usd": 40},
        {"job_id": "j3", "material": "PETG", "weight_g": 180, "print_time_steps": 14, "priority": 3, "deadline_steps": 30, "price_usd": 75},
        {"job_id": "j4", "material": "ABS",  "weight_g": 250, "print_time_steps": 18, "priority": 2, "deadline_steps": 50, "price_usd": 55},
        {"job_id": "j5", "material": "PLA",  "weight_g": 120, "print_time_steps": 10, "priority": 1, "deadline_steps": None, "price_usd": 20},
        {"job_id": "j6", "material": "PETG", "weight_g": 160, "print_time_steps": 13, "priority": 3, "deadline_steps": 55, "price_usd": 65},
    ],
    initial_inventory_g={"PLA": 1500.0, "PETG": 1000.0, "ABS": 1000.0},
    scheduled_failures=[],
    stochastic_failure_rates={"_all": "profile_default"},
    success_criteria={
        "min_net_profit_usd": 150,
        "max_catastrophic_failures": 0,
        "max_operator_queue_overflow_steps": 0,
    },
))


# --- Task 2: Sensor Trust Calibration ---------------------------------------
# Theme: Zero-Trust (Theme 5). Blocking task.
# 5 printers, 4 jobs, 60 steps. 3 scheduled sensor failures.
# All stochastic rates zeroed — only scheduled faults fire.
_register(TaskConfig(
    task_id="task_2",
    name="Sensor Trust Calibration",
    episode_length_steps=60,
    printer_count=5,
    printer_profiles=["bambu_x1c", "prusa_mk4", "prusa_mk4", "creality_k1", "voron_24"],
    printer_overrides=[],
    operator_roster=[
        {"operator_id": "op_j1", "skill_level": "junior", "shift_window": [0, 60]},
        {"operator_id": "op_s1", "skill_level": "senior", "shift_window": [0, 60]},
    ],
    job_queue=[
        {"job_id": "j1", "material": "PLA",  "weight_g": 180, "print_time_steps": 14, "priority": 2, "deadline_steps": 30, "price_usd": 40},
        {"job_id": "j2", "material": "PETG", "weight_g": 220, "print_time_steps": 18, "priority": 3, "deadline_steps": 40, "price_usd": 70},
        {"job_id": "j3", "material": "PLA",  "weight_g": 150, "print_time_steps": 12, "priority": 2, "deadline_steps": 45, "price_usd": 35},
        {"job_id": "j4", "material": "ABS",  "weight_g": 250, "print_time_steps": 20, "priority": 2, "deadline_steps": 55, "price_usd": 55},
    ],
    initial_inventory_g={"PLA": 2000.0, "PETG": 1000.0, "ABS": 1000.0},
    scheduled_failures=[
        {"printer_id": 1, "mode": "thermistor_open",              "trigger_step": 8,  "duration": 10},
        {"printer_id": 2, "mode": "filament_sensor_false_runout", "trigger_step": 18, "duration": 5},
        {"printer_id": 3, "mode": "webcam_freeze",                "trigger_step": 25, "duration": 12},
    ],
    scheduled_operator_reports=[],
    stochastic_failure_rates={"_all": "zero"},
    success_criteria={
        "min_net_profit_usd": 100,
        "max_catastrophic_failures": 0,
        "max_unnecessary_diagnostic_cost_usd": 2.00,  # ≤ 4 unnecessary RUN_DIAGNOSTICs
    },
))


# --- Task 3: Disagreement Resolution ----------------------------------------
# Theme: Multi-Agent + Zero-Trust. Blocking task.
# Printer 1: telemetry lies (thermistor_open), operator knows it's fine.
# Printer 3: real progress_drift, operator detects it; telemetry looks normal.
_register(TaskConfig(
    task_id="task_3",
    name="Disagreement Resolution",
    episode_length_steps=60,
    printer_count=4,
    printer_profiles=["bambu_x1c", "prusa_mk4", "creality_k1", "voron_24"],
    printer_overrides=[],
    operator_roster=[
        {"operator_id": "op_j1", "skill_level": "junior", "shift_window": [0, 60]},
        {"operator_id": "op_s1", "skill_level": "senior", "shift_window": [0, 60]},
    ],
    job_queue=[
        {"job_id": "j1", "material": "PLA",  "weight_g": 180, "print_time_steps": 14, "priority": 2, "deadline_steps": 30, "price_usd": 40},
        {"job_id": "j2", "material": "PETG", "weight_g": 250, "print_time_steps": 20, "priority": 3, "deadline_steps": 42, "price_usd": 80},
        {"job_id": "j3", "material": "ABS",  "weight_g": 200, "print_time_steps": 16, "priority": 2, "deadline_steps": 50, "price_usd": 50},
    ],
    initial_inventory_g={"PLA": 1500.0, "PETG": 1000.0, "ABS": 1000.0},
    scheduled_failures=[
        # Printer 1: telemetry lies (thermistor_open); operator is right
        {"printer_id": 1, "mode": "thermistor_open",  "trigger_step": 6,  "duration": 20},
        # Printer 3: real slowdown; sensor looks fine but operator notices
        {"printer_id": 3, "mode": "progress_drift",   "trigger_step": 20, "duration": 15},
    ],
    scheduled_operator_reports=[
        # These are injected as synthetic REPORT_ANOMALY events at specified steps
        {"step": 10, "operator_id": "op_j1", "printer_id": 1, "report": "printing fine"},
        {"step": 24, "operator_id": "op_s1", "printer_id": 3, "report": "seems slow"},
    ],
    stochastic_failure_rates={"_all": "zero"},
    success_criteria={
        "min_net_profit_usd": 80,
        "max_catastrophic_failures": 0,
    },
))


# --- Task 4: Long-Horizon Maintenance Planning (stretch) --------------------
# Theme: Long-Horizon (Theme 2). 90-step episode.
# Two printers near maintenance threshold at start; a third develops
# bed_level_drift mid-episode. Dispatcher must schedule maintenance
# at natural idle windows without stalling the job pipeline.
_register(TaskConfig(
    task_id="task_4",
    name="Long-Horizon Maintenance Planning",
    episode_length_steps=90,
    printer_count=6,
    printer_profiles=[
        "bambu_x1c", "bambu_x1c",
        "prusa_mk4", "prusa_mk4",
        "creality_k1", "voron_24",
    ],
    printer_overrides=[
        {"printer_id": 1, "maintenance_due_in": 12},
        {"printer_id": 4, "maintenance_due_in": 18},
        # Printers 2, 3, 5, 6 default to maintenance_due_in=50
    ],
    operator_roster=[
        {"operator_id": "op_j1", "skill_level": "junior", "shift_window": [0, 90]},
        {"operator_id": "op_j2", "skill_level": "junior", "shift_window": [0, 90]},
        {"operator_id": "op_s1", "skill_level": "senior", "shift_window": [0, 90]},
        {"operator_id": "op_l1", "skill_level": "lead",   "shift_window": [0, 90]},
    ],
    job_queue=[
        {"job_id": "j1", "material": "PLA",  "weight_g": 200, "print_time_steps": 16, "priority": 2, "deadline_steps": 25,  "price_usd": 40},
        {"job_id": "j2", "material": "PETG", "weight_g": 300, "print_time_steps": 24, "priority": 3, "deadline_steps": 55,  "price_usd": 85},
        {"job_id": "j3", "material": "PLA",  "weight_g": 220, "print_time_steps": 18, "priority": 2, "deadline_steps": 60,  "price_usd": 45},
        {"job_id": "j4", "material": "ABS",  "weight_g": 280, "print_time_steps": 22, "priority": 3, "deadline_steps": 80,  "price_usd": 75},
        {"job_id": "j5", "material": "PETG", "weight_g": 180, "print_time_steps": 14, "priority": 2, "deadline_steps": 75,  "price_usd": 50},
        {"job_id": "j6", "material": "PLA",  "weight_g": 240, "print_time_steps": 19, "priority": 2, "deadline_steps": 85,  "price_usd": 48},
    ],
    initial_inventory_g={"PLA": 1500.0, "PETG": 1200.0, "ABS": 1000.0},
    scheduled_failures=[
        {"printer_id": 3, "mode": "bed_level_drift", "trigger_step": 40, "duration": 50},
    ],
    stochastic_failure_rates={"_all": "profile_default"},
    success_criteria={
        "min_net_profit_usd": 150,
        "max_catastrophic_failures": 0,
        "min_avg_reliability_end_of_episode": 0.80,
    },
))


# --- Task 5: Economic Stress Test (stretch / held-out eval) -----------------
# Theme: World Modeling + P&L. Do NOT include in SFT/DPO training data.
# 15 jobs, limited inventory, 90 steps. Correct play involves triage.
_register(TaskConfig(
    task_id="task_5",
    name="Economic Stress Test",
    episode_length_steps=90,
    printer_count=6,
    printer_profiles=[
        "bambu_x1c", "bambu_x1c",
        "prusa_mk4",
        "creality_k1", "creality_k1",
        "voron_24",
    ],
    printer_overrides=[],
    operator_roster=[
        {"operator_id": "op_j1", "skill_level": "junior", "shift_window": [0, 90]},
        {"operator_id": "op_j2", "skill_level": "junior", "shift_window": [0, 90]},
        {"operator_id": "op_s1", "skill_level": "senior", "shift_window": [0, 90]},
        {"operator_id": "op_l1", "skill_level": "lead",   "shift_window": [0, 90]},
    ],
    job_queue=[
        # 4 urgent (priority 3)
        {"job_id": "u1", "material": "PLA",  "weight_g": 200, "print_time_steps": 15, "priority": 3, "deadline_steps": 25,  "price_usd": 90},
        {"job_id": "u2", "material": "PETG", "weight_g": 250, "print_time_steps": 20, "priority": 3, "deadline_steps": 40,  "price_usd": 100},
        {"job_id": "u3", "material": "ABS",  "weight_g": 220, "print_time_steps": 18, "priority": 3, "deadline_steps": 55,  "price_usd": 85},
        {"job_id": "u4", "material": "PLA",  "weight_g": 180, "print_time_steps": 14, "priority": 3, "deadline_steps": 70,  "price_usd": 75},
        # 8 normal (priority 2)
        {"job_id": "n1", "material": "PLA",  "weight_g": 150, "print_time_steps": 12, "priority": 2, "deadline_steps": 40,  "price_usd": 35},
        {"job_id": "n2", "material": "PLA",  "weight_g": 180, "print_time_steps": 14, "priority": 2, "deadline_steps": 45,  "price_usd": 40},
        {"job_id": "n3", "material": "PETG", "weight_g": 160, "print_time_steps": 13, "priority": 2, "deadline_steps": 50,  "price_usd": 42},
        {"job_id": "n4", "material": "PETG", "weight_g": 200, "print_time_steps": 16, "priority": 2, "deadline_steps": 60,  "price_usd": 48},
        {"job_id": "n5", "material": "ABS",  "weight_g": 180, "print_time_steps": 14, "priority": 2, "deadline_steps": 65,  "price_usd": 45},
        {"job_id": "n6", "material": "ABS",  "weight_g": 220, "print_time_steps": 18, "priority": 2, "deadline_steps": 75,  "price_usd": 50},
        {"job_id": "n7", "material": "PLA",  "weight_g": 150, "print_time_steps": 12, "priority": 2, "deadline_steps": 80,  "price_usd": 38},
        {"job_id": "n8", "material": "PETG", "weight_g": 130, "print_time_steps": 10, "priority": 2, "deadline_steps": 85,  "price_usd": 32},
        # 3 low (priority 1)
        {"job_id": "l1", "material": "PLA",  "weight_g": 120, "print_time_steps": 10, "priority": 1, "deadline_steps": None,  "price_usd": 18},
        {"job_id": "l2", "material": "PLA",  "weight_g": 100, "print_time_steps": 8,  "priority": 1, "deadline_steps": None,  "price_usd": 15},
        {"job_id": "l3", "material": "PETG", "weight_g": 110, "print_time_steps": 9,  "priority": 1, "deadline_steps": None,  "price_usd": 20},
    ],
    initial_inventory_g={"PLA": 1200.0, "PETG": 1000.0, "ABS": 800.0},
    scheduled_failures=[
        {"printer_id": 3, "mode": "bed_level_drift", "trigger_step": 30, "duration": 60},
    ],
    stochastic_failure_rates={"_all": "profile_default"},
    success_criteria={
        "min_net_profit_usd": 400,
        "max_catastrophic_failures": 0,
    },
))


# ---------------------------------------------------------------------------
#  Task loader — builds PrinterObservation + PrintJob list from TaskConfig
# ---------------------------------------------------------------------------

def load_task(task_id: str) -> "FarmObservation":
    """Build the initial FarmObservation for a task.

    The full ground-truth internal state (PrinterInternal objects, operator NPCs)
    is constructed inside PrintFarmEnvironment.reset(). This function returns
    only the *observation* layer (the corrupted/public view) and the task config
    so env.py can build the rest.
    """
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"Unknown task_id '{task_id}'. Known tasks: {list(TASK_CONFIGS)}")

    cfg = TASK_CONFIGS[task_id]
    return cfg   # env.py consumes the TaskConfig to bootstrap everything


def get_task_config(task_id: str) -> TaskConfig:
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"Unknown task_id '{task_id}'.")
    return TASK_CONFIGS[task_id]


# ---------------------------------------------------------------------------
#  TaskGrader — dollar-denominated scoring (aligned with economics.py)
# ---------------------------------------------------------------------------

class TaskGrader:
    """
    Tracks per-episode economic metrics and computes a normalised [0,1] score.

    The primary signal is net_profit_usd. A normalised score is also computed
    for cross-task comparability:
        normalised = (actual - naive_floor) / (clairvoyant_ceiling - naive_floor)

    naive_floor and clairvoyant_ceiling are stored per task after baseline runs.
    For now they default so the score roughly reflects profit vs a do-nothing policy.
    """

    # Per-task normalisation baselines.
    # Calibrated from 5-episode runs (seed=42) on 2026-04-21.
    # Floor  = naive-greedy mean * 1.5 (generous worst-case).
    # Ceiling = clairvoyant-greedy mean (prescient upper bound, not true optimal).
    # Update these after each full 200-episode calibration run.
    _NAIVE_FLOOR: Dict[str, float] = {
        "task_0_1": -50.0,   # naive mean: +32.8
        "task_0_2": -70.0,   # naive mean: -46.0
        "task_0_3": -265.0,  # naive mean: -175.5
        "task_1":   -710.0,  # naive mean: -471.8
        "task_2":   -250.0,  # naive mean: -166.0
        "task_3":   -215.0,  # naive mean: -142.0
        "task_4":   -430.0,  # naive mean: -283.4
        "task_5":   -830.0,  # naive mean: -553.0
    }
    _CLAIRVOYANT_CEILING: Dict[str, float] = {
        "task_0_1":  68.0,    # clairvoyant mean: +68.0
        "task_0_2": -233.0,   # clairvoyant mean: -232.7
        "task_0_3": -213.0,   # clairvoyant mean: -213.4
        "task_1":   -879.0,   # clairvoyant mean: -878.5
        "task_2":   -669.0,   # clairvoyant mean: -669.0
        "task_3":   -498.0,   # clairvoyant mean: -498.4
        "task_4":  -1319.0,   # clairvoyant mean: -1318.5
        "task_5":  -1932.0,   # clairvoyant mean: -1931.6
    }

    def __init__(self, task_id: str):
        self.task_id            = task_id
        self.failed_actions     = 0
        self.wasted_waits       = 0
        self.unnecessary_diags  = 0
        self.caught_faults      = 0
        self.catastrophic_count = 0
        self.operator_queue_overflow_steps = 0
        self.completion_step: Dict[str, int] = {}

    def step_update(
        self,
        action:          Optional[FarmAction],
        action_handled:  bool,
        state:           FarmObservation,
        time_step:       int,
        extra:           Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called each step from env.py with the action taken and resulting state."""
        extra = extra or {}

        # Count failed (rejected) actions
        if action and not action_handled and action.action != FarmActionEnum.WAIT:
            self.failed_actions += 1

        # Count wasted WAITs (there is actionable work)
        if action and action.action == FarmActionEnum.WAIT:
            has_pending = any(
                j.state in (JobState.PENDING, JobState.PAUSED)
                for j in state.active_queue
            )
            if has_pending:
                self.wasted_waits += 1

        # Diagnostic tracking
        if extra.get("diagnostic_was_unnecessary"):
            self.unnecessary_diags += 1
        if extra.get("diagnostic_caught_fault"):
            self.caught_faults += 1

        # Catastrophic failures
        self.catastrophic_count += extra.get("catastrophic_this_step", 0)

        # Operator queue overflow (any operator queue_size > capacity)
        for op in state.operators:
            if op.queue_size >= op.queue_capacity:
                self.operator_queue_overflow_steps += 1

        # Record completion steps
        for job in state.active_queue:
            if job.state == JobState.COMPLETED and job.job_id not in self.completion_step:
                self.completion_step[job.job_id] = time_step

    def get_score(self, state: FarmObservation) -> float:
        """
        Returns a normalised score in (0, 1).

        Primary signal: net_profit_usd from the FarmObservation economic summary.
        Action-quality penalties applied on top.
        """
        raw_profit = state.net_profit_usd
        floor     = self._NAIVE_FLOOR.get(self.task_id, -100.0)
        ceiling   = self._CLAIRVOYANT_CEILING.get(self.task_id, 500.0)

        if ceiling <= floor:
            normalised = 0.5
        else:
            normalised = (raw_profit - floor) / (ceiling - floor)

        # Soft penalise bad action hygiene (does not dominate the profit signal)
        action_penalty = (
            self.failed_actions * 0.005
            + self.wasted_waits * 0.002
            + self.unnecessary_diags * 0.003
        )

        score = normalised - action_penalty
        return _clamp(score)

    def check_success_criteria(self, state: FarmObservation, cfg: TaskConfig) -> Dict[str, Any]:
        """Check explicit success/failure criteria from TaskConfig."""
        criteria = cfg.success_criteria
        results: Dict[str, Any] = {}

        if "min_net_profit_usd" in criteria:
            results["profit_ok"] = state.net_profit_usd >= criteria["min_net_profit_usd"]

        if "max_catastrophic_failures" in criteria:
            results["safety_ok"] = self.catastrophic_count <= criteria["max_catastrophic_failures"]

        if "max_operator_queue_overflow_steps" in criteria:
            results["queue_ok"] = self.operator_queue_overflow_steps <= criteria["max_operator_queue_overflow_steps"]

        if "max_unnecessary_diagnostic_cost_usd" in criteria:
            cost = self.unnecessary_diags * 0.50  # DIAGNOSTIC_BASE_COST
            results["diagnostic_ok"] = cost <= criteria["max_unnecessary_diagnostic_cost_usd"]

        if "min_avg_reliability_end_of_episode" in criteria:
            if state.printers:
                avg_rel = sum(p.reliability for p in state.printers) / len(state.printers)
                results["reliability_ok"] = avg_rel >= criteria["min_avg_reliability_end_of_episode"]

        results["all_passed"] = all(v for v in results.values() if isinstance(v, bool))
        return results


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _priority_weight(priority: int) -> float:
    return {1: 0.5, 2: 1.0, 3: 2.0}.get(priority, 1.0)


def _clamp(value: float) -> float:
    """Clamp to (0, 1) — OpenEnv validator requires strictly between 0 and 1."""
    EPSILON = 0.001
    return max(EPSILON, min(1.0 - EPSILON, value))
