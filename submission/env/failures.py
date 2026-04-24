"""Failure mode engine: injection, duration tracking, and sensor corruption layer."""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
#  Fault instance (internal env state)
# ---------------------------------------------------------------------------

@dataclass
class FaultInstance:
    mode: str
    printer_id: int
    injected_step: int
    duration_remaining: int    # steps until auto-expiry; -1 = never auto-expires
    payload: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Failure mode definitions
# ---------------------------------------------------------------------------
# condition:
#   None              — no extra check
#   "spool_gt_200"    — only if spool_weight_g > 200
#   "long_print"      — only if current job.print_time_steps > 8
#   "drift_counter"   — deterministic: triggers when bed_drift_counter >= 1.0

FAILURE_MODES: Dict[str, Dict] = {
    "thermistor_open": {
        "rate": 1e-3, "duration_mean": 8,
        "clears_on": {"diagnostic_physical", "expiry"},
        "condition": None,
    },
    "thermistor_short": {
        "rate": 5e-4, "duration_mean": 8,
        "clears_on": {"diagnostic_physical", "expiry"},
        "condition": None,
    },
    "filament_sensor_false_runout": {
        "rate": 2e-3, "duration_mean": 5,
        "clears_on": {"operator_reject", "expiry"},
        "condition": "spool_gt_200",
    },
    "filament_sensor_missed_runout": {
        "rate": 3e-4, "duration_mean": -1,
        "clears_on": {"stall_detected"},
        "condition": None,
    },
    "webcam_freeze": {
        "rate": 3e-3, "duration_mean": 10,
        "clears_on": {"run_diagnostic", "expiry"},
        "condition": None,
    },
    "klipper_mcu_disconnect": {
        "rate": 5e-4, "duration_mean": 6,
        "clears_on": {"self_clear", "diagnostic_physical", "expiry"},
        "condition": None,
    },
    "progress_drift": {
        "rate": 1e-3, "duration_mean": 6,
        "clears_on": {"run_diagnostic", "completion", "expiry"},
        "condition": "long_print",
    },
    "fan_rpm_ghost": {
        "rate": 2e-3, "duration_mean": -1,
        "clears_on": {"idle"},
        "condition": None,
    },
    "bed_level_drift": {
        "rate": 0,  "duration_mean": -1,
        "clears_on": {"maintenance_basic", "maintenance_full_rebuild"},
        "condition": "drift_counter",
    },
}


def _sample_duration(mode: str, rng: random.Random) -> int:
    mean = FAILURE_MODES[mode]["duration_mean"]
    if mean < 0:
        return -1
    return max(1, int(rng.expovariate(1.0 / mean)))


# ---------------------------------------------------------------------------
#  Injection
# ---------------------------------------------------------------------------

def try_inject(
    mode: str,
    printer,           # PrinterInternal from env.py
    job,               # Optional[PrintJob] — current job (may be None)
    current_step: int,
    rng: random.Random,
    override_rate: Optional[float] = None,
) -> Optional[FaultInstance]:
    """Roll for fault injection. Returns FaultInstance or None."""

    if mode in printer.active_faults:
        return None  # already active on this printer

    mdef = FAILURE_MODES[mode]
    base_rate = override_rate if override_rate is not None else mdef["rate"]
    multiplier = printer.profile.failure_rate_multipliers.get(mode, 1.0)
    effective_rate = base_rate * multiplier

    # Condition gates
    cond = mdef["condition"]
    if cond == "spool_gt_200" and printer.spool_weight_g <= 200:
        return None
    if cond == "long_print":
        if job is None or job.print_time_steps <= 8:
            return None
    if cond == "drift_counter":
        if printer.bed_drift_counter < 1.0:
            return None
        # Deterministic injection
        return FaultInstance(
            mode=mode, printer_id=printer.printer_id,
            injected_step=current_step, duration_remaining=-1,
        )

    if effective_rate <= 0 or rng.random() >= effective_rate:
        return None

    # Build mode-specific payload (snapshot for corruption layer)
    payload: Dict[str, Any] = {}
    if mode == "webcam_freeze":
        payload["frozen_hash"] = printer.webcam_hash
    elif mode == "klipper_mcu_disconnect":
        payload["frozen"] = {
            "hotend_temp": printer.hotend_temp,
            "spool_weight_g": printer.spool_weight_g,
            "state": printer.state.value,
            "telemetry_ts": printer.telemetry_ts,
            "fan_rpm": printer.fan_rpm,
        }
    elif mode == "filament_sensor_missed_runout":
        payload["frozen_spool"] = printer.spool_weight_g
    elif mode == "progress_drift":
        payload["frozen_progress"] = job.progress_steps if job else 0

    return FaultInstance(
        mode=mode, printer_id=printer.printer_id,
        injected_step=current_step,
        duration_remaining=_sample_duration(mode, rng),
        payload=payload,
    )


# ---------------------------------------------------------------------------
#  Scheduled injection (for task configs with trigger_step)
# ---------------------------------------------------------------------------

def inject_scheduled(printer, mode: str, duration: int, current_step: int,
                     job=None) -> FaultInstance:
    payload: Dict[str, Any] = {}
    if mode == "webcam_freeze":
        payload["frozen_hash"] = printer.webcam_hash
    elif mode == "klipper_mcu_disconnect":
        payload["frozen"] = {
            "hotend_temp": printer.hotend_temp,
            "spool_weight_g": printer.spool_weight_g,
            "state": printer.state.value,
            "telemetry_ts": printer.telemetry_ts,
            "fan_rpm": printer.fan_rpm,
        }
    elif mode == "filament_sensor_missed_runout":
        payload["frozen_spool"] = printer.spool_weight_g
    elif mode == "progress_drift":
        payload["frozen_progress"] = job.progress_steps if job else 0

    return FaultInstance(
        mode=mode, printer_id=printer.printer_id,
        injected_step=current_step, duration_remaining=duration,
        payload=payload,
    )


# ---------------------------------------------------------------------------
#  Duration advance
# ---------------------------------------------------------------------------

def advance_faults(active_faults: Dict[str, FaultInstance]) -> Dict[str, FaultInstance]:
    """Decrement duration counters; remove expired faults. Returns new dict."""
    alive = {}
    for mode, fault in active_faults.items():
        if fault.duration_remaining == -1:
            alive[mode] = fault  # persistent
        elif fault.duration_remaining > 1:
            fault.duration_remaining -= 1
            alive[mode] = fault
        # else: expired, drop it
    return alive


def clear_fault(active_faults: Dict[str, FaultInstance], mode: str) -> None:
    active_faults.pop(mode, None)


def clear_fault_class(active_faults: Dict[str, FaultInstance], clears_on: str) -> None:
    """Remove all faults whose clears_on set contains the given trigger."""
    to_remove = [
        m for m, f in active_faults.items()
        if clears_on in FAILURE_MODES[m]["clears_on"]
    ]
    for m in to_remove:
        del active_faults[m]


# ---------------------------------------------------------------------------
#  Sensor corruption layer
# ---------------------------------------------------------------------------

def apply_corruption(printer, job=None) -> Dict[str, Any]:
    """Return overrides dict for corrupted fields in PrinterObservation.

    Keys present in the returned dict should replace the ground-truth value.
    Keys absent mean no corruption for that field.
    """
    overrides: Dict[str, Any] = {}
    faults = printer.active_faults

    if "thermistor_open" in faults:
        overrides["hotend_temp"] = 0.0

    if "thermistor_short" in faults:
        overrides["hotend_temp"] = 450.0

    if "filament_sensor_false_runout" in faults:
        from .models import PrinterState
        overrides["state"] = PrinterState.PAUSED_RUNOUT

    if "filament_sensor_missed_runout" in faults:
        overrides["spool_weight_g"] = faults["filament_sensor_missed_runout"].payload.get(
            "frozen_spool", printer.spool_weight_g
        )

    if "webcam_freeze" in faults:
        overrides["webcam_hash"] = faults["webcam_freeze"].payload.get(
            "frozen_hash", printer.webcam_hash
        )

    if "klipper_mcu_disconnect" in faults:
        frozen = faults["klipper_mcu_disconnect"].payload.get("frozen", {})
        for k in ("hotend_temp", "spool_weight_g", "fan_rpm"):
            if k in frozen:
                overrides[k] = frozen[k]
        if "telemetry_ts" in frozen:
            overrides["telemetry_ts"] = frozen["telemetry_ts"]
        if "state" in frozen:
            from .models import PrinterState
            overrides["state"] = PrinterState(frozen["state"])

    if "fan_rpm_ghost" in faults:
        overrides["fan_rpm"] = 0

    # bed_level_drift has no telemetry corruption (only detectable via repeat failures)
    # progress_drift is applied at job level in env._build_observation

    return overrides


def get_corrupted_progress(printer, job) -> Optional[int]:
    """If progress_drift is active, return the frozen progress to show the dispatcher."""
    if job is None:
        return None
    if "progress_drift" in printer.active_faults:
        return printer.active_faults["progress_drift"].payload.get(
            "frozen_progress", job.progress_steps
        )
    return None
