"""
r_novel_fault — reward for catching novel (non-taxonomy) faults preemptively.

Novel fault types (hybrid_thermal_humidity, intermittent_mcu, degraded_webcam)
are outside the 9-fault taxonomy and require LLM reasoning over text + telemetry.

Reward:
  +0.4  preemptive action on a novel fault with observable evidence
   0.0  preemptive action but no evidence (don't reward guessing)
   0.0  non-preemptive action or no novel fault present
"""

from typing import Any, Dict, List, Optional
from submission.shared.parse_action import AgentAction


NOVEL_FAULT_TYPES = {
    "hybrid_thermal_humidity", "intermittent_mcu", "degraded_webcam",
    "slow_extrusion_drift", "oscillating_z_wobble", "partial_bed_delamination",
    "resonance_buildup", "filament_moisture_absorption",
}
PREEMPTIVE_ACTIONS = {"REQUEST_MAINTENANCE", "DISPATCH_TICKET", "PAUSE_JOB", "RUN_DIAGNOSTIC"}


def r_novel_fault(
    action: Optional[AgentAction],
    anomaly_tags: List[Dict[str, Any]],
    observation: Optional[Dict[str, Any]] = None,
) -> float:
    """Reward catching novel (non-taxonomy) faults before they cascade.

    Args:
        action: the parsed agent action, or None.
        anomaly_tags: ground-truth tags for anomaly_flags this step.
        observation: structured observation dict (for evidence checking).

    Returns:
        +0.4 for evidence-based preemptive action on novel fault, 0.0 otherwise.
    """
    if action is None:
        return 0.0

    if action.action_type not in PREEMPTIVE_ACTIONS:
        return 0.0

    target_pid = action.printer_id
    if target_pid is None:
        return 0.0

    # Is there a novel fault on the targeted printer?
    novel_tag = None
    for tag in anomaly_tags:
        if (tag.get("printer_id") == target_pid
                and tag.get("fault_type") in NOVEL_FAULT_TYPES):
            novel_tag = tag
            break

    if novel_tag is None:
        return 0.0  # no novel fault here; this reward doesn't apply

    # Was there observable evidence?
    evidence = _has_subtle_evidence(observation, target_pid, novel_tag["fault_type"])
    if not evidence:
        return 0.0  # agent had no signal; don't reward guessing

    return +0.4  # evidence-based preemptive action on novel fault


def _has_subtle_evidence(
    observation: Optional[Dict[str, Any]],
    target_id: int,
    fault_type: str,
) -> bool:
    """Was there a SUBTLE signal — note, anomaly flag, or telemetry — pointing here?"""
    if observation is None:
        return False

    # Check visible anomaly flags (List[str]) — novel faults emit anomaly text
    for flag in observation.get("anomaly_flags", []):
        text = flag if isinstance(flag, str) else str(flag)
        if f"P{target_id}" in text or f"Printer {target_id}" in text:
            return True

    # Check visible operator notes (List[str])
    for note in observation.get("operator_notes", []):
        text = note if isinstance(note, str) else str(note)
        if f"P{target_id}" in text or f"Printer {target_id}" in text:
            return True

    # Type-specific telemetry evidence checks
    printer = None
    for p in observation.get("printers", []):
        pid = p.get("printer_id") if isinstance(p, dict) else getattr(p, "printer_id", None)
        if pid == target_id:
            printer = p
            break

    if printer is None:
        return False

    if isinstance(printer, dict):
        get = printer.get
    else:
        get = lambda k, d=None: getattr(printer, k, d)

    if fault_type == "hybrid_thermal_humidity":
        # Env emits "unusual thermal pattern" anomaly flag text for this,
        # but also check if printer fatigue is elevated
        fatigue = get("fatigue_level", 0.0)
        if fatigue >= 3:
            return True
    elif fault_type == "intermittent_mcu":
        # Env emits "intermittent communication" anomaly flag text
        # Stale telemetry timestamp can indicate MCU issues
        pass  # covered by anomaly flag text check above
    elif fault_type == "degraded_webcam":
        # Env emits "webcam quality degrading" anomaly flag text
        pass  # covered by anomaly flag text check above

    return False
