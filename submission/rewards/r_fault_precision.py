"""
r_fault_precision — reward for correctly investigating faults with evidence gating.

If the action was RUN_DIAGNOSTIC, REQUEST_MAINTENANCE, or DISPATCH_TICKET
for a diagnostic, was the target printer actually at risk?

Evidence gating ensures the agent is rewarded for reasoning over observable
signals (operator notes, anomaly flags, telemetry), not blind guessing.

  +0.4  evidence present + real anomaly (correct investigation)
  -0.15 evidence present + no anomaly (red herring — should have judged benign)
   0.0  no evidence + real anomaly (got lucky — neutral, no learning signal)
  -0.05 no evidence + no anomaly (blind guess)
  -0.1  malformed investigation (no target / non-existent target)
   0.0  action wasn't an investigation

Uses hidden ground-truth anomaly tags AND visible observation for evidence.
"""

from typing import Any, Dict, List, Optional
from submission.shared.parse_action import AgentAction


_INVESTIGATION_ACTIONS = {
    "RUN_DIAGNOSTIC", "REQUEST_MAINTENANCE", "DISPATCH_TICKET",
}

_INVESTIGATION_TICKET_TYPES = {
    "diagnostic_physical", "maintenance_basic", "maintenance_full_rebuild",
}


def r_fault_precision(
    action: Optional[AgentAction],
    anomaly_tags: List[Dict[str, Any]],
    note_tags: List[Dict[str, Any]],
    observation: Optional[Dict[str, Any]] = None,
    printer_id: Optional[int] = None,
) -> float:
    """Return fault precision reward with evidence gating.

    Args:
        action: the parsed agent action, or None.
        anomaly_tags: ground-truth tags for anomaly_flags this step.
        note_tags: ground-truth tags for operator_notes this step.
        observation: structured observation dict (for evidence checking).
        printer_id: override printer id (from action or obs).

    Returns:
        Reward in [-0.15, +0.4] based on evidence and anomaly presence.
    """
    if action is None:
        return 0.0

    is_investigation = False

    if action.action_type in ("RUN_DIAGNOSTIC", "REQUEST_MAINTENANCE"):
        is_investigation = True
    elif action.action_type == "DISPATCH_TICKET":
        if action.ticket_type in _INVESTIGATION_TICKET_TYPES:
            is_investigation = True

    if not is_investigation:
        return 0.0

    target_pid = action.printer_id or printer_id
    if target_pid is None:
        return -0.1  # investigation without a target is malformed

    # Check if there's a genuine risk on the targeted printer (ground truth)
    has_anomaly = False

    for tag in anomaly_tags:
        if tag.get("printer_id") == target_pid and tag.get("correct_action") == "investigate":
            has_anomaly = True
            break

    if not has_anomaly:
        for tag in note_tags:
            if (tag.get("printer_id") == target_pid
                    and tag.get("category") == "predictive"
                    and tag.get("ground_truth_action") == "investigate"):
                has_anomaly = True
                break

    # Check if there's observable evidence justifying this investigation
    evidence = _has_evidence_for_target(observation, target_pid) if observation else False

    # Evidence-gated reward matrix
    if not evidence and not has_anomaly:
        return -0.05  # blind guess, no evidence, no anomaly
    if not evidence and has_anomaly:
        return 0.0  # got lucky but no observable signal to learn from
    if evidence and has_anomaly:
        return +0.4  # evidence present, agent acted correctly
    if evidence and not has_anomaly:
        return -0.15  # red herring; agent should have judged it benign

    return 0.0  # unreachable


def _has_evidence_for_target(observation: Optional[Dict[str, Any]], target_id: int) -> bool:
    """Did the observation contain a signal pointing at this specific printer?

    Checks visible operator_notes, anomaly_flags, and printer telemetry.
    """
    if observation is None:
        return False

    target_str = str(target_id)

    # Check visible operator notes (List[str])
    for note in observation.get("operator_notes", []):
        text = note if isinstance(note, str) else str(note)
        if f"P{target_id}" in text or f"Printer {target_id}" in text or f"printer {target_id}" in text:
            return True

    # Check visible anomaly flags (List[str])
    for flag in observation.get("anomaly_flags", []):
        text = flag if isinstance(flag, str) else str(flag)
        if f"P{target_id}" in text or f"Printer {target_id}" in text:
            return True

    # Check printer telemetry for anomalous readings
    for p in observation.get("printers", []):
        pid = p.get("printer_id") if isinstance(p, dict) else getattr(p, "printer_id", None)
        if pid == target_id and _is_telemetry_anomalous(p):
            return True

    return False


def _is_telemetry_anomalous(printer: Any) -> bool:
    """Cheap heuristic — flag obviously anomalous telemetry values."""
    if isinstance(printer, dict):
        get = printer.get
    else:
        get = lambda k, d=None: getattr(printer, k, d)

    # Hotend temp out of normal range (normally ~200°C)
    hotend = get("hotend_temp", 200.0)
    if hotend < 150.0 or hotend > 260.0:
        return True

    # Fan RPM anomalous (normally ~3000)
    fan = get("fan_rpm", 3000)
    if fan < 1000 or fan > 6000:
        return True

    # Fatigue level high
    fatigue = get("fatigue_level", 0.0)
    if fatigue >= 7:
        return True

    # Reliability degraded
    reliability = get("reliability", 0.95)
    if reliability < 0.7:
        return True

    return False
