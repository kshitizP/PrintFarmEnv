"""
r_fault_precision — reward for correctly investigating faults with evidence gating
                    PLUS negligence penalty for ignoring visible threats.

The reward matrix has TWO axes:
  1. Did the agent investigate? (action ∈ INVESTIGATION_ACTIONS)
  2. Was there visible evidence of a real anomaly?

  +0.4   investigated + visible evidence + real anomaly (correct)
  -0.15  investigated + visible evidence + no anomaly   (red herring; benign)
   0.0   investigated + no visible evidence + real anomaly (lucky guess)
  -0.05  investigated + no visible evidence + no anomaly (blind guess)
  -0.1   investigated + no/invalid target (malformed investigation)
  -0.4   did NOT investigate + visible evidence + real anomaly  ← NEGLIGENCE
   0.0   did NOT investigate + no real anomaly (correct inaction)

The negligence penalty closes the EV trap that let "always ASSIGN_JOB" beat
investigation: ignoring a fire is now strictly worse than guessing wrong.
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
    """Return fault precision reward with evidence gating + negligence penalty.

    Args:
        action: the parsed agent action, or None.
        anomaly_tags: ground-truth tags for anomaly_flags this step.
        note_tags: ground-truth tags for operator_notes this step.
        observation: structured observation dict (for evidence checking).
        printer_id: override printer id (from action or obs).

    Returns:
        Reward in [-0.4, +0.4] based on evidence, anomaly presence, and action.
    """
    if action is None:
        # Unparseable output — composite already penalises via r_format. We
        # return 0 here to avoid double-counting; negligence requires a real
        # action to be observed.
        return 0.0

    # Determine if the agent investigated
    is_investigation = False
    if action.action_type in ("RUN_DIAGNOSTIC", "REQUEST_MAINTENANCE"):
        is_investigation = True
    elif action.action_type == "DISPATCH_TICKET":
        if action.ticket_type in _INVESTIGATION_TICKET_TYPES:
            is_investigation = True

    # Determine if there is a visible threat anywhere in the observation
    # (any printer with a real anomaly that the agent could see)
    visible_threat_pids = _find_visible_threats(observation, anomaly_tags, note_tags)
    has_visible_threat = bool(visible_threat_pids)

    if not is_investigation:
        # Agent chose a non-investigation action.
        # If a visible threat exists somewhere on the farm, this is negligence.
        if has_visible_threat:
            return -0.4  # NEGLIGENCE: ignored a visible fire
        return 0.0  # no threats visible, no need to investigate

    # Agent investigated — apply the original 4-quadrant matrix on the target
    target_pid = action.printer_id or printer_id
    if target_pid is None:
        return -0.1  # investigation without a target is malformed

    # Did the targeted printer actually have an anomaly?
    has_anomaly_at_target = False
    for tag in anomaly_tags:
        if tag.get("printer_id") == target_pid and tag.get("correct_action") == "investigate":
            has_anomaly_at_target = True
            break
    if not has_anomaly_at_target:
        for tag in note_tags:
            if (tag.get("printer_id") == target_pid
                    and tag.get("category") == "predictive"
                    and tag.get("ground_truth_action") == "investigate"):
                has_anomaly_at_target = True
                break

    # Was there observable evidence at the target?
    evidence = _has_evidence_for_target(observation, target_pid) if observation else False

    if not evidence and not has_anomaly_at_target:
        return -0.05  # blind guess
    if not evidence and has_anomaly_at_target:
        return 0.0    # lucky guess, no learning signal
    if evidence and has_anomaly_at_target:
        return +0.4   # correct investigation
    if evidence and not has_anomaly_at_target:
        return -0.15  # red herring (resolved alarm decoy or benign-looking note)

    return 0.0  # unreachable


def _find_visible_threats(
    observation: Optional[Dict[str, Any]],
    anomaly_tags: List[Dict[str, Any]],
    note_tags: List[Dict[str, Any]],
) -> set:
    """Return set of printer_ids that have a visible, real anomaly the agent
    could have acted on.

    A "visible threat" requires:
      - a real anomaly (correct_action == "investigate" in tags), AND
      - observable evidence in the observation (note/flag mentions that pid,
        or anomalous telemetry).
    """
    threat_pids = set()
    if observation is None:
        return threat_pids

    # Real anomalies from anomaly_flags
    for tag in anomaly_tags:
        pid = tag.get("printer_id")
        if pid is not None and tag.get("correct_action") == "investigate":
            if _has_evidence_for_target(observation, pid):
                threat_pids.add(pid)

    # Real anomalies from predictive operator notes
    for tag in note_tags:
        pid = tag.get("printer_id")
        if (pid is not None
                and tag.get("category") == "predictive"
                and tag.get("ground_truth_action") == "investigate"):
            if _has_evidence_for_target(observation, pid):
                threat_pids.add(pid)

    return threat_pids


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
