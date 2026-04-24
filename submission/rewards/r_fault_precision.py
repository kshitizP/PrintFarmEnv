"""
r_fault_precision — reward for correctly investigating faults.

If the action was RUN_DIAGNOSTIC, REQUEST_MAINTENANCE, or DISPATCH_TICKET
for a diagnostic, was the target printer actually at risk?

  +0.2 if yes (correct investigation)
  -0.1 if no (false positive / unnecessary action)
   0.0 if the action wasn't an investigation

Uses hidden ground-truth anomaly tags and printer fault state.
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
    printer_id: Optional[int] = None,
) -> float:
    """Return fault precision reward.

    Args:
        action: the parsed agent action, or None.
        anomaly_tags: ground-truth tags for anomaly_flags this step.
        note_tags: ground-truth tags for operator_notes this step.
        printer_id: the printer targeted by the action (from action or obs).

    Returns:
        +0.2 for correct investigation, -0.1 for false positive, 0.0 otherwise.
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

    # Check if there's a genuine risk on the targeted printer
    has_risk = False

    # Check anomaly flags
    for tag in anomaly_tags:
        if tag.get("printer_id") == target_pid and tag.get("correct_action") == "investigate":
            has_risk = True
            break

    # Check operator notes (predictive category)
    if not has_risk:
        for tag in note_tags:
            if (tag.get("printer_id") == target_pid
                    and tag.get("category") == "predictive"
                    and tag.get("ground_truth_action") == "investigate"):
                has_risk = True
                break

    return 0.2 if has_risk else -0.1
