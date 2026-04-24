"""
r_unnecessary_action — penalty for spamming actions on healthy printers.

-0.05 for DIAGNOSTIC on a healthy printer with no notes/flags.
 0.0  otherwise.
"""

from typing import Any, Dict, List, Optional
from submission.shared.parse_action import AgentAction


def r_unnecessary_action(
    action: Optional[AgentAction],
    anomaly_tags: List[Dict[str, Any]],
    note_tags: List[Dict[str, Any]],
) -> float:
    """Return unnecessary action penalty.

    Args:
        action: the parsed agent action, or None.
        anomaly_tags: ground-truth tags for anomaly_flags this step.
        note_tags: ground-truth tags for operator_notes this step.

    Returns:
        -0.05 for unnecessary action, 0.0 otherwise.
    """
    if action is None:
        return 0.0

    # Only penalize investigation actions (RUN_DIAGNOSTIC, maintenance requests)
    if action.action_type not in ("RUN_DIAGNOSTIC", "REQUEST_MAINTENANCE"):
        return 0.0

    target_pid = action.printer_id

    # Check if there's ANY signal justifying this action
    has_signal = False

    for tag in anomaly_tags:
        if tag.get("printer_id") == target_pid:
            has_signal = True
            break

    if not has_signal:
        for tag in note_tags:
            if tag.get("printer_id") == target_pid and tag.get("category") in ("predictive", "ambiguous"):
                has_signal = True
                break

    return 0.0 if has_signal else -0.05
