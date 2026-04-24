"""
Composite reward function — sums all components and logs them separately.

Components:
  r_economic:           net P&L delta vs rules baseline     [-1, +1]
  r_format:             did output parse?                   {-0.1, 0.0}
  r_fault_precision:    correct investigation?              {-0.1, 0.0, +0.2}
  r_message_handling:   correct customer response?          [-0.05, +0.15]
  r_unnecessary_action: penalty for spam                    {-0.05, 0.0}

Total = sum of all components.
"""

import math
from typing import Any, Dict, List, Optional

from submission.shared.parse_action import AgentAction
from .r_format import r_format
from .r_economic import r_economic
from .r_fault_precision import r_fault_precision
from .r_message_handling import r_message_handling
from .r_unnecessary_action import r_unnecessary_action


def compute_reward(
    parsed_action: Optional[AgentAction],
    llm_reward_delta: float,
    rules_reward_delta: float,
    ground_truth_tags: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, float]:
    """Compute all reward components and return as a dict.

    Args:
        parsed_action: the parsed agent action, or None if unparseable.
        llm_reward_delta: reward delta from LLM action + K rule steps.
        rules_reward_delta: reward delta from K+1 rule steps (baseline).
        ground_truth_tags: {operator_notes: [...], customer_messages: [...], anomaly_flags: [...]}

    Returns:
        Dict with keys: r_economic, r_format, r_fault_precision,
        r_message_handling, r_unnecessary_action, total.
    """
    note_tags = ground_truth_tags.get("operator_notes", [])
    msg_tags = ground_truth_tags.get("customer_messages", [])
    anomaly_tags = ground_truth_tags.get("anomaly_flags", [])

    components = {
        "r_economic": r_economic(llm_reward_delta, rules_reward_delta),
        "r_format": r_format(parsed_action),
        "r_fault_precision": r_fault_precision(parsed_action, anomaly_tags, note_tags),
        "r_message_handling": r_message_handling(parsed_action, msg_tags),
        "r_unnecessary_action": r_unnecessary_action(parsed_action, anomaly_tags, note_tags),
    }

    total = sum(components.values())

    # Safety: NaN check
    if not math.isfinite(total):
        total = 0.0
        for k in components:
            if not math.isfinite(components[k]):
                components[k] = 0.0

    components["total"] = total
    return components
