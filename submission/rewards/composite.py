"""
Composite reward function — sums all components and logs them separately.

Components:
  r_format:             format + anti-echo                  [-0.3, +0.1]
  r_economic:           net P&L delta vs rules baseline     [-0.4, +0.4]
  r_fault_precision:    evidence-gated investigation        [-0.15, +0.4]
  r_message_handling:   correct customer response?          [-0.1, +0.4]
  r_unnecessary_action: penalty for spam                    {-0.05, 0.0}
  r_novel_fault:        catching novel faults preemptively  {0.0, +0.4}

Total = sum of all components.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from submission.shared.parse_action import AgentAction
from .r_format import r_format
from .r_economic import r_economic
from .r_fault_precision import r_fault_precision
from .r_message_handling import r_message_handling
from .r_unnecessary_action import r_unnecessary_action
from .r_novel_fault import r_novel_fault


@dataclass
class RewardBreakdown:
    """Returned for logging — each component logged separately in W&B."""
    format: float
    economic: float
    fault_precision: float
    message_handling: float
    unnecessary_action: float
    novel_fault: float

    @property
    def total(self) -> float:
        return (
            self.format
            + self.economic
            + self.fault_precision
            + self.message_handling
            + self.unnecessary_action
            + self.novel_fault
        )

    def to_dict(self) -> dict:
        return {
            "reward/format": self.format,
            "reward/economic": self.economic,
            "reward/fault_precision": self.fault_precision,
            "reward/message_handling": self.message_handling,
            "reward/unnecessary_action": self.unnecessary_action,
            "reward/novel_fault": self.novel_fault,
            "reward/total": self.total,
        }


def compute_reward(
    parsed_action: Optional[AgentAction],
    llm_reward_delta: float,
    rules_reward_delta: float,
    ground_truth_tags: Dict[str, List[Dict[str, Any]]],
    model_output: str = "",
    observation_text: str = "",
    observation: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Compute all reward components and return as a dict.

    Args:
        parsed_action: the parsed agent action, or None if unparseable.
        llm_reward_delta: reward delta from LLM action + K rule steps.
        rules_reward_delta: reward delta from K+1 rule steps (baseline).
        ground_truth_tags: {operator_notes: [...], customer_messages: [...], anomaly_flags: [...]}
        model_output: raw model completion text (for anti-echo detection).
        observation_text: the observation text sent in the prompt (for anti-echo detection).
        observation: structured observation dict (for evidence gating in fault/novel rewards).

    Returns:
        Dict with keys: r_format, r_economic, r_fault_precision,
        r_message_handling, r_unnecessary_action, r_novel_fault, total.
    """
    note_tags = ground_truth_tags.get("operator_notes", [])
    msg_tags = ground_truth_tags.get("customer_messages", [])
    anomaly_tags = ground_truth_tags.get("anomaly_flags", [])

    fmt = r_format(parsed_action, model_output=model_output, observation_text=observation_text)

    if parsed_action is None:
        breakdown = RewardBreakdown(
            format=fmt,
            economic=0.0,
            fault_precision=0.0,
            message_handling=0.0,
            unnecessary_action=0.0,
            novel_fault=0.0,
        )
    else:
        breakdown = RewardBreakdown(
            format=fmt,
            economic=r_economic(llm_reward_delta, rules_reward_delta),
            fault_precision=r_fault_precision(parsed_action, anomaly_tags, note_tags, observation=observation),
            message_handling=r_message_handling(parsed_action, msg_tags),
            unnecessary_action=r_unnecessary_action(parsed_action, anomaly_tags, note_tags),
            novel_fault=r_novel_fault(parsed_action, anomaly_tags, observation=observation),
        )

    components = {
        "r_format": breakdown.format,
        "r_economic": breakdown.economic,
        "r_fault_precision": breakdown.fault_precision,
        "r_message_handling": breakdown.message_handling,
        "r_unnecessary_action": breakdown.unnecessary_action,
        "r_novel_fault": breakdown.novel_fault,
    }

    total = breakdown.total

    # Safety: NaN check
    if not math.isfinite(total):
        total = 0.0
        for k in components:
            if not math.isfinite(components[k]):
                components[k] = 0.0

    components["total"] = total
    return components
