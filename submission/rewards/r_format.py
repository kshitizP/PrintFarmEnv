"""
r_format — format compliance reward.

Did the model's output parse into a valid AgentAction?
  - Parseable: 0.0 (no bonus, just not penalized)
  - Unparseable: -0.1

Capped to prevent format-farming.
"""

from submission.shared.parse_action import AgentAction
from typing import Optional


def r_format(parsed_action: Optional[AgentAction]) -> float:
    """Return format reward. -0.1 if unparseable, 0.0 if valid."""
    if parsed_action is None:
        return -0.1
    return 0.0
