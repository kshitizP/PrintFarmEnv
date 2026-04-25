"""
r_format — format compliance reward with anti-echo detection.

Reward signal:
  - No <action> tag at all:            -0.3  (heavy penalty)
  - <action> tag but invalid content:  -0.2
  - Echoing (>50% token overlap):      -0.2
  - Clean parseable action:            +0.1

The anti-echo penalty ensures the model cannot find a local optimum by
copying the observation text.
"""

import json
import re
from typing import Optional

from submission.shared.parse_action import AgentAction, _ACTION_TAG_RE


def r_format(
    parsed_action: Optional[AgentAction],
    model_output: str = "",
    observation_text: str = "",
) -> float:
    """Return format reward with anti-echo detection.

    Args:
        parsed_action: the parsed AgentAction, or None if unparseable.
        model_output: raw model completion text.
        observation_text: the observation text that was in the prompt.
    """
    # Check for <action> tag presence
    has_tag = bool(_ACTION_TAG_RE.search(model_output)) if model_output else False

    if not has_tag and parsed_action is None:
        return -0.3  # No tag, no parseable action

    if not has_tag and parsed_action is not None:
        # Parsed via fallback (raw JSON) but no tag — mild penalty
        return -0.1

    if has_tag and parsed_action is None:
        return -0.2  # Tag present but content invalid

    # At this point: has_tag=True and parsed_action is valid
    # Check for echoing
    if model_output and observation_text and len(observation_text) > 20:
        output_tokens = set(model_output.lower().split())
        obs_tokens = set(observation_text.lower().split())
        if output_tokens:
            overlap = len(output_tokens & obs_tokens) / len(output_tokens)
            if overlap > 0.5:
                return -0.2  # Heavy overlap = likely echoing
    
    return +0.1  # Clean parseable action with tag
