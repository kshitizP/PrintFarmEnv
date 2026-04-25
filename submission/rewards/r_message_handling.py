"""
r_message_handling — reward for correctly handling customer messages.

If a customer message was present and the action addressed it correctly:
  +0.4  if action matches ground-truth correct action AND correct target
  +0.15 if right action type but wrong job target
  -0.1  if message was present but action is irrelevant/wrong
   0.0  if no message was present

Requires correct action class AND correct target job_id.
"""

from typing import Any, Dict, List, Optional
from submission.shared.parse_action import AgentAction


# Mapping from ground_truth_action to acceptable action_types
_CORRECT_ACTIONS = {
    "accept_rush": {"ASSIGN_JOB", "RESUME_JOB"},  # prioritize the job
    "standard_queue": {"WAIT", "ASSIGN_JOB"},      # don't rush, normal processing
    "decline": {"CANCEL_JOB", "WAIT"},
    "accept_substitute": {"REQUEST_SPOOL_SWAP", "ASSIGN_JOB"},
}


def r_message_handling(
    action: Optional[AgentAction],
    message_tags: List[Dict[str, Any]],
) -> float:
    """Return message handling reward.

    Args:
        action: the parsed agent action, or None.
        message_tags: ground-truth tags for customer_messages this step.

    Returns:
        +0.4 for exact match, +0.15 right action wrong job, -0.1 wrong, 0.0 if no message.
    """
    if not message_tags:
        return 0.0

    if action is None:
        return -0.1  # Had a message but couldn't even parse the action

    total_reward = 0.0

    for tag in message_tags:
        gt_action = tag.get("ground_truth_action", "standard_queue")
        acceptable = _CORRECT_ACTIONS.get(gt_action, set())

        if action.action_type in acceptable:
            # Check job_id match for targeted actions
            tag_job = tag.get("job_id")
            if action.action_type in ("ASSIGN_JOB", "CANCEL_JOB", "RESUME_JOB"):
                if tag_job and action.job_id and action.job_id != tag_job:
                    # Right action type but wrong job — partial credit
                    total_reward += 0.15
                else:
                    total_reward += 0.4
            else:
                # WAIT / REQUEST_SPOOL_SWAP — correct action type is enough
                total_reward += 0.4
        else:
            total_reward -= 0.1

    return total_reward
