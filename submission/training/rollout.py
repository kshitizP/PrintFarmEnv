"""
GRPO Rollout Generator for PrintFarmEnv.

Generates decision-point prompts from the environment and evaluates
model completions via the composite reward function.

Used by GRPOTrainer as the reward function provider.
"""

import copy
import random
from typing import Any, Dict, List, Tuple

from submission.env.decision_point import DecisionPointEnv, _rules_action, K_HORIZON
from submission.env.models import FarmAction, FarmActionEnum
from submission.shared.serialize import serialize_obs
from submission.shared.parse_action import parse_action, action_to_farm_action, AgentAction
from submission.shared.prompt import SYSTEM_PROMPT
from submission.rewards.composite import compute_reward


def generate_decision_prompts(
    n_prompts: int = 50,
    tasks: List[str] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate decision-point prompts from the environment.

    Returns a list of dicts, each with:
        - prompt: the chat-formatted prompt string
        - messages: list of message dicts for chat template
        - task_id: which task this came from
        - seed: the seed used
        - ground_truth_tags: hidden tags for reward computation
        - decision_obs: the raw observation at the decision point
    """
    if tasks is None:
        tasks = ["task_1", "task_2", "task_3", "task_4", "task_5"]

    rng = random.Random(seed)
    prompts = []

    for i in range(n_prompts):
        task_id = tasks[i % len(tasks)]
        ep_seed = rng.randint(0, 2**31)

        dp_env = DecisionPointEnv(k_horizon=K_HORIZON)
        serialized, obs = dp_env.reset(seed=ep_seed, task_id=task_id)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current State:\n{serialized}"},
        ]

        prompts.append({
            "prompt": serialized,
            "messages": messages,
            "task_id": task_id,
            "seed": ep_seed,
            "ground_truth_tags": dp_env.get_decision_tags(),
            "decision_obs": obs,
        })

    return prompts


def evaluate_completion(
    completion: str,
    prompt_info: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate a model completion against the environment.

    Runs the completion's action + K rule steps, then compares against
    rules-only counterfactual.

    Returns reward components dict.
    """
    parsed = parse_action(completion)
    gt_tags = prompt_info["ground_truth_tags"]

    # Run LLM action in env
    dp_env_llm = DecisionPointEnv(k_horizon=K_HORIZON)
    dp_env_llm.reset(seed=prompt_info["seed"], task_id=prompt_info["task_id"])

    if parsed is not None:
        farm_action = action_to_farm_action(parsed)
    else:
        farm_action = FarmAction(action=FarmActionEnum.WAIT)

    llm_delta, _ = dp_env_llm.step(farm_action)

    # Run rules-only counterfactual
    dp_env_rules = DecisionPointEnv(k_horizon=K_HORIZON)
    _, obs = dp_env_rules.reset(seed=prompt_info["seed"], task_id=prompt_info["task_id"])

    rules_action = _rules_action(obs)
    rules_delta, _ = dp_env_rules.step(rules_action)

    # Compute composite reward
    return compute_reward(parsed, llm_delta, rules_delta, gt_tags)


def reward_function(completions: List[str], prompt_info: Dict[str, Any]) -> List[float]:
    """Batch reward function for GRPOTrainer.

    Args:
        completions: list of model-generated completions.
        prompt_info: the prompt metadata from generate_decision_prompts().

    Returns:
        list of total rewards, one per completion.
    """
    rewards = []
    for c in completions:
        components = evaluate_completion(c, prompt_info)
        rewards.append(components["total"])
    return rewards
