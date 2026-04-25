"""
GRPO Rollout Generator for PrintFarmEnv.

Generates decision-point prompts from the environment and evaluates
model completions via the composite reward function.

Used by GRPOTrainer as the reward function provider.
"""

import copy
import json
import random
from typing import Any, Dict, List, Tuple

from submission.env.decision_point import (
    DecisionPointEnv, _rules_action, K_HORIZON, SIGNAL_TYPES, signal_present,
)
from submission.env.models import FarmAction, FarmActionEnum
from submission.shared.serialize import serialize_obs
from submission.shared.parse_action import parse_action, action_to_farm_action, AgentAction
from submission.shared.prompt import SYSTEM_PROMPT
from submission.shared.obs_formatter import format_observation_as_text
from submission.rewards.composite import compute_reward


def _compress_obs_json(serialized: str) -> str:
    """Compress serialized observation JSON to reduce token count.

    Removes completed jobs, zero/default printer fields, trims logs,
    and strips low-value top-level fields. Keeps all decision-critical
    signals (operator_notes, customer_messages, anomaly_flags).
    """
    try:
        obs = json.loads(serialized)
    except (json.JSONDecodeError, ValueError):
        return serialized

    # Remove completed/cancelled jobs from active_queue
    if "active_queue" in obs:
        obs["active_queue"] = [
            j for j in obs["active_queue"]
            if j.get("state") not in ("COMPLETED", "CANCELLED")
        ]
        for j in obs["active_queue"]:
            for k in list(j.keys()):
                if j[k] in (0, 0.0, False, None, ""):
                    del j[k]

    # Drop default/zero printer fields
    printer_defaults = {
        "warmup_remaining": 0, "offline_remaining": 0,
        "bed_drift_counter": 0.0, "reliability_penalty_active": False,
        "outstanding_ticket_id": None, "current_job_id": None,
        "revealed_this_step": False,
    }
    if "printers" in obs:
        for p in obs["printers"]:
            for k, default_val in printer_defaults.items():
                if p.get(k) == default_val:
                    p.pop(k, None)
            if p.get("current_material") is None:
                p.pop("current_material", None)

    # Trim oversight_log to last 3 entries
    if "oversight_log" in obs:
        obs["oversight_log"] = obs["oversight_log"][-3:]

    # Remove low-value top-level fields
    for drop_key in ("ticket_events", "reward_breakdown", "total_labor_billed"):
        obs.pop(drop_key, None)

    # Compress operator entries
    if "operators" in obs:
        for op in obs["operators"]:
            if op.get("current_ticket_id") is None:
                op.pop("current_ticket_id", None)
            if not op.get("pattern_recommendations"):
                op.pop("pattern_recommendations", None)
            if op.get("printer_visit_counts") == {}:
                op.pop("printer_visit_counts", None)

    return json.dumps(obs, separators=(",", ":"))


def generate_decision_prompts(
    n_prompts: int = 50,
    tasks: List[str] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate decision-point prompts from the environment.

    Round-robins across SIGNAL_TYPES so that each signal type (notes, messages,
    anomalies, structured) gets equal representation. Without this, the ~40%
    per-step probability of operator_notes crowds out messages (20%) and
    anomaly_flags (15%), causing those reward components to never fire.

    Returns a list of dicts, each with:
        - prompt: the chat-formatted prompt string
        - messages: list of message dicts for chat template
        - task_id: which task this came from
        - seed: the seed used
        - target_signal: which signal type triggered this decision point
        - ground_truth_tags: hidden tags for reward computation
        - decision_obs: the raw observation at the decision point
    """
    if tasks is None:
        tasks = ["task_1", "task_2", "task_3", "task_4", "task_5"]

    rng = random.Random(seed)
    prompts = []
    skipped = 0
    attempts = 0
    max_attempts = n_prompts * 4  # allow retries for rare signals

    signal_idx = 0  # cycles through SIGNAL_TYPES

    while len(prompts) < n_prompts and attempts < max_attempts:
        attempts += 1
        task_id = tasks[attempts % len(tasks)]
        ep_seed = rng.randint(0, 2**31)

        target_signal = SIGNAL_TYPES[signal_idx % len(SIGNAL_TYPES)]
        signal_idx += 1

        dp_env = DecisionPointEnv(k_horizon=K_HORIZON)
        serialized, obs = dp_env.reset(
            seed=ep_seed, task_id=task_id, target_signal=target_signal,
        )

        # Skip if the target signal did not appear within max_steps_to_decision
        if not signal_present(obs, target_signal):
            skipped += 1
            continue

        compressed = _compress_obs_json(serialized)
        obs_text = format_observation_as_text(compressed)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current state:\n{obs_text}"},
        ]

        prompts.append({
            "prompt": serialized,
            "messages": messages,
            "task_id": task_id,
            "seed": ep_seed,
            "target_signal": target_signal,
            "ground_truth_tags": dp_env.get_decision_tags(),
            "decision_obs": obs,
            "observation_text": obs_text,
        })

    if skipped > 0:
        print(f"[rollout] Skipped {skipped}/{attempts} attempts "
              f"(target signal not found in {90} steps)")
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

    # Compute composite reward (with anti-echo detection + evidence gating)
    obs_text = prompt_info.get("observation_text", "")

    # Build structured observation dict for evidence-gated rewards
    decision_obs = prompt_info.get("decision_obs")
    obs_dict = None
    if decision_obs is not None:
        if hasattr(decision_obs, "model_dump"):
            obs_dict = decision_obs.model_dump()
        elif isinstance(decision_obs, dict):
            obs_dict = decision_obs

    return compute_reward(
        parsed, llm_delta, rules_delta, gt_tags,
        model_output=completion, observation_text=obs_text,
        observation=obs_dict,
    )


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
