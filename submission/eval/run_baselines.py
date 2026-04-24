"""
Baseline evaluator — measures performance of deterministic policies on the v2 env.

Records:
  - random policy
  - rule-based policy (uses structured signals only)
  - clairvoyant (uses ground truth)

Output: eval/baselines.json
"""

import json
import random
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submission.env.env import PrintFarmEnvironment
from submission.env.models import FarmAction, FarmActionEnum, FarmObservation
from submission.env.decision_point import DecisionPointEnv, _rules_action
from submission.shared.serialize import serialize_obs
from submission.shared.parse_action import AgentAction
from submission.rewards.composite import compute_reward


TASKS = ["task_1", "task_2", "task_3", "task_4", "task_5"]
SEEDS = [42, 123, 456, 789, 1010]
N_STEPS = 20  # Steps per episode for decision-point eval


def evaluate_policy(policy_fn, label, n_episodes=25):
    """Run a policy and collect metrics.

    policy_fn(obs) -> FarmAction
    """
    rng = random.Random(42)
    results = []
    action_dist = defaultdict(int)

    for ep in range(n_episodes):
        task_id = TASKS[ep % len(TASKS)]
        seed = SEEDS[ep % len(SEEDS)] + ep

        env = PrintFarmEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)

        ep_steps = 0
        while not obs.done:
            action = policy_fn(obs, rng)
            action_dist[action.action.value] += 1
            obs = env.step(action)
            ep_steps += 1

        results.append({
            "task_id": task_id,
            "seed": seed,
            "reward": obs.reward,
            "steps": ep_steps,
        })

    rewards = [r["reward"] for r in results]
    avg = sum(rewards) / len(rewards) if rewards else 0.0

    return {
        "label": label,
        "n_episodes": len(results),
        "avg_reward": avg,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "action_distribution": dict(action_dist),
        "episodes": results,
    }


def evaluate_decision_point_policy(policy_fn, label, n_episodes=25, k=10):
    """Evaluate a policy on decision points only (for LLM comparison).

    policy_fn(obs, gt_tags) -> AgentAction or None
    """
    rng = random.Random(42)
    results = []
    total_components = defaultdict(float)
    count = 0

    for ep in range(n_episodes):
        task_id = TASKS[ep % len(TASKS)]
        seed = SEEDS[ep % len(SEEDS)] + ep

        dp_env = DecisionPointEnv(k_horizon=k)
        try:
            serialized, obs = dp_env.reset(seed=seed, task_id=task_id)
        except Exception:
            continue

        gt_tags = dp_env.get_decision_tags()
        parsed = policy_fn(obs, gt_tags, rng)

        if parsed is not None:
            from submission.shared.parse_action import action_to_farm_action
            farm_action = action_to_farm_action(parsed)
        else:
            farm_action = FarmAction(action=FarmActionEnum.WAIT)

        llm_delta, _ = dp_env.step(farm_action)

        # Rules counterfactual
        dp_env2 = DecisionPointEnv(k_horizon=k)
        _, obs2 = dp_env2.reset(seed=seed, task_id=task_id)
        rules_action = _rules_action(obs2)
        rules_delta, _ = dp_env2.step(rules_action)

        components = compute_reward(parsed, llm_delta, rules_delta, gt_tags)
        for k_name, v in components.items():
            total_components[k_name] += v
        count += 1

        results.append({
            "task_id": task_id,
            "seed": seed,
            "reward_total": components["total"],
            "components": components,
        })

    if count > 0:
        avg_components = {k: v / count for k, v in total_components.items()}
    else:
        avg_components = {}

    return {
        "label": label,
        "n_episodes": count,
        "avg_reward": avg_components.get("total", 0),
        "avg_components": avg_components,
        "episodes": results,
    }


def random_policy(obs, rng):
    """Random valid action."""
    action_type = rng.choice(list(FarmActionEnum))
    action = FarmAction(action=action_type)
    if action_type == FarmActionEnum.ASSIGN_JOB:
        if obs.printers and obs.active_queue:
            action.printer_id = obs.printers[0].printer_id
            action.job_id = obs.active_queue[0].job_id
    elif action_type in (FarmActionEnum.RUN_DIAGNOSTIC, FarmActionEnum.PAUSE_JOB,
                         FarmActionEnum.RESUME_JOB):
        if obs.printers:
            action.printer_id = obs.printers[0].printer_id
    elif action_type == FarmActionEnum.REQUEST_SPOOL_SWAP:
        if obs.printers:
            action.printer_id = obs.printers[0].printer_id
            action.material = "PLA"
    return action


def rules_policy(obs, rng):
    """Rule-based agent (structured signals only)."""
    return _rules_action(obs)


def wait_policy(obs, rng):
    """Always wait."""
    return FarmAction(action=FarmActionEnum.WAIT)


# Decision point policies (get ground truth)
def dp_random_policy(obs, gt_tags, rng):
    """Random for decision point evaluation."""
    return AgentAction(
        action_type=rng.choice([e.value for e in FarmActionEnum])
    )


def dp_rules_policy(obs, gt_tags, rng):
    """Rules agent as AgentAction."""
    action = _rules_action(obs)
    return AgentAction(action_type=action.action.value)


def dp_wait_policy(obs, gt_tags, rng):
    """Always wait at decision points."""
    return AgentAction(action_type="WAIT")


def main():
    print("=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    output_dir = Path(__file__).resolve().parent
    all_results = {}

    # Full-episode baselines
    print("\n--- Full Episode Baselines ---")
    for name, fn in [("random", random_policy), ("rules", rules_policy), ("wait", wait_policy)]:
        print(f"\nEvaluating {name}...")
        result = evaluate_policy(fn, name, n_episodes=15)
        all_results[f"full_{name}"] = result
        print(f"  {name}: avg_reward={result['avg_reward']:.4f} "
              f"[{result['min_reward']:.4f}, {result['max_reward']:.4f}]")
        print(f"  Actions: {result['action_distribution']}")

    # Decision-point baselines
    print("\n--- Decision Point Baselines ---")
    for name, fn in [("dp_random", dp_random_policy),
                     ("dp_rules", dp_rules_policy),
                     ("dp_wait", dp_wait_policy)]:
        print(f"\nEvaluating {name}...")
        result = evaluate_decision_point_policy(fn, name, n_episodes=20)
        all_results[name] = result
        print(f"  {name}: avg_reward={result['avg_reward']:.4f}")
        if result['avg_components']:
            for k, v in result['avg_components'].items():
                if k != 'total':
                    print(f"    {k}: {v:.4f}")

    # Save results
    output_file = output_dir / "baselines.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Baselines saved to {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
