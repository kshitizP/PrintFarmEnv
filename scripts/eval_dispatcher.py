"""
Evaluate a trained Dispatcher model against baselines on Round 2 tasks.

Usage:
    python scripts/eval_dispatcher.py --model <path_or_hf_id> --tasks task_1 task_2 task_3 --episodes 20

Outputs a comparison table: trained model vs naive-greedy vs clairvoyant-greedy.
"""

import argparse
import json
import sys
from pathlib import Path

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmAction, FarmActionEnum
from baselines.naive_greedy import naive_action
from baselines.clairvoyant_greedy import clairvoyant_action

def run_episodes(env, policy_fn, task_id, episodes, seed=42, needs_env=False):
    """Run N episodes and return list of (score, profit) tuples."""
    results = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep, task_id=task_id)
        while not obs.done:
            action = policy_fn(env) if needs_env else policy_fn(obs)
            obs = env.step(action)
        results.append((obs.reward, obs.net_profit_usd))
    return results

def make_llm_policy(model_id: str):
    """Create a policy function that calls an LLM for actions."""

    import os
    os.environ.setdefault("MODEL_NAME", model_id)
    from inference import extract_action

    def policy(obs):
        return extract_action(obs.model_dump_json())
    return policy

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dispatcher model")
    parser.add_argument("--model", type=str, default=None,
                        help="HF model ID or local path (uses inference.py)")
    parser.add_argument("--tasks", nargs="+", default=["task_1", "task_2", "task_3"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    env = PrintFarmEnvironment()
    all_results = {}

    policies = {
        "naive_greedy": (naive_action, False),
        "clairvoyant_greedy": (clairvoyant_action, True),
    }

    if args.model:
        policies["trained_model"] = (make_llm_policy(args.model), False)

    for task_id in args.tasks:
        all_results[task_id] = {}
        for name, (policy_fn, needs_env) in policies.items():
            results = run_episodes(env, policy_fn, task_id, args.episodes,
                                   seed=args.seed, needs_env=needs_env)
            scores = [r[0] for r in results]
            profits = [r[1] for r in results]
            all_results[task_id][name] = {
                "mean_score": sum(scores) / len(scores),
                "mean_profit": sum(profits) / len(profits),
                "min_score": min(scores),
                "max_score": max(scores),
            }

    # Print comparison table
    print(f"\n{'Task':<12} {'Policy':<22} {'Mean Score':>10} {'Mean Profit':>12} {'Min':>8} {'Max':>8}")
    print("-" * 74)
    for task_id in args.tasks:
        for name, stats in all_results[task_id].items():
            print(f"{task_id:<12} {name:<22} {stats['mean_score']:>10.4f} "
                  f"${stats['mean_profit']:>+10.2f} {stats['min_score']:>8.4f} {stats['max_score']:>8.4f}")
        print()

    # Gap analysis
    for task_id in args.tasks:
        naive = all_results[task_id]["naive_greedy"]["mean_score"]
        clairv = all_results[task_id]["clairvoyant_greedy"]["mean_score"]
        gap = clairv - naive
        if "trained_model" in all_results[task_id] and gap > 0:
            trained = all_results[task_id]["trained_model"]["mean_score"]
            pct = (trained - naive) / gap * 100
            print(f"  {task_id}: trained captures {pct:.1f}% of naive→clairvoyant gap")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
