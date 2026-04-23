"""
run_baselines.py
================
Runs the naive-greedy and clairvoyant-greedy baselines on the requested tasks
and writes full step-level trajectories to disk as JSONL.

Outputs:
    data/baselines/{naive,clairvoyant}/{task_id}/episodes.jsonl   # trajectories
    data/baselines/{naive,clairvoyant}/{task_id}/summary.json      # aggregate stats
    data/baselines/summary.json                                    # master summary

Each line of episodes.jsonl is one full episode:
    {
      "task_id": "task_1",
      "seed": 42,
      "episode": 0,
      "final_score": 0.31,
      "final_profit_usd": 123.2,
      "final_step": 150,
      "steps": [
        {
          "step": 0,
          "observation": {...FarmObservation JSON...},
          "action": {...FarmAction JSON...},
          "reward": 0.5,
          "step_reward_usd": -0.1,
          "net_profit_usd": -0.1
        },
        ...
      ]
    }

Usage:
    # Full run per plan §9 (100 naive + 200 clairvoyant × 8 tasks)
    python scripts/run_baselines.py

    # Quick smoke test (one task, few episodes)
    python scripts/run_baselines.py --tasks task_1 --naive-episodes 3 --clairvoyant-episodes 3

    # Subset
    python scripts/run_baselines.py --tasks task_1 task_2 task_3 \
        --naive-episodes 100 --clairvoyant-episodes 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmAction, FarmObservation

from baselines.naive_greedy import naive_action
from baselines.clairvoyant_greedy import clairvoyant_action


ALL_TASKS = [
    "task_0_1", "task_0_2", "task_0_3",
    "task_1", "task_2", "task_3",
    "task_4", "task_5",
]


def _run_one_episode(
    env: PrintFarmEnvironment,
    task_id: str,
    seed: int,
    episode: int,
    policy_name: str,
    policy_fn: Callable,
) -> dict:
    """Run a single episode, collecting full step-level trace."""
    obs = env.reset(seed=seed + episode, task_id=task_id)
    steps = []
    step_idx = 0

    while not obs.done:
        if policy_name == "naive":
            action = policy_fn(obs)
        else:  # clairvoyant needs env, not obs
            action = policy_fn(env)

        # Capture pre-step observation + action
        obs_json = obs.model_dump(mode="json")
        action_json = action.model_dump(mode="json")

        obs = env.step(action)

        step_reward_usd = obs.metadata.get("step_reward_usd", 0.0) if obs.metadata else 0.0

        steps.append({
            "step": step_idx,
            "observation": obs_json,
            "action": action_json,
            "reward": obs.reward,
            "step_reward_usd": step_reward_usd,
            "net_profit_usd": obs.net_profit_usd,
        })
        step_idx += 1

    return {
        "task_id": task_id,
        "policy": policy_name,
        "seed": seed + episode,
        "episode": episode,
        "final_score": obs.reward,
        "final_profit_usd": obs.net_profit_usd,
        "final_step": step_idx,
        "steps": steps,
    }


def _run_policy_on_task(
    out_root: Path,
    task_id: str,
    policy_name: str,
    policy_fn: Callable,
    episodes: int,
    seed: int,
) -> dict:
    """Run a policy on a task for N episodes. Stream JSONL to disk."""
    out_dir = out_root / policy_name / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "episodes.jsonl"
    summary_path = out_dir / "summary.json"

    env = PrintFarmEnvironment()
    scores, profits = [], []
    t0 = time.time()

    with jsonl_path.open("w") as f:
        for ep in range(episodes):
            row = _run_one_episode(env, task_id, seed, ep, policy_name, policy_fn)
            f.write(json.dumps(row) + "\n")
            f.flush()
            scores.append(row["final_score"])
            profits.append(row["final_profit_usd"])
            if (ep + 1) % 10 == 0 or ep == episodes - 1:
                elapsed = time.time() - t0
                print(
                    f"    [{policy_name:<11} {task_id:<9} {ep+1:>3}/{episodes}] "
                    f"mean_score={sum(scores)/len(scores):.3f} "
                    f"mean_profit=${sum(profits)/len(profits):+7.2f} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )

    summary = {
        "task_id": task_id,
        "policy": policy_name,
        "episodes": episodes,
        "seed_base": seed,
        "mean_score": round(sum(scores) / len(scores), 4),
        "mean_profit_usd": round(sum(profits) / len(profits), 2),
        "best_score": round(max(scores), 4),
        "worst_score": round(min(scores), 4),
        "best_profit_usd": round(max(profits), 2),
        "worst_profit_usd": round(min(profits), 2),
        "wall_clock_seconds": round(time.time() - t0, 1),
        "jsonl_path": str(jsonl_path.relative_to(out_root.parent.parent)),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run baseline policies and save trajectories")
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                    help=f"Which tasks to run (default: all 8 = {ALL_TASKS})")
    ap.add_argument("--naive-episodes", type=int, default=100)
    ap.add_argument("--clairvoyant-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-naive", action="store_true")
    ap.add_argument("--skip-clairvoyant", action="store_true")
    ap.add_argument("--output-root", default="data/baselines",
                    help="Root directory for all output")
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output root:           {out_root}")
    print(f"Tasks:                 {', '.join(args.tasks)}")
    print(f"Naive episodes:        {args.naive_episodes if not args.skip_naive else 'SKIPPED'}")
    print(f"Clairvoyant episodes:  {args.clairvoyant_episodes if not args.skip_clairvoyant else 'SKIPPED'}")
    print(f"Seed base:             {args.seed}")
    print()

    master_summary = {"naive": {}, "clairvoyant": {}}

    if not args.skip_naive:
        print("== NAIVE-GREEDY ==")
        for task_id in args.tasks:
            s = _run_policy_on_task(
                out_root, task_id, "naive", naive_action, args.naive_episodes, args.seed,
            )
            master_summary["naive"][task_id] = s
        print()

    if not args.skip_clairvoyant:
        print("== CLAIRVOYANT-GREEDY ==")
        for task_id in args.tasks:
            s = _run_policy_on_task(
                out_root, task_id, "clairvoyant", clairvoyant_action,
                args.clairvoyant_episodes, args.seed,
            )
            master_summary["clairvoyant"][task_id] = s
        print()

    with (out_root / "summary.json").open("w") as f:
        json.dump(master_summary, f, indent=2)

    # Pretty print summary table
    print("=" * 78)
    print(f"  {'Task':<10} {'Naive $':>10} {'Clairv $':>10} {'Δ $':>10} {'Naive s':>9} {'Clairv s':>9}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*9} {'-'*9}")
    for task_id in args.tasks:
        n = master_summary["naive"].get(task_id, {})
        c = master_summary["clairvoyant"].get(task_id, {})
        n_p = n.get("mean_profit_usd", float("nan"))
        c_p = c.get("mean_profit_usd", float("nan"))
        d = (c_p - n_p) if (n_p == n_p and c_p == c_p) else float("nan")
        print(f"  {task_id:<10} {n_p:>10.2f} {c_p:>10.2f} {d:>+10.2f} "
              f"{n.get('mean_score', float('nan')):>9.3f} {c.get('mean_score', float('nan')):>9.3f}")
    print("=" * 78)
    print(f"\n  Wrote master summary to {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
