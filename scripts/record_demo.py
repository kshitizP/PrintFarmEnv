"""
Record a demo trajectory with side-by-side naive vs clairvoyant comparison
and oversight narration overlay. Outputs JSON for visualization.

Usage:
    python scripts/record_demo.py --task task_3 --seed 42 --output demo_trajectory.json
"""

import argparse
import json
from pathlib import Path

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmActionEnum
from baselines.naive_greedy import naive_action
from baselines.clairvoyant_greedy import clairvoyant_action
from scripts.oversight_narrate import OVERSIGHT_CHECKS

def record_trajectory(task_id, seed, policy_fn, policy_name, needs_env=False):
    """Record a full episode trajectory with step-by-step detail."""
    env = PrintFarmEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    prev_obs = None
    steps = []

    while not obs.done:
        action = policy_fn(env) if needs_env else policy_fn(obs)
        prev_obs = obs
        obs = env.step(action)

        # Collect oversight flags
        flags = []
        for check in OVERSIGHT_CHECKS:
            finding = check(action, obs, prev_obs)
            if finding:
                flags.append(finding)

        steps.append({
            "step": obs.time_step,
            "action": action.action.value,
            "printer_id": action.printer_id,
            "job_id": action.job_id,
            "step_reward_usd": obs.metadata.get("step_reward_usd", 0.0),
            "net_profit_usd": obs.net_profit_usd,
            "reward": obs.reward,
            "reward_breakdown": obs.reward_breakdown,
            "oversight_flags": flags,
            "active_printers": [
                {"id": p.printer_id, "state": p.state.value, "job": p.current_job_id}
                for p in obs.printers
            ],
        })

    return {
        "policy": policy_name,
        "task_id": task_id,
        "seed": seed,
        "final_score": obs.reward,
        "final_profit_usd": obs.net_profit_usd,
        "total_steps": obs.time_step,
        "total_oversight_flags": sum(len(s["oversight_flags"]) for s in steps),
        "steps": steps,
    }

def main():
    parser = argparse.ArgumentParser(description="Record demo trajectory")
    parser.add_argument("--task", type=str, default="task_3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="demo_trajectory.json")
    args = parser.parse_args()

    print(f"Recording {args.task} (seed={args.seed})...\n")

    naive_traj = record_trajectory(args.task, args.seed, naive_action, "naive_greedy")
    clair_traj = record_trajectory(args.task, args.seed, clairvoyant_action, "clairvoyant_greedy", needs_env=True)

    demo = {
        "task_id": args.task,
        "seed": args.seed,
        "comparison": {
            "naive_greedy": {
                "score": naive_traj["final_score"],
                "profit_usd": naive_traj["final_profit_usd"],
                "oversight_flags": naive_traj["total_oversight_flags"],
            },
            "clairvoyant_greedy": {
                "score": clair_traj["final_score"],
                "profit_usd": clair_traj["final_profit_usd"],
                "oversight_flags": clair_traj["total_oversight_flags"],
            },
        },
        "trajectories": {
            "naive_greedy": naive_traj,
            "clairvoyant_greedy": clair_traj,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(demo, f, indent=2)

    print(f"{'Policy':<22} {'Score':>8} {'Profit':>10} {'Flags':>6}")
    print("-" * 50)
    for name in ["naive_greedy", "clairvoyant_greedy"]:
        c = demo["comparison"][name]
        print(f"{name:<22} {c['score']:>8.4f} ${c['profit_usd']:>+8.2f} {c['oversight_flags']:>6}")

    print(f"\nSaved → {args.output}")

if __name__ == "__main__":
    main()
