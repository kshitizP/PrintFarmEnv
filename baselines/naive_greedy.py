"""
Naive-Greedy Baseline
=====================
Plays PrintFarmEnv using OBSERVATION-ONLY, FIFO assignment.
No diagnostics, no maintenance, no operator orchestration.
This is what most LLMs do zero-shot — the "floor" reference.

Per ROUND2_MANUAL.md §9:
  "Naive-greedy: assign jobs FIFO to any available printer,
   ignore sensor warnings, trust everything."

Usage:
    python baselines/naive_greedy.py
    python baselines/naive_greedy.py --tasks task_1 task_2 task_3 --episodes 100
    python baselines/naive_greedy.py --tasks task_1 --episodes 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmAction, FarmActionEnum, JobState, PrinterState

DEFAULT_TASKS    = ["task_1", "task_2", "task_3"]
DEFAULT_EPISODES = 10


# ---------------------------------------------------------------------------
#  Naive policy — observation-only
# ---------------------------------------------------------------------------

def naive_action(obs) -> FarmAction:
    """
    FIFO job assignment. No diagnostics. No maintenance. No ticket dispatch.
    Trusts all telemetry. Ignores operator system entirely.

    Decision order:
      1. If a printer is IDLE and a PENDING job matches its material → ASSIGN
      2. WAIT
    """
    # Build lookup of idle printers by material
    idle_by_material: dict[str, list] = {}
    for p in obs.printers:
        if p.state == PrinterState.IDLE and p.current_material:
            idle_by_material.setdefault(p.current_material, []).append(p)

    # FIFO: pick first PENDING job that has a matching idle printer
    for job in obs.active_queue:
        if job.state != JobState.PENDING:
            continue
        candidates = idle_by_material.get(job.material_required, [])
        if candidates:
            printer = candidates.pop(0)
            return FarmAction(
                action=FarmActionEnum.ASSIGN_JOB,
                printer_id=printer.printer_id,
                job_id=job.job_id,
            )

    return FarmAction(action=FarmActionEnum.WAIT)


# ---------------------------------------------------------------------------
#  Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: PrintFarmEnvironment,
    task_id: str,
    episode: int,
    seed: int,
    verbose: bool,
) -> float:
    obs = env.reset(seed=seed + episode, task_id=task_id)

    if verbose:
        print(f"    Episode {episode:3d}  ", end="", flush=True)

    step = 0
    while not obs.done:
        action = naive_action(obs)
        obs = env.step(action)
        step += 1

    if verbose:
        print(f"score={obs.reward:.4f}  profit=${obs.net_profit_usd:+.2f}  steps={step}")

    return obs.reward


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Naive-greedy baseline (floor reference)")
    parser.add_argument("--tasks",    nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--episodes", type=int,  default=DEFAULT_EPISODES)
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--output",   type=str,  default=None,
                        help="JSON output path for results")
    parser.add_argument("--verbose",  "-v", action="store_true")
    args = parser.parse_args()

    env     = PrintFarmEnvironment()
    results = {}

    print(f"Naive-Greedy Baseline (floor)")
    print(f"Tasks:    {', '.join(args.tasks)}")
    print(f"Episodes: {args.episodes} per task")
    print()

    for task_id in args.tasks:
        print(f"  {task_id}:")
        scores = []
        for ep in range(args.episodes):
            score = run_episode(env, task_id, ep, args.seed, args.verbose)
            scores.append(score)
            if not args.verbose and (ep + 1) % 10 == 0:
                print(f"    [{ep+1:3d}/{args.episodes}] mean={sum(scores)/len(scores):.4f}",
                      flush=True)

        mean  = sum(scores) / len(scores)
        results[task_id] = {
            "mean": round(mean, 4),
            "best": round(max(scores), 4),
            "worst": round(min(scores), 4),
            "scores": [round(s, 4) for s in scores],
        }
        print(f"    → mean={mean:.4f}  best={max(scores):.4f}  worst={min(scores):.4f}\n")

    # Summary
    print("=" * 55)
    print(f"  {'Task':<14} {'Mean':>8} {'Best':>8} {'Worst':>8}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8}")
    for task_id, r in results.items():
        print(f"  {task_id:<14} {r['mean']:>8.4f} {r['best']:>8.4f} {r['worst']:>8.4f}")
    print("=" * 55)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
