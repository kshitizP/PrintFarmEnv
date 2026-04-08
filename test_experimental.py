"""
Test runner for experimental tasks.

Usage:
  python test_experimental.py                  # Run all experimental tasks
  python test_experimental.py exp_5 exp_7      # Run specific tasks
"""

import os
import sys
import json
from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.experimental_tasks import load_experimental_task, ExperimentalGrader
from printfarm_env.models import FarmAction, FarmActionEnum

# Reuse inference setup from main module
from inference import extract_action


class ExperimentalEnvironment(PrintFarmEnvironment):
    """Thin wrapper that loads experimental tasks instead of standard ones."""

    def reset(self, seed=None, episode_id=None, **kwargs):
        task_id = kwargs.get("task_id", episode_id or "exp_4")
        self.current_task_id = task_id
        self._state = load_experimental_task(task_id)
        self.time_step = 0
        self.max_steps = self._state.max_steps
        self.grader = ExperimentalGrader(task_id)
        import random
        self._rng = random.Random(seed if seed is not None else 42)
        return self._state


def run_experimental_task(task_id: str, env: ExperimentalEnvironment) -> float:
    print(f"\n{'='*72}")
    print(f"  EXPERIMENTAL TASK: {task_id}")
    print(f"{'='*72}")

    observation = env.reset(episode_id=task_id)
    step_num = 0

    while not observation.done:
        action = extract_action(observation.model_dump_json())

        observation = env.step(action)
        step_num += 1

        print(f"  [STEP {step_num:2d}] reward={observation.reward:.4f}", end="")
        if observation.metadata.get("error"):
            print(f"  ERROR: {observation.metadata['error']}", end="")
        if observation.metadata.get("warning"):
            print(f"  WARN: {observation.metadata['warning']}", end="")
        print()

    final_score = observation.reward
    print(f"\n  FINAL: {task_id} score={final_score:.4f} steps={step_num}")
    return final_score


def main():
    all_tasks = ["exp_4", "exp_5", "exp_6", "exp_7", "exp_8"]

    if len(sys.argv) > 1:
        tasks = [t for t in sys.argv[1:] if t in all_tasks]
        if not tasks:
            print(f"Unknown tasks. Available: {', '.join(all_tasks)}")
            sys.exit(1)
    else:
        tasks = all_tasks

    env = ExperimentalEnvironment()
    scores = {}

    for t in tasks:
        scores[t] = run_experimental_task(t, env)

    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    for t, s in scores.items():
        bar = "█" * int(s * 40) + "░" * (40 - int(s * 40))
        print(f"  {t}: {s:.4f}  {bar}")

    # Sort by score descending to see difficulty
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Easiest → Hardest (by score):")
    for i, (t, s) in enumerate(ranked, 1):
        print(f"    {i}. {t} = {s:.4f}")


if __name__ == "__main__":
    main()
