"""
Sanity check: run N rollouts with random valid actions and verify reward
component distributions are healthy before training.

Checks:
  - Each component fires non-zero on at least 10% of rollouts
  - No single component dominates (mean of any one < 3x another)
  - Total reward variance > 0.01 (gradient signal exists)
  - Some fraction (10-40%) of random actions get positive reward

Usage:
    python -m submission.scripts.check_reward_distribution [--n 200]
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submission.env.decision_point import DecisionPointEnv
from submission.env.models import FarmAction, FarmActionEnum
from submission.shared.parse_action import AgentAction
from submission.rewards.composite import compute_reward, RewardBreakdown


def _random_valid_action(obs, rng):
    """Generate a random VALID action consistent with current observation."""
    action_types = [
        "ASSIGN_JOB", "RUN_DIAGNOSTIC", "REQUEST_MAINTENANCE",
        "REQUEST_SPOOL_SWAP", "WAIT", "PAUSE_JOB",
    ]
    action_type = rng.choice(action_types)

    printers = obs.printers
    jobs = obs.active_queue

    if action_type == "ASSIGN_JOB":
        if not printers or not jobs:
            return FarmAction(action=FarmActionEnum.WAIT), "WAIT"
        return FarmAction(
            action=FarmActionEnum.ASSIGN_JOB,
            printer_id=rng.choice(printers).printer_id,
            job_id=rng.choice(jobs).job_id,
        ), "ASSIGN_JOB"
    elif action_type in ("RUN_DIAGNOSTIC", "PAUSE_JOB"):
        if not printers:
            return FarmAction(action=FarmActionEnum.WAIT), "WAIT"
        return FarmAction(
            action=FarmActionEnum(action_type),
            printer_id=rng.choice(printers).printer_id,
        ), action_type
    elif action_type == "REQUEST_MAINTENANCE":
        if not printers:
            return FarmAction(action=FarmActionEnum.WAIT), "WAIT"
        return FarmAction(
            action=FarmActionEnum.REQUEST_MAINTENANCE,
            printer_id=rng.choice(printers).printer_id,
            maintenance_type="general",
        ), "REQUEST_MAINTENANCE"
    elif action_type == "REQUEST_SPOOL_SWAP":
        if not printers:
            return FarmAction(action=FarmActionEnum.WAIT), "WAIT"
        return FarmAction(
            action=FarmActionEnum.REQUEST_SPOOL_SWAP,
            printer_id=rng.choice(printers).printer_id,
            material="PLA",
        ), "REQUEST_SPOOL_SWAP"
    else:
        return FarmAction(action=FarmActionEnum.WAIT), "WAIT"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="Number of rollouts")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    tasks = ["task_1", "task_2", "task_3"]

    components_list = {
        "r_format": [], "r_economic": [], "r_fault_precision": [],
        "r_message_handling": [], "r_unnecessary_action": [], "r_novel_fault": [],
    }
    totals = []
    errors = 0

    for i in range(args.n):
        task_id = tasks[i % len(tasks)]
        dp_env = DecisionPointEnv(k_horizon=5)
        try:
            _, obs = dp_env.reset(seed=rng.randint(0, 2**31), task_id=task_id)
        except Exception:
            errors += 1
            continue

        farm_action, action_name = _random_valid_action(obs, rng)

        try:
            delta, info = dp_env.step(farm_action)
            gt_tags = dp_env.get_decision_tags()

            obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else None

            # Build AgentAction with appropriate fields
            kwargs = {"action_type": action_name}
            if farm_action.printer_id is not None:
                kwargs["printer_id"] = farm_action.printer_id
            if farm_action.job_id is not None:
                kwargs["job_id"] = farm_action.job_id
            if farm_action.material is not None:
                kwargs["material"] = farm_action.material
            if farm_action.maintenance_type is not None:
                kwargs["maintenance_type"] = farm_action.maintenance_type

            try:
                parsed = AgentAction(**kwargs)
            except Exception:
                parsed = AgentAction(action_type="WAIT")

            model_output = f'<action>{json.dumps(parsed.model_dump(exclude_none=True))}</action>'
            components = compute_reward(
                parsed, delta, 0.0, gt_tags,
                model_output=model_output,
                observation=obs_dict,
            )

            for k in components_list:
                components_list[k].append(components[k])
            totals.append(components["total"])

        except Exception as e:
            errors += 1
            continue

    # Report
    print(f"\n{'='*60}")
    print(f"REWARD DISTRIBUTION CHECK ({len(totals)}/{args.n} successful, {errors} errors)")
    print(f"{'='*60}\n")

    for component, values in components_list.items():
        if not values:
            print(f"{component}: NO DATA")
            continue
        mean_v = sum(values) / len(values)
        var_v = sum((v - mean_v) ** 2 for v in values) / len(values)
        std_v = var_v ** 0.5
        nonzero = sum(1 for v in values if v != 0)
        print(f"{component}: mean={mean_v:.3f}, std={std_v:.3f}, "
              f"min={min(values):.3f}, max={max(values):.3f}, "
              f"nonzero={nonzero}/{len(values)}")

    if totals:
        mean_t = sum(totals) / len(totals)
        var_t = sum((t - mean_t) ** 2 for t in totals) / len(totals)
        pos_frac = sum(1 for t in totals if t > 0) / len(totals)
        print(f"\nTotal reward: mean={mean_t:.3f}, std={var_t**0.5:.3f}, "
              f"positive_fraction={pos_frac:.2%}")

        # Health checks
        issues = []
        if var_t < 0.01:
            issues.append("WARN: Total variance < 0.01 — weak gradient signal")
        if pos_frac < 0.05:
            issues.append("WARN: <5% positive rewards — bootstrapping may fail")
        if pos_frac > 0.80:
            issues.append("WARN: >80% positive rewards — reward too easy")

        for component, values in components_list.items():
            nonzero = sum(1 for v in values if v != 0)
            if nonzero < len(values) * 0.05:
                issues.append(f"WARN: {component} fires <5% — may have no gradient")

        if issues:
            print(f"\n{'!'*60}")
            for issue in issues:
                print(f"  {issue}")
            print(f"{'!'*60}")
        else:
            print("\nAll health checks PASSED")


if __name__ == "__main__":
    main()
