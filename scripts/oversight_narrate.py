"""
Oversight Agent — narrates Dispatcher decisions using the oversight log.

Produces natural-language audit commentary for questionable decisions:
  - Dispatcher overrides an operator's REPORT_ANOMALY
  - Dispatcher ignores a sensor warning without diagnosing
  - Dispatcher assigns a job against suspicious telemetry
  - WAIT spam when actionable work exists

Usage:
    python scripts/oversight_narrate.py --task task_3 --episodes 1 --seed 42

Can also be imported and called programmatically:
    from scripts.oversight_narrate import narrate_episode
    narration = narrate_episode(task_id="task_3", seed=42)
"""

import argparse
import json
from typing import List, Dict, Any

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmActionEnum
from baselines.clairvoyant_greedy import clairvoyant_action

# ── Oversight rules (deterministic — no LLM needed for basic narration) ──────

def _check_override_without_diagnostic(action, obs, prev_obs) -> str | None:
    """Flag: Dispatcher overrode an operator ticket without running a diagnostic first."""
    if action.action == FarmActionEnum.OVERRIDE_OPERATOR:
        return (f"⚠️ Step {obs.time_step}: OVERRIDE_OPERATOR on ticket {action.ticket_id}. "
                f"Reason given: '{action.reason or 'none'}'. "
                f"No RUN_DIAGNOSTIC preceded this override — consider verifying sensor state first.")
    return None

def _check_assign_on_suspicious_telemetry(action, obs, prev_obs) -> str | None:
    """Flag: Dispatcher assigned a job to a printer with suspicious telemetry."""
    if action.action != FarmActionEnum.ASSIGN_JOB or action.printer_id is None:
        return None
    for p in obs.printers:
        if p.printer_id == action.printer_id and not p.revealed_this_step:
            issues = []
            if p.hotend_temp == 0.0:
                issues.append("hotend_temp=0 (possible thermistor_open)")
            if p.hotend_temp > 400:
                issues.append(f"hotend_temp={p.hotend_temp} (possible thermistor_short)")
            if p.fan_rpm == 0:
                issues.append("fan_rpm=0 (possible fan_rpm_ghost)")
            if issues:
                return (f"⚠️ Step {obs.time_step}: ASSIGN_JOB to printer {action.printer_id} "
                        f"with unverified suspicious telemetry: {'; '.join(issues)}. "
                        f"Consider RUN_DIAGNOSTIC first.")
    return None

def _check_ignored_anomaly_report(action, obs, prev_obs) -> str | None:
    """Flag: An operator reported an anomaly but the Dispatcher took no diagnostic action."""
    if prev_obs is None:
        return None
    # Check if any ticket_events in prev observation were anomaly reports
    for evt in prev_obs.ticket_events:
        if evt.get("type") == "REPORT_ANOMALY":
            printer_id = evt.get("printer_id")
            if action.action not in (FarmActionEnum.RUN_DIAGNOSTIC, FarmActionEnum.PAUSE_JOB):
                return (f"⚠️ Step {obs.time_step}: Operator reported anomaly on printer {printer_id} "
                        f"but Dispatcher chose {action.action.value} instead of investigating.")
    return None

def _check_wait_spam(action, obs, prev_obs) -> str | None:
    """Flag: WAIT when there are PENDING jobs and IDLE printers."""
    if action.action != FarmActionEnum.WAIT:
        return None
    pending = [j for j in obs.active_queue if j.state.value == "PENDING"]
    idle = [p for p in obs.printers if p.state.value == "IDLE"]
    if pending and idle:
        return (f"⚠️ Step {obs.time_step}: WAIT while {len(pending)} jobs are PENDING "
                f"and {len(idle)} printers are IDLE. Potential missed opportunity.")
    return None

OVERSIGHT_CHECKS = [
    _check_override_without_diagnostic,
    _check_assign_on_suspicious_telemetry,
    _check_ignored_anomaly_report,
    _check_wait_spam,
]

def narrate_episode(task_id: str, seed: int = 42, policy_fn=None,
                    verbose: bool = True) -> List[str]:
    """
    Run one episode and produce oversight narration entries.

    Args:
        task_id: Task to run.
        seed: Random seed.
        policy_fn: Policy function (takes env, returns FarmAction).
                   Defaults to clairvoyant_action for demo purposes.
        verbose: Print narration as it's generated.

    Returns:
        List of narration strings (one per flagged event).
    """
    if policy_fn is None:
        policy_fn = clairvoyant_action

    env = PrintFarmEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    prev_obs = None
    narration: List[str] = []

    while not obs.done:
        action = policy_fn(env)

        prev_obs = obs
        obs = env.step(action)

        # Run all oversight checks
        for check in OVERSIGHT_CHECKS:
            finding = check(action, obs, prev_obs)
            if finding:
                narration.append(finding)
                if verbose:
                    print(finding)

    # Episode summary
    summary = (
        f"\n📊 Episode summary ({task_id}, seed={seed}):\n"
        f"   Final score    : {obs.reward:.4f}\n"
        f"   Net profit     : ${obs.net_profit_usd:+.2f}\n"
        f"   Oversight flags: {len(narration)}\n"
        f"   Steps          : {obs.time_step}"
    )
    narration.append(summary)
    if verbose:
        print(summary)

    return narration

def main():
    parser = argparse.ArgumentParser(description="Oversight Agent narration")
    parser.add_argument("--task", type=str, default="task_3")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for ep in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep} — {args.task} (seed={args.seed + ep})")
        print(f"{'='*60}")
        narrate_episode(args.task, seed=args.seed + ep)

if __name__ == "__main__":
    main()
