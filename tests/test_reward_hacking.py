"""
Reward-hacking adversarial policy tests.

Each test runs a policy that attempts to exploit the reward function —
spamming WAIT, ignoring sensor alerts, hammering diagnostics,
cancelling all jobs, or flooding operator queues — and asserts that
the environment's multi-component reward stays below the honest
(naive-greedy) baseline.

These tests serve two purposes:
  1. CI gate: catch regressions where env changes accidentally open
     new reward-hacking loopholes.
  2. Pitch / blog evidence (FAQ #57): "Do not optimize a reward you
     have not tried to break yourself first."

Run:
    pytest tests/test_reward_hacking.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import (
    FarmAction, FarmActionEnum, FarmObservation,
    JobState, PrinterState,
)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

ADVERSARIAL_STEPS = 30   # enough to see hacking without running full episode
TASK  = "task_1"
SEED  = 42

# ---------------------------------------------------------------------------
#  Core helpers
# ---------------------------------------------------------------------------

def run_adversarial(policy_fn, task_id=TASK, seed=SEED, steps=ADVERSARIAL_STEPS):
    """Run an adversarial policy. Returns (episode_reward, step_rewards)."""
    env = PrintFarmEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    step_rewards = []
    for _ in range(steps):
        if obs.done:
            break
        action = policy_fn(env, obs)
        obs    = env.step(action)
        step_rewards.append(obs.metadata.get("step_reward_usd", 0.0))
    return obs.reward, step_rewards


def honest_baseline_reward(task_id=TASK, seed=SEED, steps=ADVERSARIAL_STEPS):
    """Run naive-greedy for the same step budget and return episode reward."""
    from baselines.naive_greedy import naive_action
    env = PrintFarmEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    for _ in range(steps):
        if obs.done:
            break
        obs = env.step(naive_action(obs))   # naive_action takes observation, not env
    return obs.reward


# ---------------------------------------------------------------------------
#  Adversarial policy 1: WAIT spam
#  Caught by: WAIT_COST (−$0.10/step) + SLA penalties for idle printers
# ---------------------------------------------------------------------------

def _wait_spam(env, obs):
    return FarmAction(action=FarmActionEnum.WAIT)


def test_wait_spam_stays_below_honest_baseline():
    """Spamming WAIT should not outscore honest play."""
    adversarial, _ = run_adversarial(_wait_spam)
    honest         = honest_baseline_reward()
    assert adversarial <= honest + 0.05, (
        f"WAIT spam scored {adversarial:.4f}, honest={honest:.4f} — "
        "WAIT_COST or SLA penalties are not penalising correctly"
    )


# ---------------------------------------------------------------------------
#  Adversarial policy 2: Diagnostic bomb
#  Caught by: DIAGNOSTIC_BASE_COST (−$0.50 on healthy printers, no bonus)
# ---------------------------------------------------------------------------

def _diagnostic_bomb(env, obs):
    """Run RUN_DIAGNOSTIC on every printer regardless of health."""
    if obs.printers:
        return FarmAction(
            action     = FarmActionEnum.RUN_DIAGNOSTIC,
            printer_id = obs.printers[0].printer_id,
        )
    return FarmAction(action=FarmActionEnum.WAIT)


def test_diagnostic_bomb_is_unprofitable():
    """Spamming diagnostics on healthy printers must crater P&L."""
    _, step_rewards = run_adversarial(_diagnostic_bomb)
    cumulative = sum(step_rewards)
    assert cumulative < 0, (
        f"Diagnostic bomb produced cumulative step_reward={cumulative:.2f} ≥ 0; "
        "DIAGNOSTIC_BASE_COST is not penalising unnecessary diagnostics"
    )


# ---------------------------------------------------------------------------
#  Adversarial policy 3: Sensor ignorer
#  Never responds to ERROR/PAUSED_RUNOUT; assigns blindly
#  Caught by: CATASTROPHIC_COST (−$250) + SLA penalties
# ---------------------------------------------------------------------------

def _sensor_ignorer(env, obs):
    """Assign jobs blindly; never dispatch maintenance or spool swaps."""
    pending = [j for j in obs.active_queue if j.state == JobState.PENDING]
    for p in obs.printers:
        if p.state == PrinterState.IDLE and pending:
            job = pending.pop(0)
            return FarmAction(
                action     = FarmActionEnum.ASSIGN_JOB,
                printer_id = p.printer_id,
                job_id     = job.job_id,
            )
    return FarmAction(action=FarmActionEnum.WAIT)


def test_sensor_ignorer_scores_below_honest():
    """Ignoring sensor alerts should underperform honest play."""
    adversarial, _ = run_adversarial(_sensor_ignorer, steps=ADVERSARIAL_STEPS)
    honest         = honest_baseline_reward()
    assert adversarial <= honest + 0.10, (
        f"Sensor-ignorer scored {adversarial:.4f} vs honest {honest:.4f}; "
        "CATASTROPHIC_COST or fault penalties may not be firing"
    )


# ---------------------------------------------------------------------------
#  Adversarial policy 4: Cancel everything
#  Caught by: scrap cost + revenue clawback on CANCEL_JOB
# ---------------------------------------------------------------------------

def _cancel_all(env, obs):
    """Cancel every active job the moment it appears."""
    for j in obs.active_queue:
        if j.state in (JobState.PENDING, JobState.PRINTING, JobState.PAUSED):
            return FarmAction(
                action = FarmActionEnum.CANCEL_JOB,
                job_id = j.job_id,
            )
    return FarmAction(action=FarmActionEnum.WAIT)


def test_cancel_all_is_worse_than_honest():
    """Cancelling all jobs should be strictly unprofitable."""
    adversarial, step_rewards = run_adversarial(_cancel_all)
    honest                    = honest_baseline_reward()
    assert adversarial < honest, (
        f"Cancel-all scored {adversarial:.4f} ≥ honest {honest:.4f}; "
        "scrap_cost or revenue clawback is not deterring cancellations"
    )
    assert sum(step_rewards) <= 0.01, (
        "Cancel-all should produce non-positive cumulative step reward"
    )


# ---------------------------------------------------------------------------
#  Adversarial policy 5: Operator queue flood
#  Dispatch maintenance to every operator every step
#  Caught by: labor deduction + ACTION_REJECT_COST on refused tickets
# ---------------------------------------------------------------------------

def _operator_flood(env, obs):
    """Dispatch maintenance to every available operator regardless of need."""
    for p in obs.printers:
        for op in obs.operators:
            if op.is_on_shift and op.queue_size < op.queue_capacity:
                return FarmAction(
                    action       = FarmActionEnum.DISPATCH_TICKET,
                    printer_id   = p.printer_id,
                    operator_id  = op.operator_id,
                    ticket_type  = "maintenance_basic",
                )
    return FarmAction(action=FarmActionEnum.WAIT)


def test_operator_flood_is_unprofitable():
    """Flooding operator queues should not improve scores."""
    adversarial, _ = run_adversarial(_operator_flood)
    honest         = honest_baseline_reward()
    assert adversarial <= honest + 0.05, (
        f"Operator-flood scored {adversarial:.4f} vs honest {honest:.4f}; "
        "labor deduction may not be large enough to deter flooding"
    )


# ---------------------------------------------------------------------------
#  Adversarial policy 6: Pause-resume cycling
#  Rapidly toggle every printing job to exploit revenue-timing gaps
#  Caught by: revenue accrues only on PRINTING steps (not PAUSED)
# ---------------------------------------------------------------------------

_pr_toggle: dict = {}

def _pause_resume_cycle(env, obs):
    """Toggle every printing job between PAUSED and PRINTING."""
    for p in obs.printers:
        pid = p.printer_id
        if p.state == PrinterState.PRINTING:
            _pr_toggle[pid] = True
            return FarmAction(action=FarmActionEnum.PAUSE_JOB, printer_id=pid)
        if p.state == PrinterState.PAUSED and _pr_toggle.get(pid):
            _pr_toggle[pid] = False
            return FarmAction(
                action     = FarmActionEnum.RESUME_JOB,
                printer_id = pid,
                job_id     = p.current_job_id,
            )
    return FarmAction(action=FarmActionEnum.WAIT)


def test_pause_resume_cycle_no_extra_revenue():
    """Pause-resume cycling should not earn more than steady printing."""
    _pr_toggle.clear()
    adversarial, _ = run_adversarial(_pause_resume_cycle)
    honest         = honest_baseline_reward()
    assert adversarial <= honest + 0.10, (
        f"Pause-resume cycle scored {adversarial:.4f} vs honest {honest:.4f}; "
        "revenue should not accrue on PAUSED steps"
    )


# ---------------------------------------------------------------------------
#  Adversarial policy 7: Override operator spam
#  Override every queued ticket to avoid labor costs
#  Caught by: OVERRIDE_COST + printers left unserviced
# ---------------------------------------------------------------------------

def _override_spam(env, obs):
    """Override every open ticket immediately using env._tickets."""
    for ticket_id, ticket in list(env._tickets.items()):
        # Only override tickets that are queued (not in-progress)
        from printfarm_env.models import TicketState
        if ticket.state == TicketState.QUEUED:
            return FarmAction(
                action    = FarmActionEnum.OVERRIDE_OPERATOR,
                ticket_id = ticket_id,
                reason    = "adversarial_override_spam",
            )
    return FarmAction(action=FarmActionEnum.WAIT)


def test_override_spam_stays_below_honest():
    """Overriding all tickets leaves printers unserviced and scores low."""
    adversarial, _ = run_adversarial(_override_spam)
    honest         = honest_baseline_reward()
    assert adversarial <= honest + 0.05, (
        f"Override-spam scored {adversarial:.4f} vs honest {honest:.4f}"
    )


# ---------------------------------------------------------------------------
#  Meta-test: honest baseline sanity check
#  Ensures comparisons above are meaningful — naive-greedy must be positive
# ---------------------------------------------------------------------------

def test_honest_baseline_is_positive():
    """Sanity check: naive-greedy should produce a positive episode reward."""
    honest = honest_baseline_reward()
    assert honest > 0, (
        f"Honest naive baseline returned {honest:.4f} ≤ 0; "
        "NAIVE_FLOOR calibration in tasks.py may need updating"
    )
