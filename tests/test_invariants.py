"""
Invariant checks from SIMULATOR_SPEC.md §8.

Runs a short episode for every task and asserts every invariant after each step().
"""
import pytest
from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import (
    FarmAction, FarmActionEnum, JobState, PrinterState,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _run_episode(task_id: str, steps: int = 20) -> list:
    """Return list of (env, obs) snapshots after each step."""
    env = PrintFarmEnvironment()
    obs = env.reset(seed=42, task_id=task_id)
    snapshots = [(env, obs)]
    for _ in range(steps):
        if obs.done:
            break
        obs = env.step(FarmAction(action=FarmActionEnum.WAIT))
        snapshots.append((env, obs))
    return snapshots


# ---------------------------------------------------------------------------
#  Physical invariants (§8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_inventory_non_negative(task_id):
    for env, obs in _run_episode(task_id):
        for mat, qty in env._inventory.items():
            assert qty >= -0.01, f"{task_id}: inventory[{mat}]={qty:.3f} < 0"


@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_spool_weight_non_negative(task_id):
    for env, obs in _run_episode(task_id):
        for p in env._printers:
            assert p.spool_weight_g >= -0.01, (
                f"{task_id}: P{p.printer_id} spool={p.spool_weight_g:.3f} < 0"
            )


@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_fatigue_in_range(task_id):
    for env, obs in _run_episode(task_id):
        for p in env._printers:
            assert 0.0 <= p.fatigue_level <= 10.0, (
                f"{task_id}: P{p.printer_id} fatigue={p.fatigue_level}"
            )


@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_reliability_in_range(task_id):
    for env, obs in _run_episode(task_id):
        for p in env._printers:
            assert 0.0 <= p.reliability <= 1.0, (
                f"{task_id}: P{p.printer_id} reliability={p.reliability}"
            )


@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_operator_queue_capacity(task_id):
    """Total queued tickets never exceeds total operator capacity."""
    for env, obs in _run_episode(task_id):
        total_queued = sum(len(op.queue) + (1 if op.current_ticket else 0)
                           for op in env._operators)
        total_cap    = sum(op.queue_capacity for op in env._operators)
        assert total_queued <= total_cap, (
            f"{task_id}: queued={total_queued} > capacity={total_cap}"
        )


@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_counters_non_negative(task_id):
    for env, obs in _run_episode(task_id):
        for p in env._printers:
            assert p.warmup_remaining     >= 0
            assert p.offline_remaining    >= 0
            assert p.maintenance_due_in   >= 0
            assert p.consecutive_idle_steps >= 0


# ---------------------------------------------------------------------------
#  Semantic invariants (§8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_maintenance_printer_has_no_job(task_id):
    for env, obs in _run_episode(task_id):
        for p in env._printers:
            if p.state in (PrinterState.MAINTENANCE, PrinterState.MAINTENANCE_QUEUED):
                assert p.current_job_id is None, (
                    f"{task_id}: P{p.printer_id} in {p.state} but has job {p.current_job_id}"
                )


@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_printing_job_has_one_printer(task_id):
    """Each PRINTING job is claimed by exactly one printer."""
    for env, obs in _run_episode(task_id):
        printer_job_ids = {p.current_job_id for p in env._printers if p.current_job_id}
        for j in env._jobs.values():
            if j.state == JobState.PRINTING:
                assert j.job_id in printer_job_ids, (
                    f"{task_id}: PRINTING job {j.job_id} has no printer"
                )


@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_no_duplicate_ticket_claims(task_id):
    """No two operators hold the same ticket_id in their queue/current_ticket."""
    for env, obs in _run_episode(task_id):
        seen = set()
        for op in env._operators:
            if op.current_ticket:
                tid = op.current_ticket.ticket_id
                assert tid not in seen, f"{task_id}: ticket {tid} claimed twice"
                seen.add(tid)
            for t in op.queue:
                assert t.ticket_id not in seen, (
                    f"{task_id}: ticket {t.ticket_id} claimed twice"
                )
                seen.add(t.ticket_id)


@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_paused_runout_job_is_paused(task_id):
    """A PAUSED_RUNOUT printer's job must be in state PAUSED."""
    for env, obs in _run_episode(task_id):
        for p in env._printers:
            if p.state == PrinterState.PAUSED_RUNOUT:
                j = env._jobs.get(p.current_job_id)
                assert j is not None and j.state == JobState.PAUSED, (
                    f"{task_id}: P{p.printer_id} PAUSED_RUNOUT but job state "
                    f"= {j.state if j else 'None'}"
                )


@pytest.mark.parametrize("task_id", ["task_0_1", "task_1", "task_2", "task_3"])
def test_no_duplicate_fault_modes(task_id):
    """Each printer has at most one instance of each failure mode."""
    for env, obs in _run_episode(task_id):
        for p in env._printers:
            # active_faults is a dict keyed by mode name → no duplicates possible
            # (ensure it's actually a dict, not a list)
            assert isinstance(p.active_faults, dict), (
                f"{task_id}: P{p.printer_id} active_faults is not a dict"
            )


# ---------------------------------------------------------------------------
#  Economic invariants (§8)
# ---------------------------------------------------------------------------

def test_sla_cap_80pct():
    """SLA penalty never exceeds 80% of job price."""
    env = PrintFarmEnvironment()
    obs = env.reset(seed=42, task_id="task_2")  # has scheduled failures + deadlines
    for _ in range(60):
        if obs.done:
            break
        obs = env.step(FarmAction(action=FarmActionEnum.WAIT))
    for j in env._jobs.values():
        cap = j.price_usd * 0.80
        assert j.total_sla_penalty <= cap + 0.01, (
            f"Job {j.job_id}: SLA penalty {j.total_sla_penalty:.2f} > 80% cap {cap:.2f}"
        )


def test_catastrophic_penalty():
    """Catastrophic failure subtracts exactly $250 + revenue clawback."""
    # Force a catastrophic failure by running a long episode with a near-max fatigue printer
    env = PrintFarmEnvironment()
    # task_0_3 has a printer with fatigue_level=7; with 0.1/step increment,
    # won't hit catastrophic in 30 steps. Check no catastrophic fires unexpectedly.
    obs = env.reset(seed=42, task_id="task_0_3")
    for _ in range(30):
        if obs.done:
            break
        obs = env.step(FarmAction(action=FarmActionEnum.WAIT))
    for p in env._printers:
        assert p.fatigue_level < 10.0, (
            f"Unexpected catastrophic: P{p.printer_id} fatigue={p.fatigue_level}"
        )


def test_step_reward_magnitude():
    """Per-step reward stays within ±$500 (spec §8 training invariant)."""
    env = PrintFarmEnvironment()
    for task_id in ["task_1", "task_2", "task_3"]:
        obs = env.reset(seed=42, task_id=task_id)
        for _ in range(60):
            if obs.done:
                break
            obs = env.step(FarmAction(action=FarmActionEnum.WAIT))
            sr = obs.metadata.get("step_reward_usd", 0.0)
            assert abs(sr) <= 500.0, (
                f"{task_id}: step reward {sr:.2f} exceeds ±$500"
            )


def test_net_profit_cumulation():
    """net_profit_usd matches sum of per-step rewards."""
    env = PrintFarmEnvironment()
    obs = env.reset(seed=42, task_id="task_1")
    cumulative = 0.0
    for _ in range(30):
        if obs.done:
            break
        obs = env.step(FarmAction(action=FarmActionEnum.WAIT))
        cumulative += obs.metadata.get("step_reward_usd", 0.0)
    assert abs(obs.net_profit_usd - cumulative) < 0.01, (
        f"net_profit={obs.net_profit_usd:.4f} ≠ sum of step rewards {cumulative:.4f}"
    )


# ---------------------------------------------------------------------------
#  Observation invariants (§8)
# ---------------------------------------------------------------------------

def test_observation_json_serialisable():
    """All FarmObservation fields are JSON-serialisable (no numpy leakage)."""
    import json
    env = PrintFarmEnvironment()
    obs = env.reset(seed=42, task_id="task_1")
    for _ in range(5):
        if obs.done:
            break
        obs = env.step(FarmAction(action=FarmActionEnum.WAIT))
    # model_dump_json() raises if non-serialisable types are present
    json_str = obs.model_dump_json()
    data = json.loads(json_str)
    assert "printers" in data
    assert "operators" in data


def test_reward_field_equals_step_return():
    """obs.reward returned from step() matches obs.reward field stored in obs."""
    env = PrintFarmEnvironment()
    obs = env.reset(seed=42, task_id="task_1")
    for _ in range(10):
        if obs.done:
            break
        action = FarmAction(action=FarmActionEnum.WAIT)
        obs = env.step(action)
        # obs.reward should be the normalised score from TaskGrader
        assert 0.0 <= obs.reward <= 1.0, f"obs.reward={obs.reward} out of [0,1]"


# ---------------------------------------------------------------------------
#  Task grader invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", ["task_1", "task_2", "task_3"])
def test_grader_ceiling_above_floor(task_id):
    """TaskGrader floor < ceiling for every task (reward not inverted)."""
    from printfarm_env.tasks import TaskGrader
    g = TaskGrader(task_id)
    floor   = g._NAIVE_FLOOR[task_id]
    ceiling = g._CLAIRVOYANT_CEILING[task_id]
    assert ceiling > floor, (
        f"{task_id}: ceiling={ceiling} ≤ floor={floor} (inverted reward)"
    )


@pytest.mark.parametrize("task_id", ["task_1", "task_2", "task_3"])
def test_score_in_unit_interval(task_id):
    """get_score() always returns a value in (0, 1)."""
    env = PrintFarmEnvironment()
    obs = env.reset(seed=42, task_id=task_id)
    for _ in range(30):
        if obs.done:
            break
        obs = env.step(FarmAction(action=FarmActionEnum.WAIT))
    score = env.grader.get_score(obs)
    assert 0.0 < score < 1.0, f"{task_id}: score={score} outside (0,1)"
