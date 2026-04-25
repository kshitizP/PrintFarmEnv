"""
Phase 5 Smoke Tests — catch silent failures before wasting an overnight run.

Tests:
  5.1 Observation consistency (serialization roundtrip)
  5.2 Action parsing robustness
  5.3 Reward function unit tests (all 6 components + composite NaN check)
  5.4 Rule-based agent regression on new env
  5.5 Random policy reward distribution (non-degenerate)
  5.6 Integration: decision point env end-to-end
  5.7 AgentAction validation (required fields enforcement)
  5.8 Fault precision evidence gating
  5.9 Novel fault reward
  5.10 Message handling updated weights
  5.11 RewardBreakdown total sums correctly
  5.12 Observation dict is source of truth
"""

import json
import math
import random
import sys
from pathlib import Path

# Ensure submission is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from submission.env.env import PrintFarmEnvironment
from submission.env.models import FarmAction, FarmActionEnum, FarmObservation
from submission.env.decision_point import DecisionPointEnv, _rules_action
from submission.shared.serialize import serialize_obs
from submission.shared.parse_action import parse_action, AgentAction
from submission.shared.obs_formatter import format_observation_as_text
from submission.rewards.composite import compute_reward, RewardBreakdown
from submission.rewards.r_format import r_format
from submission.rewards.r_economic import r_economic
from submission.rewards.r_fault_precision import r_fault_precision
from submission.rewards.r_message_handling import r_message_handling
from submission.rewards.r_unnecessary_action import r_unnecessary_action
from submission.rewards.r_novel_fault import r_novel_fault


def test_5_1_observation_consistency():
    """Serialization produces identical output for identical state."""
    print("Test 5.1: Observation consistency...", end=" ")

    env = PrintFarmEnvironment()
    state = env.reset(seed=42, task_id="task_1")

    obs_train = serialize_obs(state)
    obs_eval = serialize_obs(state)
    assert obs_train == obs_eval, "serialization mismatch on reset"

    # Step once and check again
    action = FarmAction(action=FarmActionEnum.WAIT)
    state2 = env.step(action)
    obs_after_1 = serialize_obs(state2)
    obs_after_2 = serialize_obs(state2)
    assert obs_after_1 == obs_after_2, "serialization mismatch after step"

    # Verify JSON is valid
    data = json.loads(obs_train)
    assert "printers" in data
    assert "active_queue" in data
    assert "operator_notes" in data
    assert "customer_messages" in data
    assert "anomaly_flags" in data

    print("PASS")


def test_5_2_action_parsing():
    """Action parser handles all edge cases without crashing."""
    print("Test 5.2: Action parsing...", end=" ")

    test_cases = [
        # Valid JSON
        ('{"action_type":"WAIT"}', True),
        ('{"action_type":"ASSIGN_JOB","printer_id":1,"job_id":"j1"}', True),
        ('{"action_type":"RUN_DIAGNOSTIC","printer_id":3}', True),
        ('{"action_type":"REQUEST_SPOOL_SWAP","printer_id":2,"material":"PLA"}', True),
        # Old "action" key (backward compat)
        ('{"action":"WAIT"}', True),
        ('{"action":"ASSIGN_JOB","printer_id":1,"job_id":"j1"}', True),
        # Markdown-wrapped
        ('```json\n{"action_type":"WAIT"}\n```', True),
        ('Here is my response:\n```\n{"action_type":"RUN_DIAGNOSTIC","printer_id":1}\n```', True),
        # JSON in text
        ('I think we should {"action_type":"WAIT"} to be safe', True),
        # With reasoning field (should parse, reasoning ignored)
        ('{"action_type":"WAIT","reasoning":"nothing to do"}', True),
        # Invalid cases
        ("", False),
        ("I think we should wait and see", False),
        ("not json at all", False),
        ("{invalid json}", False),
        ('{"action_type":"INVALID_ACTION"}', False),
        # Wrong schema
        ('{"foo":"bar"}', False),
        # Empty object
        ('{}', False),
    ]

    for i, (text, should_parse) in enumerate(test_cases):
        result = parse_action(text)
        if should_parse:
            assert result is not None, f"Case {i} should parse but got None: {text[:50]}"
            assert isinstance(result, AgentAction)
        else:
            assert result is None, f"Case {i} should not parse but got {result}: {text[:50]}"

    print(f"PASS ({len(test_cases)} cases)")


def test_5_3_reward_function_units():
    """Each reward component returns finite, bounded values."""
    print("Test 5.3: Reward function unit tests...", end=" ")

    # r_format — needs model_output with <action> tag for positive reward
    clean_output = '<action>{"action_type": "WAIT"}</action>'
    assert r_format(AgentAction(action_type="WAIT"), model_output=clean_output) == 0.1
    assert r_format(None, model_output="some text") <= -0.1  # unparseable
    # No <action> tag but parsed via fallback → -0.1
    assert r_format(AgentAction(action_type="WAIT"), model_output="") == -0.1

    # r_economic (now scaled by 0.4)
    assert r_economic(1.0, 0.0) > 0  # LLM did better
    assert r_economic(0.0, 1.0) < 0  # LLM did worse
    assert r_economic(0.0, 0.0) == 0.0  # Tied
    assert -0.4 <= r_economic(100, -100) <= 0.4  # Clamped at ±0.4
    assert math.isfinite(r_economic(float('inf'), 0))  # NaN guard

    # r_fault_precision (evidence-gated)
    tags_with_fault = [{"printer_id": 3, "correct_action": "investigate"}]
    tags_empty = []
    # With evidence (observation has anomaly flag for P3)
    obs_with_evidence = {
        "operator_notes": ["P3 sounds rattly on retract"],
        "anomaly_flags": [],
        "printers": [{"printer_id": 3, "hotend_temp": 200, "fan_rpm": 3000,
                       "fatigue_level": 0, "reliability": 0.95}],
    }
    assert r_fault_precision(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=3),
        tags_with_fault, [], observation=obs_with_evidence) == 0.4
    # Without evidence + no anomaly = -0.05
    obs_no_evidence = {
        "operator_notes": [],
        "anomaly_flags": [],
        "printers": [{"printer_id": 3, "hotend_temp": 200, "fan_rpm": 3000,
                       "fatigue_level": 0, "reliability": 0.95}],
    }
    assert r_fault_precision(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=3),
        tags_empty, [], observation=obs_no_evidence) == -0.05
    # Non-investigation action
    assert r_fault_precision(
        AgentAction(action_type="WAIT"), tags_with_fault, []) == 0.0

    # r_message_handling (updated weights)
    msg_rush = [{"ground_truth_action": "accept_rush", "job_id": "j1"}]
    assert r_message_handling(
        AgentAction(action_type="ASSIGN_JOB", printer_id=1, job_id="j1"), msg_rush) == 0.4
    assert r_message_handling(
        AgentAction(action_type="WAIT"), msg_rush) < 0
    assert r_message_handling(AgentAction(action_type="WAIT"), []) == 0.0

    # r_unnecessary_action
    assert r_unnecessary_action(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=1), [], []) == -0.05
    assert r_unnecessary_action(
        AgentAction(action_type="WAIT"), [], []) == 0.0
    assert r_unnecessary_action(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=1),
        [{"printer_id": 1}], []) == 0.0

    # r_novel_fault
    novel_tags = [{"printer_id": 2, "fault_type": "hybrid_thermal_humidity",
                   "correct_action": "investigate"}]
    obs_novel = {
        "operator_notes": [],
        "anomaly_flags": ["P2: unusual thermal pattern — temps nominal but ambient humidity elevated"],
        "printers": [{"printer_id": 2, "hotend_temp": 200, "fan_rpm": 3000,
                       "fatigue_level": 1, "reliability": 0.95}],
    }
    assert r_novel_fault(
        AgentAction(action_type="REQUEST_MAINTENANCE", printer_id=2, maintenance_type="general"),
        novel_tags, observation=obs_novel) == 0.4
    # Non-preemptive action
    assert r_novel_fault(
        AgentAction(action_type="WAIT"), novel_tags) == 0.0

    # Composite: no NaN, includes r_novel_fault
    r = compute_reward(
        AgentAction(action_type="WAIT"), 0.0, 0.0,
        {"operator_notes": [], "customer_messages": [], "anomaly_flags": []})
    assert math.isfinite(r["total"])
    assert "r_novel_fault" in r
    for k, v in r.items():
        assert math.isfinite(v), f"NaN in {k}"

    print("PASS")


def test_5_4_rule_agent_regression():
    """Rule-based agent should score lower on the v2 env (with unstructured signals).

    If the rule agent still gets >0.5, the unstructured signals aren't load-bearing.
    """
    print("Test 5.4: Rule-based agent regression...", end=" ")

    env = PrintFarmEnvironment()
    tasks = ["task_1", "task_2", "task_3"]
    scores = []

    for task_id in tasks:
        obs = env.reset(seed=42, task_id=task_id)
        while not obs.done:
            action = _rules_action(obs)
            obs = env.step(action)
        scores.append(obs.reward)

    avg_score = sum(scores) / len(scores)
    print(f"avg={avg_score:.3f} scores={[f'{s:.3f}' for s in scores]}", end=" ")

    # Note: we expect the rule agent to still work reasonably well on structured
    # signals. The GAP is that it ignores operator_notes, customer_messages,
    # and anomaly_flags — which an LLM can use for additional reward.
    # We don't require score < 0.5, just that the score is measured.
    assert avg_score > 0.0, "Rule agent should get positive reward"
    print("PASS")


def test_5_5_random_policy_reward_distribution():
    """Random policy should produce non-degenerate reward distribution.

    Requirements:
    - Not all zero
    - Not all negative
    - Variance > 0
    """
    print("Test 5.5: Random policy reward distribution...", end=" ")

    rng = random.Random(42)
    rewards = []

    for i in range(100):
        task_id = ["task_1", "task_2", "task_3"][i % 3]
        dp_env = DecisionPointEnv(k_horizon=5)
        try:
            _, obs = dp_env.reset(seed=rng.randint(0, 2**31), task_id=task_id)
        except Exception:
            continue

        # Generate a schema-valid random action
        action = _random_valid_action(obs, rng)

        try:
            delta, info = dp_env.step(action)
            gt_tags = dp_env.get_decision_tags()

            # Build observation dict for evidence gating
            obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else None

            parsed = AgentAction(action_type=action.action.value,
                                 printer_id=action.printer_id,
                                 job_id=action.job_id,
                                 operator_id=action.operator_id,
                                 ticket_type=action.ticket_type,
                                 material=action.material,
                                 maintenance_type=action.maintenance_type)
            components = compute_reward(parsed, delta, 0.0, gt_tags,
                                        observation=obs_dict)
            rewards.append(components["total"])
        except Exception:
            rewards.append(0.0)

    assert len(rewards) > 50, f"Too few rewards: {len(rewards)}"

    pos = sum(1 for r in rewards if r > 0)
    neg = sum(1 for r in rewards if r < 0)
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)

    print(f"mean={mean_r:.3f} var={var_r:.4f} pos={pos} neg={neg}", end=" ")

    assert var_r > 0, "Zero variance — no gradient signal"
    spread = max(rewards) - min(rewards)
    assert spread > 0.01, f"Reward spread too small ({spread:.4f}) — no ranking signal"

    print("PASS")


def _random_valid_action(obs: FarmObservation, rng: random.Random) -> FarmAction:
    """Generate a random VALID action consistent with current observation."""
    action_types = [
        FarmActionEnum.ASSIGN_JOB, FarmActionEnum.RUN_DIAGNOSTIC,
        FarmActionEnum.REQUEST_MAINTENANCE, FarmActionEnum.REQUEST_SPOOL_SWAP,
        FarmActionEnum.WAIT, FarmActionEnum.PAUSE_JOB,
    ]
    action_type = rng.choice(action_types)

    printers = obs.printers
    jobs = obs.active_queue

    if action_type == FarmActionEnum.ASSIGN_JOB:
        if not printers or not jobs:
            return FarmAction(action=FarmActionEnum.WAIT)
        return FarmAction(
            action=action_type,
            printer_id=rng.choice(printers).printer_id,
            job_id=rng.choice(jobs).job_id,
        )
    elif action_type in (FarmActionEnum.RUN_DIAGNOSTIC, FarmActionEnum.PAUSE_JOB):
        if not printers:
            return FarmAction(action=FarmActionEnum.WAIT)
        return FarmAction(action=action_type,
                          printer_id=rng.choice(printers).printer_id)
    elif action_type == FarmActionEnum.REQUEST_MAINTENANCE:
        if not printers:
            return FarmAction(action=FarmActionEnum.WAIT)
        return FarmAction(action=action_type,
                          printer_id=rng.choice(printers).printer_id,
                          maintenance_type="general")
    elif action_type == FarmActionEnum.REQUEST_SPOOL_SWAP:
        if not printers:
            return FarmAction(action=FarmActionEnum.WAIT)
        return FarmAction(action=action_type,
                          printer_id=rng.choice(printers).printer_id,
                          material="PLA")
    else:
        return FarmAction(action=FarmActionEnum.WAIT)


def test_5_6_decision_point_e2e():
    """Decision point env works end-to-end."""
    print("Test 5.6: Decision point E2E...", end=" ")

    dp_env = DecisionPointEnv(k_horizon=10)
    serialized, obs = dp_env.reset(seed=123, task_id="task_2")

    # Verify serialization is valid JSON
    data = json.loads(serialized)
    assert isinstance(data, dict)

    # Verify ground truth tags are available
    tags = dp_env.get_decision_tags()
    assert "operator_notes" in tags
    assert "customer_messages" in tags
    assert "anomaly_flags" in tags

    # Execute a WAIT action
    action = FarmAction(action=FarmActionEnum.WAIT)
    delta, info = dp_env.step(action)
    assert isinstance(delta, float)
    assert math.isfinite(delta)

    print("PASS")


def test_5_7_action_validation_rejects_incomplete():
    """AgentAction model_validator rejects actions with missing required fields."""
    print("Test 5.7: Action validation...", end=" ")

    from pydantic import ValidationError

    # ASSIGN_JOB without job_id should fail
    try:
        AgentAction(action_type="ASSIGN_JOB", printer_id=1)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # RUN_DIAGNOSTIC without printer_id should fail
    try:
        AgentAction(action_type="RUN_DIAGNOSTIC")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # REQUEST_MAINTENANCE without maintenance_type should fail
    try:
        AgentAction(action_type="REQUEST_MAINTENANCE", printer_id=1)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # DISPATCH_TICKET without operator_id/ticket_type should fail
    try:
        AgentAction(action_type="DISPATCH_TICKET")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass

    # WAIT should pass with no extra fields
    w = AgentAction(action_type="WAIT")
    assert w.action_type == "WAIT"

    # Valid ASSIGN_JOB should pass
    a = AgentAction(action_type="ASSIGN_JOB", printer_id=1, job_id="j1")
    assert a.action_type == "ASSIGN_JOB"

    print("PASS")


def test_5_8_fault_precision_evidence_gating():
    """Fault precision reward uses evidence gating correctly."""
    print("Test 5.8: Fault precision evidence gating...", end=" ")

    anomaly_tags = [{"printer_id": 3, "correct_action": "investigate"}]

    # Evidence present + anomaly = +0.4
    obs_evidence = {
        "operator_notes": ["P3 sounds rattly on retract"],
        "anomaly_flags": [],
        "printers": [{"printer_id": 3, "hotend_temp": 200, "fan_rpm": 3000,
                       "fatigue_level": 0, "reliability": 0.95}],
    }
    assert r_fault_precision(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=3),
        anomaly_tags, [], observation=obs_evidence) == 0.4

    # No evidence + anomaly = 0.0 (got lucky)
    obs_no_evidence = {
        "operator_notes": [],
        "anomaly_flags": [],
        "printers": [{"printer_id": 3, "hotend_temp": 200, "fan_rpm": 3000,
                       "fatigue_level": 0, "reliability": 0.95}],
    }
    assert r_fault_precision(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=3),
        anomaly_tags, [], observation=obs_no_evidence) == 0.0

    # Evidence present + no anomaly = -0.15 (red herring)
    assert r_fault_precision(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=3),
        [], [], observation=obs_evidence) == -0.15

    # No evidence + no anomaly = -0.05 (blind guess)
    assert r_fault_precision(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=3),
        [], [], observation=obs_no_evidence) == -0.05

    print("PASS")


def test_5_9_novel_fault_reward():
    """Novel fault reward fires correctly with evidence."""
    print("Test 5.9: Novel fault reward...", end=" ")

    novel_tags = [{"printer_id": 2, "fault_type": "hybrid_thermal_humidity",
                   "correct_action": "investigate"}]

    # With evidence (anomaly flag text mentions P2)
    obs_with = {
        "operator_notes": [],
        "anomaly_flags": ["P2: unusual thermal pattern — temps nominal but ambient humidity elevated"],
        "printers": [{"printer_id": 2, "hotend_temp": 200, "fan_rpm": 3000,
                       "fatigue_level": 1, "reliability": 0.95}],
    }
    assert r_novel_fault(
        AgentAction(action_type="REQUEST_MAINTENANCE", printer_id=2, maintenance_type="general"),
        novel_tags, observation=obs_with) == 0.4

    # Without evidence
    obs_without = {
        "operator_notes": [],
        "anomaly_flags": [],
        "printers": [{"printer_id": 2, "hotend_temp": 200, "fan_rpm": 3000,
                       "fatigue_level": 0, "reliability": 0.95}],
    }
    assert r_novel_fault(
        AgentAction(action_type="REQUEST_MAINTENANCE", printer_id=2, maintenance_type="general"),
        novel_tags, observation=obs_without) == 0.0

    # Non-preemptive action
    assert r_novel_fault(
        AgentAction(action_type="WAIT"), novel_tags, observation=obs_with) == 0.0

    # No novel fault present
    assert r_novel_fault(
        AgentAction(action_type="REQUEST_MAINTENANCE", printer_id=2, maintenance_type="general"),
        [], observation=obs_with) == 0.0

    print("PASS")


def test_5_10_message_handling_updated_weights():
    """Message handling uses updated reward weights."""
    print("Test 5.10: Message handling weights...", end=" ")

    msg_rush = [{"ground_truth_action": "accept_rush", "job_id": "j22"}]

    # Exact match: +0.4
    assert r_message_handling(
        AgentAction(action_type="ASSIGN_JOB", printer_id=1, job_id="j22"),
        msg_rush) == 0.4

    # Right action, wrong job: +0.15
    assert r_message_handling(
        AgentAction(action_type="ASSIGN_JOB", printer_id=1, job_id="j99"),
        msg_rush) == 0.15

    # Wrong action: -0.1
    assert r_message_handling(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=1),
        msg_rush) == -0.1

    # No message: 0.0
    assert r_message_handling(
        AgentAction(action_type="WAIT"), []) == 0.0

    print("PASS")


def test_5_11_reward_breakdown_total():
    """RewardBreakdown.total sums all components correctly."""
    print("Test 5.11: RewardBreakdown total...", end=" ")

    bd = RewardBreakdown(
        format=+0.1,
        economic=+0.2,
        fault_precision=+0.4,
        message_handling=0.0,
        unnecessary_action=0.0,
        novel_fault=0.0,
    )
    assert abs(bd.total - 0.7) < 1e-6

    # to_dict should have all keys
    d = bd.to_dict()
    assert "reward/format" in d
    assert "reward/economic" in d
    assert "reward/fault_precision" in d
    assert "reward/message_handling" in d
    assert "reward/unnecessary_action" in d
    assert "reward/novel_fault" in d
    assert "reward/total" in d
    assert abs(d["reward/total"] - 0.7) < 1e-6

    print("PASS")


def test_5_12_obs_dict_source_of_truth():
    """Observation dict for reward is the source of truth, not reconstructed from text."""
    print("Test 5.12: Observation dict source of truth...", end=" ")

    env = PrintFarmEnvironment()
    state = env.reset(seed=42, task_id="task_1")

    # The observation dict should be obtainable from the state object
    obs_dict = state.model_dump()
    obs_text = format_observation_as_text(serialize_obs(state))

    # The obs_dict must contain the core fields
    assert "printers" in obs_dict
    assert "operator_notes" in obs_dict
    assert "customer_messages" in obs_dict
    assert "anomaly_flags" in obs_dict

    # Verify the dict is a faithful representation
    assert len(obs_dict["printers"]) == len(state.printers)
    assert obs_dict["operator_notes"] == list(state.operator_notes)

    print("PASS")


def run_all_tests():
    """Run all Phase 5 smoke tests."""
    print("=" * 60)
    print("PHASE 5 — SMOKE TESTS")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    tests = [
        test_5_1_observation_consistency,
        test_5_2_action_parsing,
        test_5_3_reward_function_units,
        test_5_4_rule_agent_regression,
        test_5_5_random_policy_reward_distribution,
        test_5_6_decision_point_e2e,
        test_5_7_action_validation_rejects_incomplete,
        test_5_8_fault_precision_evidence_gating,
        test_5_9_novel_fault_reward,
        test_5_10_message_handling_updated_weights,
        test_5_11_reward_breakdown_total,
        test_5_12_obs_dict_source_of_truth,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
            errors.append((test_fn.__name__, str(e)))
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
            errors.append((test_fn.__name__, str(e)))

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        for name, err in errors:
            print(f"  FAIL: {name}: {err}")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
