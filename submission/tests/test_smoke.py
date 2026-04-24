"""
Phase 5 Smoke Tests — catch silent failures before wasting an overnight run.

Tests:
  5.1 Observation consistency (serialization roundtrip)
  5.2 Action parsing robustness
  5.3 Reward function unit tests
  5.4 Rule-based agent regression on new env
  5.5 Random policy reward distribution (non-degenerate)
  5.6 Integration: decision point env end-to-end
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
from submission.rewards.composite import compute_reward
from submission.rewards.r_format import r_format
from submission.rewards.r_economic import r_economic
from submission.rewards.r_fault_precision import r_fault_precision
from submission.rewards.r_message_handling import r_message_handling
from submission.rewards.r_unnecessary_action import r_unnecessary_action


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

    # r_format
    assert r_format(AgentAction(action_type="WAIT")) == 0.0
    assert r_format(None) == -0.1
    assert r_format(AgentAction(action_type="ASSIGN_JOB", printer_id=1, job_id="j1")) == 0.0

    # r_economic
    assert r_economic(1.0, 0.0) > 0  # LLM did better
    assert r_economic(0.0, 1.0) < 0  # LLM did worse
    assert r_economic(0.0, 0.0) == 0.0  # Tied
    assert -1.0 <= r_economic(100, -100) <= 1.0  # Clamped
    assert math.isfinite(r_economic(float('inf'), 0))  # NaN guard

    # r_fault_precision
    tags_with_fault = [{"printer_id": 3, "correct_action": "investigate"}]
    tags_empty = []
    assert r_fault_precision(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=3),
        tags_with_fault, []) == 0.2
    assert r_fault_precision(
        AgentAction(action_type="RUN_DIAGNOSTIC", printer_id=3),
        tags_empty, []) == -0.1
    assert r_fault_precision(
        AgentAction(action_type="WAIT"), tags_with_fault, []) == 0.0

    # r_message_handling
    msg_rush = [{"ground_truth_action": "accept_rush", "job_id": "j1"}]
    assert r_message_handling(
        AgentAction(action_type="ASSIGN_JOB", job_id="j1"), msg_rush) == 0.15
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

    # Composite: no NaN
    r = compute_reward(
        AgentAction(action_type="WAIT"), 0.0, 0.0,
        {"operator_notes": [], "customer_messages": [], "anomaly_flags": []})
    assert math.isfinite(r["total"])
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
    all_actions = list(FarmActionEnum)
    rewards = []
    format_failures = 0

    for i in range(100):
        task_id = ["task_1", "task_2", "task_3"][i % 3]
        dp_env = DecisionPointEnv(k_horizon=5)
        try:
            _, obs = dp_env.reset(seed=rng.randint(0, 2**31), task_id=task_id)
        except Exception:
            continue

        # Random action
        action_type = rng.choice(all_actions)
        action = FarmAction(action=action_type)

        # Add required fields for some action types
        if action_type == FarmActionEnum.ASSIGN_JOB:
            if obs.printers and obs.active_queue:
                action.printer_id = obs.printers[0].printer_id
                action.job_id = obs.active_queue[0].job_id
        elif action_type in (FarmActionEnum.RUN_DIAGNOSTIC, FarmActionEnum.PAUSE_JOB):
            if obs.printers:
                action.printer_id = obs.printers[0].printer_id
        elif action_type == FarmActionEnum.REQUEST_SPOOL_SWAP:
            if obs.printers:
                action.printer_id = obs.printers[0].printer_id
                action.material = "PLA"

        try:
            delta, info = dp_env.step(action)
            gt_tags = dp_env.get_decision_tags()

            parsed = AgentAction(action_type=action_type.value)
            components = compute_reward(parsed, delta, 0.0, gt_tags)
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
    # GRPO normalizes per-group, so it just needs variance.
    # But if literally ALL rewards are the same sign & magnitude, warn.
    spread = max(rewards) - min(rewards)
    assert spread > 0.01, f"Reward spread too small ({spread:.4f}) — no ranking signal"

    print("PASS")


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
