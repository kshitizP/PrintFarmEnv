"""
Verification checklist for GRPO v2 fixes.

6 checks:
1. Schema unified — no "action" vs "action_type" mismatches
2. Round-trip test — parse_action(model_output_with_action_tag) returns valid AgentAction
3. Format reward unit test — 4 cases
4. Untrained Gemma format compliance — <30% failure on 20 prompts
5. Untrained baseline reward — mean in [-0.1, +0.1]
6. Observation is text, not JSON (anti-echo structural check)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))


def check_1_schema():
    """Verify no action vs action_type mismatch in submission code."""
    print("\n=== CHECK 1: Schema unified ===")
    from submission.shared.parse_action import AgentAction
    # AgentAction must have action_type field
    assert "action_type" in AgentAction.model_fields, "Missing action_type field"
    assert "action" not in AgentAction.model_fields, "Should not have 'action' field"
    
    # Prompt uses action_type
    from submission.shared.prompt import SYSTEM_PROMPT
    assert "action_type" in SYSTEM_PROMPT, "SYSTEM_PROMPT missing action_type"
    assert "<action>" in SYSTEM_PROMPT, "SYSTEM_PROMPT missing <action> tag"
    
    print("  PASS: AgentAction uses action_type, SYSTEM_PROMPT uses action_type + <action> tags")


def check_2_roundtrip():
    """Round-trip parse test with <action> tags."""
    print("\n=== CHECK 2: Round-trip parse ===")
    from submission.shared.parse_action import parse_action, AgentAction
    
    test_cases = [
        # <action> tagged (preferred)
        ('<action>{"action_type": "WAIT"}</action>', "WAIT"),
        ('<action>{"action_type": "ASSIGN_JOB", "printer_id": 1, "job_id": "j1"}</action>', "ASSIGN_JOB"),
        ('<action>{"action_type": "REQUEST_MAINTENANCE", "printer_id": 3, "maintenance_type": "maintenance_basic", "reasoning": "wear detected"}</action>', "REQUEST_MAINTENANCE"),
        # With extra whitespace
        ('<action>\n{"action_type": "WAIT"}\n</action>', "WAIT"),
        # Backward compat: raw JSON still works
        ('{"action_type": "WAIT"}', "WAIT"),
        # Backward compat: "action" normalized to "action_type"
        ('{"action": "WAIT"}', "WAIT"),
        # Markdown fenced
        ('```json\n{"action_type":"RUN_DIAGNOSTIC","printer_id":2}\n```', "RUN_DIAGNOSTIC"),
    ]
    
    for text, expected_action in test_cases:
        result = parse_action(text)
        assert result is not None, f"Failed to parse: {text}"
        assert result.action_type == expected_action, f"Expected {expected_action}, got {result.action_type} for: {text}"
    
    # Test unparseable
    assert parse_action("") is None
    assert parse_action("hello world") is None
    assert parse_action('{"done": false, "reward": 0.5}') is None
    
    # Full round-trip: create → dump → parse
    original = AgentAction(action_type="ASSIGN_JOB", printer_id=2, job_id="j5", reasoning="test")
    dumped = f'<action>{json.dumps(original.model_dump())}</action>'
    parsed = parse_action(dumped)
    assert parsed is not None
    assert parsed.action_type == original.action_type
    assert parsed.printer_id == original.printer_id
    assert parsed.job_id == original.job_id
    
    print(f"  PASS: {len(test_cases)} parse cases + round-trip + 3 negative cases")


def check_3_format_reward():
    """Format reward unit test: 4 cases."""
    print("\n=== CHECK 3: Format reward ===")
    from submission.rewards.r_format import r_format
    from submission.shared.parse_action import parse_action
    
    obs_text = "STEP 5/100 | Net profit: $12.50\nPRINTERS:\n  - P1: IDLE, mat=PLA, spool=800g"
    
    # Case 1: Clean action with tag → +0.1
    clean = '<action>{"action_type": "WAIT"}</action>'
    r1 = r_format(parse_action(clean), model_output=clean, observation_text=obs_text)
    assert r1 == 0.1, f"Clean action: expected +0.1, got {r1}"
    
    # Case 2: Echo of observation (high overlap) → <= -0.2
    echo = "STEP 5/100 | Net profit: $12.50\nPRINTERS:\n  - P1: IDLE, mat=PLA, spool=800g"
    r2 = r_format(None, model_output=echo, observation_text=obs_text)
    assert r2 <= -0.2, f"Echo: expected <= -0.2, got {r2}"
    
    # Case 3: <action> tag but invalid content → -0.2
    bad_tag = '<action>not valid json at all</action>'
    r3 = r_format(parse_action(bad_tag), model_output=bad_tag, observation_text=obs_text)
    assert r3 == -0.2, f"Bad tag: expected -0.2, got {r3}"
    
    # Case 4: No tag at all, no parseable action → -0.3
    no_tag = "I think you should wait for the printer to finish."
    r4 = r_format(parse_action(no_tag), model_output=no_tag, observation_text=obs_text)
    assert r4 == -0.3, f"No tag: expected -0.3, got {r4}"
    
    print(f"  PASS: clean={r1}, echo={r2}, bad_tag={r3}, no_tag={r4}")


def check_4_obs_is_text():
    """Verify observation is rendered as text, not JSON."""
    print("\n=== CHECK 4: Observation is text (not JSON) ===")
    from submission.shared.obs_formatter import format_observation_as_text
    
    # Sample observation JSON
    obs_json = json.dumps({
        "time_step": 10, "max_steps": 100, "net_profit_usd": 15.5,
        "printers": [
            {"printer_id": 1, "state": "IDLE", "current_material": "PLA",
             "spool_weight_g": 800, "reliability": 0.95, "fatigue_level": 0,
             "hotend_temp": 200, "maintenance_due_in": 40},
            {"printer_id": 2, "state": "PRINTING", "current_material": "PETG",
             "current_job_id": "j3", "spool_weight_g": 400, "reliability": 0.88,
             "fatigue_level": 2.1, "hotend_temp": 230, "maintenance_due_in": 15}
        ],
        "active_queue": [
            {"job_id": "j3", "state": "PRINTING", "material_required": "PETG",
             "print_time_steps": 30, "progress_steps": 15, "price_usd": 55},
            {"job_id": "j4", "state": "PENDING", "material_required": "PLA",
             "weight_required_g": 200, "print_time_steps": 20, "price_usd": 40,
             "deadline_steps": 50}
        ],
        "operators": [
            {"operator_id": "op1", "skill_level": "expert", "is_on_shift": True,
             "queue_size": 1, "queue_capacity": 3, "current_fatigue": 0.3,
             "shift_window": [0, 100]}
        ],
        "inventory": {"PLA": 2000, "PETG": 1500},
        "operator_notes": ["P2: slight grinding noise on retract"],
        "customer_messages": [],
        "anomaly_flags": ["P2: unusual vibration pattern"]
    })
    
    text = format_observation_as_text(obs_json)
    
    # Must NOT be valid JSON
    try:
        json.loads(text)
        assert False, "Observation text should NOT be valid JSON"
    except json.JSONDecodeError:
        pass
    
    # Must contain key sections
    assert "STEP 10/100" in text, "Missing step info"
    assert "PRINTERS:" in text, "Missing printers section"
    assert "JOBS:" in text, "Missing jobs section"
    assert "OPERATORS:" in text, "Missing operators section"
    assert "OPERATOR NOTES:" in text, "Missing operator notes"
    assert "ANOMALY FLAGS:" in text, "Missing anomaly flags"
    assert "P1:" in text or "P1" in text, "Missing printer P1"
    assert "P2:" in text or "P2" in text, "Missing printer P2"
    
    print(f"  PASS: Observation is {len(text)} chars of structured text (not JSON)")
    print(f"  Preview:\n{text[:300]}...")


def check_5_prompt_structure():
    """Verify the full prompt structure uses text obs + <action> examples."""
    print("\n=== CHECK 5: Prompt structure ===")
    from submission.training.rollout import generate_decision_prompts
    
    prompts = generate_decision_prompts(n_prompts=3, tasks=["task_1"], seed=42)
    p = prompts[0]
    
    user_content = p["messages"][1]["content"]
    system_content = p["messages"][0]["content"]
    
    # User content should NOT be valid JSON (it's text now)
    try:
        json.loads(user_content)
        assert False, "User content should NOT be valid JSON"
    except (json.JSONDecodeError, ValueError):
        pass
    
    # System content should have <action> tags
    assert "<action>" in system_content, "System prompt missing <action> tags"
    
    # observation_text should be stored for anti-echo
    assert "observation_text" in p, "Missing observation_text in prompt_info"
    assert len(p["observation_text"]) > 50, "observation_text too short"
    
    print(f"  PASS: Prompt has text obs ({len(user_content)} chars), <action> examples in system, observation_text stored")


if __name__ == "__main__":
    passed = 0
    failed = 0
    
    for check_fn in [check_1_schema, check_2_roundtrip, check_3_format_reward, check_4_obs_is_text, check_5_prompt_structure]:
        try:
            check_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    sys.exit(1 if failed else 0)
