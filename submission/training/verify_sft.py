"""
verify_sft.py — Quick eyeball test for the SFT warm-start adapter.

Generates 10 fresh prompts (NOT in the SFT training set), runs the warm-started
model on them, and checks whether the model learned the Safety > Service
hierarchy. The test passes if:

  - On TRUE-POSITIVE prompts (real anomaly visible): ≥80% of outputs are
    investigation actions (RUN_DIAGNOSTIC / REQUEST_MAINTENANCE).
  - On FALSE-POSITIVE prompts (resolved-alarm decoy): ≥70% of outputs are
    NON-investigation (the model correctly ignores the decoy).
  - Overall action diversity: ≥3 distinct action types across the 10 prompts.

If any criterion fails, the SFT data needs work — fix it before GRPO.

Usage:
    python -m submission.training.verify_sft \
        --adapter grpo_runs/sft_warm/final_adapter \
        --base google/gemma-3-1b-it \
        --n 10
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

_here = Path(__file__).resolve().parent
for _candidate in [_here.parent, _here.parent.parent]:
    if (_candidate / "submission" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from submission.training.build_sft_dataset import (
    classify_scenario, oracle_action,
)
from submission.training.rollout import _compress_obs_json
from submission.shared.obs_formatter import format_observation_as_text
from submission.shared.parse_action import parse_action, _ACTION_TAG_RE
from submission.shared.prompt import SYSTEM_PROMPT
from submission.env.decision_point import (
    DecisionPointEnv, K_HORIZON, SIGNAL_TYPES, signal_present,
)


INVESTIGATION_ACTIONS = {"RUN_DIAGNOSTIC", "REQUEST_MAINTENANCE", "DISPATCH_TICKET"}


GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; BOLD = "\033[1m"; RESET = "\033[0m"


def parse_args():
    p = argparse.ArgumentParser(description="Verify SFT warm-start adapter")
    p.add_argument("--adapter", required=True, help="Path to SFT adapter")
    p.add_argument("--base", default="google/gemma-3-1b-it", help="Base model")
    p.add_argument("--n", type=int, default=10, help="Number of test prompts")
    p.add_argument("--seed_offset", type=int, default=999_000)
    p.add_argument("--max_new_tokens", type=int, default=80)
    return p.parse_args()


def generate_test_prompts(n: int, seed_offset: int):
    """Generate prompts NOT seen during SFT (different seed range)."""
    import random
    rng = random.Random(seed_offset)
    prompts = []
    sig_idx = 0
    attempts = 0
    while len(prompts) < n and attempts < n * 30:
        attempts += 1
        target_signal = SIGNAL_TYPES[sig_idx % len(SIGNAL_TYPES)]
        sig_idx += 1
        ep_seed = rng.randint(0, 2**31)
        env = DecisionPointEnv(k_horizon=K_HORIZON)
        try:
            serialized, obs = env.reset(
                seed=ep_seed, task_id=f"task_{(attempts % 5) + 1}",
                target_signal=target_signal,
            )
        except Exception:
            continue
        if not signal_present(obs, target_signal):
            continue
        gt = env.get_decision_tags()
        scenario = classify_scenario(gt)
        if scenario == "other":
            continue
        compressed = _compress_obs_json(serialized)
        obs_text = format_observation_as_text(compressed)
        prompts.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current state:\n{obs_text}"},
            ],
            "scenario": scenario,
            "expected_action": oracle_action(obs, gt),
        })
    return prompts


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print(f"{BOLD}SFT Verification — adapter @ {args.adapter}{RESET}")
    print(f"{'='*70}\n")

    # Load model + adapter
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif is_mps:
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map=device)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    print(f"Loaded base + adapter on {device}")

    # Generate test prompts
    test_prompts = generate_test_prompts(args.n, args.seed_offset)
    print(f"Generated {len(test_prompts)} test prompts (seed_offset={args.seed_offset})\n")

    by_scenario = Counter()
    correct_by_scenario = Counter()
    investigation_by_scenario = Counter()
    actions = Counter()
    parse_ok = 0

    for i, p in enumerate(test_prompts):
        prompt_str = tokenizer.apply_chat_template(
            p["messages"], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                temperature=0.3, do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        parsed = parse_action(response)
        action_type = parsed.action_type if parsed else "INVALID"
        actions[action_type] += 1
        scenario = p["scenario"]
        by_scenario[scenario] += 1
        if parsed is not None:
            parse_ok += 1
        if action_type in INVESTIGATION_ACTIONS:
            investigation_by_scenario[scenario] += 1
        # Correctness: scenario-specific
        if scenario == "true_positive" and action_type in INVESTIGATION_ACTIONS:
            correct_by_scenario[scenario] += 1
        elif scenario == "false_positive" and action_type not in INVESTIGATION_ACTIONS:
            correct_by_scenario[scenario] += 1
        elif scenario == "true_negative" and action_type not in INVESTIGATION_ACTIONS:
            correct_by_scenario[scenario] += 1

        flag = GREEN + "✓" + RESET if (
            (scenario == "true_positive" and action_type in INVESTIGATION_ACTIONS) or
            (scenario != "true_positive" and action_type not in INVESTIGATION_ACTIONS)
        ) else RED + "✗" + RESET
        expected_act = p["expected_action"]["action_type"] if p["expected_action"] else "—"
        print(f"  [{i+1:2d}] {flag} {scenario:15s} expected={expected_act:20s} got={action_type:20s}")
        if parsed is None:
            print(f"        {YELLOW}(unparseable: {response[:80]!r}){RESET}")

    # Summary
    print(f"\n{'─'*70}")
    print(f"{BOLD}Results{RESET}")
    print(f"{'─'*70}")
    n = len(test_prompts)
    print(f"Parse rate:       {parse_ok}/{n}  ({parse_ok/max(n,1)*100:.0f}%)")
    print(f"Action diversity: {len(actions)} unique  ({dict(actions.most_common())})")

    tp_rate = correct_by_scenario["true_positive"] / max(by_scenario["true_positive"], 1)
    tn_rate = correct_by_scenario["true_negative"] / max(by_scenario["true_negative"], 1)
    fp_rate = correct_by_scenario["false_positive"] / max(by_scenario["false_positive"], 1)

    def colour(rate, threshold):
        return GREEN if rate >= threshold else (YELLOW if rate >= threshold - 0.2 else RED)

    print(f"\nPer-scenario correctness:")
    print(f"  TRUE_POSITIVE  (real anomaly → investigate):  {colour(tp_rate, 0.8)}{correct_by_scenario['true_positive']}/{by_scenario['true_positive']} = {tp_rate*100:.0f}%{RESET}  (target ≥80%)")
    print(f"  TRUE_NEGATIVE  (msg only → respond/wait):     {colour(tn_rate, 0.7)}{correct_by_scenario['true_negative']}/{by_scenario['true_negative']} = {tn_rate*100:.0f}%{RESET}  (target ≥70%)")
    print(f"  FALSE_POSITIVE (decoy → ignore):              {colour(fp_rate, 0.7)}{correct_by_scenario['false_positive']}/{by_scenario['false_positive']} = {fp_rate*100:.0f}%{RESET}  (target ≥70%)")

    # Verdict
    pass_tp = tp_rate >= 0.80
    pass_tn = tn_rate >= 0.70
    pass_fp = fp_rate >= 0.70
    pass_div = len(actions) >= 3
    pass_parse = parse_ok / max(n, 1) >= 0.8

    print(f"\n{BOLD}VERDICT:{RESET}")
    print(f"  TP rate ≥ 80%:        {GREEN if pass_tp else RED}{'PASS' if pass_tp else 'FAIL'}{RESET}")
    print(f"  TN rate ≥ 70%:        {GREEN if pass_tn else RED}{'PASS' if pass_tn else 'FAIL'}{RESET}")
    print(f"  FP rate ≥ 70%:        {GREEN if pass_fp else RED}{'PASS' if pass_fp else 'FAIL'}{RESET}")
    print(f"  Action diversity ≥ 3: {GREEN if pass_div else RED}{'PASS' if pass_div else 'FAIL'}{RESET}")
    print(f"  Parse rate ≥ 80%:     {GREEN if pass_parse else RED}{'PASS' if pass_parse else 'FAIL'}{RESET}")

    overall = pass_tp and pass_tn and pass_fp and pass_div and pass_parse
    if overall:
        print(f"\n  {GREEN}{BOLD}✓ READY FOR GRPO{RESET}")
        sys.exit(0)
    else:
        print(f"\n  {RED}{BOLD}✗ NOT READY — improve SFT data and re-train{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
