"""
Evaluate an LLM checkpoint on the PrintFarmEnv decision points.

Usage:
    # Evaluate untrained Gemma baseline
    python -m submission.eval.eval_model --model google/gemma-3-1b-it

    # Evaluate a trained LoRA adapter
    python -m submission.eval.eval_model --model google/gemma-3-1b-it --adapter ./grpo_runs/latest/final_adapter

    # Quick eval
    python -m submission.eval.eval_model --model google/gemma-3-1b-it --n_episodes 5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

_here = Path(__file__).resolve().parent
for _candidate in [_here.parent, _here.parent.parent]:
    if (_candidate / "submission" / "__init__.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from submission.env.decision_point import DecisionPointEnv, _rules_action
from submission.env.models import FarmAction, FarmActionEnum
from submission.shared.serialize import serialize_obs
from submission.shared.parse_action import parse_action, action_to_farm_action
from submission.shared.prompt import SYSTEM_PROMPT
from submission.rewards.composite import compute_reward


TASKS = ["task_1", "task_2", "task_3", "task_4", "task_5"]


def load_model(model_name, adapter_path=None):
    """Load model + tokenizer, optionally with LoRA adapter."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path or model_name,
            max_seq_length=3500,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print(f"Loaded with Unsloth: {model_name}" +
              (f" + adapter {adapter_path}" if adapter_path else ""))
        return model, tokenizer, "unsloth"
    except ImportError:
        pass

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"Loaded adapter from {adapter_path}")

    model.eval()
    print(f"Loaded: {model_name} on {device} (dtype={dtype})")
    return model, tokenizer, "transformers"


def generate_action(model, tokenizer, obs_text, backend="transformers"):
    """Generate an action from the model given observation text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current State:\n{obs_text}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return completion


def eval_model(model, tokenizer, backend, n_episodes=20, seeds=None):
    """Evaluate model on decision points."""
    if seeds is None:
        seeds = list(range(42, 42 + n_episodes))

    results = []
    total_components = defaultdict(float)
    action_dist = defaultdict(int)
    format_failures = 0

    for ep in range(n_episodes):
        task_id = TASKS[ep % len(TASKS)]
        seed = seeds[ep % len(seeds)]

        dp_env = DecisionPointEnv(k_horizon=10)
        try:
            serialized, obs = dp_env.reset(seed=seed, task_id=task_id)
        except Exception as e:
            print(f"  Episode {ep}: reset failed: {e}")
            continue

        gt_tags = dp_env.get_decision_tags()

        # Generate action
        t0 = time.time()
        completion = generate_action(model, tokenizer, serialized, backend)
        gen_time = time.time() - t0

        parsed = parse_action(completion)
        if parsed is None:
            format_failures += 1
            farm_action = FarmAction(action=FarmActionEnum.WAIT)
        else:
            farm_action = action_to_farm_action(parsed)
            action_dist[parsed.action_type] += 1

        # LLM step
        llm_delta, _ = dp_env.step(farm_action)

        # Rules counterfactual
        dp_env2 = DecisionPointEnv(k_horizon=10)
        _, obs2 = dp_env2.reset(seed=seed, task_id=task_id)
        rules_delta, _ = dp_env2.step(_rules_action(obs2))

        components = compute_reward(parsed, llm_delta, rules_delta, gt_tags)
        for k, v in components.items():
            total_components[k] += v

        results.append({
            "episode": ep,
            "task_id": task_id,
            "seed": seed,
            "completion": completion[:200],
            "parsed": parsed.model_dump() if parsed else None,
            "reward": components["total"],
            "components": components,
            "gen_time": gen_time,
        })

        action_str = parsed.action_type if parsed else "PARSE_FAIL"
        print(f"  Ep {ep}: {action_str:>20s} reward={components['total']:+.3f} "
              f"({gen_time:.1f}s)")

    n = len(results)
    if n > 0:
        avg_components = {k: v / n for k, v in total_components.items()}
    else:
        avg_components = {}

    summary = {
        "n_episodes": n,
        "format_failure_rate": format_failures / max(n, 1),
        "avg_reward": avg_components.get("total", 0),
        "avg_components": avg_components,
        "action_distribution": dict(action_dist),
        "episodes": results,
    }

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-3-1b-it")
    p.add_argument("--adapter", default=None, help="LoRA adapter path")
    p.add_argument("--n_episodes", type=int, default=20)
    p.add_argument("--output", default=None,
                   help="Output JSON file (default: eval/model_eval.json)")
    args = p.parse_args()

    print("=" * 60)
    print("LLM EVALUATION")
    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print(f"Episodes: {args.n_episodes}")
    print("=" * 60)

    model, tokenizer, backend = load_model(args.model, args.adapter)
    summary = eval_model(model, tokenizer, backend, args.n_episodes)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  Avg reward: {summary['avg_reward']:.4f}")
    print(f"  Format failure rate: {summary['format_failure_rate']:.1%}")
    print(f"  Action distribution: {summary['action_distribution']}")
    if summary['avg_components']:
        print(f"  Components:")
        for k, v in summary['avg_components'].items():
            if k != 'total':
                print(f"    {k}: {v:.4f}")
    print(f"{'='*60}")

    output_file = args.output or str(Path(__file__).resolve().parent / "model_eval.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
