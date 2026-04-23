"""
grpo_rollout.py
===============
Two roles:

  1. CLI / library: Collect (prompt, completion, reward) rollouts from a
     reference policy and write a TRL-compatible dataset to disk.

  2. Reward function factory: `make_env_reward_fn()` returns a callable
     with the signature TRL GRPOTrainer expects:
         fn(prompts, completions, *, task_id, seed, prior_actions, **kwargs)
              -> list[float]
     It replays the env to the stored state and evaluates each completion.

Dataset schema (one row = one env step)
----------------------------------------
  prompt_messages  list[dict]   chat messages [{role, content}, ...]
  completion       str          reference action JSON (ground-truth label)
  reward           float        step_reward_usd for that action
  task_id          str          e.g. "task_1"
  seed             int          episode seed
  step_idx         int          0-based step index within the episode
  prior_actions    list[str]    JSON strings of all FarmActions taken before this step

Usage
-----
  # Collect from clairvoyant policy (bootstrapping before model is trained)
  python scripts/grpo_rollout.py --policy clairvoyant \\
      --tasks task_1 task_2 task_3 --episodes 50 --output data/grpo

  # Collect from a trained model (HF-compatible checkpoint)
  python scripts/grpo_rollout.py --policy model --model-path ./sft_output \\
      --tasks task_1 --episodes 20 --output data/grpo

  # Smoke test
  python scripts/grpo_rollout.py --tasks task_1 --episodes 2 --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmAction, FarmActionEnum, FarmObservation
from printfarm_env.tasks import TASK_CONFIGS

# Re-use the system prompt from inference.py
from inference import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

DEFAULT_TASKS    = ["task_1", "task_2", "task_3"]
DEFAULT_EPISODES = 20
DEFAULT_SEED     = 42


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def parse_action(text: str) -> FarmAction:
    """Parse a FarmAction from raw model text (JSON, fenced, or inline)."""
    text = text.strip()
    data = None
    for pattern in [
        lambda t: json.loads(t),
        lambda t: json.loads(re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL).group(1)),
        lambda t: json.loads(re.search(r"\{[^{}]*\}", t).group(0)),
    ]:
        try:
            data = pattern(text)
            break
        except Exception:
            continue
    if data is None:
        return FarmAction(action=FarmActionEnum.WAIT)
    try:
        return FarmAction(**data)
    except Exception:
        return FarmAction(action=FarmActionEnum.WAIT)


def obs_to_prompt_messages(obs: FarmObservation) -> List[Dict[str, str]]:
    """Format a FarmObservation as a two-message chat list for GRPOTrainer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Current State:\n{obs.model_dump_json()}"},
    ]


def replay_to_step(
    env: PrintFarmEnvironment,
    task_id: str,
    seed: int,
    prior_actions_json: List[str],
) -> None:
    """Reset env and replay stored actions to reconstruct state at a given step."""
    env.reset(seed=seed, task_id=task_id)
    for action_json in prior_actions_json:
        env.step(parse_action(action_json))


# ---------------------------------------------------------------------------
#  Clairvoyant policy import (lazy to avoid circular import issues)
# ---------------------------------------------------------------------------

def _clairvoyant_policy_fn(env: PrintFarmEnvironment) -> FarmAction:
    from baselines.clairvoyant_greedy import clairvoyant_action
    return clairvoyant_action(env)


def _naive_policy_fn(env: PrintFarmEnvironment) -> FarmAction:
    from baselines.naive_greedy import naive_action
    return naive_action(env)


def _build_model_policy_fn(model_path: str) -> Callable:
    """Return a policy function that uses a local HF model checkpoint."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=200,
        temperature=0.2,
        do_sample=True,
    )

    def policy_fn(env: PrintFarmEnvironment) -> FarmAction:
        obs = env.state
        messages = obs_to_prompt_messages(obs)
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = pipe(formatted)[0]["generated_text"]
        # Strip the prompt prefix — pipeline returns full text
        completion = out[len(formatted):].strip()
        return parse_action(completion)

    return policy_fn


# ---------------------------------------------------------------------------
#  Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: PrintFarmEnvironment,
    policy_fn: Callable,
    task_id: str,
    seed: int,
    episode_idx: int,
) -> List[Dict[str, Any]]:
    """
    Run one full episode and return a list of step records.
    Each record has the columns needed by the GRPO dataset.
    """
    obs = env.reset(seed=seed, task_id=task_id)
    records: List[Dict[str, Any]] = []
    prior_actions: List[str] = []

    while not obs.done:
        action = policy_fn(env)
        action_json = action.model_dump_json()
        step_reward = obs.metadata.get("step_reward_usd", 0.0)  # reward BEFORE this action

        records.append({
            "task_id":        task_id,
            "seed":           seed,
            "episode":        episode_idx,
            "step_idx":       env.time_step,
            "prompt_messages": obs_to_prompt_messages(obs),
            "completion":     action_json,
            "reward":         float(step_reward),
            "prior_actions":  list(prior_actions),
        })

        prior_actions.append(action_json)
        obs = env.step(action)

    # Back-fill episode reward onto all records so caller can filter
    episode_reward = float(obs.reward)
    for r in records:
        r["episode_reward"] = episode_reward

    return records


# ---------------------------------------------------------------------------
#  Collection driver
# ---------------------------------------------------------------------------

def collect_rollouts(
    policy_fn: Callable,
    tasks: List[str],
    n_episodes: int,
    base_seed: int = DEFAULT_SEED,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    env = PrintFarmEnvironment()
    all_records: List[Dict[str, Any]] = []

    for task_id in tasks:
        ep_rewards = []
        for ep in range(n_episodes):
            seed = base_seed + ep
            records = run_episode(env, policy_fn, task_id, seed, ep)
            all_records.extend(records)
            ep_rewards.append(records[-1]["episode_reward"] if records else 0.0)

            if verbose:
                er = ep_rewards[-1]
                print(f"  {task_id}  ep={ep:3d}  steps={len(records):3d}  "
                      f"ep_reward={er:.4f}  step_records={len(records)}")
            elif (ep + 1) % 10 == 0:
                mean = sum(ep_rewards) / len(ep_rewards)
                print(f"  {task_id}  [{ep+1:3d}/{n_episodes}]  "
                      f"mean_ep_reward={mean:.4f}  total_steps={len(all_records)}")

    return all_records


# ---------------------------------------------------------------------------
#  TRL GRPOTrainer reward function (online mode)
# ---------------------------------------------------------------------------

def make_env_reward_fn() -> Callable:
    """
    Returns a reward function compatible with TRL GRPOTrainer.

    The dataset must contain columns: task_id, seed, step_idx, prior_actions.
    TRL passes extra dataset columns as kwargs to reward functions.

    Example notebook usage:
        from scripts.grpo_rollout import make_env_reward_fn
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[make_env_reward_fn()],
            args=grpo_config,
            train_dataset=dataset,
        )
    """
    def _reward_fn(
        prompts: List[str],
        completions: List[str],
        *,
        task_id: List[str],
        seed: List[int],
        prior_actions: List[List[str]],
        **kwargs,
    ) -> List[float]:
        env = PrintFarmEnvironment()
        rewards: List[float] = []

        for completion, tid, s, pa in zip(completions, task_id, seed, prior_actions):
            try:
                replay_to_step(env, tid, int(s), pa)
                action = parse_action(completion)
                obs = env.step(action)
                rewards.append(obs.metadata.get("step_reward_usd", 0.0))
            except Exception:
                rewards.append(-1.0)  # penalise broken completions

        return rewards

    return _reward_fn


# ---------------------------------------------------------------------------
#  Dataset serialisation
# ---------------------------------------------------------------------------

def save_dataset(records: List[Dict[str, Any]], output_dir: str) -> None:
    """Save records to JSONL and attempt to push as HF dataset."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    jsonl_path = out / "grpo_steps.jsonl"
    with open(jsonl_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} step records → {jsonl_path}")

    # Summary stats
    summary = {
        "total_steps":   len(records),
        "tasks":         list({r["task_id"] for r in records}),
        "episodes":      len({(r["task_id"], r["episode"]) for r in records}),
        "mean_step_reward": sum(r["reward"] for r in records) / max(len(records), 1),
        "mean_ep_reward":   sum(r["episode_reward"] for r in records) / max(len(records), 1),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary}")


def load_as_hf_dataset(jsonl_path: str):
    """Load saved JSONL as a HuggingFace Dataset."""
    from datasets import Dataset
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GRPO rollouts from PrintFarmEnv")
    parser.add_argument("--policy",     default="clairvoyant",
                        choices=["clairvoyant", "naive", "model"],
                        help="Reference policy to collect from")
    parser.add_argument("--model-path", default=None,
                        help="Path to HF model checkpoint (required for --policy model)")
    parser.add_argument("--tasks",      nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--episodes",   type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    parser.add_argument("--output",     default="data/grpo")
    parser.add_argument("--verbose",    "-v", action="store_true")
    args = parser.parse_args()

    if args.policy == "clairvoyant":
        policy_fn = _clairvoyant_policy_fn
    elif args.policy == "naive":
        policy_fn = _naive_policy_fn
    else:
        if not args.model_path:
            parser.error("--model-path required when --policy model")
        policy_fn = _build_model_policy_fn(args.model_path)

    print(f"Policy:   {args.policy}")
    print(f"Tasks:    {', '.join(args.tasks)}")
    print(f"Episodes: {args.episodes} per task")
    print(f"Output:   {args.output}")
    print()

    records = collect_rollouts(
        policy_fn=policy_fn,
        tasks=args.tasks,
        n_episodes=args.episodes,
        base_seed=args.seed,
        verbose=args.verbose,
    )
    save_dataset(records, args.output)


if __name__ == "__main__":
    main()
