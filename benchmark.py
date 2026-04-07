"""
Benchmark runner — runs inference across multiple models and prints consolidated results.

Usage:
    python benchmark.py
    python benchmark.py --models gpt-5.4 gpt-4.1 gpt-4o
    python benchmark.py --tasks task_1 task_3
    python benchmark.py --api-base https://router.huggingface.co/v1 --models meta-llama/Llama-3.3-70B-Instruct
"""

import os
import sys
import json
import time
import argparse
from openai import OpenAI
from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmAction, FarmActionEnum

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "gpt-5.4",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]

DEFAULT_TASKS = ["task_1", "task_2", "task_3"]

SYSTEM_PROMPT = """You are an automated Floor Manager for a 3D printing farm.
Your goal is to manage incoming jobs, inventory, and hardware issues to maximise
the total score. Higher-priority jobs and meeting deadlines matter most.

Available actions (respond with exactly ONE as JSON):

1. ASSIGN_JOB — Start a pending job on an idle printer.
   {"action": "ASSIGN_JOB", "printer_id": <int>, "job_id": "<string>"}
   Requirements: printer must be IDLE, material must match, spool must have
   enough filament for the job.  There is a 1-step warmup before printing
   begins (2 steps if purge_needed is true).

2. SWAP_FILAMENT — Replace a printer's spool from inventory.
   {"action": "SWAP_FILAMENT", "printer_id": <int>, "material": "<string>"}
   Printer must be IDLE or ERROR. Gives a fresh 1000g spool. Sets purge_needed.

3. CANCEL_JOB — Cancel a pending or in-progress job.
   {"action": "CANCEL_JOB", "job_id": "<string>"}

4. PERFORM_MAINTENANCE — Service a printer (takes 3 steps, resets reliability).
   {"action": "PERFORM_MAINTENANCE", "printer_id": <int>}
   Printer must be IDLE or ERROR.

5. WAIT — Do nothing (penalised by the grader).
   {"action": "WAIT"}

Key mechanics:
- Each printer has a reliability score. Lower reliability = higher chance of
  random failure each printing step. Failed jobs revert to PENDING and can be
  re-assigned.
- Maintenance resets the reliability counter and adds +5% reliability.
- Jobs have priority (1=low, 2=normal, 3=urgent) and optional deadlines.
  Completing urgent jobs on time is worth much more than low-priority jobs.
- Printers with maintenance_due_in <= 0 degrade reliability each step.

Strategy tips:
- Assign urgent/deadline jobs to reliable printers first.
- Swap filament proactively — don't wait for runout.
- Perform maintenance on printers with low maintenance_due_in before they fail.
- Avoid assigning critical jobs to unreliable printers.

Respond with ONLY valid JSON. No explanations."""


# ---------------------------------------------------------------------------
#  Agent
# ---------------------------------------------------------------------------

def make_client(api_base: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=api_base, api_key=api_key)


def _is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning/codex variant that needs special handling."""
    m = model.lower()
    # o-series reasoning models
    if m.startswith(("o1", "o3", "o4")):
        return True
    # gpt-5+ codex and pro variants
    if "codex" in m or "-pro" in m:
        return True
    # gpt-5 base and mini (not gpt-5.x)
    if m in ("gpt-5", "gpt-5-mini"):
        return True
    return False


def _call_llm(client: OpenAI, model: str, messages: list) -> str | None:
    """Try multiple parameter combos to handle model-specific quirks."""
    reasoning = _is_reasoning_model(model)

    if reasoning:
        # Reasoning models: no temperature, no system role, use developer role
        reasoning_messages = []
        for msg in messages:
            if msg["role"] == "system":
                reasoning_messages.append({"role": "developer", "content": msg["content"]})
            else:
                reasoning_messages.append(msg)

        combos = [
            {"response_format": {"type": "json_object"}, "max_completion_tokens": 400},
            {"max_completion_tokens": 400},
        ]
        for kwargs in combos:
            try:
                response = client.chat.completions.create(
                    model=model, messages=reasoning_messages, **kwargs,
                )
                return response.choices[0].message.content
            except Exception:
                continue

        # Last resort: plain user message, no special params
        try:
            plain = [{"role": "user", "content":
                       messages[0]["content"] + "\n\n" + messages[1]["content"]
                       + "\n\nRespond with ONLY valid JSON."}]
            response = client.chat.completions.create(
                model=model, messages=plain, max_completion_tokens=400,
            )
            return response.choices[0].message.content
        except Exception:
            pass

        return None

    # Standard models: try all combos with temperature
    combos = [
        {"response_format": {"type": "json_object"}, "max_completion_tokens": 200},
        {"response_format": {"type": "json_object"}, "max_tokens": 200},
        {"max_completion_tokens": 200},
        {"max_tokens": 200},
    ]
    for kwargs in combos:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.2, **kwargs,
            )
            return response.choices[0].message.content
        except Exception:
            continue
    return None


def _parse_json_from_text(text: str) -> dict | None:
    """Extract JSON from text that may contain markdown fences or extra text."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code fence
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } block
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def extract_action(client: OpenAI, model: str, state_json: str) -> FarmAction:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current State:\n{state_json}"},
    ]

    content = _call_llm(client, model, messages)
    if content is None:
        print(f"      [LLM Error] All API call variants failed")
        return FarmAction(action=FarmActionEnum.WAIT)

    action_data = _parse_json_from_text(content)
    if action_data is None:
        print(f"      [Parse Error] Could not extract JSON from response")
        return FarmAction(action=FarmActionEnum.WAIT)

    try:
        return FarmAction(**action_data)
    except Exception as e:
        print(f"      [Parse Error] {e}")
        return FarmAction(action=FarmActionEnum.WAIT)


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, model: str, task_id: str, verbose: bool) -> tuple[float, float]:
    """Returns (score, elapsed_seconds)."""
    env = PrintFarmEnvironment()
    obs = env.reset(episode_id=task_id)
    t0 = time.time()

    while not obs.done:
        action = extract_action(client, model, obs.model_dump_json())
        if verbose:
            label = action.action.value
            if action.printer_id:
                label += f" p={action.printer_id}"
            if action.job_id:
                label += f" j={action.job_id}"
            if action.material:
                label += f" m={action.material}"
            print(f"      Step {env.time_step:2d}: {label}")

        obs = env.step(action)
        if verbose and obs.metadata.get("error"):
            print(f"             -> {obs.metadata['error']}")

    elapsed = time.time() - t0
    return obs.reward, elapsed


def run_benchmark(models: list[str], tasks: list[str],
                  api_base: str, api_key: str, verbose: bool):
    client = make_client(api_base, api_key)

    # results[model][task] = (score, time)
    results: dict[str, dict[str, tuple[float, float]]] = {}

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model}")
        print(f"{'=' * 60}")
        results[model] = {}

        for task_id in tasks:
            print(f"    {task_id} ... ", end="", flush=True)
            if verbose:
                print()
            try:
                score, elapsed = run_task(client, model, task_id, verbose)
                results[model][task_id] = (score, elapsed)
                print(f"score={score:.3f}  ({elapsed:.1f}s)")
            except Exception as e:
                results[model][task_id] = (0.0, 0.0)
                print(f"FAILED: {e}")

    # ------------------------------------------------------------------
    #  Consolidated results
    # ------------------------------------------------------------------
    print(f"\n\n{'=' * 60}")
    print("  CONSOLIDATED RESULTS")
    print(f"{'=' * 60}\n")

    # Header
    task_headers = "".join(f"  {t:>10s}" for t in tasks)
    print(f"  {'Model':<40s}{task_headers}  {'Total':>8s}  {'Time':>8s}")
    print(f"  {'-' * 40}{'-' * (12 * len(tasks))}{'-' * 10}{'-' * 10}")

    for model in models:
        row = f"  {model:<40s}"
        total_score = 0.0
        total_time = 0.0
        for task_id in tasks:
            score, elapsed = results[model].get(task_id, (0.0, 0.0))
            total_score += score
            total_time += elapsed
            row += f"  {score:>10.3f}"
        row += f"  {total_score:>8.3f}"
        row += f"  {total_time:>7.1f}s"
        print(row)

    print(f"\n  Tasks: {', '.join(tasks)}")
    print(f"  API: {api_base}")
    print()

    # ------------------------------------------------------------------
    #  JSON output
    # ------------------------------------------------------------------
    json_results = {}
    for model in models:
        json_results[model] = {}
        for task_id in tasks:
            score, elapsed = results[model].get(task_id, (0.0, 0.0))
            json_results[model][task_id] = {"score": round(score, 4), "time_s": round(elapsed, 1)}
        scores = [results[model].get(t, (0.0, 0.0))[0] for t in tasks]
        json_results[model]["total"] = round(sum(scores), 4)

    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Results saved to {out_path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PrintFarmEnv benchmark runner")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model names to benchmark")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        help="Task IDs to run")
    parser.add_argument("--api-base", default=None,
                        help="API base URL (default: from API_BASE_URL env or OpenAI)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: from HF_TOKEN / OPENAI_API_KEY / API_KEY env)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each agent action")
    args = parser.parse_args()

    api_base = args.api_base or os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    api_key = (args.api_key
               or os.getenv("HF_TOKEN")
               or os.getenv("OPENAI_API_KEY")
               or os.getenv("API_KEY")
               or "")

    if not api_key:
        print("Error: No API key found. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY.")
        sys.exit(1)

    print(f"  API base: {api_base}")
    print(f"  Models:   {', '.join(args.models)}")
    print(f"  Tasks:    {', '.join(args.tasks)}")

    run_benchmark(args.models, args.tasks, api_base, api_key, args.verbose)


if __name__ == "__main__":
    main()
