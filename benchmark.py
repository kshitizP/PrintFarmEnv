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

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_DUMMY_KEY = "ollama"  # Ollama ignores the API key

DEFAULT_TASKS = ["task_1", "task_2", "task_3"]

SYSTEM_PROMPT = """You are an automated Floor Manager for a 3D printing farm.
Your goal is to manage incoming jobs, inventory, and hardware issues to maximise
the total score. Higher-priority jobs and meeting deadlines matter most.

Available actions (respond with exactly ONE as JSON):

1. ASSIGN_JOB — Start a pending job on an idle printer.
   {"action": "ASSIGN_JOB", "printer_id": <int>, "job_id": "<string>"}
   Requirements: printer must be IDLE, material must match. There is a 1-step
   warmup before printing begins. If the spool has less filament than the job
   requires, the print will start but may run out mid-print (PAUSED_RUNOUT).

2. SWAP_FILAMENT — Replace a printer's spool from inventory.
   {"action": "SWAP_FILAMENT", "printer_id": <int>, "material": "<string>"}
   Printer must be IDLE, ERROR, or PAUSED_RUNOUT. Loads a fresh spool (950g
   after 50g purge cost). IMPORTANT: costs 2 timesteps of warmup (changeover
   cost). Plan swaps carefully — each one burns time and material.

3. CANCEL_JOB — Cancel a pending, printing, or paused job.
   {"action": "CANCEL_JOB", "job_id": "<string>"}

4. PERFORM_MAINTENANCE — Service a printer (takes 3 steps).
   {"action": "PERFORM_MAINTENANCE", "printer_id": <int>}
   Printer must be IDLE or ERROR. Resets fatigue_level to 0, resets
   maintenance_due_in to 50, adds +5% reliability.

5. RESUME_JOB — Resume a paused job after filament swap.
   {"action": "RESUME_JOB", "printer_id": <int>, "job_id": "<string>"}
   Printer must be IDLE, job must be PAUSED. Resumes printing from where it
   left off (no warmup).

6. WAIT — Do nothing (penalised by the grader).
   {"action": "WAIT"}

Key mechanics:
- CHANGEOVER COST: Every SWAP_FILAMENT costs 2 timesteps and 50g of material.
  Batch jobs by material to minimise swaps.
- CONTINUOUS LATENCY DECAY: For every timestep a job is late past its deadline,
  its value drops by 5%, bottoming at 10%. Formula:
  late_credit = weight * max(0.1, 1.0 - 0.05 * steps_late)
- FILAMENT RUNOUT: If a spool empties mid-print, the printer enters
  PAUSED_RUNOUT and the job becomes PAUSED. You must SWAP_FILAMENT then
  RESUME_JOB to continue. Do NOT cancel — you lose all progress.
- MACHINE FATIGUE: Each printing step increases fatigue_level by 1. If fatigue
  reaches 10, the printer suffers CATASTROPHIC FAILURE: the job is destroyed,
  the machine goes OFFLINE for 10 steps. Use PERFORM_MAINTENANCE to reset
  fatigue before it's too late.
- RELIABILITY: Lower reliability = higher chance of random failure per step.
  Maintenance resets the counter and adds +5%.
- PRIORITY: Jobs have priority 1=low, 2=normal, 3=urgent. Urgent jobs with
  deadlines are worth the most.

Strategy tips:
- Check fatigue_level before assigning long jobs. If fatigue + print_time >= 10,
  run maintenance FIRST.
- Batch same-material jobs on the same printer to avoid changeover costs.
- If a spool runs out, swap and resume — never cancel a partially-printed job.
- Assign urgent/deadline jobs to reliable printers first.
- Every wasted step bleeds points via latency decay. Act decisively.

Respond with ONLY valid JSON. No explanations."""


# ---------------------------------------------------------------------------
#  Agent
# ---------------------------------------------------------------------------

def make_client(api_base: str, api_key: str, timeout: float = 120.0) -> OpenAI:
    return OpenAI(base_url=api_base, api_key=api_key, timeout=timeout)


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
            {"response_format": {"type": "json_object"}, "max_completion_tokens": 1024},
            {"max_completion_tokens": 1024},
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
    # Use 2048 tokens to accommodate reasoning/thinking models (e.g., gemma4)
    # that consume internal thinking tokens before producing visible output.
    # max_tokens first (broader compatibility with Ollama / local servers)
    combos = [
        {"max_tokens": 2048},
        {"response_format": {"type": "json_object"}, "max_tokens": 2048},
        {"max_completion_tokens": 2048},
    ]
    for kwargs in combos:
        try:
            response = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.2, **kwargs,
            )
            content = response.choices[0].message.content
            # If model returned empty (e.g., all thinking tokens), try next combo
            if content and content.strip():
                return content
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


def _summarize_state(state_json: str) -> str:
    """Condense the full state JSON into a compact text summary.

    This dramatically reduces token count for local/small models while
    keeping all decision-relevant information.
    """
    try:
        s = json.loads(state_json)
    except json.JSONDecodeError:
        return state_json  # Fallback to raw JSON

    lines = [f"Step {s.get('time_step', '?')}/{s.get('max_steps', '?')}"]

    # Inventory
    inv = s.get("inventory", {})
    inv_str = ", ".join(f"{k}={v:.0f}g" for k, v in inv.items())
    lines.append(f"Inventory: {inv_str}")

    # Printers (only show relevant ones: non-empty or non-IDLE)
    lines.append("Printers:")
    for p in s.get("printers", []):
        pid = p["printer_id"]
        state = p["state"]
        mat = p.get("current_material") or "empty"
        spool = p.get("spool_weight_g", 0)
        rel = p.get("reliability", 0.95)
        fatigue = p.get("fatigue_level", 0)
        maint = p.get("maintenance_due_in", 50)
        job = p.get("current_job_id") or ""
        warmup = p.get("warmup_remaining", 0)
        offline = p.get("offline_remaining", 0)

        # Skip empty idle printers with nothing interesting
        if (state == "IDLE" and mat == "empty" and not job
                and fatigue == 0 and maint >= 40 and rel >= 0.94):
            continue

        detail = f"  P{pid}: {state}"
        if mat != "empty":
            detail += f" {mat}({spool:.0f}g)"
        if job:
            detail += f" job={job}"
        if fatigue > 0:
            detail += f" fatigue={fatigue}"
        if maint < 20:
            detail += f" maint_in={maint}"
        if rel < 0.94:
            detail += f" rel={rel:.0%}"
        if warmup > 0:
            detail += f" warmup={warmup}"
        if offline > 0:
            detail += f" offline={offline}"
        lines.append(detail)

    # Count hidden printers
    all_printers = s.get("printers", [])
    shown = sum(1 for l in lines if l.startswith("  P"))
    hidden = len(all_printers) - shown
    if hidden > 0:
        lines.append(f"  ({hidden} more empty/idle printers available)")

    # Jobs
    lines.append("Jobs:")
    for j in s.get("active_queue", []):
        jid = j["job_id"]
        state = j["state"]
        mat = j["material_required"]
        weight = j["weight_required_g"]
        steps = j["print_time_steps"]
        prio = j["priority"]
        progress = j.get("progress_steps", 0)
        deadline = j.get("deadline_steps")

        prio_label = {1: "low", 2: "normal", 3: "URGENT"}[prio]
        detail = f"  {jid}: {state} {mat} {weight:.0f}g {steps}steps prio={prio_label}"
        if progress > 0:
            detail += f" progress={progress}/{steps}"
        if deadline:
            detail += f" deadline={deadline}"
        lines.append(detail)

    return "\n".join(lines)


def extract_action(client: OpenAI, model: str, state_json: str,
                   compact: bool = False, verbose: bool = False) -> FarmAction:
    state_text = _summarize_state(state_json) if compact else state_json
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Current State:\n{state_text}"},
    ]

    content = _call_llm(client, model, messages)
    if content is None:
        print(f"      [LLM Error] All API call variants failed", flush=True)
        return FarmAction(action=FarmActionEnum.WAIT)

    action_data = _parse_json_from_text(content)
    if action_data is None:
        preview = repr(content[:200]) if content else repr(content)
        print(f"      [Parse Error] Could not extract JSON from response", flush=True)
        if verbose:
            print(f"             raw={preview}", flush=True)
        return FarmAction(action=FarmActionEnum.WAIT)

    try:
        return FarmAction(**action_data)
    except Exception as e:
        print(f"      [Parse Error] {e}", flush=True)
        return FarmAction(action=FarmActionEnum.WAIT)


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, model: str, task_id: str, verbose: bool,
             compact: bool = False) -> tuple[float, float]:
    """Returns (score, elapsed_seconds)."""
    env = PrintFarmEnvironment()
    obs = env.reset(episode_id=task_id)
    t0 = time.time()

    while not obs.done:
        action = extract_action(client, model, obs.model_dump_json(),
                                compact=compact, verbose=verbose)
        if verbose:
            label = action.action.value
            if action.printer_id:
                label += f" p={action.printer_id}"
            if action.job_id:
                label += f" j={action.job_id}"
            if action.material:
                label += f" m={action.material}"
            elapsed_so_far = time.time() - t0
            print(f"      Step {env.time_step:2d}: {label}  ({elapsed_so_far:.1f}s)", flush=True)

        obs = env.step(action)
        if verbose and obs.metadata.get("error"):
            print(f"             -> {obs.metadata['error']}", flush=True)
        if verbose:
            print(f"             score={obs.reward:.4f}", flush=True)

    elapsed = time.time() - t0
    return obs.reward, elapsed


def run_benchmark(models: list[str], tasks: list[str],
                  api_base: str, api_key: str, verbose: bool,
                  timeout: float = 120.0, compact: bool = False):
    client = make_client(api_base, api_key, timeout=timeout)

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
                score, elapsed = run_task(client, model, task_id, verbose,
                                          compact=compact)
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
    parser = argparse.ArgumentParser(
        description="PrintFarmEnv benchmark runner",
        epilog="""
Examples:
  # Run against local Ollama (gemma4)
  python benchmark.py --ollama --models gemma4

  # Run against Ollama with multiple local models
  python benchmark.py --ollama --models gemma4 llama3.2 --verbose

  # Run a single task against Ollama
  python benchmark.py --ollama --models gemma4 --tasks task_3

  # Run against OpenAI
  python benchmark.py --models gpt-4o-mini

  # Run against any OpenAI-compatible API
  python benchmark.py --api-base http://localhost:8080/v1 --models my-model
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to benchmark")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        help="Task IDs to run")
    parser.add_argument("--ollama", action="store_true",
                        help="Use local Ollama server (http://localhost:11434/v1)")
    parser.add_argument("--api-base", default=None,
                        help="API base URL (default: from API_BASE_URL env or OpenAI)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: from HF_TOKEN / OPENAI_API_KEY / API_KEY env)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each agent action")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-request timeout in seconds (default: 120)")
    parser.add_argument("--compact", action="store_true",
                        help="Send compact state summaries instead of full JSON"
                             " (faster for local/small models, auto-enabled with --ollama)")
    args = parser.parse_args()

    # Resolve API base and key
    compact = args.compact
    if args.ollama:
        api_base = args.api_base or OLLAMA_BASE_URL
        api_key = OLLAMA_DUMMY_KEY
        models = args.models or ["gemma4"]
        compact = True  # Auto-enable compact mode for local models
    else:
        api_base = args.api_base or os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        api_key = (args.api_key
                   or os.getenv("HF_TOKEN")
                   or os.getenv("OPENAI_API_KEY")
                   or os.getenv("API_KEY")
                   or "")
        models = args.models or DEFAULT_MODELS

        if not api_key:
            print("Error: No API key found. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY,")
            print("       or use --ollama for local models.")
            sys.exit(1)

    print(f"  API base: {api_base}")
    print(f"  Models:   {', '.join(models)}")
    print(f"  Tasks:    {', '.join(args.tasks)}")

    if compact:
        print(f"  Compact:  enabled (summarized state)")

    run_benchmark(models, args.tasks, api_base, api_key, args.verbose,
                  timeout=args.timeout, compact=compact)


if __name__ == "__main__":
    main()
