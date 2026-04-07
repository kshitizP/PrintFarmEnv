import os
import json
from openai import OpenAI
from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmAction, FarmActionEnum

api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
api_key = (os.getenv("HF_TOKEN")
           or os.getenv("OPENAI_API_KEY")
           or os.getenv("API_KEY")
           or "")

client = OpenAI(
    base_url=api_base,
    api_key=api_key if api_key else "dummy_key"
)

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


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    if m.startswith(("o1", "o3", "o4")):
        return True
    if "codex" in m or "-pro" in m:
        return True
    if m in ("gpt-5", "gpt-5-mini"):
        return True
    return False


def extract_action(state_json: str) -> FarmAction:
    if not api_key:
        return FarmAction(action=FarmActionEnum.WAIT)

    import re
    reasoning = _is_reasoning_model(model_name)

    if reasoning:
        messages = [
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current State:\n{state_json}"},
        ]
        combos = [
            {"response_format": {"type": "json_object"}, "max_completion_tokens": 400},
            {"max_completion_tokens": 400},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current State:\n{state_json}"},
        ]
        combos = [
            {"response_format": {"type": "json_object"}, "max_completion_tokens": 200},
            {"response_format": {"type": "json_object"}, "max_tokens": 200},
            {"max_completion_tokens": 200},
            {"max_tokens": 200},
        ]

    content = None
    for kwargs in combos:
        try:
            call_kwargs = {"model": model_name, "messages": messages, **kwargs}
            if not reasoning:
                call_kwargs["temperature"] = 0.2
            response = client.chat.completions.create(**call_kwargs)
            content = response.choices[0].message.content
            break
        except Exception:
            continue

    if content is None:
        print(f"  [LLM Error] All API call variants failed")
        return FarmAction(action=FarmActionEnum.WAIT)

    # Parse JSON (handle markdown fences / extra text / thinking tokens)
    text = content.strip()
    action_data = None
    try:
        action_data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                action_data = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        if action_data is None:
            match = re.search(r"\{[^{}]*\}", text)
            if match:
                try:
                    action_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

    if action_data is None:
        print(f"  [Parse Error] Could not extract JSON")
        return FarmAction(action=FarmActionEnum.WAIT)

    try:
        return FarmAction(**action_data)
    except Exception as e:
        print(f"  [Parse Error] {e}")
        return FarmAction(action=FarmActionEnum.WAIT)


def run_task(task_id: str, env: PrintFarmEnvironment) -> float:
    # Emit structured [START] tag for the validator
    print(f"[START] task={task_id}", flush=True)

    observation = env.reset(episode_id=task_id)
    step_num = 0

    while not observation.done:
        action = extract_action(observation.model_dump_json())

        observation = env.step(action)
        step_num += 1

        # Emit structured [STEP] tag for the validator
        print(f"[STEP] step={step_num} reward={observation.reward:.4f}", flush=True)

        if observation.metadata.get("error"):
            print(f"  -> Error: {observation.metadata['error']}", flush=True)

    final_score = observation.reward
    # Emit structured [END] tag for the validator
    print(f"[END] task={task_id} score={final_score:.4f} steps={step_num}", flush=True)
    return final_score


if __name__ == "__main__":
    env = PrintFarmEnvironment()
    tasks = ["task_1", "task_2", "task_3"]
    scores = {}

    for t in tasks:
        scores[t] = run_task(t, env)

    # Summary (non-structured, but validator only cares about [START]/[STEP]/[END])
    print(f"\nFINAL SCORES", flush=True)
    for t, s in scores.items():
        print(f"  {t}: {s:.3f}", flush=True)
    print(f"  Total: {sum(scores.values()):.3f} / {len(tasks)}", flush=True)
