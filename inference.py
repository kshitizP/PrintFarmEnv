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
    print(f"\n{'=' * 50}")
    print(f"  STARTING {task_id.upper()}")
    print(f"{'=' * 50}")
    observation = env.reset(episode_id=task_id)

    while not observation.done:
        env.render_dashboard()
        action = extract_action(observation.model_dump_json())
        print(f"  -> Agent: {action.action.value}", end="")
        if action.printer_id:
            print(f" printer={action.printer_id}", end="")
        if action.job_id:
            print(f" job={action.job_id}", end="")
        if action.material:
            print(f" material={action.material}", end="")
        print()

        observation = env.step(action)
        if observation.metadata.get("error"):
            print(f"  -> Error: {observation.metadata['error']}")

        if observation.done:
            print(f"\n  {task_id.upper()} DONE — Score: {observation.reward:.3f}")
            env.render_dashboard()
            return observation.reward
    return 0.0


if __name__ == "__main__":
    env = PrintFarmEnvironment()
    tasks = ["task_1", "task_2", "task_3"]
    scores = {}

    for t in tasks:
        scores[t] = run_task(t, env)

    print(f"\n{'=' * 50}")
    print("  FINAL SCORES")
    print(f"{'=' * 50}")
    for t, s in scores.items():
        print(f"  {t}: {s:.3f}")
    print(f"  Total: {sum(scores.values()):.3f} / {len(tasks)}")
