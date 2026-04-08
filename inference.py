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

SYSTEM_PROMPT = """You are a Floor Manager for a 3D printing farm. Maximise your score.

Available actions (respond with exactly ONE JSON object):

ASSIGN_JOB    {"action":"ASSIGN_JOB","printer_id":<int>,"job_id":"<str>"}
              Printer must be IDLE with matching material. Job must be PENDING.

SWAP_FILAMENT {"action":"SWAP_FILAMENT","printer_id":<int>,"material":"<str>"}
              Printer must be IDLE, ERROR, or PAUSED_RUNOUT. Replaces the current
              spool. New spool loads at 950g (50g is consumed in the purge cycle).
              The printer enters WARMING_UP for 2 steps before becoming available.

CANCEL_JOB    {"action":"CANCEL_JOB","job_id":"<str>"}
              Cancels a PENDING, PRINTING, or PAUSED job.

PERFORM_MAINTENANCE {"action":"PERFORM_MAINTENANCE","printer_id":<int>}
              Printer must be IDLE or ERROR. Takes 3 steps. Resets fatigue_level
              to 0 and restores reliability to 0.95.
              IMPORTANT: If the printer has fatigue_level > 0, it must have been
              continuously IDLE for at least 3 steps (thermal cooldown) before
              maintenance can begin. Check the consecutive_idle_steps field.

RESUME_JOB    {"action":"RESUME_JOB","printer_id":<int>,"job_id":"<str>"}
              Printer must be IDLE. Job must be PAUSED. Resumes printing from
              where it left off (no warmup required).

WAIT          {"action":"WAIT"}

Equipment reference:
- Each printer tracks a fatigue_level that increments by 1 for every step spent
  printing. When fatigue_level reaches 10 the printer suffers a catastrophic
  failure: the current job is marked FAILED and the printer goes OFFLINE for
  10 steps. PERFORM_MAINTENANCE resets fatigue_level to 0.
- Printers have a stochastic failure chance each printing step based on their
  reliability rating. A random failure also marks the job FAILED.
- Each spool has a weight in grams (spool_weight_g). Printing consumes material
  each step. If the spool runs out mid-print the printer enters PAUSED_RUNOUT
  and the job becomes PAUSED. A filament swap is required to continue.
- maintenance_due_in counts down each printing step. It is informational.

Scoring:
- Completed jobs earn points scaled by priority (1=low, 2=medium, 3=urgent).
- Jobs completed after their deadline lose value progressively — the later the
  finish, the less the job is worth (down to 10% of base value).
- Failed and cancelled jobs receive reduced scores.

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
