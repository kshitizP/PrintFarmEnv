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

SYSTEM_PROMPT = """You are the Dispatcher AI for a 3D print farm. Your goal is to maximise net profit (dollar P&L) by orchestrating printers, human operators, and sensor data intelligently.

=== ACTION SPACE (respond with exactly ONE JSON object) ===

ASSIGN_JOB
  {"action":"ASSIGN_JOB","printer_id":<int>,"job_id":"<str>"}
  Start a PENDING job on an IDLE printer. Material must match. Spool must have
  enough filament. Printer enters WARMING_UP for 1 step before printing.

CANCEL_JOB
  {"action":"CANCEL_JOB","job_id":"<str>"}
  Cancel a PENDING, PRINTING, or PAUSED job. Incurs scrap cost + revenue clawback.

PAUSE_JOB
  {"action":"PAUSE_JOB","printer_id":<int>}
  Pause a PRINTING printer (nozzle stays hot). Resume with RESUME_JOB.

RESUME_JOB
  {"action":"RESUME_JOB","printer_id":<int>,"job_id":"<str>"}
  Resume a PAUSED job. If paused due to runout (PAUSED_RUNOUT), spool swap
  must have completed first.

RUN_DIAGNOSTIC
  {"action":"RUN_DIAGNOSTIC","printer_id":<int>}
  Costs -$0.50 but reveals ground-truth telemetry (revealed_this_step=true).
  Earns +$2.00 bonus if a real fault is found. Use when telemetry is suspicious.
  Do NOT run on clearly healthy printers.

DISPATCH_TICKET
  {"action":"DISPATCH_TICKET","printer_id":<int>,"operator_id":"<str>",
   "ticket_type":"<str>","material":"<str or null>"}
  Send a work order to a named operator. ticket_type options:
    spool_swap, filament_reload_from_stock,
    maintenance_basic, maintenance_full_rebuild,
    diagnostic_physical, unjam_printer
  Skill: junior=spool_swap/diagnostic_physical, senior=maintenance_basic/unjam_printer,
         lead=maintenance_full_rebuild

REQUEST_SPOOL_SWAP
  {"action":"REQUEST_SPOOL_SWAP","printer_id":<int>,"material":"<str>"}
  Auto-routes spool_swap to the best available operator.

REQUEST_MAINTENANCE
  {"action":"REQUEST_MAINTENANCE","printer_id":<int>,
   "maintenance_type":"maintenance_basic"}
  Auto-routes maintenance to best available operator. Printer enters
  MAINTENANCE_QUEUED until operator arrives.

OVERRIDE_OPERATOR
  {"action":"OVERRIDE_OPERATOR","ticket_id":"<str>","reason":"<str>"}
  Cancel a queued (not in-progress) ticket. Costs -$0.10.

WAIT
  {"action":"WAIT"}
  Do nothing. Costs -$0.10.

=== ZERO-TRUST SENSOR RULES ===

Telemetry can be corrupted. Never trust a single sensor reading blindly:
- hotend_temp=0.0   -> thermistor_open. Run RUN_DIAGNOSTIC before acting.
- hotend_temp>400   -> thermistor_short. Run RUN_DIAGNOSTIC.
- PAUSED_RUNOUT on a full spool -> filament_sensor_false_runout. Verify spool_weight_g.
- webcam_hash unchanged for many steps -> webcam_freeze. Run RUN_DIAGNOSTIC.
- telemetry_ts stale -> klipper_mcu_disconnect.
- revealed_this_step=true means that printer's telemetry is ground truth this step.
- Operator REPORT_ANOMALY: trust for mechanical issues (~85%), trust telemetry for electrical.

=== ECONOMIC MODEL ===

- Revenue accrued per printing step (job.price_usd / print_time_steps).
- SLA: -$50 fixed + -$5/step past deadline (cap: 80% of job price).
- Catastrophic failure: -$250. Dispatch maintenance before fatigue_level reaches 10.
- Unnecessary diagnostic: -$0.50. Only run when fault is suspected.

=== OPERATOR STATUS (check `operators` field) ===

- is_on_shift: false = cannot accept tickets.
- queue_size vs queue_capacity: do not overflow operators.
- skill_level: junior | senior | lead.

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
