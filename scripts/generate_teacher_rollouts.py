"""
generate_teacher_rollouts.py
============================
Rolls out a teacher LLM (Claude or GPT) as the Dispatcher, saving full
(observation, action, reward) trajectories for SFT training.

Outputs:
    data/teacher/{task_id}/episodes.jsonl    # one full episode per line
    data/teacher/cost_ledger.json            # running token + $ tally
    data/teacher/summary.json                # aggregate stats

Each episode line:
    {
      "task_id": "task_1", "episode": 0, "seed": 42,
      "final_score": 0.67, "final_profit_usd": 145.3, "final_step": 150,
      "steps": [
        {
          "step": 0,
          "observation": {...},
          "prompt_tokens": 812,
          "completion_tokens": 54,
          "raw_completion": "...",
          "parsed_action": {...},      # null if parse failed
          "parse_ok": true,
          "action": {...},              # what was actually stepped (fallback to WAIT on parse fail)
          "reward": 0.5,
          "step_reward_usd": -0.1,
          "net_profit_usd": -0.1
        }
      ]
    }

Cost cap: script aborts mid-run if ledger exceeds --max-cost-usd.

Usage:
    # Claude Sonnet 4.6, 500 eps total spread across 8 tasks, $50 cap
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_teacher_rollouts.py --provider anthropic \
        --model claude-sonnet-4-6 --episodes-per-task 60 --max-cost-usd 50

    # GPT-4o variant
    export OPENAI_API_KEY=sk-...
    python scripts/generate_teacher_rollouts.py --provider openai \
        --model gpt-4o --episodes-per-task 60

    # Smoke test
    python scripts/generate_teacher_rollouts.py --tasks task_1 --episodes-per-task 2 \
        --max-steps-per-episode 20 --max-cost-usd 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from printfarm_env.env import PrintFarmEnvironment
from printfarm_env.models import FarmAction, FarmActionEnum, FarmObservation


ALL_TASKS = [
    "task_0_1", "task_0_2", "task_0_3",
    "task_1", "task_2", "task_3",
    "task_4", "task_5",
]

SYSTEM_PROMPT = """You are the Dispatcher for a 4-printer 3D print farm. You control printers, dispatch tickets to human operators, and manage an SLA-bound job queue.

Your actions (JSON only, no prose):
- {"action": "ASSIGN_JOB", "printer_id": <int>, "job_id": "<str>"}
- {"action": "CANCEL_JOB", "job_id": "<str>"}
- {"action": "PAUSE_JOB", "printer_id": <int>}
- {"action": "RESUME_JOB", "printer_id": <int>}
- {"action": "RUN_DIAGNOSTIC", "printer_id": <int>}
- {"action": "DISPATCH_TICKET", "printer_id": <int>, "operator_id": "<str>", "ticket_type": "<str>", "material": "<optional>", "maintenance_type": "<optional>"}
- {"action": "REQUEST_SPOOL_SWAP", "printer_id": <int>, "material": "<str>"}
- {"action": "REQUEST_MAINTENANCE", "printer_id": <int>, "maintenance_type": "basic|full_rebuild"}
- {"action": "OVERRIDE_OPERATOR", "ticket_id": "<str>", "reason": "<str>"}
- {"action": "WAIT"}

Ticket types: spool_swap, filament_reload_from_stock, maintenance_basic,
  maintenance_full_rebuild, diagnostic_physical, unjam_printer.

Economics:
- Job completion pays price_usd. SLA miss: -$50 once + -$5/step overdue, capped at 80% of price.
- Catastrophic failure (fatigue hits 10): -$250 + revenue clawback.
- Sensors lie under fault modes. RUN_DIAGNOSTIC reveals ground truth (-$0.50 cost, +$2 if it finds a fault).
- WAIT costs -$0.10/step. Invalid action: -$0.20.

Goals:
1. Keep printers busy but not overrun — fatigue accumulates while PRINTING.
2. Dispatch maintenance before fatigue >= 8, spool_swap before runout.
3. Run diagnostics when telemetry looks off (stale ts, temp spikes, fan drops).
4. In ERROR state, dispatch unjam_printer ticket.

Respond with a single JSON action object. No commentary, no markdown."""


# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens) — UPDATE AS MODELS CHANGE
# ---------------------------------------------------------------------------
PRICING = {
    # Anthropic
    "claude-opus-4-7":          {"input": 15.00, "output": 75.00},
    "claude-opus-4-6":          {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":        {"input":  3.00, "output": 15.00},
    "claude-sonnet-4-5":        {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":         {"input":  1.00, "output":  5.00},
    # OpenAI
    "gpt-4o":                   {"input":  2.50, "output": 10.00},
    "gpt-4o-mini":              {"input":  0.15, "output":  0.60},
    "gpt-4.1":                  {"input":  2.00, "output":  8.00},
}


def _price(model: str, ptoks: int, ctoks: int) -> float:
    rates = PRICING.get(model)
    if rates is None:
        return 0.0
    return (ptoks / 1_000_000) * rates["input"] + (ctoks / 1_000_000) * rates["output"]


# ---------------------------------------------------------------------------
# Observation → compact user prompt
# ---------------------------------------------------------------------------

def _compact_obs(obs: FarmObservation) -> str:
    """Trim obs to what's needed for action selection (saves tokens)."""
    compact = {
        "time_step": obs.time_step,
        "max_steps": obs.max_steps,
        "net_profit_usd": round(obs.net_profit_usd, 2),
        "inventory": {k: round(v, 1) for k, v in obs.inventory.items()},
        "active_queue": [
            {
                "job_id": j.job_id,
                "state": j.state.value if hasattr(j.state, "value") else j.state,
                "material": j.material_required,
                "weight_g": j.weight_required_g,
                "print_time_steps": j.print_time_steps,
                "progress_steps": j.progress_steps,
                "priority": j.priority,
                "deadline_steps": j.deadline_steps,
                "price_usd": j.price_usd,
            }
            for j in obs.active_queue[:20]  # cap queue len
        ],
        "printers": [
            {
                "printer_id": p.printer_id,
                "profile_id": p.profile_id,
                "state": p.state.value if hasattr(p.state, "value") else p.state,
                "current_material": p.current_material,
                "current_job_id": p.current_job_id,
                "spool_weight_g": round(p.spool_weight_g, 1),
                "reliability": round(p.reliability, 3),
                "maintenance_due_in": p.maintenance_due_in,
                "fatigue_level": round(p.fatigue_level, 2),
                "warmup_remaining": p.warmup_remaining,
                "offline_remaining": p.offline_remaining,
                "hotend_temp": round(p.hotend_temp, 1),
                "fan_rpm": p.fan_rpm,
                "telemetry_ts": p.telemetry_ts,
                "outstanding_ticket_id": p.outstanding_ticket_id,
            }
            for p in obs.printers
        ],
        "operators": [
            {
                "operator_id": o.operator_id,
                "skill_level": o.skill_level,
                "is_on_shift": o.is_on_shift,
                "queue_size": o.queue_size,
                "queue_capacity": o.queue_capacity,
                "current_fatigue": round(o.current_fatigue, 2),
                "busy_until": o.busy_until,
            }
            for o in obs.operators
        ],
    }
    return json.dumps(compact, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def _parse_action(raw: str) -> tuple[Optional[FarmAction], Optional[dict]]:
    """Return (FarmAction, parsed_dict) or (None, None) on parse failure."""
    text = raw.strip()
    # Strip code fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        # take content up to trailing fence
        if text.endswith("```"):
            text = text[:-3].strip()
    # Find first '{' to last '}'
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        parsed = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None, None

    try:
        action = FarmAction(**parsed)
        return action, parsed
    except Exception:
        return None, parsed


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

class _AnthropicClient:
    def __init__(self, model: str):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic()

    def call(self, obs_payload: str) -> tuple[str, int, int]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": obs_payload}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens


class _OpenAIClient:
    def __init__(self, model: str):
        import openai
        self.model = model
        self.client = openai.OpenAI()

    def call(self, obs_payload: str) -> tuple[str, int, int]:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_payload},
            ],
        )
        text = resp.choices[0].message.content or ""
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _run_episode(
    env: PrintFarmEnvironment,
    client,
    task_id: str,
    seed: int,
    episode: int,
    max_steps: int,
    ledger: dict,
    cost_cap: float,
) -> dict:
    obs = env.reset(seed=seed + episode, task_id=task_id)
    steps = []
    step_idx = 0

    while not obs.done and step_idx < max_steps:
        obs_payload = _compact_obs(obs)
        try:
            raw, ptoks, ctoks = client.call(obs_payload)
        except Exception as e:
            print(f"      API error: {e} — falling back to WAIT", flush=True)
            raw = '{"action":"WAIT"}'
            ptoks, ctoks = 0, 0

        cost = _price(client.model, ptoks, ctoks)
        ledger["total_prompt_tokens"]   += ptoks
        ledger["total_completion_tokens"] += ctoks
        ledger["total_cost_usd"]        += cost
        ledger["api_calls"]             += 1

        parsed_action, parsed_dict = _parse_action(raw)
        if parsed_action is None:
            action = FarmAction(action=FarmActionEnum.WAIT)
            parse_ok = False
        else:
            action = parsed_action
            parse_ok = True

        obs_json    = obs.model_dump(mode="json")
        action_json = action.model_dump(mode="json")
        obs         = env.step(action)
        step_reward_usd = obs.metadata.get("step_reward_usd", 0.0) if obs.metadata else 0.0

        steps.append({
            "step": step_idx,
            "observation": obs_json,
            "prompt_tokens": ptoks,
            "completion_tokens": ctoks,
            "raw_completion": raw,
            "parsed_action": parsed_dict,
            "parse_ok": parse_ok,
            "action": action_json,
            "reward": obs.reward,
            "step_reward_usd": step_reward_usd,
            "net_profit_usd": obs.net_profit_usd,
        })
        step_idx += 1

        if ledger["total_cost_usd"] > cost_cap:
            raise RuntimeError(
                f"Cost cap ${cost_cap} exceeded (currently ${ledger['total_cost_usd']:.2f})"
            )

    return {
        "task_id": task_id,
        "model": client.model,
        "seed": seed + episode,
        "episode": episode,
        "final_score": obs.reward,
        "final_profit_usd": obs.net_profit_usd,
        "final_step": step_idx,
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser(description="Generate teacher LLM rollouts")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="Model ID (must appear in PRICING table)")
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    ap.add_argument("--episodes-per-task", type=int, default=60,
                    help="Per plan §9.1: 500/8 ≈ 60")
    ap.add_argument("--max-steps-per-episode", type=int, default=300,
                    help="Safety cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-cost-usd", type=float, default=50.0)
    ap.add_argument("--output-root", default="data/teacher")
    args = ap.parse_args()

    # Check API key
    if args.provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: export ANTHROPIC_API_KEY first", file=sys.stderr)
            sys.exit(1)
        client = _AnthropicClient(args.model)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: export OPENAI_API_KEY first", file=sys.stderr)
            sys.exit(1)
        client = _OpenAIClient(args.model)

    if args.model not in PRICING:
        print(f"WARNING: no pricing entry for {args.model} — cost tracking disabled",
              file=sys.stderr)

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ledger = {
        "provider": args.provider,
        "model": args.model,
        "api_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cost_usd": 0.0,
        "max_cost_usd": args.max_cost_usd,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ledger_path = out_root / "cost_ledger.json"

    def save_ledger():
        ledger["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with ledger_path.open("w") as f:
            json.dump(ledger, f, indent=2)

    save_ledger()

    print(f"Provider:          {args.provider}")
    print(f"Model:             {args.model}")
    print(f"Tasks:             {', '.join(args.tasks)}")
    print(f"Episodes per task: {args.episodes_per_task}")
    print(f"Cost cap:          ${args.max_cost_usd}")
    print(f"Output root:       {out_root}")
    print()

    env = PrintFarmEnvironment()
    summary = {}
    aborted = False

    try:
        for task_id in args.tasks:
            task_dir = out_root / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = task_dir / "episodes.jsonl"
            scores, profits = [], []
            t0 = time.time()

            with jsonl_path.open("w") as f:
                for ep in range(args.episodes_per_task):
                    row = _run_episode(
                        env, client, task_id, args.seed, ep,
                        args.max_steps_per_episode, ledger, args.max_cost_usd,
                    )
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    scores.append(row["final_score"])
                    profits.append(row["final_profit_usd"])
                    save_ledger()
                    print(
                        f"    [{args.model} {task_id} {ep+1}/{args.episodes_per_task}] "
                        f"score={row['final_score']:.3f} "
                        f"profit=${row['final_profit_usd']:+7.2f} "
                        f"calls={ledger['api_calls']} "
                        f"spent=${ledger['total_cost_usd']:.2f}",
                        flush=True,
                    )

            summary[task_id] = {
                "episodes": args.episodes_per_task,
                "mean_score": round(sum(scores) / len(scores), 4),
                "mean_profit_usd": round(sum(profits) / len(profits), 2),
                "wall_clock_seconds": round(time.time() - t0, 1),
            }

    except RuntimeError as e:
        aborted = True
        print(f"\nABORTED: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        aborted = True
        print("\nInterrupted by user", file=sys.stderr)
    except Exception as e:
        aborted = True
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        traceback.print_exc()

    ledger["aborted"] = aborted
    save_ledger()

    with (out_root / "summary.json").open("w") as f:
        json.dump({"ledger": ledger, "per_task": summary}, f, indent=2)

    print(f"\n== DONE ==")
    print(f"API calls: {ledger['api_calls']}")
    print(f"Tokens:    {ledger['total_prompt_tokens']:,} in / "
          f"{ledger['total_completion_tokens']:,} out")
    print(f"Cost:      ${ledger['total_cost_usd']:.2f}")
    print(f"Summary:   {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
