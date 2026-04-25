"""
Observation → structured text formatter.

Converts FarmObservation (or serialized JSON) into a compact, human-readable
text summary. This is structurally distinct from JSON, making it impossible
for the model to "echo" the input as a valid action response.

Used by: training rollouts, eval, inference. ONE source of truth.
"""

import json
from typing import Any, Dict, List, Optional, Union


def format_observation_as_text(obs: Union[str, dict, Any]) -> str:
    """Convert observation to structured text for the model prompt.

    Accepts:
      - A raw JSON string (from serialize_obs)
      - A dict
      - A Pydantic model with model_dump()

    Returns a compact multi-line text summary.
    """
    if isinstance(obs, str):
        try:
            data = json.loads(obs)
        except (json.JSONDecodeError, ValueError):
            return obs  # can't parse, return as-is
    elif hasattr(obs, "model_dump"):
        data = obs.model_dump()
    elif isinstance(obs, dict):
        data = obs
    else:
        return str(obs)

    sections = []

    # Header
    t = data.get("time_step", "?")
    t_max = data.get("max_steps", "?")
    profit = data.get("net_profit_usd", 0.0)
    sections.append(f"STEP {t}/{t_max} | Net profit: ${profit:.2f}")

    # Printers
    printers = data.get("printers", [])
    if printers:
        lines = ["PRINTERS:"]
        for p in printers:
            pid = p.get("printer_id", "?")
            state = p.get("state", "?")
            mat = p.get("current_material", "none")
            spool = p.get("spool_weight_g", 0)
            fatigue = p.get("fatigue_level", 0)
            reliability = p.get("reliability", 0.95)
            job = p.get("current_job_id")
            temp = p.get("hotend_temp", 0)
            maint = p.get("maintenance_due_in", 50)
            webcam = p.get("webcam_hash", "")

            parts = [f"P{pid}: {state}"]
            if mat and mat != "none":
                parts.append(f"mat={mat}")
            if spool > 0:
                parts.append(f"spool={spool:.0f}g")
            if job:
                parts.append(f"job={job}")
            if fatigue > 0:
                parts.append(f"fatigue={fatigue:.1f}")
            if reliability < 0.9:
                parts.append(f"reliability={reliability:.2f}")
            if temp == 0 or temp > 400:
                parts.append(f"hotend={temp:.0f}C(!)")
            if maint < 10:
                parts.append(f"maint_due_in={maint}")
            if p.get("outstanding_ticket_id"):
                parts.append(f"ticket={p['outstanding_ticket_id']}")
            if p.get("warmup_remaining", 0) > 0:
                parts.append(f"warming({p['warmup_remaining']})")
            if p.get("reliability_penalty_active"):
                parts.append("PENALTY_ACTIVE")

            lines.append(f"  - {', '.join(parts)}")
        sections.append("\n".join(lines))

    # Jobs (active queue, skip completed/cancelled)
    queue = data.get("active_queue", [])
    active_jobs = [j for j in queue if j.get("state") not in ("COMPLETED", "CANCELLED")]
    if active_jobs:
        lines = ["JOBS:"]
        for j in active_jobs:
            jid = j.get("job_id", "?")
            state = j.get("state", "?")
            mat = j.get("material_required", "?")
            weight = j.get("weight_required_g", 0)
            time_steps = j.get("print_time_steps", 0)
            progress = j.get("progress_steps", 0)
            price = j.get("price_usd", 0)
            deadline = j.get("deadline_steps")
            priority = j.get("priority", 2)

            parts = [f"{jid}: {state}"]
            parts.append(f"mat={mat}")
            if state == "PRINTING" and time_steps > 0:
                pct = progress / time_steps * 100
                parts.append(f"progress={pct:.0f}%")
            elif state == "PENDING":
                parts.append(f"weight={weight:.0f}g")
                parts.append(f"time={time_steps}steps")
            parts.append(f"${price:.0f}")
            if deadline is not None:
                remaining = deadline - data.get("time_step", 0)
                parts.append(f"deadline_in={remaining}steps")
            if priority != 2:
                parts.append(f"priority={'HIGH' if priority == 3 else 'LOW'}")

            lines.append(f"  - {', '.join(parts)}")
        sections.append("\n".join(lines))

    # Operators
    operators = data.get("operators", [])
    if operators:
        lines = ["OPERATORS:"]
        for op in operators:
            oid = op.get("operator_id", "?")
            skill = op.get("skill_level", "?")
            on_shift = op.get("is_on_shift", False)
            qsize = op.get("queue_size", 0)
            qcap = op.get("queue_capacity", 0)
            fatigue = op.get("current_fatigue", 0)
            ticket = op.get("current_ticket_id")

            status = "ON-SHIFT" if on_shift else "OFF-SHIFT"
            parts = [f"{oid}: {skill}, {status}, queue={qsize}/{qcap}"]
            if fatigue > 0.5:
                parts.append(f"fatigue={fatigue:.1f}")
            if ticket:
                parts.append(f"working_on={ticket}")
            lines.append(f"  - {', '.join(parts)}")
        sections.append("\n".join(lines))

    # Inventory
    inv = data.get("inventory", {})
    if inv:
        inv_str = ", ".join(f"{k}={v:.0f}g" for k, v in inv.items() if v > 0)
        if inv_str:
            sections.append(f"INVENTORY: {inv_str}")

    # Unstructured signals — these are the key decision triggers
    notes = data.get("operator_notes", [])
    if notes:
        lines = ["OPERATOR NOTES:"]
        for note in notes:
            lines.append(f"  - {note}")
        sections.append("\n".join(lines))

    messages = data.get("customer_messages", [])
    if messages:
        lines = ["CUSTOMER MESSAGES:"]
        for msg in messages:
            if isinstance(msg, dict):
                lines.append(f"  - [{msg.get('urgency', '?')}] job={msg.get('job_id', '?')}: {msg.get('text', '')}")
            else:
                lines.append(f"  - {msg}")
        sections.append("\n".join(lines))

    anomalies = data.get("anomaly_flags", [])
    if anomalies:
        lines = ["ANOMALY FLAGS:"]
        for flag in anomalies:
            lines.append(f"  - {flag}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
