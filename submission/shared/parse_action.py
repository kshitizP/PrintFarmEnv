"""
Strict action parser with Pydantic validation.

Handles (in priority order):
  1. <action>JSON</action> tags (preferred format)
  2. Clean JSON
  3. JSON inside markdown fences
  4. JSON buried in free text
  5. Completely unparseable output → None (caller assigns WAIT + format penalty)
"""

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class AgentAction(BaseModel):
    """Parsed agent action — strict schema for GRPO training."""
    action_type: Literal[
        "ASSIGN_JOB", "CANCEL_JOB", "PAUSE_JOB", "RESUME_JOB",
        "RUN_DIAGNOSTIC", "DISPATCH_TICKET",
        "REQUEST_SPOOL_SWAP", "REQUEST_MAINTENANCE",
        "OVERRIDE_OPERATOR", "WAIT",
    ]
    printer_id: Optional[int] = None
    job_id: Optional[str] = None
    operator_id: Optional[str] = None
    ticket_type: Optional[str] = None
    ticket_id: Optional[str] = None
    material: Optional[str] = None
    maintenance_type: Optional[str] = None
    reason: Optional[str] = None
    reasoning: Optional[str] = Field(default=None, max_length=200, exclude=True)

    @model_validator(mode="after")
    def validate_required_fields(self):
        required = {
            "ASSIGN_JOB": ["printer_id", "job_id"],
            "CANCEL_JOB": ["job_id"],
            "PAUSE_JOB": ["printer_id"],
            "RESUME_JOB": ["job_id"],
            "RUN_DIAGNOSTIC": ["printer_id"],
            "DISPATCH_TICKET": ["operator_id", "ticket_type"],
            "REQUEST_SPOOL_SWAP": ["printer_id", "material"],
            "REQUEST_MAINTENANCE": ["printer_id", "maintenance_type"],
            "OVERRIDE_OPERATOR": ["ticket_id", "reason"],
            "WAIT": [],
        }
        missing = [
            f for f in required.get(self.action_type, [])
            if getattr(self, f) is None
        ]
        if missing:
            raise ValueError(
                f"{self.action_type} requires: {missing}"
            )
        return self


# Precompiled regex for <action> tag extraction
_ACTION_TAG_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)


def parse_action(text: str) -> Optional[AgentAction]:
    """Parse model output text into an AgentAction, or return None if unparseable."""
    if not text or not text.strip():
        return None

    text = text.strip()
    data = None

    # 1. Try <action>JSON</action> tags (preferred format)
    tag_match = _ACTION_TAG_RE.search(text)
    if tag_match:
        try:
            data = json.loads(tag_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Try direct JSON parse
    if data is None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            pass

    # 3. Try extracting from markdown fences
    if data is None:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # 4. Try extracting first JSON object from text
    if data is None:
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    if data is None:
        return None

    # Normalize the "action" key to "action_type"
    if "action" in data and "action_type" not in data:
        data["action_type"] = data.pop("action")

    # Remove any unknown keys to avoid Pydantic validation errors
    known_keys = set(AgentAction.model_fields.keys())
    data = {k: v for k, v in data.items() if k in known_keys}

    try:
        return AgentAction(**data)
    except Exception:
        return None


def action_to_farm_action(agent_action: AgentAction):
    """Convert an AgentAction to a FarmAction for the environment."""
    # Lazy import to avoid circular dependency
    from submission.env.models import FarmAction, FarmActionEnum

    kwargs = {"action": FarmActionEnum(agent_action.action_type)}

    if agent_action.printer_id is not None:
        kwargs["printer_id"] = agent_action.printer_id
    if agent_action.job_id is not None:
        kwargs["job_id"] = agent_action.job_id
    if agent_action.operator_id is not None:
        kwargs["operator_id"] = agent_action.operator_id
    if agent_action.ticket_type is not None:
        kwargs["ticket_type"] = agent_action.ticket_type
    if agent_action.ticket_id is not None:
        kwargs["ticket_id"] = agent_action.ticket_id
    if agent_action.material is not None:
        kwargs["material"] = agent_action.material
    if agent_action.maintenance_type is not None:
        kwargs["maintenance_type"] = agent_action.maintenance_type
    if agent_action.reason is not None:
        kwargs["reason"] = agent_action.reason

    return FarmAction(**kwargs)
