"""
Canonical observation serializer.

ONE function, used everywhere: rollout, training, eval, tests.
This prevents the observation-format-mismatch bug that killed round 1.

Design rules:
  - Always serializes the full JSON. No compression, no field removal.
  - Deterministic output for identical state (sorted keys).
  - Returns a string, not bytes.
"""

import json
from typing import Any


def serialize_obs(obs) -> str:
    """Convert a FarmObservation (or dict) to a canonical JSON string.

    If *obs* is a Pydantic model, uses model_dump(). Otherwise treats it as
    a plain dict. Output is sorted-key JSON, no indent — compact but
    reproducible.
    """
    if hasattr(obs, "model_dump"):
        data = obs.model_dump()
    elif hasattr(obs, "dict"):          # pydantic v1 fallback
        data = obs.dict()
    elif isinstance(obs, dict):
        data = obs
    else:
        raise TypeError(f"Cannot serialize {type(obs)}")

    return json.dumps(data, default=_json_default, sort_keys=True)


def _json_default(obj: Any) -> Any:
    """Handle enums, sets, and other non-standard types."""
    if hasattr(obj, "value"):           # enum
        return obj.value
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
