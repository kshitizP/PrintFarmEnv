"""
r_economic — net P&L delta over K steps vs. rules-only baseline.

Range roughly [-0.4, +0.4]. Computed as the normalized difference between:
  - reward obtained after the LLM's action + K rule steps
  - reward obtained by running rules for K+1 steps (counterfactual)

Normalization: $10 advantage = +1.0, then scaled by 0.4 so unstructured
signals can compete.
"""

import math


def r_economic(
    llm_reward_delta: float,
    rules_reward_delta: float,
) -> float:
    """Return economic advantage of the LLM over rules.

    Args:
        llm_reward_delta: reward delta from LLM action + K rule steps.
        rules_reward_delta: reward delta from K+1 rule steps (counterfactual).

    Returns:
        Scaled reward in [-0.4, +0.4].
    """
    diff = llm_reward_delta - rules_reward_delta

    # Normalize: $10 of advantage = +1.0
    normalized = diff / 10.0
    clamped = _finite_clamp(normalized, -1.0, 1.0)
    return clamped * 0.4  # scale so unstructured signals can compete


def _finite_clamp(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(lo, min(hi, x))
