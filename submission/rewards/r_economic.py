"""
r_economic — net P&L delta over K steps vs. rules-only baseline.

Range roughly [-1, +1]. Computed as the normalized difference between:
  - reward obtained after the LLM's action + K rule steps
  - reward obtained by running rules for K+1 steps (counterfactual)

Normalization: clip to [-1, +1] for stable GRPO training.
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
        Normalized reward in [-1, +1].
    """
    diff = llm_reward_delta - rules_reward_delta

    # Normalize: $10 of advantage = +1.0
    normalized = diff / 10.0
    return _finite_clamp(normalized, -1.0, 1.0)


def _finite_clamp(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(lo, min(hi, x))
