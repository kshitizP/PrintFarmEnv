"""Dollar-denominated economic constants and helpers for the PrintFarm environment."""

from typing import Optional

# ---------------------------------------------------------------------------
#  Filament
# ---------------------------------------------------------------------------

FILAMENT_PRICE_PER_G: dict = {
    "PLA": 0.020,
    "PETG": 0.025,
    "ABS": 0.030,
}
_DEFAULT_FILAMENT_PRICE = 0.025


def filament_price(material: str) -> float:
    return FILAMENT_PRICE_PER_G.get(material, _DEFAULT_FILAMENT_PRICE)


def filament_cost(material: str, weight_g: float) -> float:
    return weight_g * filament_price(material)


# ---------------------------------------------------------------------------
#  Electricity  ($0.15/kWh × W × 1/60 h per step = W × 2.5e-6 $/step)
# ---------------------------------------------------------------------------

_KWH_RATE = 2.5e-6  # $/step/W

_ELECTRICITY_MULT = {
    "PRINTING": 1.0,
    "MAINTENANCE": 1.0,
    "WARMING_UP": 1.5,
    "PAUSED": 1.0,
    "PAUSED_RUNOUT": 0.5,
}


def electricity_cost(power_watts: float, state_name: str) -> float:
    mult = _ELECTRICITY_MULT.get(state_name, 0.0)
    return power_watts * _KWH_RATE * mult


# ---------------------------------------------------------------------------
#  Amortization
# ---------------------------------------------------------------------------

def amortization_cost(amortization_per_hour: float) -> float:
    """Cost per PRINTING step."""
    return amortization_per_hour / 60.0


# ---------------------------------------------------------------------------
#  Job economics
# ---------------------------------------------------------------------------

def step_revenue(price_usd: float, print_time_steps: int) -> float:
    return price_usd / max(print_time_steps, 1)


def scrap_cost(material: str, weight_required_g: float,
               progress_steps: int, print_time_steps: int) -> float:
    fraction = progress_steps / max(print_time_steps, 1)
    used_g = weight_required_g * fraction
    return filament_cost(material, used_g)


# ---------------------------------------------------------------------------
#  SLA penalties
# ---------------------------------------------------------------------------

SLA_FIXED_PENALTY = 50.0
SLA_PER_STEP_PENALTY = 5.0
SLA_CAP_FRACTION = 0.80  # never more than 80% of job price


def sla_penalty_this_step(
    price_usd: float,
    deadline_steps: Optional[int],
    current_step: int,
    fixed_applied: bool,
) -> tuple[float, bool]:
    """Return (penalty_this_step, fixed_now_applied).

    Caller must track fixed_applied and pass it back each step.
    Returns (0.0, False) when no deadline or not yet late.
    """
    if deadline_steps is None or current_step <= deadline_steps:
        return 0.0, fixed_applied

    cap = price_usd * SLA_CAP_FRACTION
    penalty = 0.0
    new_fixed = fixed_applied

    if not fixed_applied:
        penalty += min(SLA_FIXED_PENALTY, cap)
        new_fixed = True

    penalty += SLA_PER_STEP_PENALTY

    # Cap is enforced cumulatively by the caller (per-episode total)
    return penalty, new_fixed


# ---------------------------------------------------------------------------
#  Action costs
# ---------------------------------------------------------------------------

ACTION_REJECT_COST = 0.20
WAIT_COST = 0.10
OVERRIDE_COST = 0.10
DIAGNOSTIC_BASE_COST = 0.50
DIAGNOSTIC_BONUS = 2.00
CATASTROPHIC_COST = 250.0
