"""
OpenEnv Rubric wrappers for PrintFarm reward components.

Wraps each reward component as an OpenEnv Rubric for the evaluation system.
Falls back to a minimal Rubric base class if openenv is not installed.
"""

from typing import Any, Dict, List

from .r_format import r_format
from .r_economic import r_economic
from .r_fault_precision import r_fault_precision
from .r_message_handling import r_message_handling
from .r_unnecessary_action import r_unnecessary_action
from .r_novel_fault import r_novel_fault

# Try importing OpenEnv Rubric; fall back to a minimal base class
try:
    from openenv.rubric import Rubric
except ImportError:
    try:
        from openenv_core.rubric import Rubric
    except ImportError:
        class Rubric:
            """Minimal Rubric base class when openenv is not installed."""
            name: str = ""
            weight: float = 1.0

            def score(self, trajectory) -> float:
                raise NotImplementedError


class FormatRubric(Rubric):
    name = "format_compliance"
    weight = 1.0

    def score(self, trajectory) -> float:
        return r_format(
            trajectory.parsed_action,
            model_output=getattr(trajectory, "last_output", ""),
            observation_text=getattr(trajectory, "observation_text", ""),
        )


class EconomicRubric(Rubric):
    name = "economic_advantage"
    weight = 1.0  # weight already applied inside r_economic

    def score(self, trajectory) -> float:
        return r_economic(
            getattr(trajectory, "rollout_value", 0.0),
            getattr(trajectory, "counterfactual_value", 0.0),
        )


class FaultPrecisionRubric(Rubric):
    name = "fault_precision"
    weight = 1.0

    def score(self, trajectory) -> float:
        gt_tags = getattr(trajectory, "ground_truth_tags", {})
        return r_fault_precision(
            trajectory.parsed_action,
            gt_tags.get("anomaly_flags", []),
            gt_tags.get("operator_notes", []),
            observation=getattr(trajectory, "observation", None),
        )


class MessageHandlingRubric(Rubric):
    name = "message_handling"
    weight = 1.0

    def score(self, trajectory) -> float:
        gt_tags = getattr(trajectory, "ground_truth_tags", {})
        return r_message_handling(
            trajectory.parsed_action,
            gt_tags.get("customer_messages", []),
        )


class UnnecessaryActionRubric(Rubric):
    name = "unnecessary_action"
    weight = 1.0

    def score(self, trajectory) -> float:
        gt_tags = getattr(trajectory, "ground_truth_tags", {})
        return r_unnecessary_action(
            trajectory.parsed_action,
            gt_tags.get("anomaly_flags", []),
            gt_tags.get("operator_notes", []),
        )


class NovelFaultRubric(Rubric):
    name = "novel_fault"
    weight = 1.0

    def score(self, trajectory) -> float:
        gt_tags = getattr(trajectory, "ground_truth_tags", {})
        return r_novel_fault(
            trajectory.parsed_action,
            gt_tags.get("anomaly_flags", []),
            observation=getattr(trajectory, "observation", None),
        )


ALL_RUBRICS = [
    FormatRubric(),
    EconomicRubric(),
    FaultPrecisionRubric(),
    MessageHandlingRubric(),
    UnnecessaryActionRubric(),
    NovelFaultRubric(),
]


class PrintFarmRubric(Rubric):
    """Composite rubric that scores all components."""
    name = "printfarm_composite"
    weight = 1.0

    def __init__(self):
        self.components = ALL_RUBRICS

    def score(self, trajectory) -> Dict[str, float]:
        result = {}
        for r in self.components:
            result[r.name] = r.score(trajectory)
        result["total"] = sum(result.values())
        return result
