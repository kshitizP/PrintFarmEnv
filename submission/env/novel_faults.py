"""
Novel faults generator — inject failure modes that don't match the 9-fault taxonomy.

Three compound failure modes:
1. hybrid_thermal_humidity: thermistor normal, but humidity+temp = imminent warp
2. degraded_webcam: feed transmitting but blur rising slowly
3. intermittent_mcu: pings work 8/10 times (not a clean disconnect)

The rule-based agent's 9-fault taxonomy won't catch these.
An LLM reasoning over text+telemetry can.
"""

import random
from typing import Any, Dict, List, Tuple


# Ground truth: which anomaly flag text maps to which actual condition
NOVEL_FAULT_DEFS = {
    "hybrid_thermal_humidity": {
        "anomaly_text": "P{pid}: unusual thermal pattern — temps nominal but ambient humidity elevated",
        "detail": "Combination of normal thermistor + high humidity + ambient temp predicts warp",
        "correct_action": "investigate",  # RUN_DIAGNOSTIC or preemptive PAUSE
    },
    "degraded_webcam": {
        "anomaly_text": "P{pid}: webcam quality degrading — image hash changing but blur metrics rising",
        "detail": "Feed still transmitting but resolution/clarity dropping progressively",
        "correct_action": "investigate",  # RUN_DIAGNOSTIC to check actual print quality
    },
    "intermittent_mcu": {
        "anomaly_text": "P{pid}: intermittent communication — telemetry updates irregular",
        "detail": "MCU pings succeed 80% of the time; not a clean disconnect",
        "correct_action": "investigate",  # Needs monitoring; may need maintenance
    },
}


def generate_anomaly_flags(
    step: int,
    printers: list,
    rng: random.Random,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Generate 0-1 novel anomaly flags for this step.

    Returns:
        (visible_flags, ground_truth_tags)

        visible_flags: list of vague anomaly strings for the observation.
        ground_truth_tags: list of dicts with hidden metadata for the reward fn.
    """
    # ~15% chance of a novel fault signal per step
    if rng.random() > 0.15:
        return [], []

    # Only flag printers that are actively printing (where it matters)
    printing = [p for p in printers
                if getattr(getattr(p, 'state', ''), 'value', str(getattr(p, 'state', '')))
                in ("PRINTING", "WARMING_UP")]
    if not printing:
        return [], []

    printer = rng.choice(printing)
    pid = printer.printer_id

    fault_type = rng.choice(list(NOVEL_FAULT_DEFS.keys()))
    fdef = NOVEL_FAULT_DEFS[fault_type]

    flag_text = fdef["anomaly_text"].format(pid=pid)

    visible = [flag_text]
    tags = [{
        "fault_type": fault_type,
        "printer_id": pid,
        "correct_action": fdef["correct_action"],
        "detail": fdef["detail"],
    }]

    return visible, tags
