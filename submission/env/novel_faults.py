"""
Novel faults generator — inject failure modes that don't match the 9-fault taxonomy.

Eight compound failure modes (expanded from 3 to prevent template memorisation):
1. hybrid_thermal_humidity: thermistor normal, but humidity+temp = imminent warp
2. degraded_webcam: feed transmitting but blur rising slowly
3. intermittent_mcu: pings work 8/10 times (not a clean disconnect)
4. slow_extrusion_drift: extruder steps nominal but flow rate creeping down
5. oscillating_z_wobble: z-height varies cyclically, prints look fine until layer 30+
6. partial_bed_delamination: one bed zone losing adhesion while others are fine
7. resonance_buildup: print quality fine at low speed, ringing at high speed moves
8. filament_moisture_absorption: spool sat open; prints look fine but layer bonding weak

Plus "resolved alarm" decoys — same vocabulary as real faults but confirmed benign.
These prevent the model learning "fault phrase → investigate" as a universal rule.

The rule-based agent's 9-fault taxonomy won't catch the real novel faults.
An LLM reasoning over the text descriptions can distinguish real from resolved.
"""

import random
from typing import Any, Dict, List, Tuple


# Ground truth: which anomaly flag text maps to which actual condition
NOVEL_FAULT_DEFS = {
    "hybrid_thermal_humidity": {
        "anomaly_text": "P{pid}: unusual thermal pattern — temps nominal but ambient humidity elevated",
        "detail": "Combination of normal thermistor + high humidity + ambient temp predicts warp",
        "correct_action": "investigate",
    },
    "degraded_webcam": {
        "anomaly_text": "P{pid}: webcam quality degrading — image hash changing but blur metrics rising",
        "detail": "Feed still transmitting but resolution/clarity dropping progressively",
        "correct_action": "investigate",
    },
    "intermittent_mcu": {
        "anomaly_text": "P{pid}: intermittent communication — telemetry updates irregular",
        "detail": "MCU pings succeed 80% of the time; not a clean disconnect",
        "correct_action": "investigate",
    },
    "slow_extrusion_drift": {
        "anomaly_text": "P{pid}: extrusion flow rate drifting — stepper counts nominal but output volume trending down",
        "detail": "Partial clog or worn drive gear causing cumulative under-extrusion",
        "correct_action": "investigate",
    },
    "oscillating_z_wobble": {
        "anomaly_text": "P{pid}: z-axis height inconsistency detected — variance within tolerance but oscillating",
        "detail": "Cyclical z-height error invisible until high layer count; lead screw or coupler issue",
        "correct_action": "investigate",
    },
    "partial_bed_delamination": {
        "anomaly_text": "P{pid}: bed adhesion variance — front zone holding, rear zone losing grip",
        "detail": "Partial delamination will cause print failure mid-job on large footprint parts",
        "correct_action": "investigate",
    },
    "resonance_buildup": {
        "anomaly_text": "P{pid}: vibration signature changing — low-speed moves clean, high-speed shows ringing",
        "detail": "Frame resonance or loose belt causing speed-dependent print quality degradation",
        "correct_action": "investigate",
    },
    "filament_moisture_absorption": {
        "anomaly_text": "P{pid}: print surface texture anomaly — layer bonding metrics show micro-gaps",
        "detail": "Moisture-absorbed filament; surface looks fine but interlayer bonding compromised",
        "correct_action": "investigate",
    },
}

# Resolved-alarm decoys: anomaly flag text that LOOKS like a fault but was confirmed benign.
# Correct action for these is WAIT (no investigation needed).
# Generated alongside real faults ~30% of the time to prevent "flag text → investigate" shortcuts.
RESOLVED_ALARM_DEFS = {
    "resolved_thermal_alert": {
        "anomaly_text": "P{pid}: thermal pattern deviation flagged — ops confirmed ambient HVAC cycling, printer nominal",
        "detail": "HVAC unit nearby caused transient ambient spike; thermistor and print unaffected",
        "correct_action": "wait",  # already investigated and cleared
    },
    "resolved_webcam_glitch": {
        "anomaly_text": "P{pid}: webcam feed quality drop — IT confirmed network packet loss, camera hardware fine",
        "detail": "Switch-level packet loss caused blur metrics spike; print quality unaffected",
        "correct_action": "wait",
    },
    "resolved_comms_blip": {
        "anomaly_text": "P{pid}: telemetry gap detected — scheduler confirmed firmware update restart, now stable",
        "detail": "Planned firmware update caused one missed ping cycle; MCU healthy",
        "correct_action": "wait",
    },
    "resolved_extrusion_spike": {
        "anomaly_text": "P{pid}: extrusion rate anomaly flagged — ops confirmed new spool loaded, calibration normal",
        "detail": "New spool has slightly different diameter; re-calibrated, output nominal",
        "correct_action": "wait",
    },
    "resolved_bed_alert": {
        "anomaly_text": "P{pid}: bed adhesion variance detected — ops re-levelled and re-cleaned, holding fine now",
        "detail": "Dust on bed corner caused adhesion metric drop; after cleaning, all zones nominal",
        "correct_action": "wait",
    },
}


def generate_anomaly_flags(
    step: int,
    printers: list,
    rng: random.Random,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Generate 0-1 anomaly flags for this step.

    ~15% chance of a REAL novel fault (correct action: investigate).
    ~5% chance of a RESOLVED ALARM decoy (correct action: wait).
    Decoys use similar vocabulary to real faults — forcing the model to
    read the resolution text, not just pattern-match on fault vocabulary.

    Returns:
        (visible_flags, ground_truth_tags)

        visible_flags: list of anomaly strings for the observation.
        ground_truth_tags: list of dicts with hidden metadata for the reward fn.
    """
    roll = rng.random()

    # Only flag printers that are actively printing (where it matters)
    printing = [p for p in printers
                if getattr(getattr(p, 'state', ''), 'value', str(getattr(p, 'state', '')))
                in ("PRINTING", "WARMING_UP")]
    if not printing:
        return [], []

    printer = rng.choice(printing)
    pid = printer.printer_id

    if roll < 0.15:
        # Real novel fault
        fault_type = rng.choice(list(NOVEL_FAULT_DEFS.keys()))
        fdef = NOVEL_FAULT_DEFS[fault_type]
        flag_text = fdef["anomaly_text"].format(pid=pid)
        tags = [{
            "fault_type": fault_type,
            "printer_id": pid,
            "correct_action": fdef["correct_action"],
            "detail": fdef["detail"],
            "is_resolved": False,
        }]
    elif roll < 0.20:
        # Resolved alarm decoy — looks like a fault but already cleared
        alarm_type = rng.choice(list(RESOLVED_ALARM_DEFS.keys()))
        adef = RESOLVED_ALARM_DEFS[alarm_type]
        flag_text = adef["anomaly_text"].format(pid=pid)
        tags = [{
            "fault_type": alarm_type,
            "printer_id": pid,
            "correct_action": adef["correct_action"],  # "wait"
            "detail": adef["detail"],
            "is_resolved": True,
        }]
    else:
        return [], []

    return [flag_text], tags
