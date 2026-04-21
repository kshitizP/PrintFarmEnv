from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PrinterProfile:
    profile_id: str
    display_name: str
    reliability_base: float
    avg_power_watts: float
    amortization_per_hour: float
    failure_rate_multipliers: Dict[str, float] = field(default_factory=dict)


PROFILES: Dict[str, PrinterProfile] = {
    "bambu_x1c": PrinterProfile(
        profile_id="bambu_x1c",
        display_name="Bambu Lab X1C",
        reliability_base=0.97,
        avg_power_watts=350,
        amortization_per_hour=0.20,
        failure_rate_multipliers={
            "thermistor_open": 0.5, "thermistor_short": 0.5,
            "filament_sensor_false_runout": 1.0, "filament_sensor_missed_runout": 0.5,
            "webcam_freeze": 1.0, "klipper_mcu_disconnect": 0.3,
            "progress_drift": 0.5, "fan_rpm_ghost": 0.5, "bed_level_drift": 0.5,
        },
    ),
    "prusa_mk4": PrinterProfile(
        profile_id="prusa_mk4",
        display_name="Prusa MK4",
        reliability_base=0.95,
        avg_power_watts=280,
        amortization_per_hour=0.12,
        failure_rate_multipliers={
            "thermistor_open": 0.8, "thermistor_short": 0.8,
            "filament_sensor_false_runout": 1.0, "filament_sensor_missed_runout": 0.8,
            "webcam_freeze": 1.0, "klipper_mcu_disconnect": 0.6,
            "progress_drift": 0.8, "fan_rpm_ghost": 0.8, "bed_level_drift": 0.8,
        },
    ),
    "creality_k1": PrinterProfile(
        profile_id="creality_k1",
        display_name="Creality K1",
        reliability_base=0.90,
        avg_power_watts=300,
        amortization_per_hour=0.08,
        failure_rate_multipliers={
            "thermistor_open": 1.2, "thermistor_short": 1.2,
            "filament_sensor_false_runout": 1.5, "filament_sensor_missed_runout": 1.5,
            "webcam_freeze": 1.0, "klipper_mcu_disconnect": 1.5,
            "progress_drift": 1.2, "fan_rpm_ghost": 1.5, "bed_level_drift": 1.5,
        },
    ),
    "voron_24": PrinterProfile(
        profile_id="voron_24",
        display_name="Voron 2.4",
        reliability_base=0.88,
        avg_power_watts=400,
        amortization_per_hour=0.18,
        failure_rate_multipliers={
            "thermistor_open": 1.0, "thermistor_short": 1.0,
            "filament_sensor_false_runout": 1.0, "filament_sensor_missed_runout": 1.0,
            "webcam_freeze": 1.5, "klipper_mcu_disconnect": 2.0,
            "progress_drift": 1.0, "fan_rpm_ghost": 1.0, "bed_level_drift": 1.2,
        },
    ),
}
