from __future__ import annotations

from dataclasses import dataclass

from .constants import RadarConfig, ScenarioConfig
from .detection import Detection
from .geometry import delay_to_bistatic_range_m, doppler_hz_to_velocity_mps


@dataclass(frozen=True)
class TruthTarget:
    range_m: float
    doppler_hz: float
    velocity_mps: float


def scenario_truth(scenario: ScenarioConfig, config: RadarConfig) -> tuple[TruthTarget, ...]:
    return tuple(
        TruthTarget(
            range_m=delay_to_bistatic_range_m(target.delay_s, config),
            doppler_hz=target.doppler_hz,
            velocity_mps=doppler_hz_to_velocity_mps(target.doppler_hz, config),
        )
        for target in scenario.targets
    )


def match_detection_to_truth(
    detection: Detection,
    truths: tuple[TruthTarget, ...],
    range_tolerance_m: float,
    doppler_tolerance_hz: float,
) -> bool:
    for truth in truths:
        range_ok = abs(detection.range_m - truth.range_m) <= range_tolerance_m
        doppler_ok = abs(detection.doppler_hz - truth.doppler_hz) <= doppler_tolerance_hz
        if range_ok and doppler_ok:
            return True
    return False
