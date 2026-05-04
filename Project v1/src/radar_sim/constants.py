from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RadarConfig:
    speed_of_light: float = 3.0e8
    carrier_frequency_hz: float = 2.4e9
    sample_rate_hz: float = 20.0e6
    pri_s: float = 250.0e-6
    num_pulses: int = 128
    num_subcarriers: int = 256
    cyclic_prefix: int = 64
    direct_path_amplitude_db: float = -35.0
    clutter_amplitude_db: float = -45.0
    noise_power_db: float = -75.0
    epsilon: float = 1.0e-12

    @property
    def wavelength_m(self) -> float:
        return self.speed_of_light / self.carrier_frequency_hz

    @property
    def samples_per_pulse(self) -> int:
        return self.num_subcarriers + self.cyclic_prefix


@dataclass(frozen=True)
class CFARConfig:
    guard_cells_range: int = 2
    guard_cells_doppler: int = 2
    train_cells_range: int = 8
    train_cells_doppler: int = 8
    pfa: float = 1.0e-5
    min_abs_velocity_mps: float = 1.0


@dataclass(frozen=True)
class Target:
    delay_s: float
    doppler_hz: float
    amplitude_db: float

    @property
    def amplitude_linear(self) -> float:
        return 10.0 ** (self.amplitude_db / 20.0)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    targets: tuple[Target, ...] = field(default_factory=tuple)
    noise_power_db: float = -75.0
    clutter_amplitude_db: float = -45.0


@dataclass(frozen=True)
class AppConfig:
    seed: int
    radar: RadarConfig
    cfar: CFARConfig
    scenarios: dict[str, ScenarioConfig]


def _load_targets(items: list[dict[str, Any]]) -> tuple[Target, ...]:
    return tuple(
        Target(
            delay_s=float(item["delay_s"]),
            doppler_hz=float(item["doppler_hz"]),
            amplitude_db=float(item["amplitude_db"]),
        )
        for item in items
    )


def load_app_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    radar = RadarConfig(**raw["radar"])
    cfar = CFARConfig(**raw["cfar"])

    scenarios: dict[str, ScenarioConfig] = {}
    for key, value in raw["scenarios"].items():
        scenarios[key] = ScenarioConfig(
            name=str(value["name"]),
            targets=_load_targets(value.get("targets", [])),
            noise_power_db=float(value.get("noise_power_db", radar.noise_power_db)),
            clutter_amplitude_db=float(
                value.get("clutter_amplitude_db", radar.clutter_amplitude_db)
            ),
        )

    return AppConfig(
        seed=int(raw["seed"]),
        radar=radar,
        cfar=cfar,
        scenarios=scenarios,
    )
