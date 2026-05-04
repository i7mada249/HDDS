from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "Project v1"
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from radar_sim.constants import RadarConfig, ScenarioConfig, Target
from radar_sim.geometry import (
    bistatic_range_to_delay_s,
    velocity_mps_to_doppler_hz,
)


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")
TARGET_KEYS = {
    "distance_m",
    "range_m",
    "far_m",
    "range",
    "distance",
    "speed_mps",
    "velocity_mps",
    "radial_velocity_mps",
    "speed",
    "velocity",
    "delay_s",
    "doppler_hz",
    "doppler",
    "dpb",
    "dbp",
    "amplitude_db",
    "power_db",
    "rcs_db",
    "signal_db",
    "db",
}


@dataclass(frozen=True)
class ScenarioBundle:
    name: str
    video_path: Path
    audio_path: Path
    radar_path: Path


@dataclass(frozen=True)
class RadarObject:
    name: str
    distance_m: float
    speed_mps: float
    doppler_hz: float
    delay_s: float
    amplitude_db: float


@dataclass(frozen=True)
class RadarScenarioData:
    scenario: ScenarioConfig
    objects: tuple[RadarObject, ...]
    raw: dict[str, Any]


def discover_scenarios(directory: Path) -> list[ScenarioBundle]:
    files = [path for path in directory.iterdir() if path.is_file()]
    by_stem: dict[str, dict[str, Path]] = {}
    for path in files:
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            by_stem.setdefault(path.stem, {})["video"] = path
        elif suffix in AUDIO_EXTENSIONS:
            by_stem.setdefault(path.stem, {})["audio"] = path
        elif suffix == ".json":
            by_stem.setdefault(path.stem, {})["radar"] = path

    bundles = []
    for stem, parts in by_stem.items():
        if {"video", "audio", "radar"} <= parts.keys():
            bundles.append(
                ScenarioBundle(
                    name=stem,
                    video_path=parts["video"],
                    audio_path=parts["audio"],
                    radar_path=parts["radar"],
                )
            )
    return sorted(bundles, key=lambda item: item.name.lower())


def bundle_from_member(path: Path) -> ScenarioBundle:
    directory = path.parent
    stem = path.stem
    video_path = _find_required(directory, stem, VIDEO_EXTENSIONS, "video")
    audio_path = _find_required(directory, stem, AUDIO_EXTENSIONS, "audio")
    radar_path = directory / f"{stem}.json"
    if not radar_path.exists():
        raise FileNotFoundError(f"Missing radar JSON for scenario '{stem}': {radar_path}")
    return ScenarioBundle(
        name=stem,
        video_path=video_path,
        audio_path=audio_path,
        radar_path=radar_path,
    )


def load_radar_scenario(path: Path, radar: RadarConfig) -> RadarScenarioData:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Radar JSON must contain an object at the top level.")

    target_items = raw.get("targets") or raw.get("objects") or raw.get("detections")
    if target_items is None:
        target_items = [raw] if TARGET_KEYS.intersection(raw.keys()) else []
    if not isinstance(target_items, list):
        raise ValueError("Radar JSON 'targets' must be a list when provided.")

    objects: list[RadarObject] = []
    targets: list[Target] = []
    for index, item in enumerate(target_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Radar target {index} must be an object.")
        radar_object = _parse_radar_object(item, index=index, radar=radar)
        objects.append(radar_object)
        targets.append(
            Target(
                delay_s=radar_object.delay_s,
                doppler_hz=radar_object.doppler_hz,
                amplitude_db=radar_object.amplitude_db,
            )
        )

    scenario = ScenarioConfig(
        name=str(raw.get("name") or path.stem),
        targets=tuple(targets),
        noise_power_db=_float_from(raw, ("noise_power_db", "noise_db"), radar.noise_power_db),
        clutter_amplitude_db=_float_from(
            raw,
            ("clutter_amplitude_db", "clutter_db"),
            radar.clutter_amplitude_db,
        ),
    )
    return RadarScenarioData(scenario=scenario, objects=tuple(objects), raw=raw)


def _find_required(
    directory: Path,
    stem: str,
    extensions: tuple[str, ...],
    label: str,
) -> Path:
    for extension in extensions:
        candidate = directory / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    extensions_text = ", ".join(extensions)
    raise FileNotFoundError(
        f"Missing {label} file for scenario '{stem}'. Expected one of: {extensions_text}"
    )


def _parse_radar_object(item: dict[str, Any], index: int, radar: RadarConfig) -> RadarObject:
    distance_m = _float_from(
        item,
        ("distance_m", "range_m", "far_m", "range", "distance"),
        0.0,
    )
    speed_mps = _float_from(
        item,
        ("speed_mps", "velocity_mps", "radial_velocity_mps", "speed", "velocity"),
        0.0,
    )
    delay_s = _float_from(
        item,
        ("delay_s",),
        bistatic_range_to_delay_s(distance_m, radar),
    )
    doppler_hz = _float_from(
        item,
        ("doppler_hz", "doppler", "dpb", "dbp"),
        velocity_mps_to_doppler_hz(speed_mps, radar),
    )
    amplitude_db = _float_from(
        item,
        ("amplitude_db", "power_db", "rcs_db", "signal_db", "db"),
        -18.0,
    )
    return RadarObject(
        name=str(item.get("name") or item.get("id") or f"Object {index}"),
        distance_m=distance_m,
        speed_mps=speed_mps,
        doppler_hz=doppler_hz,
        delay_s=delay_s,
        amplitude_db=amplitude_db,
    )


def _float_from(
    item: dict[str, Any],
    keys: tuple[str, ...],
    default: float,
) -> float:
    for key in keys:
        if key in item and item[key] is not None:
            return float(item[key])
    return float(default)
