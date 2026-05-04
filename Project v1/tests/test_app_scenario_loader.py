from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from App.scenario_loader import bundle_from_member, discover_scenarios, load_radar_scenario
from radar_sim.constants import RadarConfig


def test_discover_scenarios_requires_matching_video_audio_and_json(tmp_path: Path) -> None:
    (tmp_path / "scenario-1.mp4").touch()
    (tmp_path / "scenario-1.wav").touch()
    (tmp_path / "scenario-1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "incomplete.mp4").touch()
    (tmp_path / "incomplete.json").write_text("{}", encoding="utf-8")

    bundles = discover_scenarios(tmp_path)

    assert [bundle.name for bundle in bundles] == ["scenario-1"]
    assert bundles[0].video_path == tmp_path / "scenario-1.mp4"
    assert bundles[0].audio_path == tmp_path / "scenario-1.wav"
    assert bundles[0].radar_path == tmp_path / "scenario-1.json"


def test_bundle_from_member_loads_same_stem_files(tmp_path: Path) -> None:
    (tmp_path / "scenario-2.mp4").touch()
    (tmp_path / "scenario-2.wav").touch()
    (tmp_path / "scenario-2.json").write_text("{}", encoding="utf-8")

    bundle = bundle_from_member(tmp_path / "scenario-2.wav")

    assert bundle.name == "scenario-2"
    assert bundle.video_path.name == "scenario-2.mp4"
    assert bundle.audio_path.name == "scenario-2.wav"
    assert bundle.radar_path.name == "scenario-2.json"


def test_bundle_from_member_reports_missing_parts(tmp_path: Path) -> None:
    (tmp_path / "scenario-3.mp4").touch()
    (tmp_path / "scenario-3.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing audio file"):
        bundle_from_member(tmp_path / "scenario-3.mp4")


def test_load_radar_scenario_accepts_distance_speed_and_dpb(tmp_path: Path) -> None:
    radar_json = tmp_path / "scenario-4.json"
    radar_json.write_text(
        json.dumps(
            {
                "name": "scenario-4",
                "noise_power_db": -70,
                "targets": [
                    {
                        "name": "Drone A",
                        "distance_m": 1500,
                        "speed_mps": 20,
                        "dpb": 333,
                        "power_db": -16,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    data = load_radar_scenario(radar_json, RadarConfig())

    assert data.scenario.name == "scenario-4"
    assert data.scenario.noise_power_db == pytest.approx(-70)
    assert data.objects[0].name == "Drone A"
    assert data.objects[0].distance_m == pytest.approx(1500)
    assert data.objects[0].speed_mps == pytest.approx(20)
    assert data.objects[0].doppler_hz == pytest.approx(333)
    assert data.objects[0].amplitude_db == pytest.approx(-16)
    assert data.scenario.targets[0].delay_s == pytest.approx(1500 / 3.0e8)


def test_load_radar_scenario_does_not_create_target_from_metadata_only_json(
    tmp_path: Path,
) -> None:
    radar_json = tmp_path / "scenario-empty.json"
    radar_json.write_text(json.dumps({"name": "metadata only"}), encoding="utf-8")

    data = load_radar_scenario(radar_json, RadarConfig())

    assert data.objects == ()
    assert data.scenario.targets == ()
