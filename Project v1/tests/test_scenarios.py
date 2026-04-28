from pathlib import Path

from radar_sim.constants import load_app_config


def test_default_config_has_expected_scenarios() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    app_config = load_app_config(config_path)
    assert "clear_sky" in app_config.scenarios
    assert "single_slow" in app_config.scenarios
    assert "two_targets" in app_config.scenarios
