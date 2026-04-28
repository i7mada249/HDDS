from __future__ import annotations

from .constants import AppConfig, ScenarioConfig


def list_scenarios(config: AppConfig) -> list[str]:
    return sorted(config.scenarios.keys())


def get_scenario(config: AppConfig, scenario_name: str) -> ScenarioConfig:
    try:
        return config.scenarios[scenario_name]
    except KeyError as exc:
        known = ", ".join(list_scenarios(config))
        raise KeyError(f"Unknown scenario '{scenario_name}'. Known: {known}") from exc
