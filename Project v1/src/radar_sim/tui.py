from __future__ import annotations

import argparse
from pathlib import Path

from .constants import RadarConfig, ScenarioConfig, Target, load_app_config
from .geometry import bistatic_range_to_delay_s, velocity_mps_to_doppler_hz
from .plotting import plot_processing_summary
from .runner import default_config_path, execute_scenario, format_report


def _prompt_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        print("Value is required.")


def _prompt_float(
    prompt: str,
    default: float | None = None,
    minimum: float | None = None,
) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Enter a numeric value.")
                continue

        if minimum is not None and value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        return value


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")


def _prompt_target(index: int, radar: RadarConfig) -> Target:
    print(f"\nTarget {index}")
    distance_m = _prompt_float("  Distance / bistatic range excess (m)", minimum=0.0)
    velocity_mps = _prompt_float("  Speed (m/s). Use negative for receding", default=0.0)
    amplitude_db = _prompt_float(
        "  Target amplitude (dB, less negative = stronger)",
        default=-18.0,
    )

    delay_s = bistatic_range_to_delay_s(range_m=distance_m, config=radar)
    doppler_hz = velocity_mps_to_doppler_hz(
        velocity_mps=velocity_mps,
        config=radar,
    )
    return Target(delay_s=delay_s, doppler_hz=doppler_hz, amplitude_db=amplitude_db)


def build_custom_scenario(config_path: Path) -> tuple:
    app_config = load_app_config(config_path)
    radar = app_config.radar

    print("Passive Radar TUI")
    print("=================")
    print("Create a custom scenario in physical units and run it immediately.\n")

    scenario_name = _prompt_text("Scenario name", default="Custom Scenario")
    noise_power_db = _prompt_float(
        "Noise power (dB)",
        default=radar.noise_power_db,
    )
    clutter_amplitude_db = _prompt_float(
        "Clutter amplitude (dB)",
        default=radar.clutter_amplitude_db,
    )

    targets: list[Target] = []
    target_index = 1
    while True:
        targets.append(_prompt_target(target_index, radar))
        target_index += 1
        if not _prompt_yes_no("Add another target?", default=False):
            break

    scenario = ScenarioConfig(
        name=scenario_name,
        targets=tuple(targets),
        noise_power_db=noise_power_db,
        clutter_amplitude_db=clutter_amplitude_db,
    )
    return app_config, scenario


def print_custom_scenario_summary(scenario: ScenarioConfig, radar) -> None:
    print("\nScenario summary")
    print("----------------")
    print(f"Name: {scenario.name}")
    print(f"Noise power (dB): {scenario.noise_power_db:.2f}")
    print(f"Clutter amplitude (dB): {scenario.clutter_amplitude_db:.2f}")
    print(f"Targets: {len(scenario.targets)}")
    for idx, target in enumerate(scenario.targets, start=1):
        range_m = target.delay_s * radar.speed_of_light
        velocity_mps = target.doppler_hz * radar.wavelength_m / 2.0
        print(
            f"  {idx}. distance={range_m:.2f} m, "
            f"speed={velocity_mps:.2f} m/s, "
            f"doppler={target.doppler_hz:.2f} Hz, "
            f"amplitude={target.amplitude_db:.2f} dB"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive TUI for custom radar scenarios.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib plots.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    app_config, scenario = build_custom_scenario(args.config)
    print_custom_scenario_summary(scenario, app_config.radar)

    if not _prompt_yes_no("\nRun this scenario now?", default=True):
        print("Cancelled.")
        return

    result = execute_scenario(config=app_config, scenario=scenario)
    print()
    print(
        format_report(
            processing=result.processing,
            detections=result.detections,
            truths=result.truths,
        )
    )

    if not args.no_plots:
        plot_processing_summary(
            processing=result.processing,
            detections=result.detections,
            scenario_name=result.scenario.name,
        )


if __name__ == "__main__":
    main()
