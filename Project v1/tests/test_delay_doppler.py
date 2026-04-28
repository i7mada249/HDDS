import numpy as np

from radar_sim.constants import RadarConfig, Target
from radar_sim.geometry import delay_to_bistatic_range_m, doppler_hz_to_velocity_mps


def test_delay_to_bistatic_range_mapping() -> None:
    config = RadarConfig(speed_of_light=3.0e8)
    delay_s = 5.0e-6
    range_m = delay_to_bistatic_range_m(delay_s, config)
    assert np.isclose(range_m, 1500.0)


def test_doppler_to_velocity_mapping() -> None:
    config = RadarConfig(carrier_frequency_hz=2.4e9)
    target = Target(delay_s=5.0e-6, doppler_hz=80.0, amplitude_db=-20.0)
    velocity = doppler_hz_to_velocity_mps(target.doppler_hz, config)
    assert velocity > 0.0
