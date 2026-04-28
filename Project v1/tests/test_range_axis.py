import numpy as np

from radar_sim.constants import RadarConfig
from radar_sim.geometry import build_doppler_axis_hz, build_range_axis_m


def test_range_axis_spacing_is_c_over_fs() -> None:
    config = RadarConfig(sample_rate_hz=20.0e6, speed_of_light=3.0e8)
    axis = build_range_axis_m(4, config)
    diffs = np.diff(axis)
    assert np.allclose(diffs, config.speed_of_light / config.sample_rate_hz)


def test_doppler_axis_is_centered() -> None:
    config = RadarConfig(pri_s=250.0e-6)
    axis = build_doppler_axis_hz(8, config)
    assert axis.shape == (8,)
    assert np.isclose(axis[0], -axis[-1] - (axis[1] - axis[0]))
