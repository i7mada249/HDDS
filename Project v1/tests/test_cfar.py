import numpy as np

from radar_sim.constants import CFARConfig
from radar_sim.detection import ca_cfar_2d


def test_cfar_detects_strong_peak() -> None:
    rd_map = np.zeros((32, 64), dtype=np.complex128)
    rd_map[12, 20] = 100.0 + 0.0j

    range_axis = np.arange(64, dtype=float)
    doppler_axis = np.arange(32, dtype=float) - 16.0
    velocity_axis = doppler_axis.copy()

    result = ca_cfar_2d(
        range_doppler_map=rd_map,
        range_axis_m=range_axis,
        doppler_axis_hz=doppler_axis,
        velocity_axis_mps=velocity_axis,
        config=CFARConfig(),
    )

    assert len(result.detections) >= 1
    assert any(item.range_bin == 20 and item.doppler_bin == 12 for item in result.detections)
