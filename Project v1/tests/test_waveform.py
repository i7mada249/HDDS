import numpy as np

from radar_sim.constants import RadarConfig
from radar_sim.waveform import generate_ofdm_symbol


def test_ofdm_symbol_is_normalized_after_cp_removal() -> None:
    config = RadarConfig()
    rng = np.random.default_rng(42)
    symbol = generate_ofdm_symbol(config, rng)
    useful = symbol[config.cyclic_prefix :]
    power = np.mean(np.abs(useful) ** 2)
    assert np.isclose(power, 1.0, atol=1.0e-6)
