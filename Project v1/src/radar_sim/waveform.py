from __future__ import annotations

import numpy as np

from .constants import RadarConfig


def generate_qpsk_symbols(num_subcarriers: int, rng: np.random.Generator) -> np.ndarray:
    bits_i = rng.choice((-1.0, 1.0), size=num_subcarriers)
    bits_q = rng.choice((-1.0, 1.0), size=num_subcarriers)
    return (bits_i + 1j * bits_q) / np.sqrt(2.0)


def generate_ofdm_symbol(config: RadarConfig, rng: np.random.Generator) -> np.ndarray:
    freq_symbols = generate_qpsk_symbols(config.num_subcarriers, rng)
    time_symbol = np.fft.ifft(freq_symbols)
    time_symbol /= np.sqrt(np.mean(np.abs(time_symbol) ** 2))
    cyclic_prefix = time_symbol[-config.cyclic_prefix :]
    return np.concatenate((cyclic_prefix, time_symbol))


def generate_reference_matrix(
    config: RadarConfig, rng: np.random.Generator
) -> np.ndarray:
    pulses = [
        generate_ofdm_symbol(config=config, rng=rng) for _ in range(config.num_pulses)
    ]
    return np.asarray(pulses, dtype=np.complex128)
