"""Sound-field encoding: test sources -> SH coefficient signals (ACN/SN3D).

A plane wave arriving from unit direction u is encoded at order N as

    a_n(t) = s(t) * Y_n^{SN3D}(u),

so the coefficient vector for one source is the source waveform times the
SH evaluation row at its direction. Multiple sources sum in the SH domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harmonics import SN3D, sh_matrix, sph_to_cart


@dataclass
class Source:
    """A test source with a known world-frame direction."""

    name: str
    theta: float          # polar angle from +z (up), radians
    phi: float            # azimuth from +x (front), positive left, radians
    signal: np.ndarray    # mono waveform, shape (samples,)

    @property
    def direction(self) -> np.ndarray:
        return sph_to_cart(np.array(self.theta), np.array(self.phi))


def plane_wave_encoder(order: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Encoding matrix E mapping source signals to ACN/SN3D coefficients.

    For K sources, ``a = E @ s`` with E of shape ((N+1)^2, K); equivalently
    each column is Y^{SN3D}(u_k).
    """
    return sh_matrix(order, np.atleast_1d(theta), np.atleast_1d(phi), SN3D).T


def encode_sources(order: int, sources: list[Source]) -> np.ndarray:
    """Encode a list of sources into ACN/SN3D coefficients, shape (C, T)."""
    if not sources:
        raise ValueError("at least one source is required")
    length = max(len(s.signal) for s in sources)
    theta = np.array([s.theta for s in sources])
    phi = np.array([s.phi for s in sources])
    encoder = plane_wave_encoder(order, theta, phi)
    waveforms = np.zeros((len(sources), length))
    for k, src in enumerate(sources):
        waveforms[k, : len(src.signal)] = src.signal
    return encoder @ waveforms


def make_impulse(sample_rate: int, delay_ms: float = 50.0, amplitude: float = 1.0) -> np.ndarray:
    """A single unit impulse (Dirac-like) delayed by ``delay_ms``."""
    n = int(sample_rate * 0.2)
    sig = np.zeros(n)
    sig[int(sample_rate * delay_ms / 1000.0)] = amplitude
    return sig


def make_uncorrelated_noise(
    sample_rate: int,
    duration_s: float,
    n_channels: int,
    seed: int = 0,
    level: float = 0.5,
) -> np.ndarray:
    """Independent white-noise streams, shape (n_channels, T).

    Independence is essential for spatial tests: correlated streams would
    collapse the encoded field instead of producing distinct directions.
    """
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_s)
    return level * rng.standard_normal((n_channels, n))


def make_pink_noise(
    sample_rate: int, duration_s: float, seed: int = 1, level: float = 0.5
) -> np.ndarray:
    """One pink-noise stream (Voss-McCartney style filtering of white noise)."""
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_s)
    white = rng.standard_normal(n)
    # -3 dB/octave shaping via one-pole low-passed octave sums.
    pink = np.zeros(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    from scipy.signal import lfilter

    pink = lfilter(b, a, white)
    pink /= max(np.max(np.abs(pink)), 1e-12)
    return level * pink
