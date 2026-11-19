"""Directional response and energy analysis of SH fields / decoders."""

from __future__ import annotations

import numpy as np

from .harmonics import SN3D, sh_matrix, sn3d_gram_diagonal
from .decode import decoding_matrix
from .decode import Layout


def field_directional_response(
    coefficients: np.ndarray, order: int, n_phi: int = 180, n_theta: int = 90
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstructed field amplitude p(u) = Y(u) a over a regular grid.

    Returns theta grid, phi grid and |p|^2 map, shape (n_theta, n_phi).
    Works per single time instant or with time-averaged coefficient energy.
    """
    if coefficients.ndim == 2:
        a = np.mean(coefficients ** 2, axis=1)
    else:
        a = np.asarray(coefficients, dtype=float) ** 2
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    y = sh_matrix(order, th.ravel(), ph.ravel(), SN3D)
    # For energy use mean-square reconstruction: E(u) ~ (Y sqrt(a))^2.
    p = y @ np.sqrt(a)
    return th, ph, p.reshape(n_theta, n_phi) ** 2


def speaker_beam_pattern(
    order: int, layout: Layout, speaker_index: int, n_phi: int = 360
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal-plane magnitude response of one decoder output."""
    d = decoding_matrix(order, layout)
    theta = np.full(n_phi, np.pi / 2)
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    y = sh_matrix(order, theta, phi, SN3D)
    response = y @ d[speaker_index]
    return phi, response


def sh_energy(coefficients: np.ndarray, order: int) -> dict:
    """Total and per-order signal energy (mean square).

    SN3D energy of an isotropic/omni field distributes 1/(2l+1) per channel,
    so per-order totals use 2l+1 channels naturally.
    """
    sig = np.atleast_2d(coefficients)
    channel_energy = np.mean(sig ** 2, axis=-1)
    per_order = {}
    start = 0
    for l in range(order + 1):
        size = 2 * l + 1
        per_order[l] = float(np.sum(channel_energy[start: start + size]))
        start += size
    return {
        "total": float(np.sum(channel_energy)),
        "per_order": per_order,
        "per_channel": channel_energy.tolist(),
    }


def spherical_energy_conservation(
    coefficients: np.ndarray, order: int
) -> float:
    """Integrated field energy sum_i Gram_i * mean(a_i^2) (SN3D basis)."""
    sig = np.atleast_2d(coefficients)
    channel_energy = np.mean(sig ** 2, axis=-1)
    return float(np.sum(sn3d_gram_diagonal(order) * channel_energy))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))
