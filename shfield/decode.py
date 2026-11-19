"""Loudspeaker layouts and SH decoding to transaural/binaural feeds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .harmonics import SN3D, cart_to_sph, channel_count, sh_matrix, sph_to_cart


@dataclass
class Layout:
    """A physical loudspeaker layout on the unit sphere."""

    name: str
    directions: np.ndarray  # (L, 3) unit vectors, +x front / +y left / +z up

    @property
    def count(self) -> int:
        return self.directions.shape[0]

    def angles(self) -> tuple[np.ndarray, np.ndarray]:
        return cart_to_sph(self.directions)


def _ring(elev_deg: float, n: int, start_phi: float = 0.0) -> np.ndarray:
    theta = np.full(n, np.pi / 2 - np.deg2rad(elev_deg))
    phi = start_phi + 2 * np.pi * np.arange(n) / n
    return sph_to_cart(theta, phi)


def stereo_layout() -> Layout:
    return Layout(
        "stereo headphones",
        sph_to_cart(
            np.full(2, np.pi / 2), np.array([np.deg2rad(-30), np.deg2rad(30)])
        ),
    )


def quad_layout() -> Layout:
    return Layout("quad", _ring(0, 4, start_phi=np.pi / 4))


def octagon_layout() -> Layout:
    return Layout("octagon", _ring(0, 8, start_phi=np.pi / 8))


def cube_layout() -> Layout:
    v = 1 / np.sqrt(3)
    dirs = np.array(
        [
            [v, v, v], [v, -v, v], [-v, -v, v], [-v, v, v],
            [v, v, -v], [v, -v, -v], [-v, -v, -v], [-v, v, -v],
        ]
    )
    return Layout("cube", dirs)


def enneagon_layout() -> Layout:
    """Nine horizontal speakers: the smallest ring carrying order 2."""
    return Layout("enneagon", _ring(0, 9, start_phi=np.pi / 9))


def dense_ring_layout(order: int) -> Layout:
    """Dense virtual-speaker ring for binaural rendering of a given order.

    Always carries at least (N+1)^2 speakers so the SH order is never silently
    truncated; these are virtual feeds convolved with per-direction HRIRs.
    """
    n = max(2 * order + 2, (order + 1) ** 2)
    return Layout(f"virtual-ring-{n}", _ring(0, n, start_phi=np.pi / n))


def tetra_layout() -> Layout:
    v = 1 / np.sqrt(3)
    return Layout(
        "tetrahedron",
        np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * v,
    )


NAMED_LAYOUTS = {
    "stereo": stereo_layout,
    "quad": quad_layout,
    "octagon": octagon_layout,
    "enneagon": enneagon_layout,
    "cube": cube_layout,
    "tetra": tetra_layout,
}


def max_decodable_order(layout: Layout) -> int:
    """Highest SH order a layout can carry (needs >= (N+1)^2 speakers)."""
    n_speakers = layout.count
    order = int(np.floor(np.sqrt(n_speakers))) - 1
    return max(order, 0)


def decoding_matrix(
    order: int,
    layout: Layout,
    normalization: str = SN3D,
    rE_weighted: bool = True,
    regularization: float = 1e-3,
) -> np.ndarray:
    """Sampling/Tikhonov decoder D so speaker feeds s = D a.

    Builds Y (L x C) at the speaker directions and inverts it with ridge
    regularization. Optionally applies rE (max-energy) weighting, which raises
    directional contrast; the effective order is never raised beyond what the
    speaker count supports — see :func:`max_decodable_order`.
    """
    limit = max_decodable_order(layout)
    if order > limit:
        raise ValueError(
            f"layout {layout.name!r} has {layout.count} speakers and can only "
            f"decode up to order {limit} ((N+1)^2 <= speakers); requested "
            f"order {order}. More SH channels do not magically add spatial "
            "resolution on a fixed layout."
        )
    theta, phi = layout.angles()
    y = sh_matrix(order, theta, phi, normalization)  # (L, C)
    if rE_weighted:
        # rE weight per SH order: cos(pi/(2(2l+1))) (Daniel 2001).
        weights = np.concatenate(
            [
                np.full(2 * l + 1, np.cos(np.pi / (2 * (2 * l + 1))))
                for l in range(order + 1)
            ]
        )
        y_eff = y * weights
    else:
        weights = np.ones(channel_count(order))
        y_eff = y
    # Tikhonov inversion of Y_eff (L x C):
    # D = W Y_eff^T (Y_eff W Y_eff^T + lambda I)^-1, then undo rE weights.
    gram = y_eff @ y_eff.T + regularization * np.eye(layout.count)
    d = np.linalg.solve(gram, y_eff)  # (L, C), row i solves gram d_i = y_eff_i
    return d / weights[None, :]


def decode(coefficients: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    """Apply a static decoder: feeds (L, T) = decoder (L, C) @ coefficients."""
    return decoder @ coefficients


# ---------------------------------------------------------------------------
# Binaural synthesis: every virtual speaker gets an explicit, direction-aware
# HRTF (ITD + head shadow). Rotation lives in the SH coefficients, so a simple
# L/R swap is never used to represent spatial motion.
# ---------------------------------------------------------------------------

def _ear_position(side: str) -> np.ndarray:
    # Head radius 87.5 mm normalized; ears at +/-y.
    return np.array([0.0, 1.0 if side == "L" else -1.0, 0.0])


def synthetic_hrir(
    direction: np.ndarray,
    sample_rate: int,
    head_radius: float = 0.0875,
    n_taps: int = 64,
) -> np.ndarray:
    """Minimal physics-based HRIR, shape (2, n_taps), rows L/R.

    Models interaural time difference from path-length difference and a
    one-pole head-shadow low-pass on the far ear. This is no substitute for a
    measured HRTF (e.g. KEMAR/SOFA), but every direction gets distinct,
    continuous ITD/ILD cues and the spatial cues move with the SH rotation.
    """
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    c = 343.0
    hrir = np.zeros((2, n_taps))
    for side in ("L", "R"):
        ear = _ear_position(side)
        # Woodworth-Schlosberg excess path around the sphere.
        cos_angle = float(np.clip(np.dot(d, ear), -1, 1))
        angle = np.arccos(cos_angle)
        path = head_radius * (angle + np.sin(angle))
        delay_s = path / c
        delay_i = int(round(delay_s * sample_rate))
        delay_i = min(delay_i, n_taps - 1)
        hrir[0 if side == "L" else 1, delay_i] = 1.0
        # Far-ear shadow: stronger for contralateral sources.
        contralateral = max(0.0, -cos_angle)
        alpha = 0.5 - 0.4 * contralateral  # pole moves toward low-pass
        x = hrir[0 if side == "L" else 1].copy()
        filtered = np.zeros_like(x)
        for i in range(1, len(x)):
            filtered[i] = alpha * filtered[i - 1] + (1 - alpha) * x[i]
        hrir[0 if side == "L" else 1] = filtered
    # Normalize so a frontal mono source has near-unity L+R energy.
    hrir /= np.sqrt(np.sum(hrir ** 2) / 2) + 1e-12
    return hrir


def binaural_decode(
    coefficients: np.ndarray,
    order: int,
    layout: Layout,
    sample_rate: int,
    rE_weighted: bool = True,
) -> np.ndarray:
    """Render ACN/SN3D coefficients to binaural L/R via virtual speakers.

    Pipeline: SH coefficients -> static speaker feeds -> per-speaker HRIR
    convolution -> sum at the ears. If the requested physical layout cannot
    carry ``order``, a denser virtual ring is used instead of truncating the
    field (more channels never fake resolution on a fixed layout).
    """
    if max_decodable_order(layout) < order:
        layout = dense_ring_layout(order)
    decoder = decoding_matrix(order, layout, rE_weighted=rE_weighted)
    feeds = decode(coefficients, decoder)  # (L, T)
    n_taps = max(32, sample_rate // 500)
    n_out = feeds.shape[1] + n_taps - 1
    ears = np.zeros((2, n_out))
    for k in range(layout.count):
        hrir = synthetic_hrir(layout.directions[k], sample_rate, n_taps=n_taps)
        for ear in range(2):
            ears[ear] += fftconvolve(feeds[k], hrir[ear])
    peak = np.max(np.abs(ears))
    if peak > 1.0:
        ears /= peak
    return ears
