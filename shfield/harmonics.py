"""Real spherical harmonics (SH) core.

Coordinate system (declared once and used everywhere in this package):

* Right-handed Cartesian: +x = front, +y = left, +z = up.
* Spherical direction of a point on the unit sphere:
    theta in [0, pi]  : polar/inclination angle measured from +z (up).
    phi   in [-pi, pi]: azimuth measured from +x (front), positive toward +y (left).
* A point (theta, phi) maps to:
    x = sin(theta) cos(phi), y = sin(theta) sin(phi), z = cos(theta).

Channel ordering: ACN, index n = order * order + order + m,
with order l >= 0 and degree m in [-l, l].

Two normalization conventions are supported:

* N3D (orthonormal on the sphere):
      Y_l^m = sqrt((2l+1) * (l-|m|)! / (l+|m|)!) * P_l^|m|(cos theta) * trig
  where trig = 1 for m=0, sqrt(2) cos(m phi) for m>0, sqrt(2) sin(|m| phi)
  for m<0, and P_l^|m| is the associated Legendre function WITHOUT the
  Condon-Shortley phase.  Integral Y_l^m Y_l'^m' dOmega = delta.
* SN3D (ambix): Y_SN3D = Y_N3D / sqrt(2l+1).
      Integral Y_SN3D,l^m Y_SN3D,l'^m' dOmega = delta / (2l+1).

These match the definitions used in the ambisonic literature
(AMBI/Eigenmik convention, Daniel 2001, Politis 2016).
"""

from __future__ import annotations

import numpy as np

N3D = "N3D"
SN3D = "SN3D"
NORMALIZATIONS = (N3D, SN3D)


def acn_index(order: int, degree: int) -> int:
    """ACN channel index for an (order, degree) pair."""
    if abs(degree) > order:
        raise ValueError(f"degree {degree} impossible for order {order}")
    return order * order + order + degree


def channel_count(order: int) -> int:
    """Number of ACN channels up to and including ``order``."""
    if order < 0:
        raise ValueError("order must be non-negative")
    return (order + 1) ** 2


def order_from_channels(channels: int) -> int:
    """Infer the truncation order from a perfect-square channel count."""
    n_plus_1 = int(round(np.sqrt(channels)))
    if n_plus_1 * n_plus_1 != channels:
        raise ValueError(
            f"{channels} channels is not a valid ACN channel count: "
            f"counts must be perfect squares (1, 4, 9, 16, 25, ...)."
        )
    return n_plus_1 - 1


def sph_to_cart(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Spherical (theta, phi) to unit Cartesian vectors, shape (..., 3)."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    st = np.sin(theta)
    return np.stack(
        (st * np.cos(phi), st * np.sin(phi), np.cos(theta)), axis=-1
    )


def cart_to_sph(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit Cartesian vectors (..., 3) to (theta, phi)."""
    v = np.asarray(vectors, dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi


def _assoc_legendre(order: int, x: np.ndarray) -> np.ndarray:
    """P_l^m(x) for all 0 <= m <= l, WITHOUT Condon-Shortley phase.

    Returns array of shape (order+1, order+1, *x.shape) with entry [l, m].
    Uses the standard recurrences (e.g. DLMF 14.10) without the (-1)^m factor.
    """
    x = np.asarray(x, dtype=float)
    p = np.zeros((order + 1, order + 1) + x.shape, dtype=float)
    p[0, 0] = 1.0
    if order == 0:
        return p
    # Diagonal: P_m^m = (2m-1)!! (1-x^2)^(m/2)  (no CS phase).
    p[1, 1] = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    for m in range(2, order + 1):
        p[m, m] = p[m - 1, m - 1] * (2 * m - 1) * np.sqrt(
            np.clip(1.0 - x * x, 0.0, None)
        )
    # Off-diagonal seed P_{m+1}^m = x (2m+1) P_m^m.
    for m in range(0, order):
        p[m + 1, m] = x * (2 * m + 1) * p[m, m]
    # Three-term recurrence in l.
    for l in range(2, order + 1):
        for m in range(0, l - 1):
            p[l, m] = (
                x * (2 * l - 1) * p[l - 1, m] - (l + m - 1) * p[l - 2, m]
            ) / (l - m)
    return p


def sh_matrix(
    order: int,
    theta: np.ndarray,
    phi: np.ndarray,
    normalization: str = SN3D,
) -> np.ndarray:
    """Real SH evaluation matrix Y, shape (n_directions, (order+1)^2).

    Row i holds every Y_l^m(theta_i, phi_i) in ACN order.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"unknown normalization {normalization!r}, expected one of {NORMALIZATIONS}"
        )
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have the same shape")
    n_dirs = theta.size
    legendre = _assoc_legendre(order, np.cos(theta))  # (L+1, L+1, n_dirs)
    y = np.zeros((n_dirs, channel_count(order)), dtype=float)
    for l in range(order + 1):
        for m in range(-l, l + 1):
            idx = acn_index(l, m)
            p_lm = legendre[l, abs(m)]
            if m == 0:
                trig = np.ones(n_dirs)
            elif m > 0:
                trig = np.cos(m * phi)
            else:
                trig = np.sin(abs(m) * phi)
            # Real-SH orthonormal (N3D) normalization. The sqrt(2) for m != 0
            # is the standard real/complex conversion factor, kept here rather
            # than in trig.
            n3d_scale = np.sqrt(
                (1 if m == 0 else 2)
                * (2 * l + 1)
                / (4 * np.pi)
                * _factorial(l - abs(m))
                / _factorial(l + abs(m))
            )
            if normalization == N3D:
                y[:, idx] = n3d_scale * p_lm * trig
            else:
                # Ambix SN3D. For l>=1 it equals N3D * sqrt(4*pi/(2l+1));
                # the omni component is additionally defined as W=1/sqrt(2)
                # (ambix/CSD convention), i.e. divide the l=0 value by sqrt(2).
                if l == 0:
                    y[:, idx] = p_lm * trig / np.sqrt(2.0)
                else:
                    sn3d_scale = n3d_scale * np.sqrt(4 * np.pi / (2 * l + 1))
                    y[:, idx] = sn3d_scale * p_lm * trig
    return y


def _factorial(k: int) -> int:
    result = 1
    for i in range(2, k + 1):
        result *= i
    return result


def gauss_legendre_grid(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre (theta) x uniform (phi) quadrature exact for SH up to order.

    Integration of any product of SH of degree <= order needs polynomials up to
    degree 2*order in cos(theta), so order+1 GL nodes in theta suffice; the
    phi integral of trig functions up to frequency 2*order needs 2*order+2
    equally spaced nodes.

    Returns
    -------
    theta, phi : 1-D arrays of flattened grid nodes, length (order+1)(2order+2).
    weights     : quadrature weights with sum(weights) = 4*pi.
    """
    n_theta = order + 1
    n_phi = 2 * order + 2
    z, w_gl = np.polynomial.legendre.leggauss(n_theta)  # int over [-1, 1]
    theta = np.arccos(z)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    # dOmega = sin(theta) dtheta dphi = (-dz) dphi; every phi node gets the
    # full GL weight vector, and the uniform phi spacing is 2*pi/n_phi.
    w = np.repeat(w_gl, n_phi) * (2 * np.pi / n_phi)
    return th.ravel(), ph.ravel(), w


def sn3d_gram_diagonal(order: int) -> np.ndarray:
    """Diagonal of the SN3D Gram matrix: integral Y_i Y_j dOmega."""
    diag = np.empty(channel_count(order))
    for l in range(order + 1):
        value = 2 * np.pi if l == 0 else 4 * np.pi / (2 * l + 1)
        for m in range(-l, l + 1):
            diag[acn_index(l, m)] = value
    return diag


def sn3d_to_n3d(order: int) -> np.ndarray:
    """Diagonal gain vector converting SN3D coefficients to N3D.

    For l>=1: gain sqrt((2l+1)/(4*pi)). For l=0 the ambix W=1/sqrt(2) gives
    gain (1/sqrt(4*pi)) / (1/sqrt(2)) = 1/sqrt(2*pi).
    """
    gains = np.empty(channel_count(order))
    for l in range(order + 1):
        gain = 1 / np.sqrt(2 * np.pi) if l == 0 else np.sqrt((2 * l + 1) / (4 * np.pi))
        for m in range(-l, l + 1):
            gains[acn_index(l, m)] = gain
    return gains
