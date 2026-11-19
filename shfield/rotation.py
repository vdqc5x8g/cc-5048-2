"""Rotation of spherical-harmonic sound-field coefficients.

A listener head rotation R (active rotation of directions in the fixed
world frame) transforms the SH coefficient vector a as

    a'(n) = sum_{n'} D^{(N)}_{n n'}(R) a(n'),

where D is the block-diagonal (per-order) Wigner-D matrix expressed in the
real ACN basis. The same D applies to SN3D and N3D coefficients because the
normalization factors depend only on the order l and cancel.

Construction here uses spherical quadrature projection rather than a hard-coded
recurrence: with the Gauss-Legendre x uniform grid (exact for SH products up to
the target order, see :mod:`shfield.harmonics`),

    D_{n n'} = sum_k w_k Y_n(R u_k) Y_{n'}(u_k),

which is the matrix of the pullback of a rotated field and is orthogonal to
machine precision. Numerical residuals are checked explicitly in the test
suite instead of being assumed.
"""

from __future__ import annotations

import numpy as np

from .harmonics import (
    SN3D,
    cart_to_sph,
    channel_count,
    gauss_legendre_grid,
    sh_matrix,
    sn3d_gram_diagonal,
    sph_to_cart,
)


def rotation_matrix_x(angle: float) -> np.ndarray:
    """Right-handed rotation about +x (front), e.g. head roll."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle: float) -> np.ndarray:
    """Right-handed rotation about +y (left), e.g. head pitch."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Right-handed rotation about +z (up), e.g. head yaw (positive = left)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_rotation(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Intrinsic-head Euler rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Angles in radians, right-handed; positive yaw turns the listener to the
    left, positive pitch looks up, positive roll tilts left ear down.
    """
    return rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)


def sh_rotation_matrix(order: int, rotation: np.ndarray) -> np.ndarray:
    """Real Wigner-D matrix for ACN channels 0..(order+1)^2-1.

    Parameters
    ----------
    order : truncation order N.
    rotation : 3x3 direction rotation applied to world-frame directions.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    theta, phi, weights = gauss_legendre_grid(order)
    u = sph_to_cart(theta, phi)                       # (K, 3)
    u_rot = u @ rotation.T                            # rotate each direction
    th_r, ph_r = cart_to_sph(u_rot)
    y = sh_matrix(order, theta, phi, SN3D)            # (K, C)
    y_r = sh_matrix(order, th_r, ph_r, SN3D)
    d = (y_r * weights[:, None]).T @ y
    # Invert the (diagonal) SN3D Gram to get the projected rotation operator.
    return d / sn3d_gram_diagonal(order)[:, None]


def rotate_coefficients(
    coefficients: np.ndarray,
    order: int,
    rotation: np.ndarray,
    listener: bool = False,
) -> np.ndarray:
    """Apply an SH rotation to coefficient arrays of shape (channels, ...).

    With ``listener=False`` the field itself is actively rotated by
    ``rotation`` (a source at u ends up at R u). With ``listener=True`` the
    listener's frame rotates by R, so the field is pulled by R^{-1}: the
    coefficient transform is D(R^T) = D(R)^T, which is exactly equivalent to
    moving every source by R^{-1} in the fixed world frame.
    """
    r = np.asarray(rotation, dtype=float)
    if listener:
        r = r.T
    d = sh_rotation_matrix(order, r)
    if coefficients.shape[0] != channel_count(order):
        raise ValueError(
            f"expected {channel_count(order)} channels for order {order}, "
            f"got {coefficients.shape[0]}"
        )
    return np.tensordot(d, coefficients, axes=([1], [0]))


def rotation_energy_error(d: np.ndarray, order: int) -> dict:
    """Diagnostics: orthogonality residual and per-order energy gain."""
    channels = channel_count(order)
    identity_err = float(np.max(np.abs(d @ d.T - np.eye(channels))))
    gains = []
    for l in range(order + 1):
        start = l * l
        size = 2 * l + 1
        block = d[start: start + size, start: start + size]
        gains.append(float(np.sqrt(np.mean(np.sum(block * block, axis=0)))))
    return {"orthogonality_max_err": identity_err, "per_order_rms_gain": gains}


def rotated_direction(direction: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate one or more unit direction vectors (..., 3)."""
    return np.asarray(direction, dtype=float) @ np.asarray(rotation, dtype=float).T
