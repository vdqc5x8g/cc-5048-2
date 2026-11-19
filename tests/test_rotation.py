"""Rotation matrix, equivalence and energy-conservation tests."""

import numpy as np

from shfield.analysis import spherical_energy_conservation
from shfield.harmonics import SN3D, cart_to_sph, sh_matrix, sph_to_cart
from shfield.rotation import (
    euler_rotation,
    rotate_coefficients,
    rotated_direction,
    rotation_energy_error,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    sh_rotation_matrix,
)


def test_rotation_orthogonal_all_orders():
    rng = np.random.default_rng(7)
    for order in range(1, 6):
        r = euler_rotation(*rng.uniform(-np.pi, np.pi, 3))
        d = sh_rotation_matrix(order, r)
        diag = rotation_energy_error(d, order)
        assert diag["orthogonality_max_err"] < 1e-11
        assert np.allclose(diag["per_order_rms_gain"], 1.0, atol=1e-10)


def test_rotation_matches_reencoded_direction():
    order = 4
    rng = np.random.default_rng(8)
    for name, r in (
        ("z", rotation_matrix_z(0.6)),
        ("y", rotation_matrix_y(-0.4)),
        ("x", rotation_matrix_x(0.9)),
        ("euler", euler_rotation(0.5, -0.3, 0.2)),
    ):
        theta = rng.uniform(0.1, np.pi - 0.1, 6)
        phi = rng.uniform(-np.pi, np.pi, 6)
        a = sh_matrix(order, theta, phi, SN3D)
        rotated = (sh_rotation_matrix(order, r) @ a.T).T
        u = sph_to_cart(theta, phi)
        th2, ph2 = cart_to_sph(u @ r.T)
        direct = sh_matrix(order, th2, ph2, SN3D)
        np.testing.assert_allclose(rotated, direct, atol=1e-12, err_msg=name)


def test_listener_rotation_equals_inverse_source_motion():
    # The user-facing check: turning the listener by R must be identical to
    # moving every source by R^{-1} in the world frame.
    order = 3
    r = rotation_matrix_z(0.5)
    theta0, phi0 = np.array(np.pi / 2), np.array(-np.pi / 4)
    a = sh_matrix(order, theta0, phi0, SN3D)[0]
    listener = rotate_coefficients(a, order, r, listener=True)

    u = sph_to_cart(theta0, phi0)
    th_i, ph_i = cart_to_sph(rotated_direction(u, r.T))
    inverse_moved = sh_matrix(order, th_i, ph_i, SN3D)[0]
    np.testing.assert_allclose(listener, inverse_moved, atol=1e-12)


def test_yaw_moves_front_source_to_the_right_in_listener_frame():
    # Positive yaw = turn left; a front source should shift to the right
    # (negative phi) in the listener-relative frame.
    order = 2
    a = sh_matrix(order, np.array(np.pi / 2), np.array(0.0), SN3D)[0]
    r = rotation_matrix_z(np.deg2rad(30))
    moved = rotate_coefficients(a, order, r, listener=True)
    u = sph_to_cart(np.array(np.pi / 2), np.array(0.0))
    th, ph = cart_to_sph(rotated_direction(u, r.T))
    assert abs(float(ph) - np.deg2rad(-30)) < 1e-12
    direct = sh_matrix(order, th, ph, SN3D)[0]
    np.testing.assert_allclose(moved, direct, atol=1e-12)


def test_pitch_and_roll_equivalence_other_axes():
    order = 3
    rng = np.random.default_rng(9)
    for r in (rotation_matrix_y(0.7), rotation_matrix_x(-0.6)):
        theta = rng.uniform(0.2, np.pi - 0.2, 5)
        phi = rng.uniform(-np.pi, np.pi, 5)
        a = sh_matrix(order, theta, phi, SN3D)
        listener = rotate_coefficients(a.T, order, r, listener=True).T
        th_i, ph_i = cart_to_sph(
            rotated_direction(sph_to_cart(theta, phi), r.T)
        )
        np.testing.assert_allclose(
            listener, sh_matrix(order, th_i, ph_i, SN3D), atol=1e-12
        )


def test_energy_preserved_under_rotation():
    order = 3
    rng = np.random.default_rng(10)
    coeffs = rng.standard_normal((16, 4096))
    before = spherical_energy_conservation(coeffs, order)
    r = euler_rotation(0.9, -0.4, 0.2)
    after = spherical_energy_conservation(
        rotate_coefficients(coeffs, order, r, listener=True), order
    )
    assert abs(before - after) / before < 1e-12


def test_full_rotation_returns_to_identity():
    order = 3
    rng = np.random.default_rng(11)
    a = rng.standard_normal((16, 100))
    r = euler_rotation(1.1, 0.7, -0.5)
    once = rotate_coefficients(a, order, r, listener=True)
    back = rotate_coefficients(once, order, r.T, listener=True)
    np.testing.assert_allclose(back, a, atol=1e-11)
