"""SH basis, quadrature and normalization tests (machine-precision checks)."""

import numpy as np

from shfield.harmonics import (
    N3D,
    SN3D,
    acn_index,
    channel_count,
    gauss_legendre_grid,
    order_from_channels,
    sh_matrix,
    sn3d_to_n3d,
)


def test_acn_index_layout():
    assert [acn_index(1, m) for m in (-1, 0, 1)] == [1, 2, 3]
    assert [acn_index(2, m) for m in range(-2, 3)] == [4, 5, 6, 7, 8]
    assert channel_count(3) == 16
    assert order_from_channels(25) == 4


def test_invalid_channel_count():
    import pytest

    with pytest.raises(ValueError):
        order_from_channels(7)


def test_n3d_orthonormality():
    for order in (1, 2, 4, 6):
        theta, phi, w = gauss_legendre_grid(order)
        y = sh_matrix(order, theta, phi, N3D)
        gram = (y * w[:, None]).T @ y
        assert np.max(np.abs(gram - np.eye(channel_count(order)))) < 1e-12


def test_sn3d_gram_and_quadrature_weight():
    order = 4
    theta, phi, w = gauss_legendre_grid(order)
    assert abs(w.sum() - 4 * np.pi) < 1e-12
    y = sh_matrix(order, theta, phi, SN3D)
    gram = (y * w[:, None]).T @ y
    # l=0 ambix W=1/sqrt(2) integrates to 2*pi; l>=1 gram is 4*pi/(2l+1).
    diag = []
    for l in range(order + 1):
        value = 2 * np.pi if l == 0 else 4 * np.pi / (2 * l + 1)
        diag.extend([value] * (2 * l + 1))
    expected = np.diag(diag)
    assert np.max(np.abs(gram - expected)) < 1e-11


def test_sn3d_n3d_scaling_identity():
    order = 3
    rng = np.random.default_rng(0)
    th = rng.uniform(0, np.pi, 20)
    ph = rng.uniform(-np.pi, np.pi, 20)
    yn = sh_matrix(order, th, ph, N3D)
    ys = sh_matrix(order, th, ph, SN3D)
    assert np.max(np.abs(ys * sn3d_to_n3d(order) - yn)) < 1e-12


def test_first_order_direction_values():
    # ACN order 0:W 1:Y 2:Z 3:X. Ambix: W=1/sqrt(2), axial components reach 1.
    front = sh_matrix(1, [np.pi / 2], [0.0], SN3D)[0]
    np.testing.assert_allclose(front, [1 / np.sqrt(2), 0, 0, 1], atol=1e-12)
    left = sh_matrix(1, [np.pi / 2], [np.pi / 2], SN3D)[0]
    np.testing.assert_allclose(left, [1 / np.sqrt(2), 1, 0, 0], atol=1e-12)
    up = sh_matrix(1, [0.0], [0.0], SN3D)[0]
    np.testing.assert_allclose(up, [1 / np.sqrt(2), 0, 1, 0], atol=1e-12)


def test_n3d_omni_value():
    w = sh_matrix(0, [np.pi / 2], [0.0], N3D)[0, 0]
    assert abs(w - 1 / np.sqrt(4 * np.pi)) < 1e-12
    w_sn3d = sh_matrix(0, [np.pi / 2], [0.0], SN3D)[0, 0]
    assert abs(w_sn3d - 1 / np.sqrt(2)) < 1e-12
