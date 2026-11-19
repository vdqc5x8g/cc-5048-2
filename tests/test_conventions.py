"""Convention identification and conversion tests."""

import numpy as np
import pytest

from shfield.conventions import (
    ACN,
    FUMA,
    ConventionError,
    convert_from_acn_sn3d,
    convert_to_acn_sn3d,
    fuma_to_acn_map,
    identify_convention,
    make_convention,
)
from shfield.harmonics import N3D, SN3D


def test_fuma_map_known_order_one():
    # mapping[i] = FuMa channel feeding ACN channel i.
    # ACN W(0) <- FuMa W(0); ACN Y(1) <- FuMa Y(2); ACN Z(2) <- FuMa Z(3);
    # ACN X(3) <- FuMa X(1).
    mapping = fuma_to_acn_map(1)
    assert mapping.tolist() == [0, 2, 3, 1]


def test_fuma_map_order_three_length_and_permutation():
    mapping = fuma_to_acn_map(3)
    assert sorted(mapping.tolist()) == list(range(16))
    assert mapping[:4].tolist() == [0, 2, 3, 1]


def test_roundtrip_acn_sn3d_identity():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 50))
    conv = make_convention(ACN, SN3D, 3)
    out = convert_from_acn_sn3d(convert_to_acn_sn3d(a, conv), conv)
    np.testing.assert_allclose(out, a, atol=1e-12)


def test_roundtrip_acn_n3d():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((9, 40))
    conv = make_convention(ACN, N3D, 2)
    native = convert_from_acn_sn3d(a, conv)
    back = convert_to_acn_sn3d(native, conv)
    np.testing.assert_allclose(back, a, atol=1e-12)


def test_roundtrip_fuma():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((16, 30))
    conv = make_convention(FUMA, FUMA, 3)
    native = convert_from_acn_sn3d(a, conv)
    back = convert_to_acn_sn3d(native, conv)
    np.testing.assert_allclose(back, a, atol=1e-12)


def test_fuma_first_order_known_gain():
    # W in FuMa is 0.5 for a unit plane wave; SN3D W is 1/sqrt(2).
    fuma_wxyz = np.array([0.5, 1.0, 0.0, 0.0])  # frontal plane wave, X=1
    conv = make_convention(FUMA, FUMA, 1)
    acn = convert_to_acn_sn3d(fuma_wxyz, conv)
    # ACN order W,Y,Z,X: W scaled by sqrt(2), X gain 1.
    np.testing.assert_allclose(acn, [1 / np.sqrt(2), 0, 0, 1.0], atol=1e-12)


def test_fuma_rejects_other_normalization():
    with pytest.raises(ConventionError):
        make_convention(FUMA, SN3D, 1)
    with pytest.raises(ConventionError):
        make_convention(ACN, FUMA, 1)


def test_fuma_order_limit():
    with pytest.raises(ConventionError):
        make_convention(FUMA, FUMA, 4)


def test_identify_requires_both_fields_together():
    with pytest.raises(ConventionError):
        identify_convention(4, ordering="ACN", normalization=None)


def test_identify_defaults_to_ambix_for_square_count():
    conv = identify_convention(9)
    assert conv.ordering == ACN and conv.normalization == SN3D
    assert conv.order == 2


def test_identify_fuma_channel_check():
    with pytest.raises(ConventionError):
        identify_convention(25, ordering="FuMa", normalization="FuMa")


def test_channel_count_mismatch_diagnosed():
    conv = make_convention(ACN, SN3D, 2)  # expects 9
    with pytest.raises(ConventionError, match="4 channels but convention declares 9"):
        convert_to_acn_sn3d(np.zeros((4, 10)), conv)


def test_same_channel_number_different_conventions_give_different_fields():
    # 4 channels are NOT decoded by one fixed matrix: conventions change data.
    rng = np.random.default_rng(4)
    data = rng.standard_normal(4)
    ambix = convert_to_acn_sn3d(data, make_convention(ACN, SN3D, 1))
    n3d = convert_to_acn_sn3d(data, make_convention(ACN, N3D, 1))
    fuma = convert_to_acn_sn3d(data, make_convention(FUMA, FUMA, 1))
    assert not np.allclose(ambix, n3d)
    assert not np.allclose(ambix, fuma)
    assert not np.allclose(n3d, fuma)
