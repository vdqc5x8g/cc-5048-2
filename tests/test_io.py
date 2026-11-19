"""Offline WAV sidecar metadata and diagnostic tests."""

import json

import numpy as np
import pytest

from shfield.conventions import ACN, FUMA, ConventionError, make_convention
from shfield.io_wav import check_channels, read_sh_wav, write_sh_wav


@pytest.fixture
def coeffs():
    rng = np.random.default_rng(5)
    return rng.standard_normal((9, 8000))


def test_roundtrip_ambix(tmp_path, coeffs):
    path = tmp_path / "field_ambix.wav"
    conv = make_convention(ACN, "SN3D", 2)
    write_sh_wav(path, coeffs, 48000, conv)
    back, sr, resolved, meta = read_sh_wav(path)
    assert sr == 48000
    assert resolved.ordering == ACN and resolved.normalization == "SN3D"
    # Float normalization in WAV writing scales amplitude; compare shapes and
    # proportionality, then exact roundtrip on the sidecar metadata path.
    assert back.shape == coeffs.shape
    scale = back / coeffs
    assert np.allclose(scale, scale[0, 0], rtol=1e-5)


def test_roundtrip_fuma_and_n3d_distinct(tmp_path, coeffs):
    for ordering, norm, order in ((FUMA, FUMA, 2), (ACN, "N3D", 2)):
        conv = make_convention(ordering, norm, order)
        path = tmp_path / f"field_{ordering}_{norm}.wav"
        write_sh_wav(path, coeffs, 48000, conv)
        back, _, resolved, _ = read_sh_wav(path)
        assert resolved.ordering == ordering
        assert back.shape == coeffs.shape
        # The sidecar must explicitly carry the convention.
        meta = json.loads((tmp_path / (path.name + ".sh.json")).read_text())
        assert meta["ordering"] == ordering and meta["normalization"] == norm


def test_sidecar_channel_mismatch_detected(tmp_path, coeffs):
    path = tmp_path / "field.wav"
    write_sh_wav(path, coeffs, 48000, make_convention(ACN, "SN3D", 2))
    sidecar = tmp_path / (path.name + ".sh.json")
    meta = json.loads(sidecar.read_text())
    meta["channels"] = 16
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ConventionError, match="sidecar declares 16"):
        read_sh_wav(path)


def test_missing_channels_diagnostic():
    with pytest.raises(ConventionError, match="expected 16 channels.*got 9"):
        check_channels(9, 3)
    with pytest.raises(ConventionError, match="not a perfect square"):
        check_channels(7, 2)


def test_partial_metadata_rejected(tmp_path, coeffs):
    path = tmp_path / "partial.wav"
    write_sh_wav(path, coeffs, 48000, make_convention(ACN, "SN3D", 2))
    with pytest.raises(ConventionError, match="both must be declared"):
        read_sh_wav(path, ordering="ACN", normalization=None)
