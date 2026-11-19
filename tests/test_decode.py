"""Encoding/decoding, layout limits, and anti-pattern tests."""

import numpy as np
import pytest

from shfield.analysis import rms, sh_energy, speaker_beam_pattern
from shfield.decode import (
    binaural_decode,
    decode,
    decoding_matrix,
    enneagon_layout,
    max_decodable_order,
    octagon_layout,
    quad_layout,
    stereo_layout,
    tetra_layout,
)
from shfield.encoding import (
    Source,
    encode_sources,
    make_impulse,
    make_uncorrelated_noise,
)
from shfield.harmonics import SN3D, sh_matrix
from shfield.rotation import rotation_matrix_z


def test_encoded_plane_wave_reconstructs_source_direction():
    order = 3
    theta, phi = np.array(np.pi / 2), np.array(np.deg2rad(60))
    impulse = make_impulse(48000)
    src = Source("click", theta, phi, impulse)
    coeffs = encode_sources(order, [src])
    # The coefficient snapshot at the impulse instant must equal Y(u).
    peak_idx = int(np.argmax(np.abs(coeffs), axis=1)[0])
    snapshot = coeffs[:, peak_idx]
    expected = sh_matrix(order, theta, phi, SN3D)[0]
    np.testing.assert_allclose(snapshot, expected, atol=1e-12)


def test_uncorrelated_noise_is_independent():
    noise = make_uncorrelated_noise(48000, 1.0, 3, seed=0)
    corr = np.corrcoef(noise)
    off_diag = corr - np.diag(np.diag(corr))
    assert np.max(np.abs(off_diag)) < 0.05


def test_layout_order_limits():
    assert max_decodable_order(stereo_layout()) == 0
    assert max_decodable_order(quad_layout()) == 1
    assert max_decodable_order(octagon_layout()) == 1
    assert max_decodable_order(enneagon_layout()) == 2
    assert max_decodable_order(tetra_layout()) == 1


def test_decoder_refuses_order_above_layout_capacity():
    with pytest.raises(ValueError, match="spatial"):
        decoding_matrix(3, quad_layout())  # 4 speakers cannot carry order 3


def test_decoder_speaker_feed_peaks_toward_source():
    # A source placed at a speaker direction must predominantly drive that
    # speaker rather than the opposite one.
    layout = enneagon_layout()
    order = 2
    d = decoding_matrix(order, layout)
    theta, phi = layout.angles()
    for k in (0, 3, 6):
        y = sh_matrix(order, np.atleast_1d(theta[k]), np.atleast_1d(phi[k]), SN3D)[0]
        feeds = d @ y
        assert np.argmax(np.abs(feeds)) == k


def test_binaural_output_shape_and_finite():
    order = 1
    layout = octagon_layout()
    src = Source("n", np.pi / 2, 0.0, make_uncorrelated_noise(48000, 0.2, 1)[0])
    coeffs = encode_sources(order, [src])
    ears = binaural_decode(coeffs, order, layout, 48000)
    assert ears.shape[0] == 2
    assert np.all(np.isfinite(ears))


def test_rotation_is_not_a_channel_swap():
    # The prohibited shortcut: yaw must change the full coefficient vector,
    # not merely exchange two channels. D for an arbitrary yaw is dense.
    order = 2
    d_rot = rotation_matrix_z(0.37)
    from shfield.rotation import sh_rotation_matrix

    d_sh = sh_rotation_matrix(order, d_rot.T)
    # Every order>=1 block has more than two nonzeros and is not a permutation.
    block = d_sh[1:4, 1:4]
    assert np.sum(np.abs(block) > 1e-9) > 2
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.abs(block), np.eye(3), atol=1e-6)


def test_more_channels_do_not_improve_fixed_layout():
    # On a 4-speaker layout, order 3 (16 SH channels) cannot be decoded at all;
    # the decoder rejects it rather than pretending to gain resolution.
    with pytest.raises(ValueError):
        decoding_matrix(3, quad_layout())
    # And order 1 vs order 0 on the same quad layout use the same 4 feeds.
    layout = quad_layout()
    src = Source("s", np.pi / 2, 0.3, make_impulse(48000))
    c1 = encode_sources(1, [src])
    c0 = c1[:1]
    f1 = decode(c1, decoding_matrix(1, layout))
    f0 = decode(c0, decoding_matrix(0, layout))
    assert f1.shape == f0.shape == (4, f1.shape[1])


def test_beam_pattern_function_runs():
    phi, response = speaker_beam_pattern(2, enneagon_layout(), 0)
    assert response.shape == (360,)
    assert np.all(np.isfinite(response))


def test_energy_of_encoded_sources_sums():
    sr = 48000
    sources = [
        Source("a", np.pi / 2, 0.0, make_uncorrelated_noise(sr, 0.5, 1)[0]),
        Source("b", np.pi / 2, 2.0, make_uncorrelated_noise(sr, 0.5, 1, seed=3)[0]),
    ]
    coeffs = encode_sources(2, sources)
    energy = sh_energy(coeffs, 2)
    assert energy["total"] > 0
    assert set(energy["per_order"]) == {0, 1, 2}
    assert rms(coeffs) > 0
