"""Diagnostic view: stage substitution, first deviating order, controls."""

import json

import numpy as np
import pytest

from shfield.diagnostics import (
    REFERENCE_FUMA_LM,
    apply_motion,
    asymmetric_sources,
    import_with_variant,
    encode_fuma_reference,
    listener_frame_directions,
    omni_coefficients,
    per_order_deviation,
    point_source_reference,
    rebuild_field,
    reduce_poses,
    reference_maxn_gain_error,
    restore_motion,
    run_diagnostic,
    write_project_sample,
    Pose,
)
from shfield.diagnostics import test_field_bundle as make_test_field
from shfield.encoding import encode_sources
from shfield.harmonics import cart_to_sph, channel_count
from shfield.rotation import rotation_matrix_y, rotation_matrix_z


# --- reference definitions -------------------------------------------------

def test_reference_maxn_gain_matches_basis_peaks():
    # The FuMa(max-N)->SN3D gain is exactly max|Y_SN3D| for every l>=1 channel.
    assert reference_maxn_gain_error(3) < 1e-4


def test_reference_lm_matches_authoritative_letter_order():
    # FuMa l=2 is R,S,T,U,V = m 0,+1,-1,+2,-2 (ACN 6,4,5,8,7).
    l2 = [m for (l, m) in REFERENCE_FUMA_LM if l == 2]
    assert l2 == [0, 1, -1, 2, -2]
    l3 = [m for (l, m) in REFERENCE_FUMA_LM if l == 3]
    assert l3 == [0, 1, -1, 2, -2, 3, -3]


def test_authoring_and_reimport_with_reference_is_identity():
    rng = np.random.default_rng(0)
    order = 3
    field = rng.standard_normal((channel_count(order), 5))
    native = encode_fuma_reference(field, order)
    back = import_with_variant(native, order, "renderer")
    np.testing.assert_allclose(back, field, atol=1e-12)


# --- stage substitution / first deviating order -----------------------------

def test_first_deviating_order_separates_failure_families():
    order = 3
    bundle = make_test_field(order)
    field = bundle["asymmetric"]["coefficients"]
    native = encode_fuma_reference(field, order)

    families = {}
    for variant in ("renderer", "no_mapping", "no_normalization",
                    "high_order_swap", "handedness_flip"):
        got = import_with_variant(native, order, variant)
        families[variant] = per_order_deviation(got, field, order)[
            "first_deviating_order"
        ]

    assert families["renderer"] is None
    # Handedness/parity reaches the first order ...
    assert families["handedness_flip"] == 1
    assert families["no_mapping"] == 1
    # ... a high-order channel-numbering error leaves l<=1 intact.
    assert families["high_order_swap"] == 2
    # A missing global max-N gain already fails the omni channel.
    assert families["no_normalization"] == 0


def test_high_order_swap_is_exactly_the_historical_bug():
    # The swap variant must reproduce the old table's l>=2 deviation and leave
    # l<=1 untouched, which is what "sounds like a handedness flip" means.
    order = 3
    field = np.arange(channel_count(order) * 2, dtype=float).reshape(-1, 2) + 1.0
    native = np.asarray(field, dtype=float)  # arrange-independent probe
    dev = per_order_deviation(
        import_with_variant(native, order, "high_order_swap"),
        import_with_variant(native, order, "renderer"),
        order,
    )
    assert dev["relative"][0] < 1e-12 and dev["relative"][1] < 1e-12
    assert dev["first_deviating_order"] == 2


# --- rotation reference (independent of Wigner-D) ---------------------------

def test_coefficient_rotation_matches_moved_point_sources():
    order = 3
    sources = asymmetric_sources()
    field = encode_sources(order, sources)
    pose = Pose("z+y", 40.0, 25.0)
    rotated = apply_motion(field, order, pose)
    reference = point_source_reference(sources, order, pose)
    dev = per_order_deviation(rotated, reference, order)
    assert dev["pass"]
    assert max(dev["relative"].values()) < 1e-10


def test_listener_frame_direction_is_source_moved_by_inverse():
    sources = asymmetric_sources()
    pose = Pose("z+y", 40.0, 25.0)
    rel = listener_frame_directions(sources, pose)
    r1 = rotation_matrix_z(np.deg2rad(40.0))
    r2 = rotation_matrix_y(np.deg2rad(25.0))
    th, ph = cart_to_sph(sources[0].direction @ r1 @ r2)
    assert abs(rel[0]["phi_deg"] - float(np.rad2deg(ph))) < 1e-12


# --- two-axis motion: reverse vs same order ---------------------------------

def test_reverse_order_restore_returns_to_start():
    order = 3
    field = make_test_field(order)["asymmetric"]["coefficients"]
    pose = Pose("z+y", 40.0, 25.0)
    back = restore_motion(apply_motion(field, order, pose), order, pose,
                          reverse=True)
    dev = per_order_deviation(back, field, order)
    assert dev["pass"]


def test_same_order_restore_fails_only_on_two_axes():
    order = 3
    field = make_test_field(order)["asymmetric"]["coefficients"]
    # A single axis commutes with its own inverse, so same-order is harmless.
    same_axis = Pose("z only", 40.0, 25.0, one_axis=True)
    back = restore_motion(apply_motion(field, order, same_axis), order,
                          same_axis, reverse=False)
    assert per_order_deviation(back, field, order)["pass"]
    # Two different axes do not commute.
    two_axis = Pose("z+y", 40.0, 25.0)
    back2 = restore_motion(apply_motion(field, order, two_axis), order,
                           two_axis, reverse=False)
    assert not per_order_deviation(back2, field, order)["pass"]


def test_pose_reduction_keeps_a_two_axis_failure():
    order = 3
    field = make_test_field(order)["asymmetric"]["coefficients"]
    poses = [
        Pose("yaw only", 40.0, 0.0, one_axis=True),
        Pose("two-axis a", 40.0, 25.0),
        Pose("two-axis b", -55.0, -30.0),
    ]
    res = reduce_poses(poses, order, field, reverse=False)
    assert "yaw only" not in res["reduced"]
    assert res["reduced"] and res["reduced_still_fails"]
    assert all(n.startswith("two-axis") for n in res["reduced"])


# --- omni negative control -------------------------------------------------

def test_omni_field_is_zero_above_order_zero():
    order = 3
    omni = omni_coefficients(order)
    assert np.allclose(omni[1:], 0.0)
    assert np.any(omni[0] != 0)


def test_omni_energy_unchanged_by_rotation():
    from shfield.analysis import spherical_energy_conservation

    order = 3
    omni = omni_coefficients(order)
    pose = Pose("z+y", 40.0, 25.0)
    before = spherical_energy_conservation(omni, order)
    after = spherical_energy_conservation(apply_motion(omni, order, pose), order)
    assert abs(after - before) / before < 1e-12


# --- full diagnostic -------------------------------------------------------

def test_run_diagnostic_reports_renderer_passes():
    result = run_diagnostic(order=3)
    renderer = next(s for s in result["stages"] if s["variant"] == "renderer")
    assert renderer["pass"] is True
    assert renderer["first_deviating_order"] is None
    assert result["rotation"]["pass"] is True
    assert result["motion"]["primary_roundtrip"]["pass"] is True
    assert result["omni_control"]["pass"] is True
    # Two-axis poses fail with same-order restore, none with reverse.
    assert result["motion"]["failing_with_same_order_restore"]
    assert result["motion"]["failing_with_reverse_restore"] == []


def test_run_diagnostic_bypasses_are_independent():
    result = run_diagnostic(order=3, bypass_import=True,
                            bypass_normalization=True, bypass_rotation=True)
    assert result["bypass"] == {
        "import_mapping": True, "normalization": True, "rotation": True,
    }
    assert result["rotation"]["pass"] is None
    assert result["motion"]["failing_with_same_order_restore"] == []
    applied = {s["variant"]: s["applied"] for s in result["stages"]}
    assert applied["no_mapping"] is False
    assert applied["no_normalization"] is False
    # The reference renderer result is still reported.
    assert applied["renderer"] is True


def test_run_diagnostic_rejects_unsupported_orders():
    with pytest.raises(ValueError):
        run_diagnostic(order=0)
    with pytest.raises(ValueError):
        run_diagnostic(order=4)


# --- reproducible project sample -------------------------------------------

def test_project_sample_roundtrips_exactly(tmp_path):
    order = 3
    out = write_project_sample(tmp_path, order=order)
    manifest = out["manifest"]
    bundle = make_test_field(order)
    for name in ("asymmetric", "omni"):
        rebuilt = rebuild_field(manifest, name)
        np.testing.assert_allclose(
            rebuilt, bundle[name]["coefficients"], atol=1e-12
        )
        meta = manifest["fields"][name]
        assert meta["per_order_encoding_coefficients"]
        assert all(
            str(l) in meta["per_order_encoding_coefficients"][src["name"]]
            for src in meta["sources"]
            for l in range(order + 1)
        )
        assert set(meta["coefficient_snapshot"]) == {
            str(l) for l in range(order + 1)
        }
    # Read the manifest back from disk: it must be self-contained JSON.
    on_disk = json.loads(
        (tmp_path / "field_coefficients.json").read_text()
    )
    again = rebuild_field(on_disk, "asymmetric")
    np.testing.assert_allclose(again, bundle["asymmetric"]["coefficients"],
                               atol=1e-12)


def test_project_sample_fuma_copy_imports_to_reference(tmp_path):
    from shfield.io_wav import read_sh_wav

    order = 3
    write_project_sample(tmp_path, order=order)
    manifest = json.loads((tmp_path / "field_coefficients.json").read_text())
    fuma_meta = manifest["fields"]["asymmetric_fuma"]

    native, _, _, _ = read_sh_wav(
        tmp_path / "asymmetric_field_fuma.wav",
        ordering="FuMa", normalization="FuMa",
    )
    # Undo the file's peak normalisation to recover the FuMa coefficients.
    native = native / 0.9 * fuma_meta["wav_peak_before_normalization"]

    reference = make_test_field(order)["asymmetric"]["coefficients"]
    dev = per_order_deviation(native, reference, order)
    assert dev["pass"], dev
