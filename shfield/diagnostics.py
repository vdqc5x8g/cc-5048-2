"""Stage-by-stage diagnostic view for the SH sound-field renderer.

The renderer is a chain of independently replaceable stages::

    input mapping -> normalization -> rotation -> (restore / decode)

This module lets each stage be *substituted* (applied, bypassed, or replaced by
a specific wrong hypothesis) while everything else is held at the reference,
so a failure can be localised instead of guessed at. The anchor is always an
**independent** reference, never the renderer's own round trip:

* import is compared against the canonical field the test sources encode to
  (ACN/SN3D) directly; the FuMa test file is authored from physics using a
  reference (l, m) table and max-N gains defined here, deliberately *not* the
  renderer's own table, so a wrong table cannot validate itself;
* rotation is compared against re-encoding after moving every source direction
  by the same rotation, i.e. no Wigner-D.

Per-order deviation is reported for every stage, and the **first order l that
deviates** from the reference is pinned. That first-order index separates the
two failure families this view exists to tell apart:

* handedness / parity reversal (mirror, negated m<0) reaches the **first**
  order (l = 1), so its fingerprint starts at l = 1;
* a high-order channel-numbering error (e.g. FuMa R,S,T,U,V read as if the
  letters were already m = -l..l, swapping S/T and U/V) leaves l <= 1 intact
  and first deviates at **l = 2**.

A zero-order (omni) field is included as a negative control: rotating it must
not change its energy. If the omni control fails too, the error cannot be
attributed to high-order channel numbering at all.

The two-axis motion (yaw about +z, then pitch about +y) is restored in reverse
order; a single axis commutes with its own inverse, so a single-axis test can
hide an ordering mistake that a two-axis pair exposes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import rms, spherical_energy_conservation
from .conventions import FUMA, make_convention
from .encoding import Source, encode_sources, make_impulse
from .harmonics import (
    SN3D,
    acn_index,
    cart_to_sph,
    channel_count,
    sh_matrix,
)
from .rotation import (
    rotate_coefficients,
    rotation_matrix_y,
    rotation_matrix_z,
)

DEFAULT_SR = 48000
DEV_TOL = 1e-2  # relative per-order deviation considered a failure

# ---------------------------------------------------------------------------
# Independent reference definitions (authoritative ambiX / Malham letters).
#
# FuMa = WXYZ | RSTUV | KLMNOPQ
# ACN  = WYZX | VTRSU | QOMKLNP
#
# Per FuMa index these (l, m) are the authoritative ones. The renderer's own
# table is intentionally not imported here: the reference must be able to
# disagree with the device under test.
# ---------------------------------------------------------------------------
REFERENCE_FUMA_LM = (
    (0, 0),   # W
    (1, 1),   # X  (+x front)
    (1, -1),  # Y  (+y left)
    (1, 0),   # Z  (+z up)
    (2, 0),   # R
    (2, 1),   # S
    (2, -1),  # T
    (2, 2),   # U
    (2, -2),  # V
    (3, 0),   # K
    (3, 1),   # L
    (3, -1),  # M
    (3, 2),   # N
    (3, -2),  # O
    (3, 3),   # P
    (3, -3),  # Q
)


def reference_maxn_gain(order: int) -> np.ndarray:
    """Per-ACN-channel max-N -> SN3D gain, defined from the basis peaks.

    FuMa max-N scales every non-omni component of a unit plane wave to peak at
    1. The FuMa->SN3D factor is therefore ``max|Y_l^m|`` in SN3D, which is
    *channel* dependent (m = 0 peaks at 1, m != 0 does not) from l = 2 on.
    l = 0 carries the documented -3 dB FuMa W correction (W_FuMa = 1/2).
    """
    gains = np.empty(channel_count(order))
    for l in range(order + 1):
        for m in range(-l, l + 1):
            i = acn_index(l, m)
            if l == 0:
                gains[i] = np.sqrt(2.0)
            elif l == 1 or m == 0:
                gains[i] = 1.0
            elif l == 2:
                gains[i] = np.sqrt(3.0) / 2.0
            elif l == 3:
                gains[i] = {1: np.sqrt(32.0 / 45.0), 2: np.sqrt(5.0 / 9.0),
                            3: np.sqrt(5.0 / 8.0)}[abs(m)]
            else:  # pragma: no cover - FuMa is only defined to order 3
                raise ValueError("FuMa reference is defined for order <= 3")
    return gains


def reference_maxn_gain_error(order: int, n_theta: int = 2000,
                              n_phi: int = 720) -> float:
    """Self-check: compare the analytic gains to empirical max|Y^SN3D|."""
    theta = np.arccos(np.linspace(-1.0, 1.0, n_theta))
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    y = sh_matrix(order, th.ravel(), ph.ravel(), SN3D)
    peak = np.max(np.abs(y), axis=0)
    gains = reference_maxn_gain(order)
    worst = 0.0
    for l in range(order + 1):
        for m in range(-l, l + 1):
            i = acn_index(l, m)
            if l == 0:
                continue  # W carries the deliberate -3 dB correction
            worst = max(worst, abs(float(peak[i]) - gains[i]))
    return worst


def encode_fuma_reference(canonical: np.ndarray, order: int) -> np.ndarray:
    """Author a FuMa (max-N) file from an ACN/SN3D field using the *reference*.

    Produces ``fuma[j] = canonical[acn(l_j, m_j)] / gain`` so that a correct
    import returns ``canonical``. Independent of the renderer's FuMa table.
    """
    canonical = np.asarray(canonical, dtype=float)
    gains = reference_maxn_gain(order)
    n = channel_count(order)
    fuma = np.empty_like(canonical)
    for j, (l, m) in enumerate(REFERENCE_FUMA_LM[:n]):
        i = acn_index(l, m)
        fuma[j] = canonical[i] / gains[i]
    return fuma


# ---------------------------------------------------------------------------
# Test fields
# ---------------------------------------------------------------------------

def narrowband_tone(
    sample_rate: int,
    freq_hz: float = 1500.0,
    duration_s: float = 0.30,
    amplitude: float = 0.4,
    mod_hz: float = 3.0,
) -> np.ndarray:
    """A clearly narrowband, finite signal (windowed sine + slow envelope)."""
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    envelope = 0.5 - 0.5 * np.cos(2 * np.pi * mod_hz * t)
    return amplitude * np.sin(2 * np.pi * freq_hz * t) * envelope


def asymmetric_sources(sample_rate: int = DEFAULT_SR) -> list[Source]:
    """Asymmetric test field: a front-upper impulse + a rear-oblique narrowband.

    Not left/right symmetric, so a mirror (handedness) error and a channel
    renumbering error do not coincidentally agree.
    """
    impulse = Source(
        "front_upper_impulse",
        theta=np.deg2rad(55.0),   # above the horizon, slightly forward
        phi=np.deg2rad(0.0),      # straight ahead
        signal=make_impulse(sample_rate, delay_ms=50.0, amplitude=1.0),
    )
    narrow = Source(
        "rear_oblique_narrowband",
        theta=np.deg2rad(115.0),  # below the horizon
        phi=np.deg2rad(150.0),    # oblique rear-left
        signal=narrowband_tone(sample_rate),
    )
    return [impulse, narrow]


# Signal recipes let the project sample be rebuilt exactly, without storing raw
# waveforms or depending on WAV peak normalisation.
ASYM_RECIPES = (
    {"kind": "impulse", "delay_ms": 50.0, "amplitude": 1.0},
    {"kind": "narrowband_tone", "freq_hz": 1500.0, "duration_s": 0.30,
     "amplitude": 0.4, "mod_hz": 3.0},
)
OMNI_RECIPES = (
    {"kind": "narrowband_tone", "freq_hz": 700.0, "duration_s": 0.30,
     "amplitude": 0.5, "mod_hz": 2.0},
)


def signal_from_recipe(recipe: dict, sample_rate: int) -> np.ndarray:
    if recipe["kind"] == "impulse":
        return make_impulse(
            sample_rate,
            delay_ms=float(recipe["delay_ms"]),
            amplitude=float(recipe["amplitude"]),
        )
    if recipe["kind"] == "narrowband_tone":
        return narrowband_tone(
            sample_rate,
            freq_hz=float(recipe["freq_hz"]),
            duration_s=float(recipe["duration_s"]),
            amplitude=float(recipe["amplitude"]),
            mod_hz=float(recipe["mod_hz"]),
        )
    raise ValueError(f"unknown signal recipe kind {recipe['kind']!r}")


def omni_sources(sample_rate: int = DEFAULT_SR) -> list[Source]:
    """Zero-order (omni) control: a signal with no directional dependence.

    Encoded at order 0 and padded with zeros to the requested order, so only
    the W channel is non-zero. Rotation must not change it.
    """
    return [
        Source(
            "omni_control",
            theta=np.pi / 2,
            phi=0.0,
            signal=narrowband_tone(sample_rate, freq_hz=700.0,
                                   amplitude=0.5, mod_hz=2.0),
        )
    ]


def omni_coefficients(order: int, sample_rate: int = DEFAULT_SR) -> np.ndarray:
    """A pure l = 0 field padded to ``order`` (ACN/SN3D)."""
    signal = omni_sources(sample_rate)[0].signal
    coeffs = np.zeros((channel_count(order), signal.size))
    coeffs[0] = signal * sh_matrix(0, np.array([np.pi / 2]),
                                   np.array([0.0]), SN3D)[0, 0]
    return coeffs


def test_field_bundle(order: int, sample_rate: int = DEFAULT_SR) -> dict:
    """Canonical ACN/SN3D coefficients plus provenance for both fields."""
    asym_sources = asymmetric_sources(sample_rate)
    asymmetric = encode_sources(order, asym_sources)
    omni = omni_coefficients(order, sample_rate)
    return {
        "order": order,
        "sample_rate": sample_rate,
        "asymmetric": {
            "sources": asym_sources,
            "coefficients": asymmetric,
        },
        "omni": {
            "sources": omni_sources(sample_rate),
            "coefficients": omni,
        },
    }


# ---------------------------------------------------------------------------
# Stage substitution
# ---------------------------------------------------------------------------

def _fuMa_lm_high_order_swap(n_ch: int) -> tuple:
    """Reference (l, m) but with S/T and U/V (and L/M, N/O, P/Q) swapped.

    This reproduces the historical high-order numbering error: l <= 1 correct,
    the l >= 2 letters read as m = -l..l instead of the FuMa zig-zag.
    """
    swapped = list(REFERENCE_FUMA_LM[:n_ch])
    for j in range(4, n_ch):  # only l >= 2
        l, m = swapped[j]
        if m != 0:
            swapped[j] = (l, -m)
    return tuple(swapped)


def reference_fuma_to_acn_map(order: int) -> np.ndarray:
    """Independent ACN<-FuMa permutation from :data:`REFERENCE_FUMA_LM`."""
    n = channel_count(order)
    mapping = np.empty(n, dtype=int)
    for fuma_idx, (l, m) in enumerate(REFERENCE_FUMA_LM[:n]):
        mapping[acn_index(l, m)] = fuma_idx
    return mapping


def _reference_import(native: np.ndarray, order: int,
                      table=REFERENCE_FUMA_LM) -> np.ndarray:
    """Reference import: place each FuMa channel by its (l, m) and max-N gain."""
    native = np.asarray(native, dtype=float)
    gains = reference_maxn_gain(order)
    out = np.zeros_like(native)
    for j, (l, m) in enumerate(table[:channel_count(order)]):
        i = acn_index(l, m)
        out[i] = native[j] * gains[i]
    return out


def import_with_variant(
    native: np.ndarray,
    order: int,
    variant: str = "renderer",
) -> np.ndarray:
    """Import a FuMa (max-N) file into ACN/SN3D under a named stage variant.

    Variants
    --------
    renderer            : the production ``convert_to_acn_sn3d`` code path.
    no_mapping          : reference gains only, channel order left as FuMa
                          (i.e. the permutation stage bypassed).
    no_normalization    : reference permutation only, max-N gains bypassed.
    high_order_swap     : reference table with the l >= 2 letters read as the
                          naive m = -l..l order (reproduces the old bug).
    handedness_flip     : correct import, then negate every m < 0 channel.
    """
    native = np.asarray(native, dtype=float)

    if variant == "renderer":
        from .conventions import convert_to_acn_sn3d

        return convert_to_acn_sn3d(native, make_convention(FUMA, FUMA, order))

    gains = reference_maxn_gain(order)
    mapping = reference_fuma_to_acn_map(order)

    if variant == "no_mapping":
        # FuMa channels used as if they were already ACN, gains still applied.
        return native * (gains[:, None] if native.ndim == 2 else gains)

    if variant == "no_normalization":
        return native[mapping]

    if variant == "high_order_swap":
        return _reference_import(native, order,
                                 _fuMa_lm_high_order_swap(channel_count(order)))

    if variant == "handedness_flip":
        out = native[mapping] * (gains[:, None] if native.ndim == 2 else gains)
        for l in range(1, order + 1):
            for m in range(-l, 0):
                i = acn_index(l, m)
                out[i] = -out[i]
        return out

    raise ValueError(f"unknown import variant {variant!r}")


# ---------------------------------------------------------------------------
# Motion: two-axis rotation and reverse-order restore
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pose:
    """A pair of listener rotations and how they are undone."""

    name: str
    yaw_deg: float
    pitch_deg: float
    one_axis: bool = False

    def rotations(self) -> tuple[np.ndarray, np.ndarray]:
        r1 = rotation_matrix_z(np.deg2rad(self.yaw_deg))
        if self.one_axis:
            r2 = rotation_matrix_z(np.deg2rad(self.pitch_deg))
        else:
            r2 = rotation_matrix_y(np.deg2rad(self.pitch_deg))
        return r1, r2


def apply_motion(
    coefficients: np.ndarray,
    order: int,
    pose: Pose,
    use_rotation: bool = True,
) -> np.ndarray:
    """Listener turns by R1 (yaw), then R2 (pitch), as SH coefficient maps."""
    if not use_rotation:
        return np.array(coefficients, dtype=float, copy=True)
    r1, r2 = pose.rotations()
    out = rotate_coefficients(coefficients, order, r1, listener=True)
    return rotate_coefficients(out, order, r2, listener=True)


def restore_motion(
    coefficients: np.ndarray,
    order: int,
    pose: Pose,
    reverse: bool = True,
    use_rotation: bool = True,
) -> np.ndarray:
    """Undo the motion, by default in reverse order (last rotation first).

    With ``reverse=False`` the rotations are undone in the order they were
    applied, which only equals the inverse when the two rotations commute
    (same axis). On two different axes it leaves a residual.
    """
    if not use_rotation:
        return np.array(coefficients, dtype=float, copy=True)
    r1, r2 = pose.rotations()
    if reverse:
        out = rotate_coefficients(coefficients, order, r2.T, listener=True)
        return rotate_coefficients(out, order, r1.T, listener=True)
    out = rotate_coefficients(coefficients, order, r1.T, listener=True)
    return rotate_coefficients(out, order, r2.T, listener=True)


def listener_frame_directions(sources: list[Source], pose: Pose) -> list[dict]:
    """Point-source directions in the listener frame, independent of Wigner-D.

    The listener turns R1 then R2, so a world direction u appears at
    ``u @ R1 @ R2`` (row vector). This is the ground truth the coefficient
    rotation is checked against.
    """
    r1, r2 = pose.rotations()
    out = []
    for s in sources:
        u = s.direction @ r1 @ r2
        th, ph = cart_to_sph(u)
        out.append({
            "name": s.name,
            "theta_deg": float(np.rad2deg(th)),
            "phi_deg": float(np.rad2deg(ph)),
            "world_theta_deg": float(np.rad2deg(s.theta)),
            "world_phi_deg": float(np.rad2deg(s.phi)),
        })
    return out


def point_source_reference(
    sources: list[Source], order: int, pose: Pose
) -> np.ndarray:
    """Re-encode sources at their moved directions (no Wigner-D anywhere)."""
    r1, r2 = pose.rotations()
    moved = []
    for s in sources:
        u = s.direction @ r1 @ r2
        th, ph = cart_to_sph(u)
        moved.append(Source(s.name, float(th), float(ph), s.signal))
    return encode_sources(order, moved)


# ---------------------------------------------------------------------------
# Per-order deviation and first deviating order
# ---------------------------------------------------------------------------

def per_order_deviation(
    actual: np.ndarray, reference: np.ndarray, order: int, tol: float = DEV_TOL
) -> dict:
    """Relative per-order deviation and the first order exceeding ``tol``.

    ``reference`` is the independent ground truth, not the renderer's own
    starting point. ``first_deviating_order`` is the smallest l whose relative
    deviation exceeds ``tol`` (``None`` if everything is within tolerance).
    """
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    # Global scale so a block that is *structurally* zero (e.g. l >= 1 of the
    # omni control) is judged against the field, not against round-off.
    global_scale = max(float(np.linalg.norm(reference)), 1e-30)
    floor = 1e-12 * global_scale
    rel = {}
    first = None
    for l in range(order + 1):
        sl = slice(l * l, (l + 1) ** 2)
        a = actual[sl]
        r = reference[sl]
        denom = max(float(np.linalg.norm(r)), floor)
        value = float(np.linalg.norm(a - r) / denom)
        rel[l] = value
        if first is None and value > tol:
            first = l
    return {
        "relative": rel,
        "first_deviating_order": first,
        "pass": first is None,
        "max_relative": max(rel.values()) if rel else 0.0,
    }


# ---------------------------------------------------------------------------
# Full diagnostic
# ---------------------------------------------------------------------------

IMPORT_VARIANTS = (
    "renderer",
    "no_mapping",
    "no_normalization",
    "high_order_swap",
    "handedness_flip",
)

IMPORT_VARIANT_LABELS = {
    "renderer": "输入映射 + 归一化 (生产路径)",
    "no_mapping": "旁路输入映射 (FuMa 当 ACN)",
    "no_normalization": "旁路归一化 (缺 max-N 增益)",
    "high_order_swap": "高阶通道编号颠倒 (S/T, U/V)",
    "handedness_flip": "坐标手性反转 (m<0 取反)",
}


def _default_poses() -> list[Pose]:
    return [
        Pose("yaw only (z)", 40.0, 0.0, one_axis=True),
        Pose("pitch only (z)", 0.0, 25.0, one_axis=True),
        Pose("two-axis z+y", 40.0, 25.0),
        Pose("two-axis z - y", -55.0, -30.0),
        Pose("two-axis z+y large", 115.0, 50.0),
    ]


def reduce_poses(
    poses: list[Pose],
    order: int,
    coefficients: np.ndarray,
    reverse: bool = True,
    tol: float = DEV_TOL,
) -> dict:
    """Report each pose's failure and reduce to a minimal still-failing set.

    Deterministic greedy reduction: walk the failing poses in order and remove
    a pose only if at least one *other* remaining pose still fails. The result
    is a subset that still fails, with the redundant poses pruned. Single-axis
    poses pass the same-order restore (their two rotations commute), so they are
    removed and at least one two-axis pose survives.
    """
    details = []
    failing: list[str] = []
    for pose in poses:
        forward = apply_motion(coefficients, order, pose)
        back = restore_motion(forward, order, pose, reverse=reverse)
        dev = per_order_deviation(back, coefficients, order, tol)
        details.append({
            "name": pose.name,
            "yaw_deg": pose.yaw_deg,
            "pitch_deg": pose.pitch_deg,
            "one_axis": pose.one_axis,
            "per_order_relative": dev["relative"],
            "first_deviating_order": dev["first_deviating_order"],
            "max_relative": dev["max_relative"],
            "pass": dev["pass"],
        })
        if not dev["pass"]:
            failing.append(pose.name)

    reduced = list(failing)
    for name in list(failing):
        if len(reduced) <= 1:
            break
        if name in reduced:
            reduced.remove(name)

    return {
        "reverse": reverse,
        "details": details,
        "failing": failing,
        "reduced": reduced,
        "reduced_still_fails": bool(reduced),
    }


def run_diagnostic(
    order: int = 3,
    sample_rate: int = DEFAULT_SR,
    yaw_deg: float = 40.0,
    pitch_deg: float = 25.0,
    tol: float = DEV_TOL,
    bypass_import: bool = False,
    bypass_normalization: bool = False,
    bypass_rotation: bool = False,
) -> dict:
    """Run every stage substitution and return a JSON-ready result.

    The three ``bypass_*`` switches let each stage be turned off independently
    so a failure can be assigned to a specific stage rather than guessed.
    """
    if order < 1:
        raise ValueError("diagnostic needs order >= 1 (order 0 has no rotation)")
    if order > 3:
        raise ValueError("FuMa test file is defined for order <= 3")

    bundle = test_field_bundle(order, sample_rate)
    canonical = bundle["asymmetric"]["coefficients"]
    sources = bundle["asymmetric"]["sources"]
    omni_coeffs = bundle["omni"]["coefficients"]

    # Author the FuMa file from physics, using the independent reference.
    fuma_native = encode_fuma_reference(canonical, order)

    # Active import variants under the user's bypass switches. The named
    # hypotheses stay visible but are marked as not applied when bypassed, so
    # the panel never claims to have tested a stage that was switched off.
    active_variants = []
    for variant in IMPORT_VARIANTS:
        if bypass_import and variant in ("no_mapping", "high_order_swap"):
            continue
        if bypass_normalization and variant in ("no_normalization",):
            continue
        active_variants.append(variant)

    stages = []
    for variant in IMPORT_VARIANTS:
        applied = variant in active_variants
        if applied:
            imported = import_with_variant(fuma_native, order, variant)
            dev = per_order_deviation(imported, canonical, order, tol)
            first = dev["first_deviating_order"]
            max_rel = dev["max_relative"]
            passed = dev["pass"]
            rel = dev["relative"]
        else:
            passed = None
            first = None
            max_rel = None
            rel = None
        stages.append({
            "variant": variant,
            "label": IMPORT_VARIANT_LABELS[variant],
            "applied": applied,
            "per_order_relative": rel,
            "first_deviating_order": first,
            "max_relative": max_rel,
            "pass": passed,
        })

    # --- rotation reference independent of Wigner-D ---
    pose_rot = Pose("primary two-axis", yaw_deg, pitch_deg)
    rotated_coeffs = apply_motion(
        canonical, order, pose_rot, use_rotation=not bypass_rotation
    )
    point_ref = point_source_reference(sources, order, pose_rot)
    rotation_dev = per_order_deviation(rotated_coeffs, point_ref, order, tol)
    rel_dirs = listener_frame_directions(sources, pose_rot)

    # --- motion: reverse-order restore vs same-order restore, per pose ---
    poses = _default_poses()
    motion = []
    if bypass_rotation:
        for pose in poses:
            motion.append({
                "name": pose.name, "yaw_deg": pose.yaw_deg,
                "pitch_deg": pose.pitch_deg, "one_axis": pose.one_axis,
                "reverse_restore": None, "same_order_restore": None,
            })
        failure_reduction = {
            "reverse": reduce_poses([], order, canonical, True, tol),
            "same_order": reduce_poses([], order, canonical, False, tol),
        }
    else:
        for pose in poses:
            forward = apply_motion(canonical, order, pose)
            back_rev = restore_motion(forward, order, pose, reverse=True)
            back_same = restore_motion(forward, order, pose, reverse=False)
            dev_rev = per_order_deviation(back_rev, canonical, order, tol)
            dev_same = per_order_deviation(back_same, canonical, order, tol)
            motion.append({
                "name": pose.name,
                "yaw_deg": pose.yaw_deg,
                "pitch_deg": pose.pitch_deg,
                "one_axis": pose.one_axis,
                "reverse_restore": {
                    "max_relative": dev_rev["max_relative"],
                    "first_deviating_order": dev_rev["first_deviating_order"],
                    "pass": dev_rev["pass"],
                },
                "same_order_restore": {
                    "max_relative": dev_same["max_relative"],
                    "first_deviating_order": dev_same["first_deviating_order"],
                    "pass": dev_same["pass"],
                },
            })
        failure_reduction = {
            "reverse": reduce_poses(poses, order, canonical, True, tol),
            "same_order": reduce_poses(poses, order, canonical, False, tol),
        }

    failing_same = failure_reduction["same_order"]["failing"]
    failing_rev = failure_reduction["reverse"]["failing"]

    forward_restore = restore_motion(
        apply_motion(canonical, order, pose_rot,
                     use_rotation=not bypass_rotation),
        order, pose_rot, reverse=True,
        use_rotation=not bypass_rotation,
    )
    forward_restore_dev = per_order_deviation(
        forward_restore, canonical, order, tol
    )

    # --- negative control: omni field, energy must not change ---
    e_before = spherical_energy_conservation(omni_coeffs, order)
    omni_rot = apply_motion(omni_coeffs, order, pose_rot,
                            use_rotation=not bypass_rotation)
    e_after = spherical_energy_conservation(omni_rot, order)
    omni_back = restore_motion(omni_rot, order, pose_rot, reverse=True,
                               use_rotation=not bypass_rotation)
    omni_dev = per_order_deviation(omni_back, omni_coeffs, order, tol)
    omni_drift = abs(e_after - e_before) / max(e_before, 1e-30)
    omni = {
        "energy_before": e_before,
        "energy_after": e_after,
        "relative_drift": omni_drift,
        "roundtrip_first_deviating_order": omni_dev["first_deviating_order"],
        "max_relative": omni_dev["max_relative"],
        "pass": (omni_drift < 1e-9) and omni_dev["pass"],
    }

    # --- confusion fingerprint ---
    first_by_variant = {
        s["variant"]: s["first_deviating_order"]
        for s in stages if s["applied"]
    }
    attribution = _attribute(
        first_by_variant,
        renderer_pass=next(
            (s["pass"] for s in stages if s["variant"] == "renderer"), None
        ),
        omni_pass=omni["pass"],
        bypass_import=bypass_import,
        bypass_rotation=bypass_rotation,
    )

    return {
        "ok": True,
        "order": order,
        "sample_rate": sample_rate,
        "tolerance": tol,
        "channels": channel_count(order),
        "bypass": {
            "import_mapping": bypass_import,
            "normalization": bypass_normalization,
            "rotation": bypass_rotation,
        },
        "reference": {
            "sources": [
                {"name": s.name,
                 "theta_deg": float(np.rad2deg(s.theta)),
                 "phi_deg": float(np.rad2deg(s.phi))}
                for s in sources
            ],
            "maxn_gain_self_check": reference_maxn_gain_error(order),
            "note": "FuMa file authored by physics; compare uses direct encoding",
        },
        "stages": stages,
        "rotation": {
            "pose": {"yaw_deg": pose_rot.yaw_deg, "pitch_deg": pose_rot.pitch_deg},
            "bypassed": bypass_rotation,
            "per_order_relative": (None if bypass_rotation
                                   else rotation_dev["relative"]),
            "first_deviating_order": (None if bypass_rotation
                                      else rotation_dev["first_deviating_order"]),
            "pass": None if bypass_rotation else rotation_dev["pass"],
            "sources_listener_frame": rel_dirs,
        },
        "motion": {
            "poses": motion,
            "failing_with_same_order_restore": failing_same,
            "failing_with_reverse_restore": failing_rev,
            "same_order_pose_reduction": failure_reduction["same_order"],
            "reverse_pose_reduction": failure_reduction["reverse"],
            "primary_roundtrip": {
                "per_order_relative": forward_restore_dev["relative"],
                "first_deviating_order": forward_restore_dev[
                    "first_deviating_order"
                ],
                "pass": forward_restore_dev["pass"],
            },
        },
        "omni_control": omni,
        "confusion": {
            "first_deviating_by_variant": first_by_variant,
            "attribution": attribution,
            "rule": (
                "handedness/parity reaches l=1; a high-order channel-numbering "
                "error first deviates at l>=2; if the omni (l=0) control also "
                "fails the cause is not high-order numbering"
            ),
        },
        "fields": {
            "asymmetric": _summarize_field(canonical, order, sources),
            "omni": _summarize_field(omni_coeffs, order,
                                     bundle["omni"]["sources"]),
        },
    }


def _attribute(
    first_by_variant: dict,
    renderer_pass,
    omni_pass: bool,
    bypass_import: bool,
    bypass_rotation: bool,
) -> str:
    if bypass_rotation:
        return "旋转已旁路：只能检验输入映射与归一化"
    if renderer_pass is None or renderer_pass:
        return "生产路径在全部阶次都在容差内"
    if not omni_pass:
        return "零阶负对照也失败：不能只归因于高阶通道编号"
    first = first_by_variant.get("renderer")
    if first is None:
        return "生产路径未通过但未定位到单一阶次"
    if bypass_import:
        return f"已旁路输入映射；首个偏离阶次 l={first}"
    if first == 0:
        return "首个偏离阶次 l=0：归一化/全局增益一类"
    if first == 1:
        return "首个偏离阶次 l=1：手性/极性反转一类"
    return f"首个偏离阶次 l={first}：高阶通道编号/归一化一类"


def _summarize_field(coeffs: np.ndarray, order: int, sources) -> dict:
    return {
        "shape": list(coeffs.shape),
        "per_order_energy": {
            str(l): float(np.sum(np.mean(coeffs[l * l:(l + 1) ** 2] ** 2, axis=1)))
            for l in range(order + 1)
        },
        "rms": rms(coeffs),
        "sources": [
            {"name": s.name,
             "theta_deg": float(np.rad2deg(s.theta)),
             "phi_deg": float(np.rad2deg(s.phi))}
            for s in sources
        ],
    }


# ---------------------------------------------------------------------------
# Reproducible project sample
# ---------------------------------------------------------------------------

def _signal_recipe(signal: np.ndarray, sample_rate: int) -> dict:
    """Describe a test signal so it can be regenerated bit-exactly.

    Only the canonical recipes this module produces are recognised; anything
    else is rejected rather than silently mislabelled.
    """
    for recipe in ASYM_RECIPES + OMNI_RECIPES:
        rebuilt = signal_from_recipe(recipe, sample_rate)
        if rebuilt.shape == signal.shape and np.allclose(rebuilt, signal, atol=1e-12):
            return dict(recipe)
    raise ValueError(
        "signal does not match any declared recipe; cannot write a rebuildable "
        "project sample"
    )


def per_order_encoding_coefficients(order: int, sources: list[Source]) -> dict:
    """Per-order SH encoding coefficients ``Y_lm(u)`` for each source.

    These are the coefficients that *generate* the field (``a = E s``, one
    column per source); they are small and exact, unlike the whole time series.
    """
    out = {}
    for s in sources:
        th = np.array([s.theta])
        ph = np.array([s.phi])
        y = sh_matrix(order, th, ph, SN3D)[0]
        out[s.name] = {
            str(l): np.asarray(y[l * l:(l + 1) ** 2]).round(12).tolist()
            for l in range(order + 1)
        }
    return out


def coefficient_snapshot(coeffs: np.ndarray, order: int) -> dict:
    """A compact known-time snapshot: the peak sample of each channel block.

    For an impulse source the peak column is exactly ``Y(u) * amplitude``, so
    this pins the field numerically without storing every sample.
    """
    snapshots = {}
    for l in range(order + 1):
        block = coeffs[l * l:(l + 1) ** 2]
        idx = int(np.argmax(np.max(np.abs(block), axis=0)))
        snapshots[str(l)] = {
            "sample_index": idx,
            "values": np.asarray(block[:, idx]).round(12).tolist(),
        }
    return snapshots


def write_project_sample(
    out_dir: str | Path,
    order: int = 3,
    sample_rate: int = DEFAULT_SR,
) -> dict:
    """Write both fields, a reference FuMa copy, and the coefficient manifest.

    The WAVs are peak-normalised for portability; the JSON manifest stores the
    exact per-order coefficients and the signal recipe, so the fields can be
    rebuilt bit-for-bit without the WAV scale factor.
    """
    from scipy.io import wavfile

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = test_field_bundle(order, sample_rate)
    gain = reference_maxn_gain(order)

    manifest = {
        "description": "Reproducible asymmetric + omni diagnostic fields",
        "coordinate_system": "+x front, +y left, +z up; theta from +z, phi from +x",
        "internal_convention": "ACN/SN3D",
        "order": order,
        "sample_rate": sample_rate,
        "fields": {},
    }
    written: dict[str, str] = {}

    for name in ("asymmetric", "omni"):
        field_bundle = bundle[name]
        coeffs = field_bundle["coefficients"]
        wav_path = out / f"{name}_field.wav"
        peak = float(np.max(np.abs(coeffs))) or 1.0
        wavfile.write(str(wav_path), sample_rate,
                      (coeffs / peak * 0.9).T.astype(np.float32))
        written[name] = str(wav_path)

        sources_meta = []
        for s in field_bundle["sources"]:
            sources_meta.append({
                "name": s.name,
                "theta_deg": float(np.rad2deg(s.theta)),
                "phi_deg": float(np.rad2deg(s.phi)),
                "signal": _signal_recipe(s.signal, sample_rate),
            })
        manifest["fields"][name] = {
            "wav": wav_path.name,
            "wav_peak_before_normalization": peak,
            "convention": "ACN/SN3D",
            "sources": sources_meta,
            "per_order_encoding_coefficients": per_order_encoding_coefficients(
                order, field_bundle["sources"]
            ),
            "coefficient_snapshot": coefficient_snapshot(coeffs, order),
        }

        if name == "asymmetric":
            # A FuMa/max-N copy authored by the independent reference, so the
            # import stage can be exercised against a file whose convention is
            # not the renderer's own output.
            fuma_native = encode_fuma_reference(coeffs, order)
            fuma_path = out / "asymmetric_field_fuma.wav"
            fuma_peak = float(np.max(np.abs(fuma_native))) or 1.0
            wavfile.write(str(fuma_path), sample_rate,
                          (fuma_native / fuma_peak * 0.9).T.astype(np.float32))
            written["asymmetric_fuma"] = str(fuma_path)
            manifest["fields"]["asymmetric_fuma"] = {
                "wav": fuma_path.name,
                "source_field": "asymmetric",
                "convention": "FuMa/max-N (reference-authored)",
                "channels": channel_count(order),
                "fuMa_lm": [list(p) for p in
                            REFERENCE_FUMA_LM[:channel_count(order)]],
                "maxn_gain": [float(g) for g in gain],
                "wav_peak_before_normalization": fuma_peak,
            }

    manifest_path = out / "field_coefficients.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    written["manifest"] = str(manifest_path)
    return {"written": written, "manifest": manifest}


def rebuild_field(manifest: dict, field_name: str) -> np.ndarray:
    """Rebuild an ACN/SN3D field from the manifest, without reading the WAV."""
    order = int(manifest["order"])
    sample_rate = int(manifest["sample_rate"])
    field_meta = manifest["fields"][field_name]
    if field_name == "omni":
        return omni_coefficients(order, sample_rate)
    sources = [
        Source(
            src["name"],
            np.deg2rad(src["theta_deg"]),
            np.deg2rad(src["phi_deg"]),
            signal_from_recipe(src["signal"], sample_rate),
        )
        for src in field_meta["sources"]
    ]
    return encode_sources(order, sources)
