"""Offline WAV I/O with explicit, machine-readable SH convention metadata.

A multichannel WAV alone cannot say whether its channels are ACN/SN3D (ambix),
ACN/N3D or FuMa. Every SH file written by this package therefore gets a
sidecar ``<name>.sh.json`` describing order, ordering and normalization. On
import the sidecar is read and validated; missing/contradictory metadata
produces a concrete :class:`~shfield.conventions.ConventionError` instead of
silently plugging channels into the wrong matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .conventions import (
    Convention,
    ConventionError,
    convert_from_acn_sn3d,
    convert_to_acn_sn3d,
    identify_convention,
)
from .harmonics import channel_count, order_from_channels

SHFIELD_META_VERSION = 1


def _sidecar_path(wav_path: str | Path) -> Path:
    return Path(str(wav_path) + ".sh.json")


def write_sh_wav(
    wav_path: str | Path,
    coefficients: np.ndarray,
    sample_rate: int,
    convention: Convention,
    extra: dict | None = None,
) -> Path:
    """Write SH signals in the requested convention plus a sidecar JSON."""
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be 2-D (channels, samples)")
    if coefficients.shape[0] != convention.channels:
        raise ConventionError(
            f"cannot write {coefficients.shape[0]} channels as "
            f"{convention.describe()}"
        )
    payload = convert_from_acn_sn3d(coefficients, convention)
    peak = np.max(np.abs(payload))
    if peak > 0:
        scaled = payload / peak * 0.9
    else:
        scaled = payload
    wavfile.write(str(wav_path), sample_rate, scaled.T.astype(np.float32))
    meta = {
        "shfield_meta_version": SHFIELD_META_VERSION,
        "order": convention.order,
        "ordering": convention.ordering,
        "normalization": convention.normalization,
        "channels": convention.channels,
        "sample_rate": sample_rate,
        "peak_before_normalization": float(peak),
        "coordinate_system": "+x front, +y left, +z up; theta from +z, phi from +x",
    }
    if extra:
        meta.update(extra)
    sidecar = _sidecar_path(wav_path)
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return sidecar


def read_sh_wav(
    wav_path: str | Path,
    ordering: str | None = None,
    normalization: str | None = None,
) -> tuple[np.ndarray, int, Convention, dict]:
    """Read an SH WAV and return canonical ACN/SN3D coefficients.

    Metadata resolution order: explicit arguments, then sidecar JSON, then the
    documented ambix default (only for valid square ACN counts). Any mismatch
    between declared and actual channel counts raises.
    """
    wav_path = Path(wav_path)
    sample_rate, data = wavfile.read(str(wav_path))
    data = data.astype(np.float64)
    if data.ndim == 1:
        data = data[:, None]
    channels = data.shape[1]
    signals = data.T  # (channels, samples)

    sidecar = _sidecar_path(wav_path)
    meta: dict = {}
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        if meta.get("channels") != channels:
            raise ConventionError(
                f"{wav_path.name}: sidecar declares {meta.get('channels')} "
                f"channels but WAV contains {channels}"
            )
        if meta.get("sample_rate") not in (None, sample_rate):
            raise ConventionError(
                f"{wav_path.name}: sidecar sample rate "
                f"{meta.get('sample_rate')} != WAV {sample_rate}"
            )

    explicit = ordering is not None or normalization is not None
    if explicit and (ordering is None or normalization is None):
        raise ConventionError(
            f"partial convention metadata (ordering={ordering!r}, "
            f"normalization={normalization!r}): both must be declared together; "
            "channel numbers cannot determine the SH convention"
        )
    ord_arg = ordering if explicit else meta.get("ordering")
    norm_arg = normalization if explicit else meta.get("normalization")
    if ord_arg is None and norm_arg is None:
        convention = identify_convention(channels)
        meta["assumed_default"] = True
    else:
        convention = identify_convention(channels, ord_arg, norm_arg)

    if convention.channels != channels:
        raise ConventionError(
            f"{wav_path.name}: resolved {convention.describe()} but file has "
            f"{channels} channels"
        )
    canonical = convert_to_acn_sn3d(signals, convention)
    return canonical, sample_rate, convention, meta


def check_channels(channels: int, expected_order: int) -> None:
    """Diagnostic helper used by CLI/UI for clear order/channel errors."""
    expected = channel_count(expected_order)
    if channels != expected:
        try:
            actual_order = order_from_channels(channels)
            hint = f"it matches a complete order-{actual_order} ACN set"
        except ValueError:
            hint = (
                "it is not a perfect square, so channels are missing or it is "
                "not an ACN/SN3D file at all"
            )
        raise ConventionError(
            f"expected {expected} channels for order {expected_order}, got "
            f"{channels}; {hint}. Refusing to decode with a mismatched matrix."
        )
