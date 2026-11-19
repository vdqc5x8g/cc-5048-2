"""Ambisonic channel-order and normalization conventions.

Two independent attributes of a spherical-harmonic (SH / Ambisonics) signal
must be declared and converted correctly; channel numbers alone are NOT enough
to pick a decoding matrix:

1. Channel ordering
   * ACN  (Ambisonic Channel Number): index = l*l + l + m.
   * FuMa (Furse-Malham, Gerzon B-format): historical order
     W,X,Y,Z | R,S,T,U,V | K,L,M,N,O,P,Q, supported here for 1st/2nd/3rd
     order layouts (4, 9 or 16 channels).

2. Normalization
   * SN3D  (ambix default; W = 1/sqrt(2) for a unit plane wave).
   * N3D   (spherically orthonormal; W = 1/sqrt(4*pi)).
   * FuMa  (max-N Gerzon; W = 1/2).

The canonical internal representation of this package is ACN + SN3D.
Every import is converted to it through an explicit, test-covered transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .harmonics import SN3D, N3D, acn_index, channel_count, order_from_channels

ACN = "ACN"
FUMA = "FuMa"
ORDERINGS = (ACN, FUMA)
FUMA_NORMALIZATION = "FuMa"

# FuMa channel label -> (order l, degree m).  Sign-interleaved Gerzon naming.
_FUMA_LM = [
    (0, 0),  # W
    (1, 1),  # X : front (+x), ACN channel 3
    (1, -1),  # Y : left (+y), ACN channel 1
    (1, 0),  # Z : up (+z), ACN channel 2
    (2, 0),  # R
    (2, -1),  # S
    (2, 1),  # T
    (2, -2),  # U
    (2, 2),  # V
    (3, 0),  # K
    (3, -1),  # L
    (3, 1),  # M
    (3, -2),  # N
    (3, 2),  # O
    (3, -3),  # P
    (3, 3),  # Q
]

# Per-order gain to convert FuMa (max-N) coefficients INTO SN3D.
# References: Daniel (2001), Malham/Furse max-N weights, ambiX converter.
# SN3D = FuMa * gain[l]; check: l=0 gives SN3D W = sqrt(2) * (1/2) = 1/sqrt(2).
_FUMA_TO_SN3D_GAIN = {
    0: np.sqrt(2.0),
    1: 1.0,
    2: np.sqrt(3.0),
    3: np.sqrt(5.0 / 2.0),
}

SUPPORTED_FUMA_CHANNELS = (4, 9, 16)


class ConventionError(ValueError):
    """Raised when a multichannel file's convention is invalid or ambiguous."""


@dataclass(frozen=True)
class Convention:
    """Declared SH convention of an external multichannel signal."""

    ordering: str
    normalization: str
    order: int

    @property
    def channels(self) -> int:
        return channel_count(self.order) if self.ordering == ACN else len(
            [lm for lm in _FUMA_LM if lm[0] <= self.order]
        )

    def describe(self) -> str:
        return (
            f"order={self.order} ordering={self.ordering} "
            f"normalization={self.normalization} "
            f"channels={self.channels}"
        )


def make_convention(ordering: str, normalization: str, order: int) -> Convention:
    """Build a :class:`Convention`, validating the attribute combinations."""
    ordering = ordering.upper() if ordering == "acn" else ordering
    if ordering not in ORDERINGS:
        raise ConventionError(
            f"unknown channel ordering {ordering!r}; supported: {ORDERINGS}"
        )
    if normalization not in (SN3D, N3D, FUMA_NORMALIZATION):
        raise ConventionError(
            f"unknown normalization {normalization!r}; supported: "
            f"{SN3D!r}, {N3D!r}, {FUMA_NORMALIZATION!r}"
        )
    if order < 0:
        raise ConventionError("order must be >= 0")
    if ordering == FUMA:
        if normalization != FUMA_NORMALIZATION:
            raise ConventionError(
                "FuMa channel ordering is only defined together with FuMa "
                "(max-N) normalization; got "
                f"ordering=FuMa normalization={normalization}"
            )
        if channel_count(order) not in SUPPORTED_FUMA_CHANNELS:
            raise ConventionError(
                f"FuMa layout is only supported for orders 1..3 "
                f"(channel counts {SUPPORTED_FUMA_CHANNELS}); "
                f"order {order} would need {channel_count(order)} channels"
            )
    elif normalization == FUMA_NORMALIZATION:
        raise ConventionError(
            "FuMa normalization requires FuMa channel ordering, not ACN"
        )
    return Convention(ordering, normalization, order)


def fuma_to_acn_map(order: int) -> np.ndarray:
    """Return array ``m`` so that ACN channel i receives FuMa channel m[i]."""
    n_channels = channel_count(order)
    if n_channels not in SUPPORTED_FUMA_CHANNELS:
        raise ConventionError(
            f"unsupported FuMa order {order} ({n_channels} channels)"
        )
    mapping = np.empty(n_channels, dtype=int)
    for fuma_idx, (l, m) in enumerate(_FUMA_LM[:n_channels]):
        mapping[acn_index(l, m)] = fuma_idx
    return mapping


def acn_to_fuma_map(order: int) -> np.ndarray:
    """Inverse permutation: FuMa channel j receives ACN channel m[j]."""
    return np.argsort(fuma_to_acn_map(order))


def convert_to_acn_sn3d(
    signals: np.ndarray, convention: Convention
) -> np.ndarray:
    """Convert SH signals of any supported convention into canonical ACN/SN3D.

    Parameters
    ----------
    signals : array, shape (channels, samples) or (channels,)
    convention : declared convention of ``signals``.

    Returns
    -------
    Canonical ACN/SN3D array with the same number of channels.
    """
    signals = np.asarray(signals, dtype=float)
    if signals.shape[0] != convention.channels:
        raise ConventionError(
            f"signal has {signals.shape[0]} channels but convention declares "
            f"{convention.channels} ({convention.describe()})"
        )
    if convention.ordering == ACN and convention.normalization == SN3D:
        return signals.copy()

    if convention.ordering == FUMA:
        mapping = fuma_to_acn_map(convention.order)
        out = signals[mapping]
        gains = np.empty(convention.channels)
        for l in range(convention.order + 1):
            for m in range(-l, l + 1):
                gains[acn_index(l, m)] = _FUMA_TO_SN3D_GAIN[l]
        return out * gains[:, None] if out.ndim == 2 else out * gains

    # ACN with N3D: N3D -> SN3D is multiplication by sqrt(4*pi/(2l+1)).
    assert convention.ordering == ACN and convention.normalization == N3D
    from .harmonics import sn3d_to_n3d

    gain = sn3d_to_n3d(convention.order)
    return signals / (gain[:, None] if signals.ndim == 2 else gain)


def convert_from_acn_sn3d(
    signals: np.ndarray, convention: Convention
) -> np.ndarray:
    """Inverse of :func:`convert_to_acn_sn3d` (used when writing files)."""
    signals = np.asarray(signals, dtype=float)
    if convention.ordering == ACN and convention.normalization == SN3D:
        return signals.copy()
    if convention.ordering == FUMA:
        # First express in ACN/FuMa-normalized, then permute to FuMa order.
        gains = np.empty(convention.channels)
        for l in range(convention.order + 1):
            for m in range(-l, l + 1):
                gains[acn_index(l, m)] = _FUMA_TO_SN3D_GAIN[l]
        acn_fuma = signals / (gains[:, None] if signals.ndim == 2 else gains)
        mapping = fuma_to_acn_map(convention.order)  # ACN i <- FuMa mapping[i]
        out = np.empty_like(acn_fuma)
        for acn_idx, fuma_idx in enumerate(mapping):
            out[fuma_idx] = acn_fuma[acn_idx]
        return out
    assert convention.ordering == ACN and convention.normalization == N3D
    from .harmonics import sn3d_to_n3d

    gain = sn3d_to_n3d(convention.order)
    return signals * (gain[:, None] if signals.ndim == 2 else gain)


def identify_convention(
    channels: int,
    ordering: str | None = None,
    normalization: str | None = None,
) -> Convention:
    """Resolve a convention from explicit metadata, refusing ambiguous input.

    Channel count alone never determines a convention: e.g. 4 channels can be
    ACN/SN3D, ACN/N3D or FuMa, and decoding them with the wrong matrix yields
    a corrupted sound field. Explicit ``ordering``/``normalization`` metadata
    is therefore required; defaults are only applied when the count is a valid
    ACN square and both fields are missing, in which case ambix (ACN/SN3D) is
    assumed AND the assumption is reported via the returned object.
    """
    if ordering is None and normalization is None:
        try:
            order = order_from_channels(channels)
        except ValueError as exc:
            raise ConventionError(
                f"cannot identify an SH convention for {channels} channels: "
                f"{exc}. Provide explicit ordering/normalization metadata."
            ) from exc
        return make_convention(ACN, SN3D, order)
    if ordering is None or normalization is None:
        raise ConventionError(
            f"partial convention metadata (ordering={ordering!r}, "
            f"normalization={normalization!r}): both must be declared together; "
            "channel numbers cannot determine the SH convention"
        )
    if ordering.upper() == FUMA or ordering == FUMA:
        if channels not in SUPPORTED_FUMA_CHANNELS:
            raise ConventionError(
                f"FuMa files must have {SUPPORTED_FUMA_CHANNELS} channels "
                f"(orders 1-3), got {channels}"
            )
        order = {4: 1, 9: 2, 16: 3}[channels]
        return make_convention(FUMA, FUMA_NORMALIZATION, order)
    try:
        order = order_from_channels(channels)
    except ValueError as exc:
        raise ConventionError(
            f"ACN convention impossible for {channels} channels: {exc}"
        ) from exc
    return make_convention(ACN, normalization.upper(), order)
