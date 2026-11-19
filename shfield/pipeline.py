"""End-to-end pipeline: encode sources -> rotate field -> decode to feeds."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decode import Layout, binaural_decode, decode, decoding_matrix
from .encoding import Source, encode_sources
from .rotation import euler_rotation, rotate_coefficients


@dataclass
class PipelineResult:
    coefficients_world: np.ndarray   # ACN/SN3D before listener rotation
    coefficients_listener: np.ndarray
    rotation_matrix: np.ndarray
    feeds: np.ndarray                # speaker feeds (L, T) or binaural (2, T)
    order: int
    layout_name: str
    diagnostics: dict = field(default_factory=dict)


def render(
    sources: list[Source],
    order: int,
    layout: Layout,
    sample_rate: int,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    binaural: bool = False,
    rE_weighted: bool = True,
) -> PipelineResult:
    """Run the complete encode/rotate/decode chain in canonical ACN/SN3D."""
    world = encode_sources(order, sources)
    r = euler_rotation(yaw, pitch, roll)
    listener = rotate_coefficients(world, order, r, listener=True)
    if binaural:
        feeds = binaural_decode(listener, order, layout, sample_rate, rE_weighted)
    else:
        d = decoding_matrix(order, layout, rE_weighted=rE_weighted)
        feeds = decode(listener, d)
    return PipelineResult(
        coefficients_world=world,
        coefficients_listener=listener,
        rotation_matrix=r,
        feeds=feeds,
        order=order,
        layout_name=layout.name,
    )
