"""shfield: spherical-harmonic sound-field encoding, rotation and decoding.

Canonical internal representation everywhere: ACN channel ordering with SN3D
normalization, coordinate frame +x front / +y left / +z up.
"""

from .harmonics import (
    N3D,
    SN3D,
    acn_index,
    channel_count,
    order_from_channels,
    sh_matrix,
    sph_to_cart,
    cart_to_sph,
)
from .conventions import (
    ACN,
    FUMA,
    Convention,
    ConventionError,
    convert_to_acn_sn3d,
    convert_from_acn_sn3d,
    identify_convention,
    make_convention,
    fuma_to_acn_map,
)
from .rotation import (
    euler_rotation,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    rotate_coefficients,
    sh_rotation_matrix,
    rotated_direction,
    rotation_energy_error,
)
from .encoding import (
    Source,
    encode_sources,
    plane_wave_encoder,
    make_impulse,
    make_uncorrelated_noise,
    make_pink_noise,
)
from .decode import (
    Layout,
    NAMED_LAYOUTS,
    binaural_decode,
    decode,
    decoding_matrix,
    dense_ring_layout,
    max_decodable_order,
    stereo_layout,
    quad_layout,
    octagon_layout,
    enneagon_layout,
    cube_layout,
    tetra_layout,
    synthetic_hrir,
)
from .analysis import (
    field_directional_response,
    speaker_beam_pattern,
    sh_energy,
    spherical_energy_conservation,
    rms,
)
from .pipeline import render, PipelineResult
from .io_wav import read_sh_wav, write_sh_wav, check_channels

__all__ = [
    "N3D", "SN3D", "ACN", "FUMA",
    "acn_index", "channel_count", "order_from_channels", "sh_matrix",
    "sph_to_cart", "cart_to_sph",
    "Convention", "ConventionError", "convert_to_acn_sn3d",
    "convert_from_acn_sn3d", "identify_convention", "make_convention",
    "fuma_to_acn_map",
    "euler_rotation", "rotation_matrix_x", "rotation_matrix_y",
    "rotation_matrix_z", "rotate_coefficients", "sh_rotation_matrix",
    "rotated_direction", "rotation_energy_error",
    "Source", "encode_sources", "plane_wave_encoder", "make_impulse",
    "make_uncorrelated_noise", "make_pink_noise",
    "Layout", "NAMED_LAYOUTS", "binaural_decode", "decode",
    "decoding_matrix", "dense_ring_layout", "max_decodable_order",
    "stereo_layout", "quad_layout",
    "octagon_layout", "enneagon_layout", "cube_layout", "tetra_layout",
    "synthetic_hrir",
    "field_directional_response", "speaker_beam_pattern", "sh_energy",
    "spherical_energy_conservation", "rms",
    "render", "PipelineResult",
    "read_sh_wav", "write_sh_wav", "check_channels",
]
