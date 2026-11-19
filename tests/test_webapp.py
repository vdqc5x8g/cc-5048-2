"""Integration tests for the kernel function backing the web view."""

import numpy as np
import pytest

from shfield.conventions import ConventionError
from shfield.webapp import kernel_compute


def test_kernel_compute_impulse_binaural():
    result = kernel_compute(
        {
            "scene": "impulse",
            "order": 2,
            "yaw": 45,
            "pitch": 0,
            "roll": 0,
            "layout": "enneagon",
            "binaural": True,
            "sources": [{"name": "front", "theta": 90, "phi": 0}],
        }
    )
    assert result["ok"] is True
    assert result["channels"] == 9
    # Positive (leftward) yaw moves a front source to the right: phi = -45.
    rel = result["sources_listener_frame"][0]
    assert abs(rel["phi_deg"] - (-45.0)) < 1e-9
    # Rotation must preserve energy and be orthogonal to machine precision.
    assert result["rotation"]["orthogonality_max_err"] < 1e-11
    assert result["rotation"]["relative_energy_drift"] < 1e-12
    assert len(result["feed_rms"]) == 2  # binaural L/R
    assert isinstance(result["_wav"], bytes) and len(result["_wav"]) > 1000


def test_kernel_compute_noise_speakers():
    result = kernel_compute(
        {
            "scene": "noise",
            "duration": 0.3,
            "order": 1,
            "layout": "quad",
            "binaural": False,
            "sources": [
                {"name": "a", "theta": 90, "phi": 0},
                {"name": "b", "theta": 90, "phi": 90},
            ],
        }
    )
    assert result["ok"] is True
    assert len(result["feed_rms"]) == 4
    assert result["layout"]["feed_labels"] == ["spk0", "spk1", "spk2", "spk3"]


def test_kernel_rejects_order_above_layout():
    with pytest.raises(ValueError, match="spatial"):
        kernel_compute(
            {
                "scene": "impulse",
                "order": 3,
                "layout": "quad",
                "binaural": False,
                "sources": [{"name": "f", "theta": 90, "phi": 0}],
            }
        )


def test_kernel_rejects_bad_order_and_empty_sources():
    with pytest.raises(ConventionError, match="0..6"):
        kernel_compute(
            {"scene": "impulse", "order": 9, "layout": "quad",
             "binaural": True, "sources": [{"name": "f", "theta": 90, "phi": 0}]}
        )
    with pytest.raises(ConventionError, match="at least one source"):
        kernel_compute(
            {"scene": "impulse", "order": 1, "layout": "quad",
             "binaural": True, "sources": []}
        )


def test_power_map_shape_and_finite():
    result = kernel_compute(
        {"scene": "noise", "duration": 0.2, "order": 2, "layout": "octagon",
         "binaural": True, "sources": [{"name": "f", "theta": 90, "phi": 0}]}
    )
    grid = np.array(result["power_map"])
    assert grid.ndim == 2 and np.all(np.isfinite(grid))
