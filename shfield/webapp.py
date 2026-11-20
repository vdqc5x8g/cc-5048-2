"""Browser-based spatial sound-field painting view, driven by the SH kernel.

The server is a dependency-free :mod:`http.server` wrapper around the same
:mod:`shfield` encode/rotate/decode functions the CLI and tests use; there is
no parallel JavaScript math implementation. The browser only renders results
and plays rendered WAVs via Web Audio. Every failure (bad order, missing
channels, undersized layout) is computed by the kernel and returned as a JSON
error which the view shows in the result panel.

Start it with::

    python -m shfield.webapp [--port 8000]

then open http://localhost:8000 .
"""

from __future__ import annotations

import argparse
import io
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from scipy.io import wavfile

from .analysis import field_directional_response, sh_energy, spherical_energy_conservation
from .conventions import ConventionError
from .decode import NAMED_LAYOUTS, binaural_decode, decode, decoding_matrix
from .diagnostics import run_diagnostic as diagnose
from .encoding import (
    Source,
    encode_sources,
    make_impulse,
    make_uncorrelated_noise,
)
from .harmonics import cart_to_sph, channel_count
from .rotation import (
    euler_rotation,
    rotate_coefficients,
    rotation_energy_error,
    sh_rotation_matrix,
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
SR = 48000


def _wav_bytes(feeds: np.ndarray, sample_rate: int) -> bytes:
    peak = np.max(np.abs(feeds)) if feeds.size else 1.0
    scaled = feeds / peak * 0.9 if peak > 0 else feeds
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, scaled.T.astype(np.float32))
    return buf.getvalue()


def _build_sources(payload: dict) -> list[Source]:
    scene = payload.get("scene", "impulse")
    specs = payload.get("sources", [])
    if not specs:
        raise ConventionError("at least one source direction is required")
    sources = []
    if scene == "impulse":
        for k, item in enumerate(specs):
            theta = np.deg2rad(float(item["theta"]))
            phi = np.deg2rad(float(item["phi"]))
            sources.append(
                Source(item.get("name", f"src{k}"), theta, phi,
                       make_impulse(SR, delay_ms=40 + 15 * k))
            )
    else:
        streams = make_uncorrelated_noise(
            SR, float(payload.get("duration", 1.5)), len(specs),
            seed=int(payload.get("seed", 0)),
        )
        for k, item in enumerate(specs):
            theta = np.deg2rad(float(item["theta"]))
            phi = np.deg2rad(float(item["phi"]))
            sources.append(
                Source(item.get("name", f"src{k}"), theta, phi, streams[k])
            )
    return sources


def kernel_compute(payload: dict) -> dict:
    """Run the full kernel pipeline and serialize everything the view shows."""
    order = int(payload["order"])
    if order < 0 or order > 6:
        raise ConventionError(f"order must be in 0..6, got {order}")
    sources = _build_sources(payload)
    world = encode_sources(order, sources)

    yaw = np.deg2rad(float(payload.get("yaw", 0.0)))
    pitch = np.deg2rad(float(payload.get("pitch", 0.0)))
    roll = np.deg2rad(float(payload.get("roll", 0.0)))
    r = euler_rotation(yaw, pitch, roll)
    d_sh = sh_rotation_matrix(order, r.T)
    rot_diag = rotation_energy_error(d_sh, order)
    listener = rotate_coefficients(world, order, r, listener=True)

    energy_before = spherical_energy_conservation(world, order)
    energy_after = spherical_energy_conservation(listener, order)
    energy_drift = abs(energy_after - energy_before) / max(energy_before, 1e-30)

    layout_name = payload.get("layout", "octagon")
    if layout_name not in NAMED_LAYOUTS:
        raise ConventionError(
            f"unknown layout {layout_name!r}; choices {sorted(NAMED_LAYOUTS)}"
        )
    layout = NAMED_LAYOUTS[layout_name]()
    binaural = bool(payload.get("binaural", True))

    if binaural:
        feeds = binaural_decode(listener, order, layout, SR)
        feed_labels = ["L", "R"]
        effective_layout = layout.name
        if feeds.shape[0] == 2 and channel_count(order) > layout.count:
            from .decode import dense_ring_layout

            effective_layout = dense_ring_layout(order).name
    else:
        decoder = decoding_matrix(order, layout)  # raises if layout too small
        feeds = decode(listener, decoder)
        feed_labels = [f"spk{k}" for k in range(layout.count)]
        effective_layout = layout.name

    # Directional response map of the listener-relative field (equirect grid).
    th_grid, ph_grid, power = field_directional_response(listener, order)
    # Downsample for the browser while keeping the shape simple.
    step_t = max(1, th_grid.shape[0] // 45)
    step_p = max(1, ph_grid.shape[1] // 90)
    power_map = power[::step_t, ::step_p].tolist()
    phi_axis = np.rad2deg(ph_grid[0, ::step_p]).tolist()
    theta_axis = np.rad2deg(th_grid[::step_t, 0]).tolist()

    # Source directions in the rotated listener frame. Verified against the
    # coefficient transform: listener-frame coords of world u are u @ R, which
    # matches D(R^T) Y(u); a front source moves to negative phi (the right)
    # under a positive (leftward) yaw.
    rel_sources = []
    for src in sources:
        u = src.direction @ r
        th, ph = cart_to_sph(u)
        rel_sources.append(
            {"name": src.name,
             "theta_deg": float(np.rad2deg(th)),
             "phi_deg": float(np.rad2deg(ph)),
             "world_theta_deg": float(np.rad2deg(src.theta)),
             "world_phi_deg": float(np.rad2deg(src.phi))}
        )

    speaker_dirs = []
    theta_s, phi_s = layout.angles()
    for k in range(layout.count):
        speaker_dirs.append(
            {"index": k, "theta_deg": float(np.rad2deg(theta_s[k])),
             "phi_deg": float(np.rad2deg(phi_s[k]))}
        )

    return {
        "ok": True,
        "order": order,
        "channels": channel_count(order),
        "scene": payload.get("scene", "impulse"),
        "rotation": {
            "yaw_deg": float(np.rad2deg(yaw)),
            "pitch_deg": float(np.rad2deg(pitch)),
            "roll_deg": float(np.rad2deg(roll)),
            "orthogonality_max_err": rot_diag["orthogonality_max_err"],
            "per_order_rms_gain": rot_diag["per_order_rms_gain"],
            "spherical_energy_before": energy_before,
            "spherical_energy_after": energy_after,
            "relative_energy_drift": energy_drift,
        },
        "layout": {
            "name": layout.name,
            "effective": effective_layout,
            "speakers": speaker_dirs,
            "binaural": binaural,
            "feed_labels": feed_labels,
        },
        "sources_listener_frame": rel_sources,
        "power_map": power_map,
        "phi_axis_deg": phi_axis,
        "theta_axis_deg": theta_axis,
        "feed_rms": [float(np.sqrt(np.mean(feeds[k] ** 2))) for k in range(feeds.shape[0])],
        "energy_per_order": sh_energy(listener, order)["per_order"],
        "_wav": _wav_bytes(feeds, SR),
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "shfield/1.0"

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._serve_file(WEB_DIR / "app.js", "application/javascript; charset=utf-8")
        elif path == "/styles.css":
            self._serve_file(WEB_DIR / "styles.css", "text/css; charset=utf-8")
        elif path == "/api/options":
            self._send_json({
                "layouts": sorted(NAMED_LAYOUTS),
                "max_order": 6,
                "sample_rate": SR,
                "convention": "internal ACN/SN3D; +x front, +y left, +z up",
            })
        else:
            self._send_json({"ok": False, "error": f"unknown path {path}"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/diagnose":
            self._handle_diagnose()
            return
        if path != "/api/compute":
            self._send_json({"ok": False, "error": f"unknown path {path}"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = kernel_compute(payload)
            wav = result.pop("_wav")
            if self.headers.get("Accept", "").find("audio/wav") >= 0:
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav)))
                self.send_header(
                    "X-shfield-meta",
                    json.dumps(result).replace("\\", "\\\\").replace("\n", " "),
                )
                self.end_headers()
                self.wfile.write(wav)
            else:
                result["wav_size_bytes"] = len(wav)
                self._send_json(result)
        except ConventionError as exc:
            self._send_json({"ok": False, "error": str(exc), "kind": "ConventionError"}, 422)
        except (ValueError, KeyError, TypeError) as exc:
            self._send_json({"ok": False, "error": str(exc), "kind": exc.__class__.__name__}, 422)
        except Exception:  # pragma: no cover - surface unexpected kernel bugs
            self._send_json(
                {"ok": False, "error": traceback.format_exc(), "kind": "internal"}, 500
            )

    def _handle_diagnose(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = diagnose(
                order=int(payload.get("order", 3)),
                sample_rate=SR,
                yaw_deg=float(payload.get("yaw", 40.0)),
                pitch_deg=float(payload.get("pitch", 25.0)),
                bypass_import=bool(payload.get("bypass_import", False)),
                bypass_normalization=bool(payload.get("bypass_normalization", False)),
                bypass_rotation=bool(payload.get("bypass_rotation", False)),
            )
            self._send_json(result)
        except (ValueError, KeyError, TypeError) as exc:
            self._send_json({"ok": False, "error": str(exc),
                             "kind": exc.__class__.__name__}, 422)
        except Exception:  # pragma: no cover
            self._send_json(
                {"ok": False, "error": traceback.format_exc(), "kind": "internal"},
                500,
            )

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({"ok": False, "error": f"missing {path.name}"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # quieter logs
        pass


def serve(port: int = 8000, host: str = "127.0.0.1") -> None:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"shfield spatial view on http://{host}:{port} (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="shfield web view")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    serve(args.port, args.host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
