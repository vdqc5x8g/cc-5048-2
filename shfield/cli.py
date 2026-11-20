"""Command-line interface for the shfield sound-field tool.

Subcommands
-----------
scene     build impulse / uncorrelated-noise test scenes and encode them
render    rotate an SH field and decode to headphones or a speaker layout
inspect   read an SH WAV + sidecar and report convention, channels, energy
diag      validate an SH WAV against an expected order, with clear errors
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.io import wavfile

from .analysis import rms, sh_energy, spherical_energy_conservation
from .conventions import ConventionError, make_convention
from .decode import NAMED_LAYOUTS, binaural_decode, decode, decoding_matrix
from .encoding import (
    Source,
    encode_sources,
    make_impulse,
    make_pink_noise,
    make_uncorrelated_noise,
)
from .io_wav import check_channels, read_sh_wav, write_sh_wav
from .rotation import euler_rotation, rotate_coefficients, rotation_energy_error, sh_rotation_matrix

DEFAULT_SR = 48000


def _parse_direction(text: str) -> tuple[float, float]:
    """Parse 'theta_deg,phi_deg' (theta from up, phi from front, +left)."""
    try:
        theta_deg, phi_deg = (float(v) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"direction {text!r} must be 'theta_deg,phi_deg'"
        ) from exc
    return np.deg2rad(theta_deg), np.deg2rad(phi_deg)


def _build_sources(args) -> list[Source]:
    rng_seed = args.seed
    sources = []
    if args.scene == "impulse":
        for k, spec in enumerate(args.source):
            theta, phi = _parse_direction(spec)
            sources.append(
                Source(f"impulse_{k}", theta, phi,
                       make_impulse(args.sample_rate, delay_ms=50 + 20 * k))
            )
    elif args.scene == "noise":
        streams = make_uncorrelated_noise(
            args.sample_rate, args.duration, len(args.source), seed=rng_seed
        )
        for k, spec in enumerate(args.source):
            theta, phi = _parse_direction(spec)
            sources.append(Source(f"noise_{k}", theta, phi, streams[k]))
    elif args.scene == "pink":
        for k, spec in enumerate(args.source):
            theta, phi = _parse_direction(spec)
            sources.append(
                Source(
                    f"pink_{k}", theta, phi,
                    make_pink_noise(args.sample_rate, args.duration,
                                    seed=rng_seed + k),
                )
            )
    else:  # pragma: no cover
        raise ValueError(args.scene)
    return sources


def cmd_scene(args) -> int:
    sources = _build_sources(args)
    coefficients = encode_sources(args.order, sources)
    convention = make_convention(args.ordering, args.normalization, args.order)
    sidecar = write_sh_wav(
        args.output, coefficients, args.sample_rate, convention,
        extra={
            "scene": args.scene,
            "sources": [
                {"name": s.name, "theta_deg": float(np.rad2deg(s.theta)),
                 "phi_deg": float(np.rad2deg(s.phi))}
                for s in sources
            ],
        },
    )
    energy = sh_energy(coefficients, args.order)
    print(
        f"wrote {args.output} ({convention.describe()}) + sidecar {sidecar.name}\n"
        f"sources: {len(sources)}  scene: {args.scene}  "
        f"total channel energy: {energy['total']:.4e}"
    )
    return 0


def _load_field(path: str, ordering: str | None, normalization: str | None):
    # read_sh_wav already returns canonical ACN/SN3D coefficients.
    return read_sh_wav(path, ordering, normalization)


def cmd_render(args) -> int:
    coefficients, sr, convention, meta = _load_field(
        args.input, args.ordering, args.normalization
    )
    order = convention.order
    r = euler_rotation(
        np.deg2rad(args.yaw), np.deg2rad(args.pitch), np.deg2rad(args.roll)
    )
    d = sh_rotation_matrix(order, r.T)
    diag = rotation_energy_error(d, order)
    rotated = rotate_coefficients(coefficients, order, r, listener=True)

    before = spherical_energy_conservation(coefficients, order)
    after = spherical_energy_conservation(rotated, order)
    print(
        f"rotation orthogonality max err: {diag['orthogonality_max_err']:.2e}\n"
        f"spherical energy before={before:.6e} after={after:.6e} "
        f"relative drift={abs(after-before)/before:.2e}"
    )

    if args.layout not in NAMED_LAYOUTS:
        raise SystemExit(f"unknown layout {args.layout!r}; choose {sorted(NAMED_LAYOUTS)}")
    layout = NAMED_LAYOUTS[args.layout]()
    if args.binaural:
        feeds = binaural_decode(rotated, order, layout, sr)
        labels = ["L", "R"]
    else:
        try:
            decoder = decoding_matrix(order, layout)
        except ValueError as exc:
            raise SystemExit(f"decoder error: {exc}") from exc
        feeds = decode(rotated, decoder)
        labels = [f"spk{k}" for k in range(layout.count)]

    peak = np.max(np.abs(feeds))
    if peak > 0:
        feeds = feeds / peak * 0.9
    wavfile.write(args.output, sr, feeds.T.astype(np.float32))
    levels = ", ".join(f"{lab} rms={rms(feeds[k]):.3f}" for k, lab in enumerate(labels))
    print(f"wrote {args.output} ({feeds.shape[0]} feeds): {levels}")
    return 0


def cmd_inspect(args) -> int:
    coefficients, sr, convention, meta = _load_field(
        args.input, args.ordering, args.normalization
    )
    energy = sh_energy(coefficients, convention.order)
    print(f"file: {args.input}")
    print(f"sample rate: {sr}  channels: {coefficients.shape[0]}  samples: {coefficients.shape[1]}")
    print(f"convention: {convention.describe()}")
    if meta.get("assumed_default"):
        print("NOTE: no sidecar/metadata; assumed ambix default ACN/SN3D")
    print("per-order energy:")
    for l, value in energy["per_order"].items():
        print(f"  order {l}: {value:.4e}")
    print("per-channel RMS (canonical ACN/SN3D):")
    for k, value in enumerate(energy["per_channel"]):
        print(f"  ACN {k:2d}: {np.sqrt(value):.4e}")
    return 0


def cmd_diag(args) -> int:
    try:
        coefficients, sr, convention, meta = _load_field(
            args.input, args.ordering, args.normalization
        )
        check_channels(coefficients.shape[0], args.expected_order)
    except ConventionError as exc:
        print(f"DIAGNOSTIC FAILURE: {exc}", file=sys.stderr)
        return 2
    print(
        f"OK: {coefficients.shape[0]} channels match order {args.expected_order}; "
        f"convention {convention.ordering}/{convention.normalization}"
    )
    return 0


def _print_diagnostic(result: dict) -> None:
    print(f"order N={result['order']}  channels={result['channels']}  "
          f"tolerance={result['tolerance']:g}")
    print("\n输入映射/归一化环节（相对偏差按阶次，None=该阶首次偏离）:")
    for stage in result["stages"]:
        if not stage["applied"]:
            print(f"  {stage['variant']:18s} 已旁路")
            continue
        rel = stage["per_order_relative"]
        rel_text = " ".join(f"l{l}={v:.2e}" for l, v in rel.items())
        print(f"  {stage['variant']:18s} 首个偏离={stage['first_deviating_order']}"
              f"  {rel_text}")
    rot = result["rotation"]
    print(f"\n旋转 vs 独立点源参照: 首个偏离={rot['first_deviating_order']} "
          f"通过={rot['pass']}")
    print("运动（逆序还原 vs 同序还原）:")
    for pose in result["motion"]["poses"]:
        if pose["reverse_restore"] is None:
            print(f"  {pose['name']:20s} 旋转已旁路")
            continue
        print(f"  {pose['name']:20s} 逆序={pose['reverse_restore']['pass']} "
              f"同序={pose['same_order_restore']['pass']}")
    red = result["motion"]["same_order_pose_reduction"]
    print(f"  同序还原仍失败的最小姿态集: {red['reduced']} "
          f"(仍失败={red['reduced_still_fails']})")
    print(f"\n零阶负对照: 通过={result['omni_control']['pass']} "
          f"能量漂移={result['omni_control']['relative_drift']:.2e}")
    print(f"归因: {result['confusion']['attribution']}")


def cmd_diagnose(args) -> int:
    from .diagnostics import run_diagnostic

    try:
        result = run_diagnostic(
            order=args.order,
            sample_rate=args.sample_rate,
            yaw_deg=args.yaw,
            pitch_deg=args.pitch,
            bypass_import=args.bypass_import,
            bypass_normalization=args.bypass_normalization,
            bypass_rotation=args.bypass_rotation,
        )
    except ValueError as exc:
        print(f"DIAGNOSE ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_diagnostic(result)
    return 0


def cmd_sample(args) -> int:
    from .diagnostics import write_project_sample

    try:
        out = write_project_sample(args.out_dir, order=args.order,
                                   sample_rate=args.sample_rate)
    except ValueError as exc:
        print(f"SAMPLE ERROR: {exc}", file=sys.stderr)
        return 2
    for name, path in out["written"].items():
        print(f"wrote {name}: {path}")
    print("manifest stores per-order coefficients and signal recipes; "
          "rebuild without the WAV scale factor")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shfield",
        description="Spherical-harmonic sound-field encode/rotate/decode tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scene = sub.add_parser("scene", help="encode a test scene to an SH WAV")
    p_scene.add_argument("--scene", choices=["impulse", "noise", "pink"], required=True)
    p_scene.add_argument("--order", type=int, default=2)
    p_scene.add_argument("--source", action="append", required=True,
                         help="direction 'theta_deg,phi_deg', repeatable")
    p_scene.add_argument("--duration", type=float, default=2.0)
    p_scene.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    p_scene.add_argument("--seed", type=int, default=0)
    p_scene.add_argument("--ordering", choices=["ACN", "FuMa"], default="ACN")
    p_scene.add_argument("--normalization", choices=["SN3D", "N3D", "FuMa"], default="SN3D")
    p_scene.add_argument("--output", required=True)
    p_scene.set_defaults(func=cmd_scene)

    p_render = sub.add_parser("render", help="rotate and decode an SH WAV")
    p_render.add_argument("input")
    p_render.add_argument("--output", required=True)
    p_render.add_argument("--yaw", type=float, default=0.0)
    p_render.add_argument("--pitch", type=float, default=0.0)
    p_render.add_argument("--roll", type=float, default=0.0)
    p_render.add_argument("--layout", choices=sorted(NAMED_LAYOUTS), default="octagon")
    p_render.add_argument("--binaural", action="store_true")
    p_render.add_argument("--ordering", choices=["ACN", "FuMa"], default=None)
    p_render.add_argument("--normalization", choices=["SN3D", "N3D", "FuMa"], default=None)
    p_render.set_defaults(func=cmd_render)

    p_inspect = sub.add_parser("inspect", help="report convention and channel content")
    p_inspect.add_argument("input")
    p_inspect.add_argument("--ordering", choices=["ACN", "FuMa"], default=None)
    p_inspect.add_argument("--normalization", choices=["SN3D", "N3D", "FuMa"], default=None)
    p_inspect.set_defaults(func=cmd_inspect)

    p_diag = sub.add_parser("diag", help="validate channel count against an order")
    p_diag.add_argument("input")
    p_diag.add_argument("--expected-order", type=int, required=True)
    p_diag.add_argument("--ordering", choices=["ACN", "FuMa"], default=None)
    p_diag.add_argument("--normalization", choices=["SN3D", "N3D", "FuMa"], default=None)
    p_diag.set_defaults(func=cmd_diag)

    p_diagnose = sub.add_parser(
        "diagnose",
        help="stage-by-stage diagnostic view (bypass, first deviating order, controls)",
    )
    p_diagnose.add_argument("--order", type=int, default=3, choices=[1, 2, 3])
    p_diagnose.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    p_diagnose.add_argument("--yaw", type=float, default=40.0)
    p_diagnose.add_argument("--pitch", type=float, default=25.0)
    p_diagnose.add_argument("--bypass-import", action="store_true",
                            help="switch off the FuMa input mapping stage")
    p_diagnose.add_argument("--bypass-normalization", action="store_true",
                            help="switch off the max-N -> SN3D gain stage")
    p_diagnose.add_argument("--bypass-rotation", action="store_true",
                            help="switch off the coefficient rotation stage")
    p_diagnose.add_argument("--json", action="store_true",
                            help="emit the full result as JSON")
    p_diagnose.set_defaults(func=cmd_diagnose)

    p_sample = sub.add_parser(
        "sample", help="write the reproducible diagnostic field project"
    )
    p_sample.add_argument("out_dir")
    p_sample.add_argument("--order", type=int, default=3, choices=[1, 2, 3])
    p_sample.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    p_sample.set_defaults(func=cmd_sample)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConventionError as exc:
        print(f"CONVENTION ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
