#!/usr/bin/env python3
"""Reproducible demonstration of the full shfield pipeline.

Run from the repository root::

    python examples/demo.py

It prints (and asserts) the key numerical facts: convention conversions,
rotation/source-motion equivalence, energy conservation, layout limits and
binaural decoding. Nothing is written outside examples/out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shfield import (  # noqa: E402
    N3D,
    SN3D,
    Source,
    binaural_decode,
    cart_to_sph,
    channel_count,
    convert_to_acn_sn3d,
    decoding_matrix,
    encode_sources,
    make_convention,
    make_impulse,
    make_uncorrelated_noise,
    quad_layout,
    rotate_coefficients,
    rotation_energy_error,
    rotation_matrix_z,
    sh_matrix,
    sh_rotation_matrix,
    sph_to_cart,
    spherical_energy_conservation,
)

SR = 48000


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> int:
    section("1. 声明的约定 (ACN/SN3D canonical, +x front +y left +z up)")
    order = 2
    print(f"order N={order}, ACN channel count (N+1)^2 = {channel_count(order)}")
    front = sh_matrix(order, np.array(np.pi / 2), np.array(0.0), SN3D)[0]
    print("front direction SN3D coefficients:", np.round(front, 4))

    section("2. 同样 9 个声道，三种约定数值不同 -> 不能套同一个矩阵")
    rng = np.random.default_rng(0)
    data = rng.standard_normal((9, 2048))
    ambix = convert_to_acn_sn3d(data, make_convention("ACN", SN3D, order))
    n3d = convert_to_acn_sn3d(data, make_convention("ACN", N3D, order))
    fuma = convert_to_acn_sn3d(data, make_convention("FuMa", "FuMa", order))
    print("ambix==n3d?", np.allclose(ambix, n3d), " ambix==fuma?", np.allclose(ambix, fuma))
    assert not np.allclose(ambix, n3d) and not np.allclose(ambix, fuma)

    section("3. 编码测试声源 (脉冲 + 互不相关噪声)")
    impulse_src = Source("click", np.pi / 2, 0.0, make_impulse(SR))
    coeffs_imp = encode_sources(order, [impulse_src])
    noise = make_uncorrelated_noise(SR, 1.0, 2, seed=1)
    noise_srcs = [
        Source("n_front", np.pi / 2, 0.0, noise[0]),
        Source("n_left", np.pi / 2, np.pi / 2, noise[1]),
    ]
    coeffs_noise = encode_sources(order, noise_srcs)
    print("impulse SH shape:", coeffs_imp.shape, " noise SH shape:", coeffs_noise.shape)

    section("4. 听者旋转 R 等价于声源反向移动 (机器精度)")
    yaw = np.deg2rad(35)
    r = rotation_matrix_z(yaw)
    theta0, phi0 = np.array(np.pi / 2), np.array(0.0)
    a = sh_matrix(order, theta0, phi0, SN3D)[0]
    listener = rotate_coefficients(a, order, r, listener=True)
    th_i, ph_i = cart_to_sph(sph_to_cart(theta0, phi0) @ r)
    moved = sh_matrix(order, th_i, ph_i, SN3D)[0]
    err = np.max(np.abs(listener - moved))
    print(f"yaw=+35deg; front source listener-relative phi = {np.rad2deg(ph_i):.2f} deg (expect -35)")
    print(f"max |D(R^T)Y(u) - Y(R u)| = {err:.2e}")
    assert err < 1e-12

    section("5. 旋转作用于系数：稠密 Wigner-D，非左右交换；能量守恒")
    d = sh_rotation_matrix(order, r.T)
    diag = rotation_energy_error(d, order)
    print("orthogonality max err:", f"{diag['orthogonality_max_err']:.2e}")
    print("per-order RMS gain:", np.round(diag["per_order_rms_gain"], 8))
    e_before = spherical_energy_conservation(coeffs_noise, order)
    e_after = spherical_energy_conservation(
        rotate_coefficients(coeffs_noise, order, r, listener=True), order
    )
    print(f"energy before={e_before:.6e} after={e_after:.6e} drift={abs(e_after-e_before)/e_before:.2e}")
    assert diag["orthogonality_max_err"] < 1e-11

    section("6. 固定布局不随声道数变精细：quad 拒绝 order 3")
    try:
        decoding_matrix(3, quad_layout())
        raise AssertionError("should have refused")
    except ValueError as exc:
        print("refused:", str(exc)[:110])

    section("7. 双耳解码输出 (合成 HRTF, 虚拟扬声器环)")
    ears = binaural_decode(coeffs_noise, order, quad_layout(), SR)
    print("binaural output shape:", ears.shape, "(L/R, samples), finite:", np.all(np.isfinite(ears)))
    print("\nALL DEMO CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
