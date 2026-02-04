"""
Pipeline:
1) Read CIE 2006 2 deg LMS cone fundamentals (energy-based) from CSV.
2) Interpolate to 1 nm grid (360..830) using local cubic splines (window=9).
3) Extrapolate LMS with a decaying tail from 390 down to 360.
4) LMS -> XYZ using the provided matrix.
5) XYZ -> linear sRGB using the provided matrix.
6) Apply sRGB OETF (gamma encoding).
7) Clip to [0, global max], then normalize.
8) Print the RGB table and plot all stages.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


# LMS -> XYZ (CIE 2006 2 deg).
LMS_TO_XYZ = np.array(
    [
        [1.94735469, -1.41445123, 0.36476327],
        [0.68990272, 0.34832189, -0.03715971],
        [0.0, 0.0, 1.93485343],
    ]
)

# XYZ (D65) -> linear sRGB.
XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)

SRGB_OETF_THRESHOLD = 0.0031308


def srgb_oetf(rgb_linear: np.ndarray) -> np.ndarray:
    return np.where(
        rgb_linear <= SRGB_OETF_THRESHOLD,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1 / 2.4) - 0.055,
    )


def generate_wavelength_colormap(csv_path: Path) -> None:
    # 1) Load LMS fundamentals.
    rows: list[tuple[float, float, float, float]] = []
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            wl = float(row[0])
            l = float(row[1])
            m = float(row[2])
            s = float("nan") if row[3].strip().lower() == "nan" else float(row[3])
            rows.append((wl, l, m, s))

    wls = np.array([r[0] for r in rows])
    L = np.array([r[1] for r in rows])
    M = np.array([r[2] for r in rows])
    S = np.nan_to_num(np.array([r[3] for r in rows]), nan=0.0)

    # 2) Interpolate to 1 nm grid using local cubic splines.
    wl_min, wl_max = 360, 830
    wl_grid = np.arange(wl_min, wl_max + 1, 1.0)
    L_i = local_cubic_spline(wls, L, wl_grid)
    M_i = local_cubic_spline(wls, M, wl_grid)
    S_i = local_cubic_spline(wls, S, wl_grid)

    # 3) Extrapolate LMS with a decaying tail for 360..389 nm.
    L_i = apply_uv_decay(wl_grid, L_i, decay_start=390, fit_start=390, fit_end=405)
    M_i = apply_uv_decay(wl_grid, M_i, decay_start=390, fit_start=390, fit_end=405)
    S_i = apply_uv_decay(wl_grid, S_i, decay_start=390, fit_start=390, fit_end=405)
    LMS = np.stack([L_i, M_i, S_i], axis=0)

    # 4) LMS -> XYZ.
    XYZ = LMS_TO_XYZ @ LMS

    # 5) XYZ -> linear sRGB (matrix result).
    RGB_prime = XYZ_TO_SRGB @ XYZ

    # 6) Gamma correction (sRGB OETF).
    RGB_gamma = srgb_oetf(RGB_prime)

    # 7) Clip after gamma to [0, global max], then normalize.
    maxv = np.max(RGB_gamma)
    RGB_gamma_pos = np.clip(RGB_gamma, 0.0, maxv)
    if maxv > 0:
        RGB_norm = RGB_gamma_pos / maxv
    else:
        RGB_norm = RGB_gamma_pos * 0.0

    # 8) Emit table (for insertion into _cm_listed.py).
    lines = ["["]
    for i in range(RGB_norm.shape[1]):
        r, g, b = (
            float(RGB_norm[0, i]),
            float(RGB_norm[1, i]),
            float(RGB_norm[2, i]),
        )
        lines.append(f"    [{r:.6f}, {g:.6f}, {b:.6f}],")
    lines.append("]")
    print("\n".join(lines))

    # Plot all steps for transparency.
    fig, axes = plt.subplots(6, 1, figsize=(10, 13), sharex=True, layout="constrained")

    axes[0].plot(wl_grid, LMS[0], label="L", color="blue")
    axes[0].plot(wl_grid, LMS[1], label="M", color="green")
    axes[0].plot(wl_grid, LMS[2], label="S", color="red")
    axes[0].set_title("LMS (CIE 2006 2 deg, energy)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(wl_grid, XYZ[0], label="X", color="blue")
    axes[1].plot(wl_grid, XYZ[1], label="Y", color="green")
    axes[1].plot(wl_grid, XYZ[2], label="Z", color="red")
    axes[1].set_title("XYZ (unnormalized)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(wl_grid, RGB_prime[0], label="R'", color="red")
    axes[2].plot(wl_grid, RGB_prime[1], label="G'", color="green")
    axes[2].plot(wl_grid, RGB_prime[2], label="B'", color="blue")
    axes[2].set_title("sRGB (matrix result, before gamma)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(wl_grid, RGB_gamma[0], label="R", color="red")
    axes[3].plot(wl_grid, RGB_gamma[1], label="G", color="green")
    axes[3].plot(wl_grid, RGB_gamma[2], label="B", color="blue")
    axes[3].set_title("sRGB (gamma encoded, unclipped)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(-0.5, None)

    axes[4].plot(wl_grid, RGB_norm[0], label="R", color="red")
    axes[4].plot(wl_grid, RGB_norm[1], label="G", color="green")
    axes[4].plot(wl_grid, RGB_norm[2], label="B", color="blue")
    axes[4].set_title("sRGB (gamma encoded, clipped to max + normalized)")
    axes[4].legend(loc="upper right")
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim(0, 1.0)
    axes[4].set_xlabel("Wavelength (nm)")

    # Final colormap preview (RGB mixed).
    stripe = RGB_norm.T[np.newaxis, :, :]
    axes[5].imshow(stripe, aspect="auto", extent=[wl_min, wl_max, 0, 1])
    axes[5].set_title("Resulting colormap")
    axes[5].set_yticks([])
    axes[5].set_xlabel("Wavelength (nm)")

    plt.show()


def local_cubic_spline(
    x: np.ndarray, y: np.ndarray, x_new: np.ndarray, window: int = 9
) -> np.ndarray:
    if window % 2 == 0 or window < 4:
        raise ValueError("window must be an odd integer >= 5")
    half = window // 2
    out = np.empty_like(x_new, dtype=float)
    for i, xv in enumerate(x_new):
        idx = int(np.searchsorted(x, xv))
        idx = max(0, min(idx, len(x) - 1))
        lo = max(0, idx - half)
        hi = min(len(x), idx + half + 1)
        xs = x[lo:hi]
        ys = y[lo:hi]
        if len(xs) < 4:
            out[i] = np.interp(xv, x, y)
        else:
            spline = CubicSpline(xs, ys, bc_type="natural")
            out[i] = float(spline(xv))
    return out


def apply_uv_decay(
    wl_grid: np.ndarray,
    values: np.ndarray,
    decay_start: float,
    fit_start: float,
    fit_end: float,
) -> np.ndarray:
    mask = wl_grid < decay_start
    if not np.any(mask):
        return values
    fit_mask = (wl_grid >= fit_start) & (wl_grid <= fit_end)
    x_fit = wl_grid[fit_mask]
    y_fit = values[fit_mask]
    positive = y_fit > 0
    if np.count_nonzero(positive) < 2:
        values[mask] = 0.0
        return values
    x_fit = x_fit[positive]
    y_fit = y_fit[positive]
    log_y = np.log(y_fit)
    slope, intercept = np.polyfit(x_fit, log_y, 1)
    slope = max(slope, 0.0)
    values[mask] = np.exp(intercept + slope * wl_grid[mask])
    return values


if __name__ == "__main__":
    # Input CSV (must be present in tools/).
    # https://files.cie.co.at/CIE_lms_cf_2deg.csv
    generate_wavelength_colormap(Path("tools/CIE_lms_cf_2deg.csv"))
