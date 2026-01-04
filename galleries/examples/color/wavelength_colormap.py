"""
===================
Wavelength colormap
===================

Examples for mapping wavelengths in nanometers to colors using the
``wavelength`` colormap.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize

rng = np.random.default_rng(4)

# 1) Singular color pulls: a few noisy points per wavelength.
fig1, ax1 = plt.subplots(layout="constrained")
laser_wls = [455, 532, 650]
for wl in laser_wls:
    x = np.linspace(0, 1, 5)
    trend = 0.3 + 0.6 * x
    noise = 0.06 * rng.normal(size=x.size)
    y = trend + noise
    ax1.plot(x, y, marker="o", c=f"{wl}nm", label=f"{wl}nm")
ax1.set_title("Direct color selection with nm strings")
ax1.set_xlabel("Time (a.u.)")
ax1.set_ylabel("Signal (a.u.)")
ax1.legend()

# 2) Simulated spectrum: emission-like peaks with noise.

cmap = plt.get_cmap("wavelength").with_extremes(under="black", over="black")
norm = Normalize(360, 830)

wavelengths = np.arange(360, 831, 1.0)
baseline = 0.08
peaks = (
    0.15 * np.exp(-0.5 * ((wavelengths - 430) / 7.0) ** 2)
    + 0.10 * np.exp(-0.5 * ((wavelengths - 520) / 12.0) ** 2)
    + 0.35 * np.exp(-0.5 * ((wavelengths - 650) / 18.0) ** 2)
)
noise = 0.025 * rng.normal(size=wavelengths.size)
signal = baseline + peaks + noise

fig2, ax2 = plt.subplots(layout="constrained")
ax2.set_xlim(wavelengths.min(), wavelengths.max())
ax2.set_ylim(0, signal.max())

# Smooth colormap underlay using Gouraud-shaded pcolormesh.
X = np.vstack([wavelengths, wavelengths])
Y = np.vstack([np.zeros_like(wavelengths), signal])
C = np.vstack([wavelengths, wavelengths])
ax2.pcolormesh(X, Y, C, cmap=cmap, norm=norm, shading="gouraud")
ax2.plot(wavelengths, signal, color="black", lw=1.2)
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Intensity (a.u.)")
ax2.set_title("Spectrum with wavelength colormap underlay")
fig2.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax2,
    label="Wavelength (nm)",
)

plt.show()
