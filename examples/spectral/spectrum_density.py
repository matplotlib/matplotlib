# -*- coding: utf-8 -*-
r"""
================================
Spectrum versus Spectral density
================================

To demonstrate the difference of a spectrum and a spectral density, this
example investigates a 20 Hz sine signal with amplitude of 1 Volt
corrupted by additive white noise.
with different averaging factors. Increasing the averaging lowers
variance of spectrum (or density) as well as decreasew the frequency
resolution.

The left plot shows the spectrum scaled to amplitude (unit V) and the right
plot depicts the amplitude density (unit V :math:`/\sqrt{Hz}`) of the analytic
signal. A Hanning window with an overlap of 50% is utilized, which corresponds
to `Welch's Method <https://en.wikipedia.org/wiki/Welch's_method>`_.

Note that in the spectrum, the height of the peak at 2 Hz stays is more or less
constant over the different averaging factors. The variations are due to the
influnce of the noise. For longer signals (large `n`) the peak will converge to
its theoretical value of 1 V.
In the density, on the other hand, the noise floor has
a constant magnitude. In summary: Use a spectrum for determining height of
peaks and a density for the magnitude of the noise.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.spectral import Periodogram
plt.style.use('seaborn-darkgrid')  # less intrusive grid lines in this style

np.random.seed(2243049845)  # for reproducibility

n = 2**10  # number of samples
T = 10/n  # sampling interval (for a duration of 10 s)
t = np.arange(n) * T  # time points
f0, sig_w = 20, 2
x = 2*np.sin(2*np.pi*f0*t) + sig_w * np.random.randn(n)

PS = Periodogram(x, f_s=1/T, window='hann', unit='V')
PS.oversampling(8)

fig1, axx1 = plt.subplots(1, 2, sharex=True, num=1, clear=True)
for c, nperseg in enumerate([n//8, n//16, n//32]):
    PS.nperseg, PS.noverlap = nperseg, nperseg//2
    fa, Xa = PS.spectrum(yscale='amp')
    fd, Xd = PS.density(yscale='amp')
    axx1[0].plot(fa, Xa, alpha=.8, label=r"$%d\times$ avg." % PS.seg_num)
    axx1[1].plot(fd, Xd, alpha=.8, label=r"$%d\times$ avg." % PS.seg_num)

axx1[0].set(title="Avg. Spectrum (Analytic)", ylabel="Amplitude in V")
axx1[1].set(title="Avg. Spectral Density  (Analytic)",
            ylabel=r"Amplitude in V/$\sqrt{Hz}$")
for ax in axx1:
    ax.set_xlabel("Frequency in Hertz")
    ax.legend()
fig1.tight_layout()
plt.show()
