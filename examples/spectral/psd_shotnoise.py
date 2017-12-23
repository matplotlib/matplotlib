# -*- coding: utf-8 -*-
r"""
=================================================
Power Spectral Density (PSD) using Welch's Method
=================================================

When investigating noisy signals, Periodograms utilizing
`Welch's Method <https://en.wikipedia.org/wiki/Welch's_method>`_
(i.e., Hann Window and 50% overlap) are useful.

This example shows a signal with white noise dominating above 1 KHz and
`flickr noise <https://en.wikipedia.org/wiki/Flicker_noise>`_  (or 1/f noise)
dominating below 1 KHz. This kind of noise is typical for analog electrial
circuits.

The plot has double logarithmic scaling with the left y-scale depicting the
relative power spectral density and the right one showing the relative spectral
power.

Checkout the :ref:`sphx_glr_gallery_spectral_spectrum_density.py` example for
the influence of the avaraging on the result.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.spectral import Periodogram

np.random.seed(1648142464)  # for reproducibility

n = 2**13  # number of samples
T = 1e-5  # sampling interval
t = np.arange(n) * T  # time points
f0, sig_w0, sig_w1 = 5e3, .02, 10
# generate shot nose via low path filtering white noise:
fW = np.fft.rfftfreq(n, T)
W1 = np.random.randn(len(fW)) * sig_w1/2
W1[0] = 0
W1[1:] = 1/np.sqrt(fW[1:])
w1 = np.fft.irfft(W1) * n
# the signal:
x = np.sin(2*np.pi*f0*t) + sig_w0 * np.random.randn(n) + w1


PS = Periodogram(x, f_s=1/T, window='hann', nperseg=n//8, noverlap=n//16,
                 detrend='mean', unit='V')
fig1, axx1 = PS.plot(density=True, yscale='db', xscale='log', fig_num=1)
axx1.set_xlim(PS.f[1], PS.f_s/2)

plt.show()
