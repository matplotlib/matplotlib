# -*- coding: utf-8 -*-
r"""
===========================
Spectra of Window Functions
===========================

Windowing allows trading frequency resolution for better suppression of side
lobes. This can be illustrated by looking at the spectrum of single frequency
signal with amplitude one and length of one second, i. e.,

.. math::

    z(t) = \exp(i 2 \pi f_0 t)

In the plot the frequency :math:`f_0` has been shifted to zero by setting
the center frequency `f_c=f_0`. It shows the spectra of the standard window
functions, where :math:`\Delta f_r` denotes the distance from the maximum
to the first minimum. The high oversampling (`nfft=1024`) improves the
resolution from :math:`\Delta f=1` Hz to :math:`\Delta f=7.8` mHz (property
`df`). The left plot has amplitude scaling, the right depicts a dB-scale. Note
that with dB scaling, the minimas are not depicted correctly due to the limited
resolution in `f`.

The plot shows that the unwindowed spectrum (`window='rect'`) has the
thinnest main lobe, but the highest side lobes. The Blackman window has the
best side lobe suppression at the cost of a third of the frequency resolution.
Frequently the Hann window is used, because it is a good compromise between the
width of the main lobe and the suppression of the higher order side lobes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.spectral import Spectrum, WINDOWS

plt.style.use('seaborn-darkgrid')  # grid is less intrusive with this style

n, tau = 2**3, 1
T, f0 = tau / n, 2
t = T * np.arange(n)
z = np.exp(2j*np.pi*f0*t)  # the signal

SP = Spectrum(z, 1/T, nfft=2**10, fft_repr='twosided', f_c=f0)

fig, axx = plt.subplots(1, 2, sharex=True)
axx[0].set(title="Amplitude Spectra", ylabel="Normalized Amplitude",
           xlabel="Normalized Frequency", xlim=(-SP.f_s/2, +SP.f_s/2))
axx[1].set(title="Rel. Power Spectra", xlabel="Normalized Frequency",
           ylabel="Relative Power in dB(1)")
wins = [k for k in WINDOWS if k != 'user']  # extract valid window names
for w in wins:
    f, X_amp = SP.spectrum(yscale='amp', window=w)
    _, X_db = SP.spectrum(yscale='db')
    lb_str = w.title() + r", $\Delta f_r=%g$" % SP.f_res
    axx[0].plot(f, X_amp, label=lb_str, alpha=.7)
    axx[1].plot(f, X_db, label=lb_str, alpha=.7)
axx[1].set_ylim(-120, 5)
axx[1].legend(loc='best', framealpha=1, frameon=True, fancybox=True)
fig.tight_layout()
