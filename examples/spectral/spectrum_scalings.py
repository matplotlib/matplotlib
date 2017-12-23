# -*- coding: utf-8 -*-
r"""
======================
Scalings of a Spectrum
======================

Different scalings of a sampled cosine signal :math:`x(t)` with frequency
:math:`f_0 = 2` Hz and amplitude :math:`a = 2` V (Volt), i.e.,

.. math::

   x(t) = a\, \cos(2 \pi f_0 t)
        = a\, \frac{e^{i 2 \pi f_0 t} + e^{-i 2 \pi f_0 t}}{2}

are presented in this example.
There are three common representations of a singled-sided Spectrum or
FFT of a real signal (property `fft_repr`):

1. Don't scale the amplitudes ('single').
2. Scale the amplitude by a factor of :math:`\sqrt{2}` representing the power
   of the signal ('rms', i.e., root-mean-square).
3. Scale by a factor of :math:`2` for representing the amplitude of the
   signal ('analytic').

Furthermore, there are three common scalings of the :math:`y`-axis (`yscale`):

a. Unscaled magnitude, revealing the amplitude :math:`a` ('amp').
b. Squared magnitude, which is proportional to the signal's power ('power').
c. Relative power scaled logarithmically in Decibel. The notation dB(1V²) means
   :math:`y = 10\, \log_{10}(P_y) - 10\, \log_{10}(1V^2)` ('dB').

The first plot shows the combination of different single-sided FFT. The height
of the peak is denoted in the plot's upper right corner.
The second plot shows the two-sided amplitude spectrum,  with the scaling being
set to amplitude (`yscale='amp'`) meaning there's a frequency component of
:math:`a/2` at :math:`\pm f_0`.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.spectral import Spectrum
plt.style.use('seaborn-darkgrid')   # grid is less intrusive with this style

n = 2**7  # number of samples
T = 10/n  # sampling interval (for a duration of 10 s)
t = np.arange(n) * T  # time points
a, f0 = 2, 2
x = a*np.cos(2*np.pi*f0*t)  # the signal

SP = Spectrum(x, f_s=1/T, unit='V')


fig1, axx1 = plt.subplots(3, 3, sharex=True, num=1, figsize=(10, 5.5),
                          clear=True)
# Legend labels:
lb = [(r"$|a|/2$",   r"$|a|/\sqrt{2}$", "$|a|$"),
      (r"$|a|^2/4$", r"$|a|^2/2$",      "$|a|^2$")]
lb.append([r"$10\,\log_{10}(%s)$" % s[1:-1] for s in lb[1]])

for p, yscale in enumerate(('amp', 'power', 'dB')):
    for q, side in enumerate(('onesided', 'rms', 'analytic')):
        ax_ = SP.plot(yscale=yscale, fft_repr=side, right_yaxis=False,
                      plt_kw=dict(color=f'C{q}'), ax=axx1[p, q])
        if q != 1 or p != 2:  # only one xlabel
            ax_.set_xlabel("")
        ax_.text(.98, .9, lb[p][q], horizontalalignment='right',
                 verticalalignment='top', transform=ax_.transAxes)
axx1[0, 0].set_xlim(0, SP.f_s/2)
for p in range(3):
    axx1[0, p].set_ylim(-0.1, 2.2)
    axx1[1, p].set_ylim(-0.2, 4.4)
fig1.tight_layout()


fig2, ax2 = SP.plot(yscale='amp', fft_repr='twosided', right_yaxis=False,
                    fig_num=2)
ax2.set_xlim(-SP.f_s/2, +SP.f_s/2)
ax2.text(.98, .95, lb[0][0], horizontalalignment='right',
         verticalalignment='top', transform=ax2.transAxes)

plt.show()
