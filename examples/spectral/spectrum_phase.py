# -*- coding: utf-8 -*-
r"""
============================
Amplitude and Phase Spectrum
============================

The signal :math:`x(t)` of a damped
`harmonic oscillator <https://en.wikipedia.org/wiki/Harmonic_oscillator>`_ with
frequency :math:`f_0=10` Hz, (initial) amplitude :math:`a=10` Volt (V), and a
damping ratio of :math:`D=.01` can be written as

.. math::

   x(t) = a\, e^{-2\pi f_0 D t}\, \cos(2 \pi \sqrt{1-D^2} f_0 t)\ .

The first plot shows a two-sided spectrum with the scaling set to amplitude
(`yscale='amp'`), meaning there's a peak at :math:`\pm f_0`.
The phase spectrum in the second plot shows that the phase is rising
proportionally with the frequency except around :math:`\pm f_0`,
where the phase shifts by almost -180°. Would :math:`D` be zero, then the phase
shift would be exactly -180°.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.spectral import Spectrum
# plt.style.use('seaborn-darkgrid')

n = 2**9  # number of samples
T = 10/n  # sampling interval (for a duration of 10 s)
t = np.arange(n) * T  # time points
a, w0, D = 10, 2*np.pi*10, .01  # Amplitude, angular frequency, damping ratio

x = a * np.exp(-D * w0 * t) * np.cos(np.sqrt(1-D**2) * w0 * t)

SP = Spectrum(x, f_s=1/T, unit='V')
fig1, ax1 = SP.plot(yscale='amp', fft_repr='twosided', right_yaxis=True,
                    fig_num=1)
fig2, ax2 = SP.plot_phase(yunit='deg', fft_repr='twosided',
                          plt_kw=dict(color='C1'), fig_num=2)

plt.show()
