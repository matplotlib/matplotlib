"""
================
Inverse Log Axis
================

This is an example of assigning an inverse log-scale for the x-axis using
`InvLogLocator` and `InvLogFormatter`. This can be useful when doing Fourier
analysis, especially when scales of Fourier modes are of interest.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

fig, (axsin, axfft) = plt.subplots(2, 1)

# sampling interval
dx = 0.01

# We measure the height of a sinus-shaped curve along 10 units of x-axis.
# The sinus period is 2 units of x-axis.
period = 2
x = np.arange(0, 10, dx)
height = np.sin(2 * np.pi * x / period)

axsin.plot(x, height)
axsin.set_xlabel(r'distance')

height_fourier = np.fft.rfft(height)
fourier_amplitude = np.abs(height_fourier)
spatial_frequencies = 2 * np.pi * np.fft.rfftfreq(len(height), d=dx)

axfft.set_xscale('log')
axfft.plot(spatial_frequencies, fourier_amplitude)
axfft.set_xlabel(r'spatial frequency')

# add scale axis
xlim = axfft.get_xlim()

ax2 = axfft.twiny()
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)

ax2.set_xscale('log')
ax2.xaxis.set_major_locator(ticker.InvLogLocator(inv_factor=2*np.pi))
ax2.xaxis.set_minor_locator(ticker.InvLogLocator(inv_factor=2*np.pi,
                                                 subs='auto'))
ax2.xaxis.set_major_formatter(ticker.InvLogFormatter(inv_factor=2*np.pi))
ax2.xaxis.set_minor_formatter(ticker.InvLogFormatter(inv_factor=2*np.pi,
                                                     labelOnlyBase=False))

ax2.set_xlim(xlim)
ax2.set_xlabel(r'scale')

plt.tight_layout()

plt.show()
