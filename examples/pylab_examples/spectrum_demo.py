"""
========================
Spectrum Representations
========================

The plots show different spectrum representations of a sine signal with
additive noise. A (frequency) spectrum of a discrete-time signal is calculated
by utilizing the fast Fourier transform (FFT).
"""
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)

dt = 0.01  # sampling interval
Fs = 1/dt  # sampling frequency
t = np.arange(0, 10, dt)

# generate noise:
nse = np.random.randn(len(t))
r = np.exp(-t/0.05)
cnse = np.convolve(nse, r)*dt
cnse = cnse[:len(t)]

s = 0.1*np.sin(2*np.pi*t) + cnse  # the signal

fig, axx = plt.subplots(3, 2)

# plot time signal:
axx[0, 0].plot(t, s)
axx[0, 0].set_xlabel("Time $t$")
axx[0, 0].set_ylabel("Signal $s(t)$")

# plot different spectrum types:
axx[1, 0].magnitude_spectrum(s, Fs=Fs)
axx[2, 0].phase_spectrum(s, Fs=Fs)
axx[0, 1].remove()  # don't display empty ax
axx[1, 1].magnitude_spectrum(s, Fs=Fs, scale='dB')
axx[2, 1].angle_spectrum(s, Fs=Fs)

fig.tight_layout()
plt.show()
