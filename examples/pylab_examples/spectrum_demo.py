#!/usr/bin/env python
# python

from pylab import *

dt = 0.01
Fs = 1/dt
t = arange(0, 10, dt)
nse = randn(len(t))
r = exp(-t/0.05)

cnse = convolve(nse, r)*dt
cnse = cnse[:len(t)]
s = 0.1*sin(2*pi*t) + cnse

subplot(3, 2, 1)
plot(t, s)

subplot(3, 2, 3)
magnitude_spectrum(s, Fs=Fs)

subplot(3, 2, 4)
magnitude_spectrum(s, Fs=Fs, scale='dB')

subplot(3, 2, 5)
angle_spectrum(s, Fs=Fs)

subplot(3, 2, 6)
phase_spectrum(s, Fs=Fs)

show()
