#!/usr/bin/env python
from matplotlib.matlab import *


dt = 0.0005
t = arange(0.0, 20.0, dt)
s1 = sin(2*pi*100*t)
s2 = 2*sin(2*pi*400*t)
mask = where(logical_and(t>10, t<12), 1.0, 0.0)
s2 = s2 * mask
nse = 0.01*randn(len(t))
x = s1 + s2 + nse
NFFT = 1024
Fs = int(1.0/dt)
Noverlap = 0
Pxx, freqs, bins, im = specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)
gca().set_xticks(arange(0,21,5))
colorbar()
show()
