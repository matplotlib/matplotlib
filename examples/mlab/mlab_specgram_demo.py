"""
===========================
mlab specgram example
===========================

This example generates the spectrogram of an input signal as well as 
demonstrates how to calculate relative power 
"""

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# generating the test signal
dt = 0.01
t = np.arange(0, 30, dt)
fs = 1./dt
# white noise
nse1 = np.random.randn(len(t))
r = np.exp(-t/0.05)
# colored noise
cnse1 = np.convolve(nse1, r, mode='same')*dt   
# signal with a coherent part and a random part
s1 = 0.01*np.sin(2*np.pi*10*t) + cnse1


# calculate spectrogram
nperseg = int(fs)
[P, F, T] = mlab.specgram(s1, Fs=fs, NFFT=nperseg, window=mlab.window_hanning, 
                         noverlap=nperseg//2, mode='psd') 
# helpful in plotting in future
extent = [min(T), max(T), min(F), max(F)]
# calculate relative power in 10Hz frequency band
P_rel = np.divide(P[10],np.sum(P,0))

# plot the results 
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(t, s1, label='Raw')
ax1.set_ylabel('amp [uV]')
ax1.set_xlim([min(t), max(t)])
ax2.imshow(P,aspect='auto', extent=extent, origin='lower', cmap='Spectral_r')
ax2.set_ylabel('frequency [Hz]')
ax2.set_xlim([min(T), max(T)])
ax3.plot(T, P_rel)
ax3.set_xlabel('time [s]')
ax3.set_ylabel('relative power')
plt.tight_layout()
plt.show()