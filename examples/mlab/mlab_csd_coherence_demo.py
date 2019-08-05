"""
==============================
mlab csd and coherence example
==============================

This example shows how to calculate csd and coherence as well as 
phase shifts between two signals
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# generating two signals
dt = 0.01
t = np.arange(0, 30, dt)
fs = 1./dt
# white noise
nse1 = np.random.randn(len(t))                 
nse2 = np.random.randn(len(t))
r = np.exp(-t/0.05)
# colored noise
cnse1 = np.convolve(nse1, r, mode='same')*dt   
cnse2 = np.convolve(nse2, r, mode='same')*dt   
# two signals with a coherent part and a random part
s1 = 0.01*np.sin(2*np.pi*10*t) + cnse1
s2 = 0.01*np.sin(2*np.pi*10*t) + cnse2


# lets plot the psd, csd, coherence, phase shift bw the two signals
fig = plt.figure(constrained_layout=True, figsize=(8,6))
gs = gridspec.GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, s1, label='signal1')
ax1.plot(t, s2, label='signal2')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('amplitude')


# compute the psd for both the signals
nperseg = len(t)
Psd1, freq1 = mlab.psd(s1, Fs=fs, NFFT=nperseg, noverlap=(nperseg//2), 
                       detrend=None, scale_by_freq=True, window=mlab.window_hanning)
Psd2, freq2 = mlab.psd(s2, Fs=fs, NFFT=nperseg, noverlap=(nperseg//2), 
                       detrend=None, scale_by_freq=True, window=mlab.window_hanning) 
# plot the psd (notice the peak at 10Hz)
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(freq1, Psd1)
ax2.plot(freq2, Psd2)
ax2.set_xlabel('frequency [Hz]')
ax2.set_ylabel('PSD')
ax2.grid(True)


# compute the csd for both signals
nperseg = 1024
csd1, fcsd = mlab.csd(s1, s2, nperseg, fs) 
# plot the csd
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(fcsd, csd1)
ax3.set_xlabel('frequency [Hz]')
ax3.set_ylabel('CSD')
ax3.grid(True)

# calculate the coherence for both signals
nperseg = 1024
coh, fcoh = mlab.cohere(s1, s2, 256, fs, noverlap=128)
# plot the coherence
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(fcoh, coh)
ax4.set_xlabel('frequency [Hz]')
ax4.set_ylabel('Coherence')
ax4.grid(True)

# calculate the phase shift zommed in on 10Hz
angle = np.angle(csd1, deg=True)
angle[angle<-90] += 360
# plot the phase shift
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(fcsd, angle)
ax5.set_xlabel('frequency [Hz]')
ax5.set_ylabel('Phase Shift')
ax5.set_xlim([8,12])
ax5.grid(True)

plt.tight_layout()