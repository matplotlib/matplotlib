"""
==============================
mlab psd example
==============================

This example shows how to calculate psd as well as effect of changing 
padding, block size and noverlap on the output of psd function
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# generating two signals with 8hz and 30hz frequency
dt1 = 0.005
fs1 = 1./dt1
t1 = np.arange(0.0, 5.0, dt1)
s1 = np.sin(2 * np.pi * 8 * t1)
s1 = s1 + 1e-3*np.random.normal(0,1,len(t1))

dt2 = 0.005
fs2 = 1./dt2
t2 = np.arange(0.0, 2.0, dt2)
s2 = np.sin(2 * np.pi * 30 * t2)
s2 = s2 + 1e-3*np.random.normal(0,1,len(t2))

# compute PSD for both the signals
nperseg1 = len(t1)
Psd1, freq1 = mlab.psd(s1, Fs=fs1, NFFT=nperseg1, noverlap=(nperseg1//2), 
                       detrend=None, scale_by_freq=True, window=mlab.window_hanning)

nperseg2 = len(t2)
Psd2, freq2 = mlab.psd(s2, Fs=fs2, NFFT=nperseg2, noverlap=(nperseg2//2), 
                       detrend=None, scale_by_freq=True, window=mlab.window_hanning) 


# plotting the 10hz signal
fig, (ax1,ax2) = plt.subplots(2, 1)
ax1.plot(t1,s1)
ax1.set_xlabel('time [s]')
ax1.set_ylabel('signal')
ax2.loglog(freq1, Psd1)
ax2.set_xlabel('frequency [Hz]')
ax2.set_ylabel('PSD')
ax2.grid(True)
#plt.suptitle('PSD example1')
plt.tight_layout()
plt.show()

# lets look at the other signal (30Hz)
fig, (ax1,ax2) = plt.subplots(2, 1)
ax1.plot(t2,s2)
ax1.set_xlabel('time [s]')
ax1.set_ylabel('signal')
ax2.loglog(freq2, Psd2)
ax2.set_xlabel('frequency [hz]')
ax2.set_ylabel('PSD')
ax2.grid(True)
#plt.suptitle('PSD example2')
plt.tight_layout()
plt.show()

# lets look at the effect of different paramers like zero padding, block size, 
# and overlap (this is the same as examples from plt.psd functions)
fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t1, s1)
ax1.set_xlabel('time [s]')
ax1.set_ylabel('signal')
ax1.set_title('Raw signal')
# look at psd with varying the amount of padding
ax2 = fig.add_subplot(gs[1, 0])
psd_1, f_1 = mlab.psd(s1, NFFT=nperseg1, pad_to=nperseg1, Fs=fs1)
psd_2, f_2 = mlab.psd(s1, NFFT=nperseg1, pad_to=nperseg1*2, Fs=fs1)
psd_3, f_3 = mlab.psd(s1, NFFT=nperseg1, pad_to=nperseg1*4, Fs=fs1)
ax2.loglog(f_1, psd_1)
ax2.loglog(f_2, psd_2)
ax2.loglog(f_3, psd_3)
ax2.set_title('zero padding')
# look at psd with different block sizes
ax3 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax2)
psd_4, f_4 = mlab.psd(s1, NFFT=nperseg1, pad_to=nperseg1, Fs=fs1)
psd_5, f_5 = mlab.psd(s1, NFFT=nperseg1//2, pad_to=nperseg1, Fs=fs1)
psd_6, f_6 = mlab.psd(s1, NFFT=nperseg1//4, pad_to=nperseg1, Fs=fs1)
ax3.loglog(f_4, psd_4)
ax3.loglog(f_5, psd_5)
ax3.loglog(f_6, psd_6)
ax3.set_title('block size')
# Plot the PSD with different amounts of overlap between blocks
ax4 = fig.add_subplot(gs[1, 2], sharex=ax2, sharey=ax2)
psd_7, f_7 = mlab.psd(s1, NFFT=nperseg1//2, pad_to=nperseg1, 
        noverlap=0, Fs=fs1)
psd_8, f_8 = mlab.psd(s1, NFFT=nperseg1//2, pad_to=nperseg1,
        noverlap=int(0.05 * nperseg1/2.), Fs=fs1)
psd_9, f_9 = mlab.psd(s1, NFFT=nperseg1//2, pad_to=nperseg1,
        noverlap=int(0.2 * nperseg1/2.), Fs=fs1)
ax4.loglog(f_7, psd_7)
ax4.loglog(f_8, psd_8)
ax4.loglog(f_9, psd_9)
ax4.set_title('overlap')
plt.show()