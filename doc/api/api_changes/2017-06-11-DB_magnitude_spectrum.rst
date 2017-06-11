Correct scaling of :func:`magnitude_spectrum()`
```````````````````````````````````````````````

The functions :func:`matplotlib.mlab.magnitude_spectrum()` and :func:`matplotlib.pyplot.magnitude_spectrum()` implicitly assumed the sum
of windowing function values to be one. In Matplotlib and Numpy the
standard windowing functions are scaled to have maximum value of one,
which usually results in a sum of the order of n/2 for a n-point
signal. Thus the amplitude scaling :func:`magnitude_spectrum()` was
off by that amount when using standard windowing functions (`Bug 8417
<https://github.com/matplotlib/matplotlib/issues/8417>`_ ). Now the
behavior is consistent with :func:`matplotlib.pyplot.psd()` and
:func:`scipy.signal.welch()`. The following example demonstrates the
new and old scaling::

    import matplotlib.pyplot as plt
    import numpy as np
    
    tau, n = 10, 1024  # 10 second signal with 1024 points
    T = tau/n  # sampling interval
    t = np.arange(n)*T
    
    a = 4  # amplitude
    x = a*np.sin(40*np.pi*t)  # 20 Hz sine with amplitude a
    
    # New correct behavior: Amplitude at 20 Hz is a/2
    plt.magnitude_spectrum(x, Fs=1/T, sides='onesided', scale='linear')
    
    # Original behavior: Amplitude at 20 Hz is (a/2)*(n/2) for a Hanning window
    w = np.hanning(n)  # default window is a Hanning window
    plt.magnitude_spectrum(x*np.sum(w), Fs=1/T, sides='onesided', scale='linear')

