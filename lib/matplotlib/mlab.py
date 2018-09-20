"""
Numerical python functions written for compatibility with MATLAB
commands with the same names.

MATLAB compatible functions
---------------------------

:func:`cohere`
    Coherence (normalized cross spectral density)

:func:`csd`
    Cross spectral density using Welch's average periodogram

:func:`detrend`
    Remove the mean or best fit line from an array

:func:`psd`
    Power spectral density using Welch's average periodogram

:func:`specgram`
    Spectrogram (spectrum over segments of time)

Miscellaneous functions
-----------------------

Functions that don't exist in MATLAB, but are useful anyway:

:func:`complex_spectrum`
    Return the complex-valued frequency spectrum of a signal

:func:`magnitude_spectrum`
    Return the magnitude of the frequency spectrum of a signal

:func:`angle_spectrum`
    Return the angle (wrapped phase) of the frequency spectrum of a signal

:func:`phase_spectrum`
    Return the phase (unwrapped angle) of the frequency spectrum of a signal

:func:`detrend_mean`
    Remove the mean from a line.

:func:`detrend_linear`
    Remove the best fit line from a line.

:func:`detrend_none`
    Return the original line.

:func:`stride_windows`
    Get all windows in an array in a memory-efficient manner

:func:`stride_repeat`
    Repeat an array in a memory-efficient manner

:func:`apply_window`
    Apply a window along a given axis
"""

import copy
import csv
import operator
import os
import warnings

import numpy as np

import matplotlib.cbook as cbook
from matplotlib import docstring
from matplotlib.path import Path
import math


def window_hanning(x):
    '''
    Return x times the hanning window of len(x).

    See Also
    --------
    :func:`window_none`
        :func:`window_none` is another window algorithm.
    '''
    return np.hanning(len(x))*x


def window_none(x):
    '''
    No window function; simply return x.

    See Also
    --------
    :func:`window_hanning`
        :func:`window_hanning` is another window algorithm.
    '''
    return x


def apply_window(x, window, axis=0, return_window=None):
    '''
    Apply the given window to the given 1D or 2D array along the given axis.

    Parameters
    ----------
    x : 1D or 2D array or sequence
        Array or sequence containing the data.

    window : function or array.
        Either a function to generate a window or an array with length
        *x*.shape[*axis*]

    axis : integer
        The axis over which to do the repetition.
        Must be 0 or 1.  The default is 0

    return_window : bool
        If true, also return the 1D values of the window that was applied
    '''
    x = np.asarray(x)

    if x.ndim < 1 or x.ndim > 2:
        raise ValueError('only 1D or 2D arrays can be used')
    if axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    xshape = list(x.shape)
    xshapetarg = xshape.pop(axis)

    if np.iterable(window):
        if len(window) != xshapetarg:
            raise ValueError('The len(window) must be the same as the shape '
                             'of x for the chosen axis')
        windowVals = window
    else:
        windowVals = window(np.ones(xshapetarg, dtype=x.dtype))

    if x.ndim == 1:
        if return_window:
            return windowVals * x, windowVals
        else:
            return windowVals * x

    xshapeother = xshape.pop()

    otheraxis = (axis+1) % 2

    windowValsRep = stride_repeat(windowVals, xshapeother, axis=otheraxis)

    if return_window:
        return windowValsRep * x, windowVals
    else:
        return windowValsRep * x


def detrend(x, key=None, axis=None):
    '''
    Return x with its trend removed.

    Parameters
    ----------
    x : array or sequence
        Array or sequence containing the data.

    key : [ 'default' | 'constant' | 'mean' | 'linear' | 'none'] or function
        Specifies the detrend algorithm to use. 'default' is 'mean', which is
        the same as :func:`detrend_mean`. 'constant' is the same. 'linear' is
        the same as :func:`detrend_linear`. 'none' is the same as
        :func:`detrend_none`. The default is 'mean'. See the corresponding
        functions for more details regarding the algorithms. Can also be a
        function that carries out the detrend operation.

    axis : integer
        The axis along which to do the detrending.

    See Also
    --------
    :func:`detrend_mean`
        :func:`detrend_mean` implements the 'mean' algorithm.

    :func:`detrend_linear`
        :func:`detrend_linear` implements the 'linear' algorithm.

    :func:`detrend_none`
        :func:`detrend_none` implements the 'none' algorithm.
    '''
    if key is None or key in ['constant', 'mean', 'default']:
        return detrend(x, key=detrend_mean, axis=axis)
    elif key == 'linear':
        return detrend(x, key=detrend_linear, axis=axis)
    elif key == 'none':
        return detrend(x, key=detrend_none, axis=axis)
    elif isinstance(key, str):
        raise ValueError("Unknown value for key %s, must be one of: "
                         "'default', 'constant', 'mean', "
                         "'linear', or a function" % key)

    if not callable(key):
        raise ValueError("Unknown value for key %s, must be one of: "
                         "'default', 'constant', 'mean', "
                         "'linear', or a function" % key)

    x = np.asarray(x)

    if axis is not None and axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    if (axis is None and x.ndim == 0) or (not axis and x.ndim == 1):
        return key(x)

    # try to use the 'axis' argument if the function supports it,
    # otherwise use apply_along_axis to do it
    try:
        return key(x, axis=axis)
    except TypeError:
        return np.apply_along_axis(key, axis=axis, arr=x)


@cbook.deprecated("3.1", alternative="detrend_mean")
def demean(x, axis=0):
    '''
    Return x minus its mean along the specified axis.

    Parameters
    ----------
    x : array or sequence
        Array or sequence containing the data
        Can have any dimensionality

    axis : integer
        The axis along which to take the mean.  See numpy.mean for a
        description of this argument.

    See Also
    --------
    :func:`detrend_mean`
        This function is the same as :func:`detrend_mean` except for the
        default *axis*.
    '''
    return detrend_mean(x, axis=axis)


def detrend_mean(x, axis=None):
    '''
    Return x minus the mean(x).

    Parameters
    ----------
    x : array or sequence
        Array or sequence containing the data
        Can have any dimensionality

    axis : integer
        The axis along which to take the mean.  See numpy.mean for a
        description of this argument.

    See Also
    --------
    :func:`demean`
        This function is the same as :func:`demean` except for the default
        *axis*.

    :func:`detrend_linear`

    :func:`detrend_none`
        :func:`detrend_linear` and :func:`detrend_none` are other detrend
        algorithms.

    :func:`detrend`
        :func:`detrend` is a wrapper around all the detrend algorithms.
    '''
    x = np.asarray(x)

    if axis is not None and axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    return x - x.mean(axis, keepdims=True)


def detrend_none(x, axis=None):
    '''
    Return x: no detrending.

    Parameters
    ----------
    x : any object
        An object containing the data

    axis : integer
        This parameter is ignored.
        It is included for compatibility with detrend_mean

    See Also
    --------
    :func:`detrend_mean`

    :func:`detrend_linear`
        :func:`detrend_mean` and :func:`detrend_linear` are other detrend
        algorithms.

    :func:`detrend`
        :func:`detrend` is a wrapper around all the detrend algorithms.
    '''
    return x


def detrend_linear(y):
    '''
    Return x minus best fit line; 'linear' detrending.

    Parameters
    ----------
    y : 0-D or 1-D array or sequence
        Array or sequence containing the data

    axis : integer
        The axis along which to take the mean.  See numpy.mean for a
        description of this argument.

    See Also
    --------
    :func:`detrend_mean`

    :func:`detrend_none`
        :func:`detrend_mean` and :func:`detrend_none` are other detrend
        algorithms.

    :func:`detrend`
        :func:`detrend` is a wrapper around all the detrend algorithms.
    '''
    # This is faster than an algorithm based on linalg.lstsq.
    y = np.asarray(y)

    if y.ndim > 1:
        raise ValueError('y cannot have ndim > 1')

    # short-circuit 0-D array.
    if not y.ndim:
        return np.array(0., dtype=y.dtype)

    x = np.arange(y.size, dtype=float)

    C = np.cov(x, y, bias=1)
    b = C[0, 1]/C[0, 0]

    a = y.mean() - b*x.mean()
    return y - (b*x + a)


def stride_windows(x, n, noverlap=None, axis=0):
    '''
    Get all windows of x with length n as a single array,
    using strides to avoid data duplication.

    .. warning::

        It is not safe to write to the output array.  Multiple
        elements may point to the same piece of memory,
        so modifying one value may change others.

    Parameters
    ----------
    x : 1D array or sequence
        Array or sequence containing the data.

    n : integer
        The number of data points in each window.

    noverlap : integer
        The overlap between adjacent windows.
        Default is 0 (no overlap)

    axis : integer
        The axis along which the windows will run.

    References
    ----------
    `stackoverflow: Rolling window for 1D arrays in Numpy?
    <http://stackoverflow.com/a/6811241>`_
    `stackoverflow: Using strides for an efficient moving average filter
    <http://stackoverflow.com/a/4947453>`_
    '''
    if noverlap is None:
        noverlap = 0

    if noverlap >= n:
        raise ValueError('noverlap must be less than n')
    if n < 1:
        raise ValueError('n cannot be less than 1')

    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError('only 1-dimensional arrays can be used')
    if n == 1 and noverlap == 0:
        if axis == 0:
            return x[np.newaxis]
        else:
            return x[np.newaxis].transpose()
    if n > x.size:
        raise ValueError('n cannot be greater than the length of x')

    # np.lib.stride_tricks.as_strided easily leads to memory corruption for
    # non integer shape and strides, i.e. noverlap or n. See #3845.
    noverlap = int(noverlap)
    n = int(n)

    step = n - noverlap
    if axis == 0:
        shape = (n, (x.shape[-1]-noverlap)//step)
        strides = (x.strides[0], step*x.strides[0])
    else:
        shape = ((x.shape[-1]-noverlap)//step, n)
        strides = (step*x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def stride_repeat(x, n, axis=0):
    '''
    Repeat the values in an array in a memory-efficient manner.  Array x is
    stacked vertically n times.

    .. warning::

        It is not safe to write to the output array.  Multiple
        elements may point to the same piece of memory, so
        modifying one value may change others.

    Parameters
    ----------
    x : 1D array or sequence
        Array or sequence containing the data.

    n : integer
        The number of time to repeat the array.

    axis : integer
        The axis along which the data will run.

    References
    ----------
    `stackoverflow: Repeat NumPy array without replicating data?
    <http://stackoverflow.com/a/5568169>`_
    '''
    if axis not in [0, 1]:
        raise ValueError('axis must be 0 or 1')
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('only 1-dimensional arrays can be used')

    if n == 1:
        if axis == 0:
            return np.atleast_2d(x)
        else:
            return np.atleast_2d(x).T
    if n < 1:
        raise ValueError('n cannot be less than 1')

    # np.lib.stride_tricks.as_strided easily leads to memory corruption for
    # non integer shape and strides, i.e. n. See #3845.
    n = int(n)

    if axis == 0:
        shape = (n, x.size)
        strides = (0, x.strides[0])
    else:
        shape = (x.size, n)
        strides = (x.strides[0], 0)

    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _spectral_helper(x, y=None, NFFT=None, Fs=None, detrend_func=None,
                     window=None, noverlap=None, pad_to=None,
                     sides=None, scale_by_freq=None, mode=None):
    '''
    This is a helper function that implements the commonality between the
    psd, csd, spectrogram and complex, magnitude, angle, and phase spectrums.
    It is *NOT* meant to be used outside of mlab and may change at any time.
    '''
    if y is None:
        # if y is None use x for y
        same_data = True
    else:
        # The checks for if y is x are so that we can use the same function to
        # implement the core of psd(), csd(), and spectrogram() without doing
        # extra calculations.  We return the unaveraged Pxy, freqs, and t.
        same_data = y is x

    if Fs is None:
        Fs = 2
    if noverlap is None:
        noverlap = 0
    if detrend_func is None:
        detrend_func = detrend_none
    if window is None:
        window = window_hanning

    # if NFFT is set to None use the whole signal
    if NFFT is None:
        NFFT = 256

    if mode is None or mode == 'default':
        mode = 'psd'
    elif mode not in ['psd', 'complex', 'magnitude', 'angle', 'phase']:
        raise ValueError("Unknown value for mode %s, must be one of: "
                         "'default', 'psd', 'complex', "
                         "'magnitude', 'angle', 'phase'" % mode)

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is not 'psd'")

    # Make sure we're dealing with a numpy array. If y and x were the same
    # object to start with, keep them that way
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)

    if sides is None or sides == 'default':
        if np.iscomplexobj(x):
            sides = 'twosided'
        else:
            sides = 'onesided'
    elif sides not in ['onesided', 'twosided']:
        raise ValueError("Unknown value for sides %s, must be one of: "
                         "'default', 'onesided', or 'twosided'" % sides)

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x) < NFFT:
        n = len(x)
        x = np.resize(x, (NFFT,))
        x[n:] = 0

    if not same_data and len(y) < NFFT:
        n = len(y)
        y = np.resize(y, (NFFT,))
        y[n:] = 0

    if pad_to is None:
        pad_to = NFFT

    if mode != 'psd':
        scale_by_freq = False
    elif scale_by_freq is None:
        scale_by_freq = True

    # For real x, ignore the negative frequencies unless told otherwise
    if sides == 'twosided':
        numFreqs = pad_to
        if pad_to % 2:
            freqcenter = (pad_to - 1)//2 + 1
        else:
            freqcenter = pad_to//2
        scaling_factor = 1.
    elif sides == 'onesided':
        if pad_to % 2:
            numFreqs = (pad_to + 1)//2
        else:
            numFreqs = pad_to//2 + 1
        scaling_factor = 2.

    result = stride_windows(x, NFFT, noverlap, axis=0)
    result = detrend(result, detrend_func, axis=0)
    result, windowVals = apply_window(result, window, axis=0,
                                      return_window=True)
    result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :]
    freqs = np.fft.fftfreq(pad_to, 1/Fs)[:numFreqs]

    if not same_data:
        # if same_data is False, mode must be 'psd'
        resultY = stride_windows(y, NFFT, noverlap)
        resultY = detrend(resultY, detrend_func, axis=0)
        resultY = apply_window(resultY, window, axis=0)
        resultY = np.fft.fft(resultY, n=pad_to, axis=0)[:numFreqs, :]
        result = np.conj(result) * resultY
    elif mode == 'psd':
        result = np.conj(result) * result
    elif mode == 'magnitude':
        result = np.abs(result) / np.abs(windowVals).sum()
    elif mode == 'angle' or mode == 'phase':
        # we unwrap the phase later to handle the onesided vs. twosided case
        result = np.angle(result)
    elif mode == 'complex':
        result /= np.abs(windowVals).sum()

    if mode == 'psd':

        # Also include scaling factors for one-sided densities and dividing by
        # the sampling frequency, if desired. Scale everything, except the DC
        # component and the NFFT/2 component:

        # if we have a even number of frequencies, don't scale NFFT/2
        if not NFFT % 2:
            slc = slice(1, -1, None)
        # if we have an odd number, just don't scale DC
        else:
            slc = slice(1, None, None)

        result[slc] *= scaling_factor

        # MATLAB divides by the sampling frequency so that density function
        # has units of dB/Hz and can be integrated by the plotted frequency
        # values. Perform the same scaling here.
        if scale_by_freq:
            result /= Fs
            # Scale the spectrum by the norm of the window to compensate for
            # windowing loss; see Bendat & Piersol Sec 11.5.2.
            result /= (np.abs(windowVals)**2).sum()
        else:
            # In this case, preserve power in the segment, not amplitude
            result /= np.abs(windowVals).sum()**2

    t = np.arange(NFFT/2, len(x) - NFFT/2 + 1, NFFT - noverlap)/Fs

    if sides == 'twosided':
        # center the frequency range at zero
        freqs = np.concatenate((freqs[freqcenter:], freqs[:freqcenter]))
        result = np.concatenate((result[freqcenter:, :],
                                 result[:freqcenter, :]), 0)
    elif not pad_to % 2:
        # get the last value correctly, it is negative otherwise
        freqs[-1] *= -1

    # we unwrap the phase here to handle the onesided vs. twosided case
    if mode == 'phase':
        result = np.unwrap(result, axis=0)

    return result, freqs, t


def _single_spectrum_helper(x, mode, Fs=None, window=None, pad_to=None,
                            sides=None):
    '''
    This is a helper function that implements the commonality between the
    complex, magnitude, angle, and phase spectrums.
    It is *NOT* meant to be used outside of mlab and may change at any time.
    '''
    if mode is None or mode == 'psd' or mode == 'default':
        raise ValueError('_single_spectrum_helper does not work with %s mode'
                         % mode)

    if pad_to is None:
        pad_to = len(x)

    spec, freqs, _ = _spectral_helper(x=x, y=None, NFFT=len(x), Fs=Fs,
                                      detrend_func=detrend_none, window=window,
                                      noverlap=0, pad_to=pad_to,
                                      sides=sides,
                                      scale_by_freq=False,
                                      mode=mode)
    if mode != 'complex':
        spec = spec.real

    if spec.ndim == 2 and spec.shape[1] == 1:
        spec = spec[:, 0]

    return spec, freqs


# Split out these keyword docs so that they can be used elsewhere
docstring.interpd.update(Spectral=cbook.dedent("""
    Fs : scalar
        The sampling frequency (samples per time unit).  It is used
        to calculate the Fourier frequencies, freqs, in cycles per time
        unit. The default value is 2.

    window : callable or ndarray
        A function or a vector of length *NFFT*. To create window
        vectors see :func:`window_hanning`, :func:`window_none`,
        :func:`numpy.blackman`, :func:`numpy.hamming`,
        :func:`numpy.bartlett`, :func:`scipy.signal`,
        :func:`scipy.signal.get_window`, etc. The default is
        :func:`window_hanning`.  If a function is passed as the
        argument, it must take a data segment as an argument and
        return the windowed version of the segment.

    sides : {'default', 'onesided', 'twosided'}
        Specifies which sides of the spectrum to return.  Default gives the
        default behavior, which returns one-sided for real data and both
        for complex data.  'onesided' forces the return of a one-sided
        spectrum, while 'twosided' forces two-sided.
"""))


docstring.interpd.update(Single_Spectrum=cbook.dedent("""
    pad_to : int
        The number of points to which the data segment is padded when
        performing the FFT.  While not increasing the actual resolution of
        the spectrum (the minimum distance between resolvable peaks),
        this can give more points in the plot, allowing for more
        detail. This corresponds to the *n* parameter in the call to fft().
        The default is None, which sets *pad_to* equal to the length of the
        input signal (i.e. no padding).
"""))


docstring.interpd.update(PSD=cbook.dedent("""
    pad_to : int
        The number of points to which the data segment is padded when
        performing the FFT.  This can be different from *NFFT*, which
        specifies the number of data points used.  While not increasing
        the actual resolution of the spectrum (the minimum distance between
        resolvable peaks), this can give more points in the plot,
        allowing for more detail. This corresponds to the *n* parameter
        in the call to fft(). The default is None, which sets *pad_to*
        equal to *NFFT*

    NFFT : int
        The number of data points used in each block for the FFT.
        A power 2 is most efficient.  The default value is 256.
        This should *NOT* be used to get zero padding, or the scaling of the
        result will be incorrect. Use *pad_to* for this instead.

    detrend : {'default', 'constant', 'mean', 'linear', 'none'} or callable
        The function applied to each segment before fft-ing,
        designed to remove the mean or linear trend.  Unlike in
        MATLAB, where the *detrend* parameter is a vector, in
        matplotlib is it a function.  The :mod:`~matplotlib.mlab`
        module defines :func:`~matplotlib.mlab.detrend_none`,
        :func:`~matplotlib.mlab.detrend_mean`, and
        :func:`~matplotlib.mlab.detrend_linear`, but you can use
        a custom function as well.  You can also use a string to choose
        one of the functions.  'default', 'constant', and 'mean' call
        :func:`~matplotlib.mlab.detrend_mean`.  'linear' calls
        :func:`~matplotlib.mlab.detrend_linear`.  'none' calls
        :func:`~matplotlib.mlab.detrend_none`.

    scale_by_freq : bool, optional
        Specifies whether the resulting density values should be scaled
        by the scaling frequency, which gives density in units of Hz^-1.
        This allows for integration over the returned frequency values.
        The default is True for MATLAB compatibility.
"""))


@docstring.dedent_interpd
def psd(x, NFFT=None, Fs=None, detrend=None, window=None,
        noverlap=None, pad_to=None, sides=None, scale_by_freq=None):
    r"""
    Compute the power spectral density.

    Call signature::

        psd(x, NFFT=256, Fs=2, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0, pad_to=None,
            sides='default', scale_by_freq=None)

    The power spectral density :math:`P_{xx}` by Welch's average
    periodogram method.  The vector *x* is divided into *NFFT* length
    segments.  Each segment is detrended by function *detrend* and
    windowed by function *window*.  *noverlap* gives the length of
    the overlap between segments.  The :math:`|\mathrm{fft}(i)|^2`
    of each segment :math:`i` are averaged to compute :math:`P_{xx}`.

    If len(*x*) < *NFFT*, it will be zero padded to *NFFT*.

    Parameters
    ----------
    x : 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(PSD)s

    noverlap : integer
        The number of points of overlap between segments.
        The default value is 0 (no overlap).

    Returns
    -------
    Pxx : 1-D array
        The values for the power spectrum `P_{xx}` (real valued)

    freqs : 1-D array
        The frequencies corresponding to the elements in *Pxx*

    References
    ----------
    Bendat & Piersol -- Random Data: Analysis and Measurement Procedures, John
    Wiley & Sons (1986)

    See Also
    --------
    :func:`specgram`
        :func:`specgram` differs in the default overlap; in not returning the
        mean of the segment periodograms; and in returning the times of the
        segments.

    :func:`magnitude_spectrum`
        :func:`magnitude_spectrum` returns the magnitude spectrum.

    :func:`csd`
        :func:`csd` returns the spectral density between two signals.
    """
    Pxx, freqs = csd(x=x, y=None, NFFT=NFFT, Fs=Fs, detrend=detrend,
                     window=window, noverlap=noverlap, pad_to=pad_to,
                     sides=sides, scale_by_freq=scale_by_freq)
    return Pxx.real, freqs


@docstring.dedent_interpd
def csd(x, y, NFFT=None, Fs=None, detrend=None, window=None,
        noverlap=None, pad_to=None, sides=None, scale_by_freq=None):
    """
    Compute the cross-spectral density.

    Call signature::

        csd(x, y, NFFT=256, Fs=2, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0, pad_to=None,
            sides='default', scale_by_freq=None)

    The cross spectral density :math:`P_{xy}` by Welch's average
    periodogram method.  The vectors *x* and *y* are divided into
    *NFFT* length segments.  Each segment is detrended by function
    *detrend* and windowed by function *window*.  *noverlap* gives
    the length of the overlap between segments.  The product of
    the direct FFTs of *x* and *y* are averaged over each segment
    to compute :math:`P_{xy}`, with a scaling to correct for power
    loss due to windowing.

    If len(*x*) < *NFFT* or len(*y*) < *NFFT*, they will be zero
    padded to *NFFT*.

    Parameters
    ----------
    x, y : 1-D arrays or sequences
        Arrays or sequences containing the data

    %(Spectral)s

    %(PSD)s

    noverlap : integer
        The number of points of overlap between segments.
        The default value is 0 (no overlap).

    Returns
    -------
    Pxy : 1-D array
        The values for the cross spectrum `P_{xy}` before scaling (real valued)

    freqs : 1-D array
        The frequencies corresponding to the elements in *Pxy*

    References
    ----------
    Bendat & Piersol -- Random Data: Analysis and Measurement Procedures, John
    Wiley & Sons (1986)

    See Also
    --------
    :func:`psd`
        :func:`psd` is the equivalent to setting y=x.
    """
    if NFFT is None:
        NFFT = 256
    Pxy, freqs, _ = _spectral_helper(x=x, y=y, NFFT=NFFT, Fs=Fs,
                                     detrend_func=detrend, window=window,
                                     noverlap=noverlap, pad_to=pad_to,
                                     sides=sides, scale_by_freq=scale_by_freq,
                                     mode='psd')

    if Pxy.ndim == 2:
        if Pxy.shape[1] > 1:
            Pxy = Pxy.mean(axis=1)
        else:
            Pxy = Pxy[:, 0]
    return Pxy, freqs


@docstring.dedent_interpd
def complex_spectrum(x, Fs=None, window=None, pad_to=None,
                     sides=None):
    """
    Compute the complex-valued frequency spectrum of *x*.  Data is padded to a
    length of *pad_to* and the windowing function *window* is applied to the
    signal.

    Parameters
    ----------
    x : 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(Single_Spectrum)s

    Returns
    -------
    spectrum : 1-D array
        The values for the complex spectrum (complex valued)

    freqs : 1-D array
        The frequencies corresponding to the elements in *spectrum*

    See Also
    --------
    :func:`magnitude_spectrum`
        :func:`magnitude_spectrum` returns the absolute value of this function.

    :func:`angle_spectrum`
        :func:`angle_spectrum` returns the angle of this function.

    :func:`phase_spectrum`
        :func:`phase_spectrum` returns the phase (unwrapped angle) of this
        function.

    :func:`specgram`
        :func:`specgram` can return the complex spectrum of segments within the
        signal.
    """
    return _single_spectrum_helper(x=x, Fs=Fs, window=window, pad_to=pad_to,
                                   sides=sides, mode='complex')


@docstring.dedent_interpd
def magnitude_spectrum(x, Fs=None, window=None, pad_to=None,
                       sides=None):
    """
    Compute the magnitude (absolute value) of the frequency spectrum of
    *x*.  Data is padded to a length of *pad_to* and the windowing function
    *window* is applied to the signal.

    Parameters
    ----------
    x : 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(Single_Spectrum)s

    Returns
    -------
    spectrum : 1-D array
        The values for the magnitude spectrum (real valued)

    freqs : 1-D array
        The frequencies corresponding to the elements in *spectrum*

    See Also
    --------
    :func:`psd`
        :func:`psd` returns the power spectral density.

    :func:`complex_spectrum`
        This function returns the absolute value of :func:`complex_spectrum`.

    :func:`angle_spectrum`
        :func:`angle_spectrum` returns the angles of the corresponding
        frequencies.

    :func:`phase_spectrum`
        :func:`phase_spectrum` returns the phase (unwrapped angle) of the
        corresponding frequencies.

    :func:`specgram`
        :func:`specgram` can return the magnitude spectrum of segments within
        the signal.
    """
    return _single_spectrum_helper(x=x, Fs=Fs, window=window, pad_to=pad_to,
                                   sides=sides, mode='magnitude')


@docstring.dedent_interpd
def angle_spectrum(x, Fs=None, window=None, pad_to=None,
                   sides=None):
    """
    Compute the angle of the frequency spectrum (wrapped phase spectrum) of
    *x*.  Data is padded to a length of *pad_to* and the windowing function
    *window* is applied to the signal.

    Parameters
    ----------
    x : 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(Single_Spectrum)s

    Returns
    -------
    spectrum : 1-D array
        The values for the angle spectrum in radians (real valued)

    freqs : 1-D array
        The frequencies corresponding to the elements in *spectrum*

    See Also
    --------
    :func:`complex_spectrum`
        This function returns the angle value of :func:`complex_spectrum`.

    :func:`magnitude_spectrum`
        :func:`angle_spectrum` returns the magnitudes of the corresponding
        frequencies.

    :func:`phase_spectrum`
        :func:`phase_spectrum` returns the unwrapped version of this function.

    :func:`specgram`
        :func:`specgram` can return the angle spectrum of segments within the
        signal.
    """
    return _single_spectrum_helper(x=x, Fs=Fs, window=window, pad_to=pad_to,
                                   sides=sides, mode='angle')


@docstring.dedent_interpd
def phase_spectrum(x, Fs=None, window=None, pad_to=None,
                   sides=None):
    """
    Compute the phase of the frequency spectrum (unwrapped angle spectrum) of
    *x*.  Data is padded to a length of *pad_to* and the windowing function
    *window* is applied to the signal.

    Parameters
    ----------
    x : 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(Single_Spectrum)s

    Returns
    -------
    spectrum : 1-D array
        The values for the phase spectrum in radians (real valued)

    freqs : 1-D array
        The frequencies corresponding to the elements in *spectrum*

    See Also
    --------
    :func:`complex_spectrum`
        This function returns the angle value of :func:`complex_spectrum`.

    :func:`magnitude_spectrum`
        :func:`magnitude_spectrum` returns the magnitudes of the corresponding
        frequencies.

    :func:`angle_spectrum`
        :func:`angle_spectrum` returns the wrapped version of this function.

    :func:`specgram`
        :func:`specgram` can return the phase spectrum of segments within the
        signal.
    """
    return _single_spectrum_helper(x=x, Fs=Fs, window=window, pad_to=pad_to,
                                   sides=sides, mode='phase')


@docstring.dedent_interpd
def specgram(x, NFFT=None, Fs=None, detrend=None, window=None,
             noverlap=None, pad_to=None, sides=None, scale_by_freq=None,
             mode=None):
    """
    Compute a spectrogram.

    Compute and plot a spectrogram of data in x.  Data are split into
    NFFT length segments and the spectrum of each section is
    computed.  The windowing function window is applied to each
    segment, and the amount of overlap of each segment is
    specified with noverlap.

    Parameters
    ----------
    x : array_like
        1-D array or sequence.

    %(Spectral)s

    %(PSD)s

    noverlap : int, optional
        The number of points of overlap between blocks.  The default
        value is 128.
    mode : str, optional
        What sort of spectrum to use, default is 'psd'.
            'psd'
                Returns the power spectral density.

            'complex'
                Returns the complex-valued frequency spectrum.

            'magnitude'
                Returns the magnitude spectrum.

            'angle'
                Returns the phase spectrum without unwrapping.

            'phase'
                Returns the phase spectrum with unwrapping.

    Returns
    -------
    spectrum : array_like
        2-D array, columns are the periodograms of successive segments.

    freqs : array_like
        1-D array, frequencies corresponding to the rows in *spectrum*.

    t : array_like
        1-D array, the times corresponding to midpoints of segments
        (i.e the columns in *spectrum*).

    See Also
    --------
    psd : differs in the overlap and in the return values.
    complex_spectrum : similar, but with complex valued frequencies.
    magnitude_spectrum : similar single segment when mode is 'magnitude'.
    angle_spectrum : similar to single segment when mode is 'angle'.
    phase_spectrum : similar to single segment when mode is 'phase'.

    Notes
    -----
    detrend and scale_by_freq only apply when *mode* is set to 'psd'.

    """
    if noverlap is None:
        noverlap = 128  # default in _spectral_helper() is noverlap = 0
    if NFFT is None:
        NFFT = 256  # same default as in _spectral_helper()
    if len(x) <= NFFT:
        warnings.warn("Only one segment is calculated since parameter NFFT " +
                      "(=%d) >= signal length (=%d)." % (NFFT, len(x)))

    spec, freqs, t = _spectral_helper(x=x, y=None, NFFT=NFFT, Fs=Fs,
                                      detrend_func=detrend, window=window,
                                      noverlap=noverlap, pad_to=pad_to,
                                      sides=sides,
                                      scale_by_freq=scale_by_freq,
                                      mode=mode)

    if mode != 'complex':
        spec = spec.real  # Needed since helper implements generically

    return spec, freqs, t


_coh_error = """Coherence is calculated by averaging over *NFFT*
length segments.  Your signal is too short for your choice of *NFFT*.
"""


@docstring.dedent_interpd
def cohere(x, y, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning,
           noverlap=0, pad_to=None, sides='default', scale_by_freq=None):
    """
    The coherence between *x* and *y*.  Coherence is the normalized
    cross spectral density:

    .. math::

        C_{xy} = \\frac{|P_{xy}|^2}{P_{xx}P_{yy}}

    Parameters
    ----------
    x, y
        Array or sequence containing the data

    %(Spectral)s

    %(PSD)s

    noverlap : integer
        The number of points of overlap between blocks.  The default value
        is 0 (no overlap).

    Returns
    -------
    The return value is the tuple (*Cxy*, *f*), where *f* are the
    frequencies of the coherence vector. For cohere, scaling the
    individual densities by the sampling frequency has no effect,
    since the factors cancel out.

    See Also
    --------
    :func:`psd`, :func:`csd` :
        For information about the methods used to compute :math:`P_{xy}`,
        :math:`P_{xx}` and :math:`P_{yy}`.
    """

    if len(x) < 2 * NFFT:
        raise ValueError(_coh_error)
    Pxx, f = psd(x, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    Pyy, f = psd(y, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    Pxy, f = csd(x, y, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    Cxy = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    return Cxy, f


@cbook.deprecated('2.2')
def movavg(x, n):
    """
    Compute the len(*n*) moving average of *x*.
    """
    w = np.empty((n,), dtype=float)
    w[:] = 1.0/n
    return np.convolve(x, w, mode='valid')


# the following code was written and submitted by Fernando Perez
# from the ipython numutils package under a BSD license
# begin fperez functions

"""
A set of convenient utilities for numerical work.

Most of this module requires numpy or is meant to be used with it.

Copyright (c) 2001-2004, Fernando Perez. <Fernando.Perez@colorado.edu>
All rights reserved.

This license was generated from the BSD license template as found in:
http://www.opensource.org/licenses/bsd-license.php

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the IPython project nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""


# *****************************************************************************
# Globals
# ****************************************************************************
# function definitions
exp_safe_MIN = math.log(2.2250738585072014e-308)
exp_safe_MAX = 1.7976931348623157e+308


@cbook.deprecated("2.2", 'numpy.exp')
def exp_safe(x):
    """
    Compute exponentials which safely underflow to zero.

    Slow, but convenient to use. Note that numpy provides proper
    floating point exception handling with access to the underlying
    hardware.
    """

    if type(x) is np.ndarray:
        return np.exp(np.clip(x, exp_safe_MIN, exp_safe_MAX))
    else:
        return math.exp(x)


@cbook.deprecated("2.2", alternative='numpy.array(list(map(...)))')
def amap(fn, *args):
    """
    amap(function, sequence[, sequence, ...]) -> array.

    Works like :func:`map`, but it returns an array.  This is just a
    convenient shorthand for ``numpy.array(map(...))``.
    """
    return np.array(list(map(fn, *args)))


@cbook.deprecated("2.2")
def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.abs(a) ** 2))


@cbook.deprecated("2.2", alternative='numpy.linalg.norm(a, ord=1)')
def l1norm(a):
    """
    Return the *l1* norm of *a*, flattened out.

    Implemented as a separate function (not a call to :func:`norm` for speed).
    """
    return np.sum(np.abs(a))


@cbook.deprecated("2.2", alternative='numpy.linalg.norm(a, ord=2)')
def l2norm(a):
    """
    Return the *l2* norm of *a*, flattened out.

    Implemented as a separate function (not a call to :func:`norm` for speed).
    """
    return np.sqrt(np.sum(np.abs(a) ** 2))


@cbook.deprecated("2.2", alternative='numpy.linalg.norm(a.flat, ord=p)')
def norm_flat(a, p=2):
    """
    norm(a,p=2) -> l-p norm of a.flat

    Return the l-p norm of *a*, considered as a flat array.  This is NOT a true
    matrix norm, since arrays of arbitrary rank are always flattened.

    *p* can be a number or the string 'Infinity' to get the L-infinity norm.
    """
    # This function was being masked by a more general norm later in
    # the file.  We may want to simply delete it.
    if p == 'Infinity':
        return np.max(np.abs(a))
    else:
        return np.sum(np.abs(a) ** p) ** (1 / p)


@cbook.deprecated("2.2", 'numpy.arange')
def frange(xini, xfin=None, delta=None, **kw):
    """
    frange([start,] stop[, step, keywords]) -> array of floats

    Return a numpy ndarray containing a progression of floats. Similar to
    :func:`numpy.arange`, but defaults to a closed interval.

    ``frange(x0, x1)`` returns ``[x0, x0+1, x0+2, ..., x1]``; *start*
    defaults to 0, and the endpoint *is included*. This behavior is
    different from that of :func:`range` and
    :func:`numpy.arange`. This is deliberate, since :func:`frange`
    will probably be more useful for generating lists of points for
    function evaluation, and endpoints are often desired in this
    use. The usual behavior of :func:`range` can be obtained by
    setting the keyword *closed* = 0, in this case, :func:`frange`
    basically becomes :func:numpy.arange`.

    When *step* is given, it specifies the increment (or
    decrement). All arguments can be floating point numbers.

    ``frange(x0,x1,d)`` returns ``[x0,x0+d,x0+2d,...,xfin]`` where
    *xfin* <= *x1*.

    :func:`frange` can also be called with the keyword *npts*. This
    sets the number of points the list should contain (and overrides
    the value *step* might have been given). :func:`numpy.arange`
    doesn't offer this option.

    Examples::

      >>> frange(3)
      array([ 0.,  1.,  2.,  3.])
      >>> frange(3,closed=0)
      array([ 0.,  1.,  2.])
      >>> frange(1,6,2)
      array([1, 3, 5])   or 1,3,5,7, depending on floating point vagueries
      >>> frange(1,6.5,npts=5)
      array([ 1.   ,  2.375,  3.75 ,  5.125,  6.5  ])
    """

    # defaults
    kw.setdefault('closed', 1)
    endpoint = kw['closed'] != 0

    # funny logic to allow the *first* argument to be optional (like range())
    # This was modified with a simpler version from a similar frange() found
    # at http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/66472
    if xfin is None:
        xfin = xini + 0.0
        xini = 0.0

    if delta is None:
        delta = 1.0

    # compute # of points, spacing and return final list
    try:
        npts = kw['npts']
        delta = (xfin-xini) / (npts-endpoint)
    except KeyError:
        npts = int(np.round((xfin-xini)/delta)) + endpoint
        # round finds the nearest, so the endpoint can be up to
        # delta/2 larger than xfin.

    return np.arange(npts)*delta+xini
# end frange()


@cbook.deprecated("2.2", 'numpy.identity')
def identity(n, rank=2, dtype='l', typecode=None):
    """
    Returns the identity matrix of shape (*n*, *n*, ..., *n*) (rank *r*).

    For ranks higher than 2, this object is simply a multi-index Kronecker
    delta::

                            /  1  if i0=i1=...=iR,
        id[i0,i1,...,iR] = -|
                            \\  0  otherwise.

    Optionally a *dtype* (or typecode) may be given (it defaults to 'l').

    Since rank defaults to 2, this function behaves in the default case (when
    only *n* is given) like ``numpy.identity(n)`` -- but surprisingly, it is
    much faster.
    """
    if typecode is not None:
        dtype = typecode
    iden = np.zeros((n,)*rank, dtype)
    for i in range(n):
        idx = (i,)*rank
        iden[idx] = 1
    return iden


@cbook.deprecated("2.2")
def base_repr(number, base=2, padding=0):
    """
    Return the representation of a *number* in any given *base*.
    """
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if number < base:
        return (padding - 1) * chars[0] + chars[int(number)]
    max_exponent = int(math.log(number)/math.log(base))
    max_power = int(base) ** max_exponent
    lead_digit = int(number/max_power)
    return (chars[lead_digit] +
            base_repr(number - max_power * lead_digit, base,
                      max(padding - 1, max_exponent)))


@cbook.deprecated("2.2")
def binary_repr(number, max_length=1025):
    """
    Return the binary representation of the input *number* as a
    string.

    This is more efficient than using :func:`base_repr` with base 2.

    Increase the value of max_length for very large numbers. Note that
    on 32-bit machines, 2**1023 is the largest integer power of 2
    which can be converted to a Python float.
    """

#   assert number < 2L << max_length
    shifts = map(operator.rshift, max_length * [number],
                 range(max_length - 1, -1, -1))
    digits = list(map(operator.mod, shifts, max_length * [2]))
    if not digits.count(1):
        return 0
    digits = digits[digits.index(1):]
    return ''.join(map(repr, digits)).replace('L', '')


@cbook.deprecated("2.2", 'numpy.log2')
def log2(x, ln2=math.log(2.0)):
    """
    Return the log(*x*) in base 2.

    This is a _slow_ function but which is guaranteed to return the correct
    integer value if the input is an integer exact power of 2.
    """
    try:
        bin_n = binary_repr(x)[1:]
    except (AssertionError, TypeError):
        return math.log(x)/ln2
    else:
        if '1' in bin_n:
            return math.log(x)/ln2
        else:
            return len(bin_n)


@cbook.deprecated("2.2")
def ispower2(n):
    """
    Returns the log base 2 of *n* if *n* is a power of 2, zero otherwise.

    Note the potential ambiguity if *n* == 1: 2**0 == 1, interpret accordingly.
    """

    bin_n = binary_repr(n)[1:]
    if '1' in bin_n:
        return 0
    else:
        return len(bin_n)


@cbook.deprecated("2.2")
def isvector(X):
    """
    Like the MATLAB function with the same name, returns *True*
    if the supplied numpy array or matrix *X* looks like a vector,
    meaning it has a one non-singleton axis (i.e., it can have
    multiple axes, but all must have length 1, except for one of
    them).

    If you just want to see if the array has 1 axis, use X.ndim == 1.
    """
    return np.prod(X.shape) == np.max(X.shape)

# end fperez numutils code


# helpers for loading, saving, manipulating and viewing numpy record arrays
@cbook.deprecated("2.2", 'numpy.isnan')
def safe_isnan(x):
    ':func:`numpy.isnan` for arbitrary types'
    if isinstance(x, str):
        return False
    try:
        b = np.isnan(x)
    except NotImplementedError:
        return False
    except TypeError:
        return False
    else:
        return b


@cbook.deprecated("2.2", 'numpy.isinf')
def safe_isinf(x):
    ':func:`numpy.isinf` for arbitrary types'
    if isinstance(x, str):
        return False
    try:
        b = np.isinf(x)
    except NotImplementedError:
        return False
    except TypeError:
        return False
    else:
        return b


@cbook.deprecated("2.2")
def rec_append_fields(rec, names, arrs, dtypes=None):
    """
    Return a new record array with field names populated with data
    from arrays in *arrs*.  If appending a single field, then *names*,
    *arrs* and *dtypes* do not have to be lists. They can just be the
    values themselves.
    """
    if (not isinstance(names, str) and np.iterable(names)
            and len(names) and isinstance(names[0], str)):
        if len(names) != len(arrs):
            raise ValueError("number of arrays do not match number of names")
    else:  # we have only 1 name and 1 array
        names = [names]
        arrs = [arrs]
    arrs = list(map(np.asarray, arrs))
    if dtypes is None:
        dtypes = [a.dtype for a in arrs]
    elif not np.iterable(dtypes):
        dtypes = [dtypes]
    if len(arrs) != len(dtypes):
        if len(dtypes) == 1:
            dtypes = dtypes * len(arrs)
        else:
            raise ValueError("dtypes must be None, a single dtype or a list")
    old_dtypes = rec.dtype.descr
    newdtype = np.dtype(old_dtypes + list(zip(names, dtypes)))
    newrec = np.recarray(rec.shape, dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    for name, arr in zip(names, arrs):
        newrec[name] = arr
    return newrec


@cbook.deprecated("2.2")
def rec_drop_fields(rec, names):
    """
    Return a new numpy record array with fields in *names* dropped.
    """

    names = set(names)

    newdtype = np.dtype([(name, rec.dtype[name]) for name in rec.dtype.names
                         if name not in names])

    newrec = np.recarray(rec.shape, dtype=newdtype)
    for field in newdtype.names:
        newrec[field] = rec[field]

    return newrec


@cbook.deprecated("2.2")
def rec_keep_fields(rec, names):
    """
    Return a new numpy record array with only fields listed in names
    """

    if isinstance(names, str):
        names = names.split(',')

    arrays = []
    for name in names:
        arrays.append(rec[name])

    return np.rec.fromarrays(arrays, names=names)


@cbook.deprecated("2.2")
def rec_groupby(r, groupby, stats):
    """
    *r* is a numpy record array

    *groupby* is a sequence of record array attribute names that
    together form the grouping key.  e.g., ('date', 'productcode')

    *stats* is a sequence of (*attr*, *func*, *outname*) tuples which
    will call ``x = func(attr)`` and assign *x* to the record array
    output with attribute *outname*.  For example::

      stats = ( ('sales', len, 'numsales'), ('sales', np.mean, 'avgsale') )

    Return record array has *dtype* names for each attribute name in
    the *groupby* argument, with the associated group values, and
    for each outname name in the *stats* argument, with the associated
    stat summary output.
    """
    # build a dictionary from groupby keys-> list of indices into r with
    # those keys
    rowd = {}
    for i, row in enumerate(r):
        key = tuple([row[attr] for attr in groupby])
        rowd.setdefault(key, []).append(i)

    rows = []
    # sort the output by groupby keys
    for key in sorted(rowd):
        row = list(key)
        # get the indices for this groupby key
        ind = rowd[key]
        thisr = r[ind]
        # call each stat function for this groupby slice
        row.extend([func(thisr[attr]) for attr, func, outname in stats])
        rows.append(row)

    # build the output record array with groupby and outname attributes
    attrs, funcs, outnames = list(zip(*stats))
    names = list(groupby)
    names.extend(outnames)
    return np.rec.fromrecords(rows, names=names)


@cbook.deprecated("2.2")
def rec_summarize(r, summaryfuncs):
    """
    *r* is a numpy record array

    *summaryfuncs* is a list of (*attr*, *func*, *outname*) tuples
    which will apply *func* to the array *r*[attr] and assign the
    output to a new attribute name *outname*.  The returned record
    array is identical to *r*, with extra arrays for each element in
    *summaryfuncs*.

    """

    names = list(r.dtype.names)
    arrays = [r[name] for name in names]

    for attr, func, outname in summaryfuncs:
        names.append(outname)
        arrays.append(np.asarray(func(r[attr])))

    return np.rec.fromarrays(arrays, names=names)


@cbook.deprecated("2.2")
def rec_join(key, r1, r2, jointype='inner', defaults=None, r1postfix='1',
             r2postfix='2'):
    """
    Join record arrays *r1* and *r2* on *key*; *key* is a tuple of
    field names -- if *key* is a string it is assumed to be a single
    attribute name. If *r1* and *r2* have equal values on all the keys
    in the *key* tuple, then their fields will be merged into a new
    record array containing the intersection of the fields of *r1* and
    *r2*.

    *r1* (also *r2*) must not have any duplicate keys.

    The *jointype* keyword can be 'inner', 'outer', 'leftouter'.  To
    do a rightouter join just reverse *r1* and *r2*.

    The *defaults* keyword is a dictionary filled with
    ``{column_name:default_value}`` pairs.

    The keywords *r1postfix* and *r2postfix* are postfixed to column names
    (other than keys) that are both in *r1* and *r2*.
    """

    if isinstance(key, str):
        key = (key, )

    for name in key:
        if name not in r1.dtype.names:
            raise ValueError('r1 does not have key field %s' % name)
        if name not in r2.dtype.names:
            raise ValueError('r2 does not have key field %s' % name)

    def makekey(row):
        return tuple([row[name] for name in key])

    r1d = {makekey(row): i for i, row in enumerate(r1)}
    r2d = {makekey(row): i for i, row in enumerate(r2)}

    r1keys = set(r1d)
    r2keys = set(r2d)

    common_keys = r1keys & r2keys

    r1ind = np.array([r1d[k] for k in common_keys])
    r2ind = np.array([r2d[k] for k in common_keys])

    common_len = len(common_keys)
    left_len = right_len = 0
    if jointype == "outer" or jointype == "leftouter":
        left_keys = r1keys.difference(r2keys)
        left_ind = np.array([r1d[k] for k in left_keys])
        left_len = len(left_ind)
    if jointype == "outer":
        right_keys = r2keys.difference(r1keys)
        right_ind = np.array([r2d[k] for k in right_keys])
        right_len = len(right_ind)

    def key_desc(name):
        '''
        if name is a string key, use the larger size of r1 or r2 before
        merging
        '''
        dt1 = r1.dtype[name]
        if dt1.type != np.string_:
            return (name, dt1.descr[0][1])

        dt2 = r2.dtype[name]
        if dt1 != dt2:
            raise ValueError("The '{}' fields in arrays 'r1' and 'r2' must "
                             "have the same dtype".format(name))
        if dt1.num > dt2.num:
            return (name, dt1.descr[0][1])
        else:
            return (name, dt2.descr[0][1])

    keydesc = [key_desc(name) for name in key]

    def mapped_r1field(name):
        """
        The column name in *newrec* that corresponds to the column in *r1*.
        """
        if name in key or name not in r2.dtype.names:
            return name
        else:
            return name + r1postfix

    def mapped_r2field(name):
        """
        The column name in *newrec* that corresponds to the column in *r2*.
        """
        if name in key or name not in r1.dtype.names:
            return name
        else:
            return name + r2postfix

    r1desc = [(mapped_r1field(desc[0]), desc[1]) for desc in r1.dtype.descr
              if desc[0] not in key]
    r2desc = [(mapped_r2field(desc[0]), desc[1]) for desc in r2.dtype.descr
              if desc[0] not in key]
    all_dtypes = keydesc + r1desc + r2desc
    newdtype = np.dtype(all_dtypes)
    newrec = np.recarray((common_len + left_len + right_len,), dtype=newdtype)

    if defaults is not None:
        for thiskey in defaults:
            if thiskey not in newdtype.names:
                warnings.warn('rec_join defaults key="%s" not in new dtype '
                              'names "%s"' % (thiskey, newdtype.names))

    for name in newdtype.names:
        dt = newdtype[name]
        if dt.kind in ('f', 'i'):
            newrec[name] = 0

    if jointype != 'inner' and defaults is not None:
        # fill in the defaults enmasse
        newrec_fields = list(newrec.dtype.fields)
        for k, v in defaults.items():
            if k in newrec_fields:
                newrec[k] = v

    for field in r1.dtype.names:
        newfield = mapped_r1field(field)
        if common_len:
            newrec[newfield][:common_len] = r1[field][r1ind]
        if (jointype == "outer" or jointype == "leftouter") and left_len:
            newrec[newfield][common_len:(common_len+left_len)] = (
                r1[field][left_ind]
            )

    for field in r2.dtype.names:
        newfield = mapped_r2field(field)
        if field not in key and common_len:
            newrec[newfield][:common_len] = r2[field][r2ind]
        if jointype == "outer" and right_len:
            newrec[newfield][-right_len:] = r2[field][right_ind]

    newrec.sort(order=key)

    return newrec


@cbook.deprecated("2.2")
def recs_join(key, name, recs, jointype='outer', missing=0., postfixes=None):
    """
    Join a sequence of record arrays on single column key.

    This function only joins a single column of the multiple record arrays

    *key*
      is the column name that acts as a key

    *name*
      is the name of the column that we want to join

    *recs*
      is a list of record arrays to join

    *jointype*
      is a string 'inner' or 'outer'

    *missing*
      is what any missing field is replaced by

    *postfixes*
      if not None, a len recs sequence of postfixes

    returns a record array with columns [rowkey, name0, name1, ... namen-1].
    or if postfixes [PF0, PF1, ..., PFN-1] are supplied,
    [rowkey, namePF0, namePF1, ... namePFN-1].

    Example::

      r = recs_join("date", "close", recs=[r0, r1], missing=0.)

    """
    results = []
    aligned_iters = cbook.align_iterators(operator.attrgetter(key),
                                          *[iter(r) for r in recs])

    def extract(r):
        if r is None:
            return missing
        else:
            return r[name]

    if jointype == "outer":
        for rowkey, row in aligned_iters:
            results.append([rowkey] + list(map(extract, row)))
    elif jointype == "inner":
        for rowkey, row in aligned_iters:
            if None not in row:  # throw out any Nones
                results.append([rowkey] + list(map(extract, row)))

    if postfixes is None:
        postfixes = ['%d' % i for i in range(len(recs))]
    names = ",".join([key] + ["%s%s" % (name, postfix)
                              for postfix in postfixes])
    return np.rec.fromrecords(results, names=names)


@cbook.deprecated("2.2")
def csv2rec(fname, comments='#', skiprows=0, checkrows=0, delimiter=',',
            converterd=None, names=None, missing='', missingd=None,
            use_mrecords=False, dayfirst=False, yearfirst=False):
    """
    Load data from comma/space/tab delimited file in *fname* into a
    numpy record array and return the record array.

    If *names* is *None*, a header row is required to automatically
    assign the recarray names.  The headers will be lower cased,
    spaces will be converted to underscores, and illegal attribute
    name characters removed.  If *names* is not *None*, it is a
    sequence of names to use for the column names.  In this case, it
    is assumed there is no header row.


    - *fname*: can be a filename or a file handle.  Support for gzipped
      files is automatic, if the filename ends in '.gz'

    - *comments*: the character used to indicate the start of a comment
      in the file, or *None* to switch off the removal of comments

    - *skiprows*: is the number of rows from the top to skip

    - *checkrows*: is the number of rows to check to validate the column
      data type.  When set to zero all rows are validated.

    - *converterd*: if not *None*, is a dictionary mapping column number or
      munged column name to a converter function.

    - *names*: if not None, is a list of header names.  In this case, no
      header will be read from the file

    - *missingd* is a dictionary mapping munged column names to field values
      which signify that the field does not contain actual data and should
      be masked, e.g., '0000-00-00' or 'unused'

    - *missing*: a string whose value signals a missing field regardless of
      the column it appears in

    - *use_mrecords*: if True, return an mrecords.fromrecords record array if
      any of the data are missing

    - *dayfirst*: default is False so that MM-DD-YY has precedence over
      DD-MM-YY.  See
      http://labix.org/python-dateutil#head-b95ce2094d189a89f80f5ae52a05b4ab7b41af47
      for further information.

    - *yearfirst*: default is False so that MM-DD-YY has precedence over
      YY-MM-DD. See
      http://labix.org/python-dateutil#head-b95ce2094d189a89f80f5ae52a05b4ab7b41af47
      for further information.

      If no rows are found, *None* is returned
    """

    if converterd is None:
        converterd = dict()

    if missingd is None:
        missingd = {}

    import dateutil.parser
    import datetime

    fh = cbook.to_filehandle(fname)

    delimiter = str(delimiter)

    class FH:
        """
        For space-delimited files, we want different behavior than
        comma or tab.  Generally, we want multiple spaces to be
        treated as a single separator, whereas with comma and tab we
        want multiple commas to return multiple (empty) fields.  The
        join/strip trick below effects this.
        """
        def __init__(self, fh):
            self.fh = fh

        def close(self):
            self.fh.close()

        def seek(self, arg):
            self.fh.seek(arg)

        def fix(self, s):
            return ' '.join(s.split())

        def __next__(self):
            return self.fix(next(self.fh))

        def __iter__(self):
            for line in self.fh:
                yield self.fix(line)

    if delimiter == ' ':
        fh = FH(fh)

    reader = csv.reader(fh, delimiter=delimiter)

    def process_skiprows(reader):
        if skiprows:
            for i, row in enumerate(reader):
                if i >= (skiprows-1):
                    break

        return fh, reader

    process_skiprows(reader)

    def ismissing(name, val):
        "Should the value val in column name be masked?"
        return val == missing or val == missingd.get(name) or val == ''

    def with_default_value(func, default):
        def newfunc(name, val):
            if ismissing(name, val):
                return default
            else:
                return func(val)
        return newfunc

    def mybool(x):
        if x == 'True':
            return True
        elif x == 'False':
            return False
        else:
            raise ValueError('invalid bool')

    dateparser = dateutil.parser.parse

    def mydateparser(x):
        # try and return a datetime object
        d = dateparser(x, dayfirst=dayfirst, yearfirst=yearfirst)
        return d

    mydateparser = with_default_value(mydateparser, datetime.datetime(1, 1, 1))

    myfloat = with_default_value(float, np.nan)
    myint = with_default_value(int, -1)
    mystr = with_default_value(str, '')
    mybool = with_default_value(mybool, None)

    def mydate(x):
        # try and return a date object
        d = dateparser(x, dayfirst=dayfirst, yearfirst=yearfirst)

        if d.hour > 0 or d.minute > 0 or d.second > 0:
            raise ValueError('not a date')
        return d.date()
    mydate = with_default_value(mydate, datetime.date(1, 1, 1))

    def get_func(name, item, func):
        # promote functions in this order
        funcs = [mybool, myint, myfloat, mydate, mydateparser, mystr]
        for func in funcs[funcs.index(func):]:
            try:
                func(name, item)
            except Exception:
                continue
            return func
        raise ValueError('Could not find a working conversion function')

    # map column names that clash with builtins -- TODO - extend this list
    itemd = {
        'return': 'return_',
        'file':   'file_',
        'print':  'print_',
        }

    def get_converters(reader, comments):

        converters = None
        i = 0
        for row in reader:
            if (len(row) and comments is not None and
                    row[0].startswith(comments)):
                continue
            if i == 0:
                converters = [mybool]*len(row)
            if checkrows and i > checkrows:
                break
            i += 1

            for j, (name, item) in enumerate(zip(names, row)):
                func = converterd.get(j)
                if func is None:
                    func = converterd.get(name)
                if func is None:
                    func = converters[j]
                    if len(item.strip()):
                        func = get_func(name, item, func)
                else:
                    # how should we handle custom converters and defaults?
                    func = with_default_value(func, None)
                converters[j] = func
        return converters

    # Get header and remove invalid characters
    needheader = names is None

    if needheader:
        for row in reader:
            if (len(row) and comments is not None and
                    row[0].startswith(comments)):
                continue
            headers = row
            break

        # remove these chars
        delete = set(r"""~!@#$%^&*()-=+~\|}[]{';: /?.>,<""")
        delete.add('"')

        names = []
        seen = dict()
        for i, item in enumerate(headers):
            item = item.strip().lower().replace(' ', '_')
            item = ''.join([c for c in item if c not in delete])
            if not len(item):
                item = 'column%d' % i

            item = itemd.get(item, item)
            cnt = seen.get(item, 0)
            if cnt > 0:
                names.append(item + '_%d' % cnt)
            else:
                names.append(item)
            seen[item] = cnt+1

    else:
        if isinstance(names, str):
            names = [n.strip() for n in names.split(',')]

    # get the converter functions by inspecting checkrows
    converters = get_converters(reader, comments)
    if converters is None:
        raise ValueError('Could not find any valid data in CSV file')

    # reset the reader and start over
    fh.seek(0)
    reader = csv.reader(fh, delimiter=delimiter)
    process_skiprows(reader)

    if needheader:
        while True:
            # skip past any comments and consume one line of column header
            row = next(reader)
            if (len(row) and comments is not None and
                    row[0].startswith(comments)):
                continue
            break

    # iterate over the remaining rows and convert the data to date
    # objects, ints, or floats as appropriate
    rows = []
    rowmasks = []
    for i, row in enumerate(reader):
        if not len(row):
            continue
        if comments is not None and row[0].startswith(comments):
            continue
        # Ensure that the row returned always has the same nr of elements
        row.extend([''] * (len(converters) - len(row)))
        rows.append([func(name, val)
                     for func, name, val in zip(converters, names, row)])
        rowmasks.append([ismissing(name, val)
                         for name, val in zip(names, row)])
    fh.close()

    if not len(rows):
        return None

    if use_mrecords and np.any(rowmasks):
        r = np.ma.mrecords.fromrecords(rows, names=names, mask=rowmasks)
    else:
        r = np.rec.fromrecords(rows, names=names)
    return r


# a series of classes for describing the format intentions of various rec views
@cbook.deprecated("2.2")
class FormatObj(object):
    def tostr(self, x):
        return self.toval(x)

    def toval(self, x):
        return str(x)

    def fromstr(self, s):
        return s

    def __hash__(self):
        """
        override the hash function of any of the formatters, so that we don't
        create duplicate excel format styles
        """
        return hash(self.__class__)


@cbook.deprecated("2.2")
class FormatString(FormatObj):
    def tostr(self, x):
        val = repr(x)
        return val[1:-1]


@cbook.deprecated("2.2")
class FormatFormatStr(FormatObj):
    def __init__(self, fmt):
        self.fmt = fmt

    def tostr(self, x):
        if x is None:
            return 'None'
        return self.fmt % self.toval(x)


@cbook.deprecated("2.2")
class FormatFloat(FormatFormatStr):
    def __init__(self, precision=4, scale=1.):
        FormatFormatStr.__init__(self, '%%1.%df' % precision)
        self.precision = precision
        self.scale = scale

    def __hash__(self):
        return hash((self.__class__, self.precision, self.scale))

    def toval(self, x):
        if x is not None:
            x = x * self.scale
        return x

    def fromstr(self, s):
        return float(s)/self.scale


@cbook.deprecated("2.2")
class FormatInt(FormatObj):

    def tostr(self, x):
        return '%d' % int(x)

    def toval(self, x):
        return int(x)

    def fromstr(self, s):
        return int(s)


@cbook.deprecated("2.2")
class FormatBool(FormatObj):
    def toval(self, x):
        return str(x)

    def fromstr(self, s):
        return bool(s)


@cbook.deprecated("2.2")
class FormatPercent(FormatFloat):
    def __init__(self, precision=4):
        FormatFloat.__init__(self, precision, scale=100.)


@cbook.deprecated("2.2")
class FormatThousands(FormatFloat):
    def __init__(self, precision=4):
        FormatFloat.__init__(self, precision, scale=1e-3)


@cbook.deprecated("2.2")
class FormatMillions(FormatFloat):
    def __init__(self, precision=4):
        FormatFloat.__init__(self, precision, scale=1e-6)


@cbook.deprecated("2.2", alternative='date.strftime')
class FormatDate(FormatObj):
    def __init__(self, fmt):
        self.fmt = fmt

    def __hash__(self):
        return hash((self.__class__, self.fmt))

    def toval(self, x):
        if x is None:
            return 'None'
        return x.strftime(self.fmt)

    def fromstr(self, x):
        import dateutil.parser
        return dateutil.parser.parse(x).date()


@cbook.deprecated("2.2", alternative='datetime.strftime')
class FormatDatetime(FormatDate):
    def __init__(self, fmt='%Y-%m-%d %H:%M:%S'):
        FormatDate.__init__(self, fmt)

    def fromstr(self, x):
        import dateutil.parser
        return dateutil.parser.parse(x)


@cbook.deprecated("2.2")
def get_formatd(r, formatd=None):
    'build a formatd guaranteed to have a key for every dtype name'
    defaultformatd = {
        np.bool_: FormatBool(),
        np.int16: FormatInt(),
        np.int32: FormatInt(),
        np.int64: FormatInt(),
        np.float32: FormatFloat(),
        np.float64: FormatFloat(),
        np.object_: FormatObj(),
        np.string_: FormatString()}

    if formatd is None:
        formatd = dict()

    for i, name in enumerate(r.dtype.names):
        dt = r.dtype[name]
        format = formatd.get(name)
        if format is None:
            format = defaultformatd.get(dt.type, FormatObj())
        formatd[name] = format
    return formatd


@cbook.deprecated("2.2")
def csvformat_factory(format):
    format = copy.deepcopy(format)
    if isinstance(format, FormatFloat):
        format.scale = 1.  # override scaling for storage
        format.fmt = '%r'
    return format


@cbook.deprecated("2.2", alternative='numpy.recarray.tofile')
def rec2txt(r, header=None, padding=3, precision=3, fields=None):
    """
    Returns a textual representation of a record array.

    Parameters
    ----------
    r: numpy recarray

    header: list
        column headers

    padding:
        space between each column

    precision: number of decimal places to use for floats.
        Set to an integer to apply to all floats.  Set to a
        list of integers to apply precision individually.
        Precision for non-floats is simply ignored.

    fields : list
        If not None, a list of field names to print.  fields
        can be a list of strings like ['field1', 'field2'] or a single
        comma separated string like 'field1,field2'

    Examples
    --------

    For ``precision=[0,2,3]``, the output is ::

      ID    Price   Return
      ABC   12.54    0.234
      XYZ    6.32   -0.076
    """

    if fields is not None:
        r = rec_keep_fields(r, fields)

    if cbook.is_numlike(precision):
        precision = [precision]*len(r.dtype)

    def get_type(item, atype=int):
        tdict = {None: int, int: float, float: str}
        try:
            atype(str(item))
        except Exception:
            return get_type(item, tdict[atype])
        return atype

    def get_justify(colname, column, precision):
        ntype = column.dtype

        if np.issubdtype(ntype, np.character):
            fixed_width = int(ntype.str[2:])
            length = max(len(colname), fixed_width)
            return 0, length+padding, "%s"  # left justify

        if np.issubdtype(ntype, np.integer):
            length = max(len(colname),
                         np.max(list(map(len, list(map(str, column))))))
            return 1, length+padding, "%d"  # right justify

        if np.issubdtype(ntype, np.floating):
            fmt = "%." + str(precision) + "f"
            length = max(
                len(colname),
                np.max(list(map(len, list(map(lambda x: fmt % x, column)))))
            )
            return 1, length+padding, fmt   # right justify

        return (0,
                max(len(colname),
                    np.max(list(map(len, list(map(str, column))))))+padding,
                "%s")

    if header is None:
        header = r.dtype.names

    justify_pad_prec = [get_justify(header[i], r.__getitem__(colname),
                                    precision[i])
                        for i, colname in enumerate(r.dtype.names)]

    justify_pad_prec_spacer = []
    for i in range(len(justify_pad_prec)):
        just, pad, prec = justify_pad_prec[i]
        if i == 0:
            justify_pad_prec_spacer.append((just, pad, prec, 0))
        else:
            pjust, ppad, pprec = justify_pad_prec[i-1]
            if pjust == 0 and just == 1:
                justify_pad_prec_spacer.append((just, pad-padding, prec, 0))
            elif pjust == 1 and just == 0:
                justify_pad_prec_spacer.append((just, pad, prec, padding))
            else:
                justify_pad_prec_spacer.append((just, pad, prec, 0))

    def format(item, just_pad_prec_spacer):
        just, pad, prec, spacer = just_pad_prec_spacer
        if just == 0:
            return spacer*' ' + str(item).ljust(pad)
        else:
            if get_type(item) == float:
                item = (prec % float(item))
            elif get_type(item) == int:
                item = (prec % int(item))

            return item.rjust(pad)

    textl = []
    textl.append(''.join([format(colitem, justify_pad_prec_spacer[j])
                          for j, colitem in enumerate(header)]))
    for i, row in enumerate(r):
        textl.append(''.join([format(colitem, justify_pad_prec_spacer[j])
                              for j, colitem in enumerate(row)]))
        if i == 0:
            textl[0] = textl[0].rstrip()

    text = os.linesep.join(textl)
    return text


@cbook.deprecated("2.2", alternative='numpy.recarray.tofile')
def rec2csv(r, fname, delimiter=',', formatd=None, missing='',
            missingd=None, withheader=True):
    """
    Save the data from numpy recarray *r* into a
    comma-/space-/tab-delimited file.  The record array dtype names
    will be used for column headers.

    *fname*: can be a filename or a file handle.  Support for gzipped
      files is automatic, if the filename ends in '.gz'

    *withheader*: if withheader is False, do not write the attribute
      names in the first row

    for formatd type FormatFloat, we override the precision to store
    full precision floats in the CSV file

    See Also
    --------
    :func:`csv2rec`
        For information about *missing* and *missingd*, which can be used to
        fill in masked values into your CSV file.
    """

    delimiter = str(delimiter)

    if missingd is None:
        missingd = dict()

    def with_mask(func):
        def newfunc(val, mask, mval):
            if mask:
                return mval
            else:
                return func(val)
        return newfunc

    if r.ndim != 1:
        raise ValueError('rec2csv only operates on 1 dimensional recarrays')

    formatd = get_formatd(r, formatd)
    funcs = []
    for i, name in enumerate(r.dtype.names):
        funcs.append(with_mask(csvformat_factory(formatd[name]).tostr))

    fh, opened = cbook.to_filehandle(fname, 'wb', return_opened=True)
    writer = csv.writer(fh, delimiter=delimiter)
    header = r.dtype.names
    if withheader:
        writer.writerow(header)

    # Our list of specials for missing values
    mvals = []
    for name in header:
        mvals.append(missingd.get(name, missing))

    ismasked = False
    if len(r):
        row = r[0]
        ismasked = hasattr(row, '_fieldmask')

    for row in r:
        if ismasked:
            row, rowmask = row.item(), row._fieldmask.item()
        else:
            rowmask = [False] * len(row)
        writer.writerow([func(val, mask, mval) for func, val, mask, mval
                         in zip(funcs, row, rowmask, mvals)])
    if opened:
        fh.close()


class GaussianKDE(object):
    """
    Representation of a kernel-density estimate using Gaussian kernels.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).

    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a
        scalar, this will be used directly as `kde.factor`.  If a
        callable, it should take a `GaussianKDE` instance as only
        parameter and return a scalar. If None (default), 'scott' is used.

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.

    dim : int
        Number of dimensions.

    num_dp : int
        Number of datapoints.

    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.

    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).

    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.

    kde(points) : ndarray
        Same as kde.evaluate(points)

    """

    # This implementation with minor modification was too good to pass up.
    # from scipy: https://github.com/scipy/scipy/blob/master/scipy/stats/kde.py

    def __init__(self, dataset, bw_method=None):
        self.dataset = np.atleast_2d(dataset)
        if not np.array(self.dataset).size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.dim, self.num_dp = np.array(self.dataset).shape
        isString = isinstance(bw_method, str)

        if bw_method is None:
            pass
        elif (isString and bw_method == 'scott'):
            self.covariance_factor = self.scotts_factor
        elif (isString and bw_method == 'silverman'):
            self.covariance_factor = self.silverman_factor
        elif (np.isscalar(bw_method) and not isString):
                self._bw_method = 'use constant'
                self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            raise ValueError("`bw_method` should be 'scott', 'silverman', a "
                             "scalar or a callable")

        # Computes the covariance matrix for each Gaussian kernel using
        # covariance_factor().

        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self.data_covariance = np.atleast_2d(
                np.cov(
                    self.dataset,
                    rowvar=1,
                    bias=False))
            self.data_inv_cov = np.linalg.inv(self.data_covariance)

        self.covariance = self.data_covariance * self.factor ** 2
        self.inv_cov = self.data_inv_cov / self.factor ** 2
        self.norm_factor = np.sqrt(
            np.linalg.det(
                2 * np.pi * self.covariance)) * self.num_dp

    def scotts_factor(self):
        return np.power(self.num_dp, -1. / (self.dim + 4))

    def silverman_factor(self):
        return np.power(
            self.num_dp * (self.dim + 2.0) / 4.0, -1. / (self.dim + 4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different
                     than the dimensionality of the KDE.

        """
        points = np.atleast_2d(points)

        dim, num_m = np.array(points).shape
        if dim != self.dim:
            raise ValueError("points have dimension {}, dataset has dimension "
                             "{}".format(dim, self.dim))

        result = np.zeros((num_m,), dtype=float)

        if num_m >= self.num_dp:
            # there are more points than data, so loop over data
            for i in range(self.num_dp):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result = result + np.exp(-energy)
        else:
            # loop over points
            for i in range(num_m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result[i] = np.sum(np.exp(-energy), axis=0)

        result = result / self.norm_factor

        return result

    __call__ = evaluate


##################################################
# Code related to things in and around polygons
##################################################
@cbook.deprecated("2.2")
def inside_poly(points, verts):
    """
    *points* is a sequence of *x*, *y* points.
    *verts* is a sequence of *x*, *y* vertices of a polygon.

    Return value is a sequence of indices into points for the points
    that are inside the polygon.
    """
    # Make a closed polygon path
    poly = Path(verts)

    # Check to see which points are contained within the Path
    return [idx for idx, p in enumerate(points) if poly.contains_point(p)]


@cbook.deprecated("2.2")
def poly_below(xmin, xs, ys):
    """
    Given a sequence of *xs* and *ys*, return the vertices of a
    polygon that has a horizontal base at *xmin* and an upper bound at
    the *ys*.  *xmin* is a scalar.

    Intended for use with :meth:`matplotlib.axes.Axes.fill`, e.g.,::

      xv, yv = poly_below(0, x, y)
      ax.fill(xv, yv)
    """
    if any(isinstance(var, np.ma.MaskedArray) for var in [xs, ys]):
        numpy = np.ma
    else:
        numpy = np

    xs = numpy.asarray(xs)
    ys = numpy.asarray(ys)
    Nx = len(xs)
    Ny = len(ys)
    if Nx != Ny:
        raise ValueError("'xs' and 'ys' must have the same length")
    x = xmin*numpy.ones(2*Nx)
    y = numpy.ones(2*Nx)
    x[:Nx] = xs
    y[:Nx] = ys
    y[Nx:] = ys[::-1]
    return x, y


@cbook.deprecated("2.2")
def poly_between(x, ylower, yupper):
    """
    Given a sequence of *x*, *ylower* and *yupper*, return the polygon
    that fills the regions between them.  *ylower* or *yupper* can be
    scalar or iterable.  If they are iterable, they must be equal in
    length to *x*.

    Return value is *x*, *y* arrays for use with
    :meth:`matplotlib.axes.Axes.fill`.
    """
    if any(isinstance(var, np.ma.MaskedArray) for var in [ylower, yupper, x]):
        numpy = np.ma
    else:
        numpy = np

    Nx = len(x)
    if not np.iterable(ylower):
        ylower = ylower*numpy.ones(Nx)

    if not np.iterable(yupper):
        yupper = yupper*numpy.ones(Nx)

    x = numpy.concatenate((x, x[::-1]))
    y = numpy.concatenate((yupper, ylower[::-1]))
    return x, y


@cbook.deprecated('2.2')
def is_closed_polygon(X):
    """
    Tests whether first and last object in a sequence are the same.  These are
    presumably coordinates on a polygonal curve, in which case this function
    tests if that curve is closed.
    """
    return np.all(X[0] == X[-1])
