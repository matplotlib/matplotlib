"""

Numerical python functions written for compatability with MATLAB
commands with the same names.

MATLAB compatible functions
-------------------------------

:func:`cohere`
    Coherence (normalized cross spectral density)

:func:`csd`
    Cross spectral density uing Welch's average periodogram

:func:`detrend`
    Remove the mean or best fit line from an array

:func:`find`
    Return the indices where some condition is true;
    numpy.nonzero is similar but more general.

:func:`griddata`
    Interpolate irregularly distributed data to a
    regular grid.

:func:`prctile`
    Find the percentiles of a sequence

:func:`prepca`
    Principal Component Analysis

:func:`psd`
    Power spectral density uing Welch's average periodogram

:func:`rk4`
    A 4th order runge kutta integrator for 1D or ND systems

:func:`specgram`
    Spectrogram (spectrum over segments of time)

Miscellaneous functions
-------------------------

Functions that don't exist in MATLAB, but are useful anyway:

:func:`cohere_pairs`
    Coherence over all pairs.  This is not a MATLAB function, but we
    compute coherence a lot in my lab, and we compute it for a lot of
    pairs.  This function is optimized to do this efficiently by
    caching the direct FFTs.

:func:`rk4`
    A 4th order Runge-Kutta ODE integrator in case you ever find
    yourself stranded without scipy (and the far superior
    scipy.integrate tools)

:func:`contiguous_regions`
    Return the indices of the regions spanned by some logical mask

:func:`cross_from_below`
    Return the indices where a 1D array crosses a threshold from below

:func:`cross_from_above`
    Return the indices where a 1D array crosses a threshold from above

:func:`complex_spectrum`
    Return the complex-valued frequency spectrum of a signal

:func:`magnitude_spectrum`
    Return the magnitude of the frequency spectrum of a signal

:func:`angle_spectrum`
    Return the angle (wrapped phase) of the frequency spectrum of a signal

:func:`phase_spectrum`
    Return the phase (unwrapped angle) of the frequency spectrum of a signal

:func:`detrend_mean`
:func:`demean`
    Remove the mean from a line.  These functions differ in their defaults.

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


record array helper functions
-------------------------------

A collection of helper methods for numpyrecord arrays

.. _htmlonly:

    See :ref:`misc-examples-index`

:func:`rec2txt`
    Pretty print a record array

:func:`rec2csv`
    Store record array in CSV file

:func:`csv2rec`
    Import record array from CSV file with type inspection

:func:`rec_append_fields`
    Adds  field(s)/array(s) to record array

:func:`rec_drop_fields`
    Drop fields from record array

:func:`rec_join`
    Join two record arrays on sequence of fields

:func:`recs_join`
    A simple join of multiple recarrays using a single column as a key

:func:`rec_groupby`
    Summarize data by groups (similar to SQL GROUP BY)

:func:`rec_summarize`
    Helper code to filter rec array fields into new fields

For the rec viewer functions(e rec2csv), there are a bunch of Format
objects you can pass into the functions that will do things like color
negative values red, set percent formatting and scaling, etc.

Example usage::

    r = csv2rec('somefile.csv', checkrows=0)

    formatd = dict(
        weight = FormatFloat(2),
        change = FormatPercent(2),
        cost   = FormatThousands(2),
        )


    rec2excel(r, 'test.xls', formatd=formatd)
    rec2csv(r, 'test.csv', formatd=formatd)
    scroll = rec2gtk(r, formatd=formatd)

    win = gtk.Window()
    win.set_size_request(600,800)
    win.add(scroll)
    win.show_all()
    gtk.main()


Deprecated functions
---------------------

The following are deprecated; please import directly from numpy (with
care--function signatures may differ):


:func:`load`
    Load ASCII file - use numpy.loadtxt

:func:`save`
    Save ASCII file - use numpy.savetxt

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import map, xrange, zip
if six.PY3:
    long = int

import copy
import csv
import operator
import os
import warnings

import numpy as np
ma = np.ma
from matplotlib import verbose

import matplotlib.cbook as cbook
from matplotlib import docstring
from matplotlib.path import Path


def logspace(xmin, xmax, N):
    '''
    Return N values logarithmically spaced between xmin and xmax.

    Call signature::

        logspace(xmin, xmax, N)
    '''
    return np.exp(np.linspace(np.log(xmin), np.log(xmax), N))


def _norm(x):
    '''
    Return sqrt(x dot x).

    Call signature::

        _norm(x)
    '''
    return np.sqrt(np.dot(x, x))


def window_hanning(x):
    '''
    Return x times the hanning window of len(x).

    Call signature::

        window_hanning(x)

    .. seealso::

        :func:`window_none`
            :func:`window_none` is another window algorithm.
    '''
    return np.hanning(len(x))*x


def window_none(x):
    '''
    No window function; simply return x.

    Call signature::

        window_none(x)

    .. seealso::

        :func:`window_hanning`
            :func:`window_hanning` is another window algorithm.
    '''
    return x


def apply_window(x, window, axis=0, return_window=None):
    '''
    Apply the given window to the given 1D or 2D array along the given axis.

    Call signature::

        apply_window(x, window, axis=0, return_window=False)

      *x*: 1D or 2D array or sequence
        Array or sequence containing the data.

      *winodw*: function or array.
        Either a function to generate a window or an array with length
        *x*.shape[*axis*]

      *axis*: integer
        The axis over which to do the repetition.
        Must be 0 or 1.  The default is 0

      *return_window*: bool
        If true, also return the 1D values of the window that was applied
    '''
    x = np.asarray(x)

    if x.ndim < 1 or x.ndim > 2:
        raise ValueError('only 1D or 2D arrays can be used')
    if axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    xshape = list(x.shape)
    xshapetarg = xshape.pop(axis)

    if cbook.iterable(window):
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

    Call signature::

        detrend(x, key='mean')

      *x*: array or sequence
        Array or sequence containing the data.

      *key*: [ 'default' | 'constant' | 'mean' | 'linear' | 'none'] or function
          Specifies the detrend algorithm to use.  'default' is 'mean',
          which is the same as :func:`detrend_mean`.  'constant' is the same.
          'linear' is the same as :func:`detrend_linear`.  'none' is the same
          as :func:`detrend_none`.  The default is 'mean'.  See the
          corresponding functions for more details regarding the algorithms.
          Can also be a function that carries out the detrend operation.

      *axis*: integer
        The axis along which to do the detrending.

    .. seealso::

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
    elif cbook.is_string_like(key):
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


def demean(x, axis=0):
    '''
    Return x minus its mean along the specified axis.

    Call signature::

        demean(x, axis=0)

      *x*: array or sequence
        Array or sequence containing the data
        Can have any dimensionality

      *axis*: integer
        The axis along which to take the mean.  See numpy.mean for a
        description of this argument.

    .. seealso::

        :func:`delinear`
        :func:`denone`
            :func:`delinear` and :func:`denone` are other detrend algorithms.

        :func:`detrend_mean`
            This function is the same as as :func:`detrend_mean` except
            for the default *axis*.
    '''
    return detrend_mean(x, axis=axis)


def detrend_mean(x, axis=None):
    '''
    Return x minus the mean(x).

    Call signature::

        detrend_mean(x, axis=None)

      *x*: array or sequence
        Array or sequence containing the data
        Can have any dimensionality

      *axis*: integer
        The axis along which to take the mean.  See numpy.mean for a
        description of this argument.

    .. seealso::

        :func:`demean`
            This function is the same as as :func:`demean` except
            for the default *axis*.

        :func:`detrend_linear`
        :func:`detrend_none`
            :func:`detrend_linear` and :func:`detrend_none` are other
            detrend algorithms.

        :func:`detrend`
            :func:`detrend` is a wrapper around all the detrend algorithms.
    '''
    x = np.asarray(x)

    if axis is not None and axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    # short-circuit 0-D array.
    if not x.ndim:
        return np.array(0., dtype=x.dtype)

    # short-circuit simple operations
    if axis == 0 or axis is None or x.ndim <= 1:
        return x - x.mean(axis)

    ind = [slice(None)] * x.ndim
    ind[axis] = np.newaxis
    return x - x.mean(axis)[ind]


def detrend_none(x, axis=None):
    '''
    Return x: no detrending.

    Call signature::

        detrend_none(x, axis=None)

      *x*: any object
        An object containing the data

      *axis*: integer
        This parameter is ignored.
        It is included for compatibility with detrend_mean

    .. seealso::

        :func:`denone`
            This function is the same as as :func:`denone` except
            for the default *axis*, which has no effect.

        :func:`detrend_mean`
        :func:`detrend_linear`
            :func:`detrend_mean` and :func:`detrend_linear` are other
            detrend algorithms.

        :func:`detrend`
            :func:`detrend` is a wrapper around all the detrend algorithms.
    '''
    return x


def detrend_linear(y):
    '''
    Return x minus best fit line; 'linear' detrending.

    Call signature::

        detrend_linear(y)

      *y*: 0-D or 1-D array or sequence
        Array or sequence containing the data

      *axis*: integer
        The axis along which to take the mean.  See numpy.mean for a
        description of this argument.

    .. seealso::

        :func:`delinear`
            This function is the same as as :func:`delinear` except
            for the default *axis*.

        :func:`detrend_mean`
        :func:`detrend_none`
            :func:`detrend_mean` and :func:`detrend_none` are other
            detrend algorithms.

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

    x = np.arange(y.size, dtype=np.float_)

    C = np.cov(x, y, bias=1)
    b = C[0, 1]/C[0, 0]

    a = y.mean() - b*x.mean()
    return y - (b*x + a)


def stride_windows(x, n, noverlap=None, axis=0):
    '''
    Get all windows of x with length n as a single array,
    using strides to avoid data duplication.

    .. warning:: It is not safe to write to the output array.  Multiple
    elements may point to the same piece of memory, so modifying one value may
    change others.

    Call signature::

        stride_windows(x, n, noverlap=0)

      *x*: 1D array or sequence
        Array or sequence containing the data.

      *n*: integer
        The number of data points in each window.

      *noverlap*: integer
        The overlap between adjacent windows.
        Default is 0 (no overlap)

      *axis*: integer
        The axis along which the windows will run.

    Refs:
        `stackoverflaw: Rolling window for 1D arrays in Numpy?
        <http://stackoverflow.com/a/6811241>`_
        `stackoverflaw: Using strides for an efficient moving average filter
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

    step = n - noverlap
    if axis == 0:
        shape = (n, (x.shape[-1]-noverlap)//step)
        strides = (x.itemsize, step*x.itemsize)
    else:
        shape = ((x.shape[-1]-noverlap)//step, n)
        strides = (step*x.itemsize, x.itemsize)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def stride_repeat(x, n, axis=0):
    '''
    Repeat the values in an array in a memory-efficient manner.  Array x is
    stacked vertically n times.

    .. warning:: It is not safe to write to the output array.  Multiple
    elements may point to the same piece of memory, so modifying one value may
    change others.

    Call signature::

        stride_repeat(x, n, axis=0)

      *x*: 1D array or sequence
        Array or sequence containing the data.

      *n*: integer
        The number of time to repeat the array.

      *axis*: integer
        The axis along which the data will run.

    Refs:
        `stackoverflaw: Repeat NumPy array without replicating data?
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

    if axis == 0:
        shape = (n, x.size)
        strides = (0, x.itemsize)
    else:
        shape = (x.size, n)
        strides = (x.itemsize, 0)

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
        #The checks for if y is x are so that we can use the same function to
        #implement the core of psd(), csd(), and spectrogram() without doing
        #extra calculations.  We return the unaveraged Pxy, freqs, and t.
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

    #Make sure we're dealing with a numpy array. If y and x were the same
    #object to start with, keep them that way
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
        resultY = apply_window(resultY, window, axis=0)
        resultY = detrend(resultY, detrend_func, axis=0)
        resultY = np.fft.fft(resultY, n=pad_to, axis=0)[:numFreqs, :]
        result = np.conjugate(result) * resultY
    elif mode == 'psd':
        result = np.conjugate(result) * result
    elif mode == 'magnitude':
        result = np.absolute(result)
    elif mode == 'angle' or mode == 'phase':
        # we unwrap the phase later to handle the onesided vs. twosided case
        result = np.angle(result)
    elif mode == 'complex':
        pass

    if mode == 'psd':
        # Scale the spectrum by the norm of the window to compensate for
        # windowing loss; see Bendat & Piersol Sec 11.5.2.
        result /= (np.abs(windowVals)**2).sum()

        # Also include scaling factors for one-sided densities and dividing by
        # the sampling frequency, if desired. Scale everything, except the DC
        # component and the NFFT/2 component:
        result[1:-1] *= scaling_factor

        # MATLAB divides by the sampling frequency so that density function
        # has units of dB/Hz and can be integrated by the plotted frequency
        # values. Perform the same scaling here.
        if scale_by_freq:
            result /= Fs

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

    if len(spec.shape) == 2 and spec.shape[1] == 1:
        spec = spec[:, 0]

    return spec, freqs


#Split out these keyword docs so that they can be used elsewhere
docstring.interpd.update(Spectral=cbook.dedent("""
    Keyword arguments:

      *Fs*: scalar
          The sampling frequency (samples per time unit).  It is used
          to calculate the Fourier frequencies, freqs, in cycles per time
          unit. The default value is 2.

      *window*: callable or ndarray
          A function or a vector of length *NFFT*. To create window
          vectors see :func:`window_hanning`, :func:`window_none`,
          :func:`numpy.blackman`, :func:`numpy.hamming`,
          :func:`numpy.bartlett`, :func:`scipy.signal`,
          :func:`scipy.signal.get_window`, etc. The default is
          :func:`window_hanning`.  If a function is passed as the
          argument, it must take a data segment as an argument and
          return the windowed version of the segment.

      *sides*: [ 'default' | 'onesided' | 'twosided' ]
          Specifies which sides of the spectrum to return.  Default gives the
          default behavior, which returns one-sided for real data and both
          for complex data.  'onesided' forces the return of a one-sided
          spectrum, while 'twosided' forces two-sided.
"""))


docstring.interpd.update(Single_Spectrum=cbook.dedent("""
      *pad_to*: integer
          The number of points to which the data segment is padded when
          performing the FFT.  While not increasing the actual resolution of
          the spectrum (the minimum distance between resolvable peaks),
          this can give more points in the plot, allowing for more
          detail. This corresponds to the *n* parameter in the call to fft().
          The default is None, which sets *pad_to* equal to the length of the
          input signal (i.e. no padding).
"""))


docstring.interpd.update(PSD=cbook.dedent("""
      *pad_to*: integer
          The number of points to which the data segment is padded when
          performing the FFT.  This can be different from *NFFT*, which
          specifies the number of data points used.  While not increasing
          the actual resolution of the spectrum (the minimum distance between
          resolvable peaks), this can give more points in the plot,
          allowing for more detail. This corresponds to the *n* parameter
          in the call to fft(). The default is None, which sets *pad_to*
          equal to *NFFT*

      *NFFT*: integer
          The number of data points used in each block for the FFT.
          A power 2 is most efficient.  The default value is 256.
          This should *NOT* be used to get zero padding, or the scaling of the
          result will be incorrect. Use *pad_to* for this instead.

      *detrend*: [ 'default' | 'constant' | 'mean' | 'linear' | 'none'] or
                 callable
          The function applied to each segment before fft-ing,
          designed to remove the mean or linear trend.  Unlike in
          MATLAB, where the *detrend* parameter is a vector, in
          matplotlib is it a function.  The :mod:`~matplotlib.pylab`
          module defines :func:`~matplotlib.pylab.detrend_none`,
          :func:`~matplotlib.pylab.detrend_mean`, and
          :func:`~matplotlib.pylab.detrend_linear`, but you can use
          a custom function as well.  You can also use a string to choose
          one of the functions.  'default', 'constant', and 'mean' call
          :func:`~matplotlib.pylab.detrend_mean`.  'linear' calls
          :func:`~matplotlib.pylab.detrend_linear`.  'none' calls
          :func:`~matplotlib.pylab.detrend_none`.

      *scale_by_freq*: boolean
          Specifies whether the resulting density values should be scaled
          by the scaling frequency, which gives density in units of Hz^-1.
          This allows for integration over the returned frequency values.
          The default is True for MATLAB compatibility.
"""))


@docstring.dedent_interpd
def psd(x, NFFT=None, Fs=None, detrend=None, window=None,
        noverlap=None, pad_to=None, sides=None, scale_by_freq=None):
    """
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

      *x*: 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(PSD)s

        *noverlap*: integer
        The number of points of overlap between segments.
        The default value is 0 (no overlap).

    Returns the tuple (*Pxx*, *freqs*).

          *Pxx*: 1-D array
            The values for the power spectrum `P_{xx}` (real valued)

          *freqs*: 1-D array
            The frequencies corresponding to the elements in *Pxx*

    Refs:

        Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    .. seealso::

        :func:`specgram`
            :func:`specgram` differs in the default overlap; in not returning
            the mean of the segment periodograms; and in returning the
            times of the segments.

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

      *x*, *y*: 1-D arrays or sequences
        Arrays or sequences containing the data

    %(Spectral)s

    %(PSD)s

      *noverlap*: integer
          The number of points of overlap between segments.
          The default value is 0 (no overlap).

    Returns the tuple (*Pxy*, *freqs*):

          *Pxy*: 1-D array
            The values for the cross spectrum `P_{xy}` before scaling
            (real valued)

          *freqs*: 1-D array
            The frequencies corresponding to the elements in *Pxy*

    Refs:
        Bendat & Piersol -- Random Data: Analysis and Measurement
        Procedures, John Wiley & Sons (1986)

    .. seealso::

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

    if len(Pxy.shape) == 2:
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

      *x*: 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(Single_Spectrum)s

    Returns the tuple (*spectrum*, *freqs*):

      *spectrum*: 1-D array
        The values for the complex spectrum (complex valued)

      *freqs*: 1-D array
        The frequencies corresponding to the elements in *spectrum*

    .. seealso::

        :func:`magnitude_spectrum`
            :func:`magnitude_spectrum` returns the absolute value of this
            function.

        :func:`angle_spectrum`
            :func:`angle_spectrum` returns the angle of this
            function.

        :func:`phase_spectrum`
            :func:`phase_spectrum` returns the phase (unwrapped angle) of this
            function.

        :func:`specgram`
            :func:`specgram` can return the complex spectrum of segments
            within the signal.
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

      *x*: 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(Single_Spectrum)s

    Returns the tuple (*spectrum*, *freqs*):

      *spectrum*: 1-D array
        The values for the magnitude spectrum (real valued)

      *freqs*: 1-D array
        The frequencies corresponding to the elements in *spectrum*

    .. seealso::

        :func:`psd`
            :func:`psd` returns the power spectral density.

        :func:`complex_spectrum`
            This function returns the absolute value of
            :func:`complex_spectrum`.

        :func:`angle_spectrum`
            :func:`angle_spectrum` returns the angles of the corresponding
            frequencies.

        :func:`phase_spectrum`
            :func:`phase_spectrum` returns the phase (unwrapped angle) of the
            corresponding frequencies.

        :func:`specgram`
            :func:`specgram` can return the magnitude spectrum of segments
            within the signal.
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

      *x*: 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(Single_Spectrum)s

    Returns the tuple (*spectrum*, *freqs*):

      *spectrum*: 1-D array
        The values for the angle spectrum in radians (real valued)

      *freqs*: 1-D array
        The frequencies corresponding to the elements in *spectrum*

    .. seealso::

        :func:`complex_spectrum`
            This function returns the angle value of
            :func:`complex_spectrum`.

        :func:`magnitude_spectrum`
            :func:`angle_spectrum` returns the magnitudes of the
            corresponding frequencies.

        :func:`phase_spectrum`
            :func:`phase_spectrum` returns the unwrapped version of this
            function.

        :func:`specgram`
            :func:`specgram` can return the angle spectrum of segments
            within the signal.
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

      *x*: 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(Single_Spectrum)s

    Returns the tuple (*spectrum*, *freqs*):

      *spectrum*: 1-D array
        The values for the phase spectrum in radians (real valued)

      *freqs*: 1-D array
        The frequencies corresponding to the elements in *spectrum*

    .. seealso::

        :func:`complex_spectrum`
            This function returns the angle value of
            :func:`complex_spectrum`.

        :func:`magnitude_spectrum`
            :func:`magnitude_spectrum` returns the magnitudes of the
            corresponding frequencies.

        :func:`angle_spectrum`
            :func:`angle_spectrum` returns the wrapped version of this
            function.

        :func:`specgram`
            :func:`specgram` can return the phase spectrum of segments
            within the signal.
    """
    return _single_spectrum_helper(x=x, Fs=Fs, window=window, pad_to=pad_to,
                                   sides=sides, mode='phase')


@docstring.dedent_interpd
def specgram(x, NFFT=None, Fs=None, detrend=None, window=None,
             noverlap=None, pad_to=None, sides=None, scale_by_freq=None,
             mode=None):
    """
    Compute a spectrogram.

    Call signature::

        specgram(x, NFFT=256, Fs=2,detrend=mlab.detrend_none,
                window=mlab.window_hanning, noverlap=128,
                cmap=None, xextent=None, pad_to=None, sides='default',
                scale_by_freq=None, mode='default')

    Compute and plot a spectrogram of data in *x*.  Data are split into
    *NFFT* length segments and the spectrum of each section is
    computed.  The windowing function *window* is applied to each
    segment, and the amount of overlap of each segment is
    specified with *noverlap*.

      *x*: 1-D array or sequence
        Array or sequence containing the data

    %(Spectral)s

    %(PSD)s

      *mode*: [ 'default' | 'psd' | 'complex' | 'magnitude'
                'angle' | 'phase' ]
          What sort of spectrum to use.  Default is 'psd'. which takes the
          power spectral density.  'complex' returns the complex-valued
          frequency spectrum.  'magnitude' returns the magnitude spectrum.
          'angle' returns the phase spectrum without unwrapping.  'phase'
          returns the phase spectrum with unwrapping.

      *noverlap*: integer
          The number of points of overlap between blocks.  The default value
          is 128.

    Returns the tuple (*spectrum*, *freqs*, *t*):

      *spectrum*: 2-D array
        columns are the periodograms of successive segments

      *freqs*: 1-D array
        The frequencies corresponding to the rows in *spectrum*

      *t*: 1-D array
        The times corresponding to midpoints of segments (i.e the columns
        in *spectrum*).

    .. note::

        *detrend* and *scale_by_freq* only apply when *mode* is set to
        'psd'

    .. seealso::

        :func:`psd`
            :func:`psd` differs in the default overlap; in returning
            the mean of the segment periodograms; and in not returning
            times.

        :func:`complex_spectrum`
            A single spectrum, similar to having a single segment when
            *mode* is 'complex'.

        :func:`magnitude_spectrum`
            A single spectrum, similar to having a single segment when
            *mode* is 'magnitude'.

        :func:`angle_spectrum`
            A single spectrum, similar to having a single segment when
            *mode* is 'angle'.

        :func:`phase_spectrum`
            A single spectrum, similar to having a single segment when
            *mode* is 'phase'.
    """
    if noverlap is None:
        noverlap = 128

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

    *x*, *y*
        Array or sequence containing the data

    %(Spectral)s

    %(PSD)s

      *noverlap*: integer
          The number of points of overlap between blocks.  The default value
          is 0 (no overlap).

    The return value is the tuple (*Cxy*, *f*), where *f* are the
    frequencies of the coherence vector. For cohere, scaling the
    individual densities by the sampling frequency has no effect,
    since the factors cancel out.

    .. seealso::

        :func:`psd` and :func:`csd`
            For information about the methods used to compute
            :math:`P_{xy}`, :math:`P_{xx}` and :math:`P_{yy}`.
    """

    if len(x) < 2 * NFFT:
        raise ValueError(_coh_error)
    Pxx, f = psd(x, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    Pyy, f = psd(y, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)
    Pxy, f = csd(x, y, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
                 scale_by_freq)

    Cxy = np.divide(np.absolute(Pxy)**2, Pxx*Pyy)
    Cxy.shape = (len(f),)
    return Cxy, f


def donothing_callback(*args):
    pass


def cohere_pairs( X, ij, NFFT=256, Fs=2, detrend=detrend_none,
                  window=window_hanning, noverlap=0,
                  preferSpeedOverMemory=True,
                  progressCallback=donothing_callback,
                  returnPxx=False):

    """
    Call signature::

      Cxy, Phase, freqs = cohere_pairs( X, ij, ...)

    Compute the coherence and phase for all pairs *ij*, in *X*.

    *X* is a *numSamples* * *numCols* array

    *ij* is a list of tuples.  Each tuple is a pair of indexes into
    the columns of X for which you want to compute coherence.  For
    example, if *X* has 64 columns, and you want to compute all
    nonredundant pairs, define *ij* as::

      ij = []
      for i in range(64):
          for j in range(i+1,64):
              ij.append( (i,j) )

    *preferSpeedOverMemory* is an optional bool. Defaults to true. If
    False, limits the caching by only making one, rather than two,
    complex cache arrays. This is useful if memory becomes critical.
    Even when *preferSpeedOverMemory* is False, :func:`cohere_pairs`
    will still give significant performace gains over calling
    :func:`cohere` for each pair, and will use subtantially less
    memory than if *preferSpeedOverMemory* is True.  In my tests with
    a 43000,64 array over all nonredundant pairs,
    *preferSpeedOverMemory* = True delivered a 33% performance boost
    on a 1.7GHZ Athlon with 512MB RAM compared with
    *preferSpeedOverMemory* = False.  But both solutions were more
    than 10x faster than naively crunching all possible pairs through
    :func:`cohere`.

    Returns::

       (Cxy, Phase, freqs)

    where:

      - *Cxy*: dictionary of (*i*, *j*) tuples -> coherence vector for
        that pair.  I.e., ``Cxy[(i,j) = cohere(X[:,i], X[:,j])``.
        Number of dictionary keys is ``len(ij)``.

      - *Phase*: dictionary of phases of the cross spectral density at
        each frequency for each pair.  Keys are (*i*, *j*).

      - *freqs*: vector of frequencies, equal in length to either the
         coherence or phase vectors for any (*i*, *j*) key.

    e.g., to make a coherence Bode plot::

          subplot(211)
          plot( freqs, Cxy[(12,19)])
          subplot(212)
          plot( freqs, Phase[(12,19)])

    For a large number of pairs, :func:`cohere_pairs` can be much more
    efficient than just calling :func:`cohere` for each pair, because
    it caches most of the intensive computations.  If :math:`N` is the
    number of pairs, this function is :math:`O(N)` for most of the
    heavy lifting, whereas calling cohere for each pair is
    :math:`O(N^2)`.  However, because of the caching, it is also more
    memory intensive, making 2 additional complex arrays with
    approximately the same number of elements as *X*.

    See :file:`test/cohere_pairs_test.py` in the src tree for an
    example script that shows that this :func:`cohere_pairs` and
    :func:`cohere` give the same results for a given pair.

    .. seealso::

        :func:`psd`
            For information about the methods used to compute
            :math:`P_{xy}`, :math:`P_{xx}` and :math:`P_{yy}`.
    """
    numRows, numCols = X.shape

    # zero pad if X is too short
    if numRows < NFFT:
        tmp = X
        X = np.zeros( (NFFT, numCols), X.dtype)
        X[:numRows,:] = tmp
        del tmp

    numRows, numCols = X.shape
    # get all the columns of X that we are interested in by checking
    # the ij tuples
    allColumns = set()
    for i,j in ij:
        allColumns.add(i); allColumns.add(j)
    Ncols = len(allColumns)

    # for real X, ignore the negative frequencies
    if np.iscomplexobj(X): numFreqs = NFFT
    else: numFreqs = NFFT//2+1

    # cache the FFT of every windowed, detrended NFFT length segement
    # of every channel.  If preferSpeedOverMemory, cache the conjugate
    # as well
    if cbook.iterable(window):
        assert(len(window) == NFFT)
        windowVals = window
    else:
        windowVals = window(np.ones(NFFT, X.dtype))
    ind = list(xrange(0, numRows-NFFT+1, NFFT-noverlap))
    numSlices = len(ind)
    FFTSlices = {}
    FFTConjSlices = {}
    Pxx = {}
    slices = range(numSlices)
    normVal = np.linalg.norm(windowVals)**2
    for iCol in allColumns:
        progressCallback(i/Ncols, 'Cacheing FFTs')
        Slices = np.zeros( (numSlices,numFreqs), dtype=np.complex_)
        for iSlice in slices:
            thisSlice = X[ind[iSlice]:ind[iSlice]+NFFT, iCol]
            thisSlice = windowVals*detrend(thisSlice)
            Slices[iSlice,:] = np.fft.fft(thisSlice)[:numFreqs]

        FFTSlices[iCol] = Slices
        if preferSpeedOverMemory:
            FFTConjSlices[iCol] = np.conjugate(Slices)
        Pxx[iCol] = np.divide(np.mean(abs(Slices)**2, axis=0), normVal)
    del Slices, ind, windowVals

    # compute the coherences and phases for all pairs using the
    # cached FFTs
    Cxy = {}
    Phase = {}
    count = 0
    N = len(ij)
    for i,j in ij:
        count +=1
        if count%10==0:
            progressCallback(count/N, 'Computing coherences')

        if preferSpeedOverMemory:
            Pxy = FFTSlices[i] * FFTConjSlices[j]
        else:
            Pxy = FFTSlices[i] * np.conjugate(FFTSlices[j])
        if numSlices>1: Pxy = np.mean(Pxy, axis=0)
        #Pxy = np.divide(Pxy, normVal)
        Pxy /= normVal
        #Cxy[(i,j)] = np.divide(np.absolute(Pxy)**2, Pxx[i]*Pxx[j])
        Cxy[i,j] = abs(Pxy)**2 / (Pxx[i]*Pxx[j])
        Phase[i,j] =  np.arctan2(Pxy.imag, Pxy.real)

    freqs = Fs/NFFT*np.arange(numFreqs)
    if returnPxx:
        return Cxy, Phase, freqs, Pxx
    else:
        return Cxy, Phase, freqs

def entropy(y, bins):
    r"""
    Return the entropy of the data in *y*.

    .. math::

      \sum p_i \log_2(p_i)

    where :math:`p_i` is the probability of observing *y* in the
    :math:`i^{th}` bin of *bins*.  *bins* can be a number of bins or a
    range of bins; see :func:`numpy.histogram`.

    Compare *S* with analytic calculation for a Gaussian::

      x = mu + sigma * randn(200000)
      Sanalytic = 0.5 * ( 1.0 + log(2*pi*sigma**2.0) )
    """
    n, bins = np.histogram(y, bins)
    n = n.astype(np.float_)

    n = np.take(n, np.nonzero(n)[0])         # get the positive

    p = np.divide(n, len(y))

    delta = bins[1] - bins[0]
    S = -1.0 * np.sum(p * np.log(p)) + np.log(delta)
    return S

def normpdf(x, *args):
    """
    Return the normal pdf evaluated at array-like *x*;
    args provides *mu*, *sigma*"
    where *mu* is the mean or expectation of the distribution and
    *sigma is the standard deviation.
    """
    mu, sigma = args
    return 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5 * (1./sigma*(x - mu))**2)


def levypdf(x, gamma, alpha):
    """
    Return the levy pdf evaluated at *x* for params *gamma*, *alpha*.
    *x* has to be of even length.
    """
    N = len(x)

    if N % 2 != 0:
        raise ValueError('x must be an even length array; try\n' + \
              'x = np.linspace(minx, maxx, N), where N is even')

    dx = x[1] - x[0]

    f = 1/(N*dx)*np.arange(-N / 2, N / 2, dtype=np.float_)

    ind = np.concatenate([np.arange(N // 2, N),
                          np.arange(0, N // 2)])
    df = f[1] - f[0]
    cfl = np.exp(-gamma * np.absolute(2 * np.pi * f) ** alpha)

    px = np.fft.fft(np.take(cfl, ind) * df).astype(np.float_)
    return np.take(px, ind)


def find(condition):
    "Return the indices where ravel(condition) is true"
    res, = np.nonzero(np.ravel(condition))
    return res


def longest_contiguous_ones(x):
    """
    Return the indices of the longest stretch of contiguous ones in *x*,
    assuming *x* is a vector of zeros and ones.  If there are two
    equally long stretches, pick the first.
    """
    x = np.ravel(x)
    if len(x)==0:
        return np.array([])

    ind = (x==0).nonzero()[0]
    if len(ind)==0:
        return np.arange(len(x))
    if len(ind)==len(x):
        return np.array([])

    y = np.zeros( (len(x)+2,), x.dtype)
    y[1:-1] = x
    dif = np.diff(y)
    up = (dif ==  1).nonzero()[0];
    dn = (dif == -1).nonzero()[0];
    i = (dn-up == max(dn - up)).nonzero()[0][0]
    ind = np.arange(up[i], dn[i])

    return ind

def longest_ones(x):
    '''alias for longest_contiguous_ones'''
    return longest_contiguous_ones(x)

def prepca(P, frac=0):
    """

    WARNING: this function is deprecated -- please see class PCA instead

    Compute the principal components of *P*.  *P* is a (*numVars*,
    *numObs*) array.  *frac* is the minimum fraction of variance that a
    component must contain to be included.

    Return value is a tuple of the form (*Pcomponents*, *Trans*,
    *fracVar*) where:

      - *Pcomponents* : a (numVars, numObs) array

      - *Trans* : the weights matrix, ie, *Pcomponents* = *Trans* *
         *P*

      - *fracVar* : the fraction of the variance accounted for by each
         component returned

    A similar function of the same name was in the MATLAB
    R13 Neural Network Toolbox but is not found in later versions;
    its successor seems to be called "processpcs".
    """
    warnings.warn('This function is deprecated -- see class PCA instead')
    U,s,v = np.linalg.svd(P)
    varEach = s**2/P.shape[1]
    totVar = varEach.sum()
    fracVar = varEach/totVar
    ind = slice((fracVar>=frac).sum())
    # select the components that are greater
    Trans = U[:,ind].transpose()
    # The transformed data
    Pcomponents = np.dot(Trans,P)
    return Pcomponents, Trans, fracVar[ind]


class PCA:
    def __init__(self, a, standardize=True):
        """
        compute the SVD of a and store data for PCA.  Use project to
        project the data onto a reduced set of dimensions

        Inputs:

          *a*: a numobservations x numdims array
          *standardize*: True if input data are to be standardized. If False, only centering will be
          carried out.

        Attrs:

          *a* a centered unit sigma version of input a

          *numrows*, *numcols*: the dimensions of a

          *mu* : a numdims array of means of a. This is the vector that points to the 
          origin of PCA space. 

          *sigma* : a numdims array of standard deviation of a

          *fracs* : the proportion of variance of each of the principal components
        
          *s* : the actual eigenvalues of the decomposition

          *Wt* : the weight vector for projecting a numdims point or array into PCA space

          *Y* : a projected into PCA space


        The factor loadings are in the Wt factor, ie the factor
        loadings for the 1st principal component are given by Wt[0].
        This row is also the 1st eigenvector.

        """
        n, m = a.shape
        if n<m:
            raise RuntimeError('we assume data in a is organized with numrows>numcols')

        self.numrows, self.numcols = n, m
        self.mu = a.mean(axis=0)
        self.sigma = a.std(axis=0)
        self.standardize = standardize

        a = self.center(a)

        self.a = a

        U, s, Vh = np.linalg.svd(a, full_matrices=False)

        # Note: .H indicates the conjugate transposed / Hermitian.
        
        # The SVD is commonly written as a = U s V.H.
        # If U is a unitary matrix, it means that it satisfies U.H = inv(U).
        
        # The rows of Vh are the eigenvectors of a.H a.
        # The columns of U are the eigenvectors of a a.H. 
        # For row i in Vh and column i in U, the corresponding eigenvalue is s[i]**2.
         
        self.Wt = Vh
        
        # save the transposed coordinates
        Y = np.dot(Vh, a.T).T
        self.Y = Y
        
        # save the eigenvalues
        self.s = s**2
        
        # and now the contribution of the individual components
        vars = self.s/float(len(s))
        self.fracs = vars/vars.sum()


    def project(self, x, minfrac=0.):
        'project x onto the principle axes, dropping any axes where fraction of variance<minfrac'
        x = np.asarray(x)

        ndims = len(x.shape)

        if (x.shape[-1]!=self.numcols):
            raise ValueError('Expected an array with dims[-1]==%d'%self.numcols)


        Y = np.dot(self.Wt, self.center(x).T).T
        mask = self.fracs>=minfrac
        if ndims==2:
            Yreduced = Y[:,mask]
        else:
            Yreduced = Y[mask]
        return Yreduced



    def center(self, x):
        'center and optionally standardize the data using the mean and sigma from training set a'
        if self.standardize:
            return (x - self.mu)/self.sigma
        else:
            return (x - self.mu)



    @staticmethod
    def _get_colinear():
        c0 = np.array([
            0.19294738,  0.6202667 ,  0.45962655,  0.07608613,  0.135818  ,
            0.83580842,  0.07218851,  0.48318321,  0.84472463,  0.18348462,
            0.81585306,  0.96923926,  0.12835919,  0.35075355,  0.15807861,
            0.837437  ,  0.10824303,  0.1723387 ,  0.43926494,  0.83705486])

        c1 = np.array([
            -1.17705601, -0.513883  , -0.26614584,  0.88067144,  1.00474954,
            -1.1616545 ,  0.0266109 ,  0.38227157,  1.80489433,  0.21472396,
            -1.41920399, -2.08158544, -0.10559009,  1.68999268,  0.34847107,
            -0.4685737 ,  1.23980423, -0.14638744, -0.35907697,  0.22442616])

        c2 = c0 + 2*c1
        c3 = -3*c0 + 4*c1
        a = np.array([c3, c0, c1, c2]).T
        return a

def prctile(x, p = (0.0, 25.0, 50.0, 75.0, 100.0)):
    """
    Return the percentiles of *x*.  *p* can either be a sequence of
    percentile values or a scalar.  If *p* is a sequence, the ith
    element of the return sequence is the *p*(i)-th percentile of *x*.
    If *p* is a scalar, the largest value of *x* less than or equal to
    the *p* percentage point in the sequence is returned.
    """

    # This implementation derived from scipy.stats.scoreatpercentile
    def _interpolate(a, b, fraction):
        """Returns the point at the given fraction between a and b, where
        'fraction' must be between 0 and 1.
        """
        return a + (b - a)*fraction

    scalar = True
    if cbook.iterable(p):
        scalar = False
    per = np.array(p)
    values = np.array(x).ravel()  # copy
    values.sort()

    idxs = per /100. * (values.shape[0] - 1)
    ai = idxs.astype(np.int)
    bi = ai + 1
    frac = idxs % 1

    # handle cases where attempting to interpolate past last index
    cond = bi >= len(values)
    if scalar:
        if cond:
            ai -= 1
            bi -= 1
            frac += 1
    else:
        ai[cond] -= 1
        bi[cond] -= 1
        frac[cond] += 1

    return _interpolate(values[ai],values[bi],frac)

def prctile_rank(x, p):
    """
    Return the rank for each element in *x*, return the rank
    0..len(*p*).  e.g., if *p* = (25, 50, 75), the return value will be a
    len(*x*) array with values in [0,1,2,3] where 0 indicates the
    value is less than the 25th percentile, 1 indicates the value is
    >= the 25th and < 50th percentile, ... and 3 indicates the value
    is above the 75th percentile cutoff.

    *p* is either an array of percentiles in [0..100] or a scalar which
    indicates how many quantiles of data you want ranked.
    """

    if not cbook.iterable(p):
        p = np.arange(100.0/p, 100.0, 100.0/p)
    else:
        p = np.asarray(p)

    if p.max()<=1 or p.min()<0 or p.max()>100:
        raise ValueError('percentiles should be in range 0..100, not 0..1')

    ptiles = prctile(x, p)
    return np.searchsorted(ptiles, x)

def center_matrix(M, dim=0):
    """
    Return the matrix *M* with each row having zero mean and unit std.

    If *dim* = 1 operate on columns instead of rows.  (*dim* is
    opposite to the numpy axis kwarg.)
    """
    M = np.asarray(M, np.float_)
    if dim:
        M = (M - M.mean(axis=0)) / M.std(axis=0)
    else:
        M = (M - M.mean(axis=1)[:,np.newaxis])
        M = M / M.std(axis=1)[:,np.newaxis]
    return M



def rk4(derivs, y0, t):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.

    *y0*
        initial state vector

    *t*
        sample times

    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``


    Example 1 ::

        ## 2D system

        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2::

        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)

        y0 = 1
        yout = rk4(derivs, y0, t)


    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try: Ny = len(y0)
    except TypeError:
        yout = np.zeros( (len(t),), np.float_)
    else:
        yout = np.zeros( (len(t), Ny), np.float_)


    yout[0] = y0
    i = 0

    for i in np.arange(len(t)-1):

        thist = t[i]
        dt = t[i+1] - thist
        dt2 = dt/2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist))
        k2 = np.asarray(derivs(y0 + dt2*k1, thist+dt2))
        k3 = np.asarray(derivs(y0 + dt2*k2, thist+dt2))
        k4 = np.asarray(derivs(y0 + dt*k3, thist+dt))
        yout[i+1] = y0 + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
    return yout


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.

    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp( -z/(2*(1-rho**2))) / denom

def get_xyz_where(Z, Cond):
    """
    *Z* and *Cond* are *M* x *N* matrices.  *Z* are data and *Cond* is
    a boolean matrix where some condition is satisfied.  Return value
    is (*x*, *y*, *z*) where *x* and *y* are the indices into *Z* and
    *z* are the values of *Z* at those indices.  *x*, *y*, and *z* are
    1D arrays.
    """
    X,Y = np.indices(Z.shape)
    return X[Cond], Y[Cond], Z[Cond]

def get_sparse_matrix(M,N,frac=0.1):
    """
    Return a *M* x *N* sparse matrix with *frac* elements randomly
    filled.
    """
    data = np.zeros((M,N))*0.
    for i in range(int(M*N*frac)):
        x = np.random.randint(0,M-1)
        y = np.random.randint(0,N-1)
        data[x,y] = np.random.rand()
    return data

def dist(x,y):
    """
    Return the distance between two points.
    """
    d = x-y
    return np.sqrt(np.dot(d,d))

def dist_point_to_segment(p, s0, s1):
    """
    Get the distance of a point to a segment.

      *p*, *s0*, *s1* are *xy* sequences

    This algorithm from
    http://softsurfer.com/Archive/algorithm_0102/algorithm_0102.htm#Distance%20to%20Ray%20or%20Segment
    """
    p = np.asarray(p, np.float_)
    s0 = np.asarray(s0, np.float_)
    s1 = np.asarray(s1, np.float_)
    v = s1 - s0
    w = p - s0

    c1 = np.dot(w,v);
    if ( c1 <= 0 ):
        return dist(p, s0);

    c2 = np.dot(v,v)
    if ( c2 <= c1 ):
        return dist(p, s1);

    b = c1 / c2
    pb = s0 + b * v;
    return dist(p, pb)

def segments_intersect(s1, s2):
    """
    Return *True* if *s1* and *s2* intersect.
    *s1* and *s2* are defined as::

      s1: (x1, y1), (x2, y2)
      s2: (x3, y3), (x4, y4)
    """
    (x1, y1), (x2, y2) = s1
    (x3, y3), (x4, y4) = s2

    den = ((y4-y3) * (x2-x1)) - ((x4-x3)*(y2-y1))

    n1 = ((x4-x3) * (y1-y3)) - ((y4-y3)*(x1-x3))
    n2 = ((x2-x1) * (y1-y3)) - ((y2-y1)*(x1-x3))

    if den == 0:
        # lines parallel
        return False

    u1 = n1/den
    u2 = n2/den

    return 0.0 <= u1 <= 1.0 and 0.0 <= u2 <= 1.0


def fftsurr(x, detrend=detrend_none, window=window_none):
    """
    Compute an FFT phase randomized surrogate of *x*.
    """
    if cbook.iterable(window):
        x=window*detrend(x)
    else:
        x = window(detrend(x))
    z = np.fft.fft(x)
    a = 2.*np.pi*1j
    phase = a * np.random.rand(len(x))
    z = z*np.exp(phase)
    return np.fft.ifft(z).real


class FIFOBuffer:
    """
    A FIFO queue to hold incoming *x*, *y* data in a rotating buffer
    using numpy arrays under the hood.  It is assumed that you will
    call asarrays much less frequently than you add data to the queue
    -- otherwise another data structure will be faster.

    This can be used to support plots where data is added from a real
    time feed and the plot object wants to grab data from the buffer
    and plot it to screen less freqeuently than the incoming.

    If you set the *dataLim* attr to
    :class:`~matplotlib.transforms.BBox` (eg
    :attr:`matplotlib.Axes.dataLim`), the *dataLim* will be updated as
    new data come in.

    TODO: add a grow method that will extend nmax

    .. note::

      mlab seems like the wrong place for this class.
    """
    @cbook.deprecated('1.3', name='FIFOBuffer', obj_type='class')
    def __init__(self, nmax):
        """
        Buffer up to *nmax* points.
        """
        self._xa = np.zeros((nmax,), np.float_)
        self._ya = np.zeros((nmax,), np.float_)
        self._xs = np.zeros((nmax,), np.float_)
        self._ys = np.zeros((nmax,), np.float_)
        self._ind = 0
        self._nmax = nmax
        self.dataLim = None
        self.callbackd = {}

    def register(self, func, N):
        """
        Call *func* every time *N* events are passed; *func* signature
        is ``func(fifo)``.
        """
        self.callbackd.setdefault(N, []).append(func)

    def add(self, x, y):
        """
        Add scalar *x* and *y* to the queue.
        """
        if self.dataLim is not None:
            xy = np.asarray([(x,y),])
            self.dataLim.update_from_data_xy(xy, None)

        ind = self._ind % self._nmax
        #print 'adding to fifo:', ind, x, y
        self._xs[ind] = x
        self._ys[ind] = y

        for N,funcs in six.iteritems(self.callbackd):
            if (self._ind%N)==0:
                for func in funcs:
                    func(self)

        self._ind += 1

    def last(self):
        """
        Get the last *x*, *y* or *None*.  *None* if no data set.
        """
        if self._ind==0: return None, None
        ind = (self._ind-1) % self._nmax
        return self._xs[ind], self._ys[ind]

    def asarrays(self):
        """
        Return *x* and *y* as arrays; their length will be the len of
        data added or *nmax*.
        """
        if self._ind<self._nmax:
            return self._xs[:self._ind], self._ys[:self._ind]
        ind = self._ind % self._nmax

        self._xa[:self._nmax-ind] = self._xs[ind:]
        self._xa[self._nmax-ind:] = self._xs[:ind]
        self._ya[:self._nmax-ind] = self._ys[ind:]
        self._ya[self._nmax-ind:] = self._ys[:ind]

        return self._xa, self._ya

    def update_datalim_to_current(self):
        """
        Update the *datalim* in the current data in the fifo.
        """
        if self.dataLim is None:
            raise ValueError('You must first set the dataLim attr')
        x, y = self.asarrays()
        self.dataLim.update_from_data(x, y, True)


def movavg(x,n):
    """
    Compute the len(*n*) moving average of *x*.
    """
    w = np.empty((n,), dtype=np.float_)
    w[:] = 1.0/n
    return np.convolve(x, w, mode='valid')


### the following code was written and submitted by Fernando Perez
### from the ipython numutils package under a BSD license
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

import math


#*****************************************************************************
# Globals

#****************************************************************************
# function definitions
exp_safe_MIN = math.log(2.2250738585072014e-308)
exp_safe_MAX = 1.7976931348623157e+308

def exp_safe(x):
    """
    Compute exponentials which safely underflow to zero.

    Slow, but convenient to use. Note that numpy provides proper
    floating point exception handling with access to the underlying
    hardware.
    """

    if type(x) is np.ndarray:
        return np.exp(np.clip(x,exp_safe_MIN,exp_safe_MAX))
    else:
        return math.exp(x)

def amap(fn,*args):
    """
    amap(function, sequence[, sequence, ...]) -> array.

    Works like :func:`map`, but it returns an array.  This is just a
    convenient shorthand for ``numpy.array(map(...))``.
    """
    return np.array(list(map(fn,*args)))


def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.absolute(a)**2))

def l1norm(a):
    """
    Return the *l1* norm of *a*, flattened out.

    Implemented as a separate function (not a call to :func:`norm` for speed).
    """
    return np.sum(np.absolute(a))

def l2norm(a):
    """
    Return the *l2* norm of *a*, flattened out.

    Implemented as a separate function (not a call to :func:`norm` for speed).
    """
    return np.sqrt(np.sum(np.absolute(a)**2))

def norm_flat(a,p=2):
    """
    norm(a,p=2) -> l-p norm of a.flat

    Return the l-p norm of *a*, considered as a flat array.  This is NOT a true
    matrix norm, since arrays of arbitrary rank are always flattened.

    *p* can be a number or the string 'Infinity' to get the L-infinity norm.
    """
    # This function was being masked by a more general norm later in
    # the file.  We may want to simply delete it.
    if p=='Infinity':
        return np.amax(np.absolute(a))
    else:
        return (np.sum(np.absolute(a)**p))**(1.0/p)

def frange(xini,xfin=None,delta=None,**kw):
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
    setting the keyword *closed* = 0, in this
