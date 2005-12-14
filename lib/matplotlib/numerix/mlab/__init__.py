from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.linear_algebra.mlab import *
elif which[0] == "numeric":
    from MLab import *
elif which[0] == "scipy":
    from scipy import *
    from scipy.basic.linalg import svd, eig
    try:
        from scipy.integrate import trapz
        from scipy.signal.signaltools import \
            hanning, kaiser, blackman, bartlett, \
            hamming
        from scipy.special import sinc
    except ImportError:
        pass

    def mean(m,axis=0):
        """mean(m,axis=0) returns the mean of m along the given dimension.
           If m is of integer type, returns a floating point answer.
        """
        m = asarray(m)
        return add.reduce(m,axis)/float(m.shape[axis])


    def std(m,axis=0):
        """std(m,axis=0) returns the standard deviation along the given
        dimension of m.  The result is unbiased with division by N-1.
        If m is of integer type returns a floating point answer.
        """
        x = asarray(m)
        n = float(x.shape[axis])
        mx = asarray(mean(x,axis))
        if axis < 0:
            axis = len(x.shape) + axis
        mx.shape = mx.shape[:axis] + (1,) + mx.shape[axis:]
        x = x - mx
        return sqrt(add.reduce(x*x,axis)/(n-1.0))

    def cov(m,y=None, rowvar=0, bias=0):
        """Estimate the covariance matrix.

        If m is a vector, return the variance.  For matrices where each row
        is an observation, and each column a variable, return the covariance
        matrix.  Note that in this case diag(cov(m)) is a vector of
        variances for each column.

        cov(m) is the same as cov(m, m)

        Normalization is by (N-1) where N is the number of observations
        (unbiased estimate).  If bias is 1 then normalization is by N.

        If rowvar is zero, then each row is a variable with
        observations in the columns.
        """
        if y is None:
            y = m
        else:
            y = y
        if rowvar:
            m = transpose(m)
            y = transpose(y)
        if (m.shape[0] == 1):
            m = transpose(m)
        if (y.shape[0] == 1):
            y = transpose(y)
        N = m.shape[0]
        if (y.shape[0] != N):
            raise ValueError, "x and y must have the same number of observations."
        m = m - mean(m,axis=0)
        y = y - mean(y,axis=0)
        if bias:
            fact = N*1.0
        else:
            fact = N-1.0
        #
        val = squeeze(dot(transpose(m),conjugate(y)) / fact)
        return val

    def bartlett(M):
        """bartlett(M) returns the M-point Bartlett window.
        """
        n = arange(0,M)
        return where(less_equal(n,(M-1)/2.0),2.0*n/(M-1),2.0-2.0*n/(M-1))

    def hanning(M):
        """hanning(M) returns the M-point Hanning window.
        """
        n = arange(0,M)
        return 0.5-0.5*cos(2.0*pi*n/(M-1))

    def hamming(M):
        """hamming(M) returns the M-point Hamming window.
        """
        n = arange(0,M)
        return 0.54-0.46*cos(2.0*pi*n/(M-1))

    def sinc(x):
        """sinc(x) returns sin(pi*x)/(pi*x) at all points of array x.
        """
        y = pi* where(x == 0, 1.0e-20, x)
        return sin(y)/y

    def msort(a):
        return sort(a, axis=0)
else:
    raise RuntimeError("invalid numerix selector")

if which[0] != "scipy":
    # for easy access to these functions w/o clobbering builtins;
    # scipy already has amin, amax
    amin = min
    amax = max
