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
    # Make min/max = scipy.amin/max
    min = amin
    max = amax
else:
    raise RuntimeError("invalid numerix selector")

# for easy access to these functions w/o clobbering builtins
amin = min
amax = max
