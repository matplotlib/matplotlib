from matplotlib.numerix import which

if which[0] == "numarray":
    from numarray.linear_algebra.mlab import *
elif which[0] == "numeric":
    from MLab import *
elif which[0] == "scipy":
    from scipy import *
    from scipy.linalg import svd, eig
    from scipy.integrate import trapz
    from scipy.signal.signaltools import \
        hanning, kaiser, blackman, bartlett, \
        hamming
    from scipy.special import sinc
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
