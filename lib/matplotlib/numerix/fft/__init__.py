try:
    from numpy.oldnumeric.fft import *
except ImportError:
    from numpy.dft.old import *
