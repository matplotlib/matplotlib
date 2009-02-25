try:
    from numpy.oldnumeric import Int8, UInt8, \
         Int16, UInt16, \
         Int32, UInt32, \
         Float32, Float64, \
         Complex32, Complex64, \
         Float, Int, Complex
except ImportError:
    from numpy import Int8, UInt8, \
         Int16, UInt16, \
         Int32, UInt32, \
         Float32, Float64, \
         Complex32, Complex64, \
         Float, Int, Complex

class _TypeNamespace:
    """Numeric compatible type aliases for use with extension functions."""
    Int8          = Int8
    UInt8         = UInt8
    Int16         = Int16
    UInt16        = UInt16
    Int32         = Int32
    UInt32        = UInt32
    Float32       = Float32
    Float64       = Float64
    Complex32     = Complex32
    Complex64     = Complex64

nx = _TypeNamespace()

from numpy import inf, infty, Infinity
from numpy.random import rand, randn
infinity = Infinity
from numpy import all, isnan, any
