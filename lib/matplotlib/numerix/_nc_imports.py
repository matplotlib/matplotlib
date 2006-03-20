from Numeric import array, ravel, reshape, shape
from Numeric import Int8, UInt8, Int16, UInt16, Int32, UInt32, \
     Float32, Float64, Complex32, Complex64, Float, Int, Complex
from matplotlib._isnan import isnan64 as _isnan

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

def isnan(a):
    """y = isnan(x) returns True where x is Not-A-Number"""
    return reshape(array([_isnan(i) for i in ravel(a)],'b'), shape(a))


# inf is useful for testing infinities in results of array divisions
# (which don't raise exceptions)

inf = infty = infinity = Infinity = (array([1])/0.0)[0]
