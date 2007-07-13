from Numeric import array, ravel, reshape, shape, alltrue, sometrue
from Numeric import Int8, UInt8, Int16, UInt16, Int32, UInt32, \
     Float32, Float64, Complex32, Complex64, Float, Int, Complex
from numpy import isnan as _isnan

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

def all(a, axis=None):
    '''Numpy-compatible version of all()'''
    if axis is None:
        return alltrue(ravel(a))
    else:
        return alltrue(a, axis)

def any(a, axis=None):
    if axis is None:
        return sometrue(ravel(a))
    else:
        return sometrue(a, axis)


# inf is useful for testing infinities in results of array divisions
# (which don't raise exceptions)

inf = infty = infinity = Infinity = (array([1])/0.0)[0]
