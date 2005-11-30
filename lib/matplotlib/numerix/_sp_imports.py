# Not sure why all the equivalences don't exist ...
from scipy import (
    Int8, UnsignedInt8,
    Int16, UnsignedInt16,
    Int32, uint32,
    Float32, Float64,
    Complex32, Complex64,
    Float, Int, Complex)

UInt8 = UnsignedInt8
UInt16 = UnsignedInt16
UInt32 = uint32

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

from scipy import (inf, infty, Infinity)
infinity = Infinity
