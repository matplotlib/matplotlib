"""Imports from Numeric for numerix, the numarray/Numeric interchangeability
module.  These array functions are used when Numeric is chosen.
"""
from Numeric import *
import MLab
from MLab import *
from FFT import fft
import LinearAlgebra
from LinearAlgebra import inverse, eigenvectors
import RandomArray
from Matrix import Matrix        

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



# inf is useful for testing infinities in results of array divisions
# (which don't raise exceptions)

inf = infty = Infinity = (array([1])/0.0)[0]

