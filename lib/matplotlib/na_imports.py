"""Imports from numarray for numerix, the numarray/Numeric interchangeability
module.  These array functions are used when numarray is chosen.
"""

from numarray.linear_algebra.mlab import *
from numarray import *
import numarray.linear_algebra as LinearAlgebra
import numarray.linear_algebra.mlab as MLab
from numarray.linear_algebra import inverse, eigenvectors
from numarray.convolve import convolve
from numarray.fft import fft
import numarray.random_array as RandomArray
from numarray.numeric import nonzero

import numarray.ieeespecial as _ieee
inf = infty = Infinity = _ieee.inf

class _TypeNamespace:
    """Numeric compatible type aliases for use with extension functions."""
    Int8          = typecode[Int8]
    UInt8         = typecode[UInt8]
    Int16         = typecode[Int16]
    UInt16        = typecode[UInt16]
    Int32         = typecode[Int32]
    #UInt32        = typecode[UInt32]  # Todd: this appears broken
    Float32       = typecode[Float32]
    Float64       = typecode[Float64]
    Complex32     = typecode[Complex32]
    Complex64     = typecode[Complex64]

nx = _TypeNamespace()

class _Matrix(NumArray):
    """_Matrix is a ported, stripped down version of the Numeric Matrix
    class which supplies only matrix multiplication.
    """
    def _rc(self, a):
        if len(shape(a)) == 0:
            return a
        else:
            return Matrix(a)
            
    def __mul__(self, other):
        aother = asarray(other)
        #if len(aother.shape) == 0:
        #    return self._rc(self*aother)
        #else:
        #    return self._rc(dot(self, aother))
        #return self._rc(dot(self, aother))
        return dot(self, aother)

    def __rmul__(self, other):
        aother = asarray(other)
        if len(aother.shape) == 0:
            return self._rc(aother*self)
        else:
            return self._rc(dot(aother, self))

    def __imul__(self,other):
        aother = asarray(other)
        self[:] = dot(self, aother)
        return self

def Matrix(data, typecode=None, copy=1, savespace=0):
    """Matrix constructs new matrices from 2D nested lists of numbers"""
    if isinstance(data, type("")):
        raise TypeError("numerix Matrix does not support Numeric matrix string notation.  Use nested lists.")
    a = fromlist(data, type=typecode)
    if a.rank == 0:
        a.shape = (1,1)
    elif a.rank == 1:
        a.shape = (1,) + a.shape
    a.__class__ = _Matrix
    return a




def _main():
        m = Matrix( [[1,2,3],[11,12,13],[21,22,23]])
        print m*m

if __name__ == '__main__':
    _main()
