from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import six

import matplotlib.colors as mcolors
import numpy as np

"""
My example image for testing is 1sq degree around ra=131, dec=19 of the POSS
plates
q = queryPoss.DownloadPossPlates()
fits = q.query(131, 19, 1)

"""


# Inherits from Normalise because _axes.py:pcolormesh() tests for isinstance()
class ImageStretch(mcolors.Normalize):
    """
    Map a range of values to the domain [0,1], scaling as appropriate to
    emphasise certain regions of the range.
    """
    def __init__(self, func, vmin=None, vmax=None):
        """

        Inputs:
        -------------
        func
            Function to stretch input values to emphaise certain ranges of
            values. Examples would include log10() or sqrt(). Function must
            take an array and operate element wise (as numpy ufuncs do)

        vmin, vmax
            (floats) Max and min values that should map to zero to one. Values
            < vmin will always map to zero, values > vmax will map to 1.

        If *vmin* or *vmax* is not given, they are taken from the input's
        minimum and maximum value respectively.
        """

        self.func = func
        self.vmin = vmin
        self.vmax = vmax

        if vmin is not None:
            self.vminTransformed = func(vmin)
            self.vmaxTransformed = func(vmax)

    def __call__(self, values, clip=False):
        """Map a range of values to the domain [0,1]"""
        assert np.all(np.isfinite(values))

        # Convert a float to an array
        if not hasattr(values, "__len__"):
            values = np.array(values)

        self.autoscale(values)

        # Apply the appropriate image stretch to emphaise the values of
        # interest then normalise to [0,1]
        norm = self.normalise(self.func(values))

        # Cast as masked array to behave as mcolors.Normalize() does
        return np.ma.MaskedArray(norm)

    def autoscale(self, values):
        """Use input values to determine values to map to 0 and 1"""
        if self.vmin is None:
            self.vmin = np.min(values)
            self.vmax = np.max(values)

            self.vminTransformed = self.func(self.vmin)
            self.vmaxTransformed = self.func(self.vmax)

    def autoscale_None(self, A):
        """mcolors.Normalize() has one of these, so we do to"""
        self.autoscale(A)

    def scaled(self):
        """mcolors.Normalize() has one of these, so we do to"""
        return self.vmin is not None and self.vmax is not None

    def inverse(self, norm):
        """Compute the inverse transform to get from [0,1] to the original
        range of values.

        Note that an array is not mapped exactly back to itself by this
        class. If some values in input are below vmin they will be mapped
        to 0 and back to vmin, not their original values.
        """
        y = np.linspace(self.vmin, self.vmax, 1000)
        x = self.func(y)

        stretch = self.denormalise(norm)
        return np.interp(stretch, x, y)

    def normalise(self, values):
        """Map values to [0,1] clipping outside the range [vmin,vmax]"""
        norm = (values-self.vminTransformed) / \
            (self.vmaxTransformed - self.vminTransformed)
        norm[norm < 0] = 0
        norm[norm > 1] = 1
        return norm

    def denormalise(self, norm):
        """Map values from [0,1] to [vmin, vmax]"""
        return self.vminTransformed + \
            norm*(self.vmaxTransformed - self.vminTransformed)


# Some example implementations
class LinearStretch(ImageStretch):
    def __init__(self, vmin=None, vmax=None):
        def func(x): return x  # A one-to-one relationship
        super(LinearStretch, self).__init__(func, vmin, vmax)


class LogStretch(ImageStretch):
    def __init__(self,  vmin=None, vmax=None):
        super(LogStretch, self).__init__(np.log10, vmin, vmax)


class SqrtStretch(ImageStretch):
    def __init__(self,  vmin=None, vmax=None):
        super(SqrtStretch, self).__init__(np.sqrt, vmin, vmax)


class HistEquStretch(LinearStretch):
    def __init__(self,  values, lwr, upr):
        vmin, vmax = np.percentile(values, [lwr, upr])
        super(HistEquStretch, self).__init__(vmin, vmax)


def test_LinearStretch():
    obj = LinearStretch(0, 10)
    y = obj(np.linspace(-5, 15, 6))
    print(np.linspace(-5, 15, 6))
    assert(arrayEquals(y, [.25, .01, .09, .49, 1, 1], tol=1e-2,
                       msg="Forward transform failed"))

    yy = obj.inverse(y)
    assert(arrayEquals(yy, [5, 1, 3, 7, 10, 10], tol=1e-2,
                       msg="Reverse transform failed"))
    return obj


def test_LogStretch():
    obj = LogStretch(1, 1000)
    x = np.logspace(0, 4, 5)

    y = obj(x)
    assert(arrayEquals(y, [0, .333, .666, 1, 1], tol=1e-2,
                       msg="Forward transform failed"))

    yy = obj.inverse(y)
    assert(arrayEquals(yy, [1e0, 1e1, 1e2, 1e3, 1e3], tol=1e-2,
                       msg="Reverse transform failed"))
    return obj


def arrayEquals(a, b, tol=0, msg=None):
    """Test every element of array 'a' is equal to element in 'b' within
    tolerance"""
    if len(a) != len(b):
        return False

    for i in range(len(a)):
        if np.fabs(a[i]-b[i]) > tol:
            print("Elt %i: %.3f!=%.3f within tol %.3e" % (i, a[i], b[i], tol))
            if msg is not None:
                print(msg)
            return False

    return True
