"""
Tests for the colors module.
"""

from __future__ import print_function
import numpy as np
from numpy.testing.utils import assert_array_equal, assert_array_almost_equal
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def test_colormap_endian():
    """
    Github issue #1005: a bug in putmask caused erroneous
    mapping of 1.0 when input from a non-native-byteorder
    array.
    """
    cmap = cm.get_cmap("jet")
    # Test under, over, and invalid along with values 0 and 1.
    a = [-0.5, 0, 0.5, 1, 1.5, np.nan]
    for dt in ["f2", "f4", "f8"]:
        anative = np.ma.masked_invalid(np.array(a, dtype=dt))
        aforeign = anative.byteswap().newbyteorder()
        #print(anative.dtype.isnative, aforeign.dtype.isnative)
        assert_array_equal(cmap(anative), cmap(aforeign))

def test_BoundaryNorm():
    """
    Github issue #1258: interpolation was failing with numpy
    1.7 pre-release.
    """
    # TODO: expand this into a more general test of BoundaryNorm.
    boundaries = [0, 1.1, 2.2]
    vals = [-1, 0, 2, 2.2, 4]
    expected = [-1, 0, 2, 3, 3]
    # ncolors != len(boundaries) - 1 triggers interpolation
    ncolors = len(boundaries)
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)
    
def test_LogNorm():
    """
    LogNorm igornoed clip, now it has the same
    behavior as Normalize, e.g. values > vmax are bigger than 1
    without clip, with clip they are 1.
    """
    ln = mcolors.LogNorm(clip=True, vmax=5)
    assert_array_equal(ln([1, 6]), [0, 1.0])

def test_Normalize():
    norm = mcolors.Normalize()
    vals = np.arange(-10, 10, 1, dtype=np.float)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)


def test_SymLogNorm():
    """
    Test SymLogNorm behavior
    """
    norm = mcolors.SymLogNorm(3, vmax=5, linscale=1.2)
    vals = np.array([-30, -1, 2, 6], dtype=np.float)
    normed_vals = norm(vals)
    expected = [ 0., 0.53980074, 0.826991, 1.02758204]
    assert_array_almost_equal(normed_vals, expected)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)


def _inverse_tester(norm_instance, vals):
    """
    Checks if the inverse of the given normalization is working.
    """
    assert_array_almost_equal(norm_instance.inverse(norm_instance(vals)), vals)

def _scalar_tester(norm_instance, vals):
    """
    Checks if scalars and arrays are handled the same way.
    Tests only for float.
    """
    scalar_result = [norm_instance(float(v)) for v in vals]
    assert_array_almost_equal(scalar_result, norm_instance(vals))

def _mask_tester(norm_instance, vals):
    """
    Checks mask handling
    """
    masked_array = np.ma.array(vals)
    masked_array[0] = np.ma.masked
    assert_array_equal(masked_array.mask, norm_instance(masked_array).mask)
