"""
Tests for the colors module.
"""

from __future__ import print_function
import numpy as np
from numpy.testing.utils import assert_array_equal
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

