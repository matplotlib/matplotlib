from __future__ import print_function
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal
from matplotlib.transforms import Affine2D, BlendedGenericTransform
from matplotlib.path import Path
from matplotlib.scale import LogScale
from matplotlib.testing.decorators import cleanup
import numpy as np

import matplotlib.transforms as mtrans
import matplotlib.pyplot as plt



@cleanup
def test_non_affine_caching():
    class AssertingNonAffineTransform(mtrans.Transform):
        """
        This transform raises an assertion error when called when it
        shouldn't be and self.raise_on_transform is True.

        """
        input_dims = output_dims = 2
        is_affine = False
        def __init__(self, *args, **kwargs):
            mtrans.Transform.__init__(self, *args, **kwargs)
            self.raise_on_transform = False
            self.underlying_transform = mtrans.Affine2D().scale(10, 10)

        def transform_path_non_affine(self, path):
            if self.raise_on_transform:
                assert False, ('Invalidated affine part of transform '
                                    'unnecessarily.')
            return self.underlying_transform.transform_path(path)
        transform_path = transform_path_non_affine

        def transform_non_affine(self, path):
            if self.raise_on_transform:
                assert False, ('Invalidated affine part of transform '
                                    'unnecessarily.')
            return self.underlying_transform.transform(path)
        transform = transform_non_affine

    my_trans = AssertingNonAffineTransform()
    ax = plt.axes()
    plt.plot(range(10), transform=my_trans + ax.transData)
    plt.draw()
    # enable the transform to raise an exception if it's non-affine transform
    # method is triggered again.
    my_trans.raise_on_transform = True
    ax.transAxes.invalidate()
    plt.draw()


def test_Affine2D_from_values():
    points = [ [0,0],
               [10,20],
               [-1,0],
               ]

    t = Affine2D.from_values(1,0,0,0,0,0)
    actual = t.transform(points)
    expected = np.array( [[0,0],[10,0],[-1,0]] )
    assert_almost_equal(actual,expected)

    t = Affine2D.from_values(0,2,0,0,0,0)
    actual = t.transform(points)
    expected = np.array( [[0,0],[0,20],[0,-2]] )
    assert_almost_equal(actual,expected)

    t = Affine2D.from_values(0,0,3,0,0,0)
    actual = t.transform(points)
    expected = np.array( [[0,0],[60,0],[0,0]] )
    assert_almost_equal(actual,expected)

    t = Affine2D.from_values(0,0,0,4,0,0)
    actual = t.transform(points)
    expected = np.array( [[0,0],[0,80],[0,0]] )
    assert_almost_equal(actual,expected)

    t = Affine2D.from_values(0,0,0,0,5,0)
    actual = t.transform(points)
    expected = np.array( [[5,0],[5,0],[5,0]] )
    assert_almost_equal(actual,expected)

    t = Affine2D.from_values(0,0,0,0,0,6)
    actual = t.transform(points)
    expected = np.array( [[0,6],[0,6],[0,6]] )
    assert_almost_equal(actual,expected)

def test_clipping_of_log():
    # issue 804
    M,L,C = Path.MOVETO, Path.LINETO, Path.CLOSEPOLY
    points = [ (0.2, -99), (0.4, -99), (0.4, 20), (0.2, 20), (0.2, -99) ]
    codes  = [          M,          L,        L,         L,          C  ]
    path = Path(points, codes)

    # something like this happens in plotting logarithmic histograms
    trans = BlendedGenericTransform(Affine2D(),
                                    LogScale.Log10Transform('clip'))
    tpath = trans.transform_path_non_affine(path)
    result = tpath.iter_segments(trans.get_affine(),
                                 clip=(0, 0, 100, 100),
                                 simplify=False)

    tpoints, tcodes = zip(*result)
    # Because y coordinate -99 is outside the clip zone, the first
    # line segment is effectively removed. That means that the closepoly
    # operation must be replaced by a move to the first point.
    assert np.allclose(tcodes, [ M, M, L, L, L ])
    assert np.allclose(tpoints[-1], tpoints[0])
