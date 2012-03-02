from __future__ import print_function
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal
from matplotlib.transforms import Affine2D, BlendedGenericTransform
from matplotlib.path import Path
from matplotlib.scale import LogScale
from matplotlib.testing.decorators import cleanup, image_comparison
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


@cleanup
def test_external_transform_api():
    class ScaledBy(object):
        def __init__(self, scale_factor):
            self._scale_factor = scale_factor
            
        def _as_mpl_transform(self, axes):
            return mtrans.Affine2D().scale(self._scale_factor) + axes.transData

    ax = plt.axes()
    line, = plt.plot(range(10), transform=ScaledBy(10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    # assert that the top transform of the line is the scale transform.
    np.testing.assert_allclose(line.get_transform()._a.get_matrix(), mtrans.Affine2D().scale(10).get_matrix())
    

@image_comparison(baseline_images=['pre_transform_data'])
def test_pre_transform_plotting():
    # a catch-all for as many as possible plot layouts which handle pre-transforming the data
    # NOTE: The axis range is important in this plot. It should be x10 what the data suggests it should be 
    ax = plt.axes()
    times10 = mtrans.Affine2D().scale(10)
    
    ax.contourf(np.arange(48).reshape(6, 8), transform=times10 + ax.transData)
    
    ax.pcolormesh(np.linspace(0, 4, 7), np.linspace(5.5, 8, 9), np.arange(48).reshape(6, 8),
                  transform=times10 + ax.transData)
    
    ax.scatter(np.linspace(0, 10), np.linspace(10, 0), transform=times10 + ax.transData)
    
    
    x = np.linspace(8, 10, 20)
    y = np.linspace(1, 5, 20)
    u = 2*np.sin(x) + np.cos(y[:, np.newaxis])
    v = np.sin(x) - np.cos(y[:, np.newaxis])

    ax.streamplot(x, y, u, v, transform=times10 + ax.transData,
                  density=(1, 1), linewidth=u**2 + v**2)
    
    # reduce the vector data down a bit for barb and quiver plotting
    x, y = x[::3], y[::3]
    u, v = u[::3, ::3], v[::3, ::3]
    
    ax.quiver(x, y + 5, u, v, transform=times10 + ax.transData)
    
    ax.barbs(x - 3, y + 5, u**2, v**2, transform=times10 + ax.transData)
    

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


if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)