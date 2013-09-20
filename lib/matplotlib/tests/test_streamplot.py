import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison, cleanup
import matplotlib.transforms as mtransforms


def velocity_field():
    Y, X = np.mgrid[-3:3:100j, -3:3:100j]
    U = -1 - X**2 + Y
    V = 1 + X - Y**2
    return X, Y, U, V


@image_comparison(baseline_images=['streamplot_colormap_test_image'])
def test_colormap():
    X, Y, U, V = velocity_field()
    plt.streamplot(X, Y, U, V, color=U, density=0.6, linewidth=2,
                   cmap=plt.cm.autumn)
    plt.colorbar()


@image_comparison(baseline_images=['streamplot_linewidth_test_image'])
def test_linewidth():
    X, Y, U, V = velocity_field()
    speed = np.sqrt(U*U + V*V)
    lw = 5*speed/speed.max()
    plt.streamplot(X, Y, U, V, density=[0.5, 1], color='k', linewidth=lw)


@image_comparison(baseline_images=['streamplot_masks_and_nans_test_image'])
def test_masks_and_nans():
    X, Y, U, V = velocity_field()
    mask = np.zeros(U.shape, dtype=bool)
    mask[40:60, 40:60] = 1
    U = np.ma.array(U, mask=mask)
    U[:20, :20] = np.nan
    plt.streamplot(X, Y, U, V, color=U, cmap=plt.cm.Blues)


@cleanup
def test_streamplot_limits():
    ax = plt.axes()
    x = np.linspace(-5, 10, 20)
    y = np.linspace(-2, 4, 10)
    y, x = np.meshgrid(y, x)
    trans = mtransforms.Affine2D().translate(25, 32) + ax.transData
    plt.barbs(x, y, np.sin(x), np.cos(y), transform=trans)
    # The calculated bounds are approximately the bounds of the original data,
    # this is because the entire path is taken into account when updating the
    # datalim.
    assert_array_almost_equal(ax.dataLim.bounds, (20, 30, 15, 6),
                              decimal=2)


if __name__=='__main__':
    import nose
    nose.runmodule()
