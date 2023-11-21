import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import (image_comparison,
                                           remove_ticks_and_titles)
import matplotlib.colors as mcolors


@image_comparison(["multivariate_pcolormesh_alpha.png"])
def test_multivariate_pcolormesh_alpha():
    """
    Check that the the alpha keyword works for pcolormesh

    This test covers all plotting modes that use the same pipeline
    (inherit from Collection).
    """
    x_0 = np.arange(25, dtype='float32').reshape(5, 5) % 5
    x_1 = np.arange(25, dtype='float32').reshape(5, 5).T % 5
    x_2 = np.arange(25, dtype='float32').reshape(5, 5) % 6

    fig, axes = plt.subplots(2, 3)

    axes[0, 0].pcolormesh(x_1,  alpha=0.5)
    axes[0, 1].pcolormesh((x_0, x_1), cmap='BiPeak', alpha=0.5)
    axes[0, 2].pcolormesh((x_0, x_1, x_2), cmap='3VarAddA', alpha=0.5)

    al = np.arange(25, dtype='float32').reshape(5, 5)[::-1].T % 6 / 5

    axes[1, 0].pcolormesh(x_1,  alpha=al)
    axes[1, 1].pcolormesh((x_0, x_1), cmap='BiPeak', alpha=al)
    axes[1, 2].pcolormesh((x_0, x_1, x_2), cmap='3VarAddA', alpha=al)

    remove_ticks_and_titles(fig)


@image_comparison(["multivariate_pcolormesh_norm.png"])
def test_multivariate_pcolormesh_norm():
    """
    Test vmin, vmax and norm
    Norm is checked via a LogNorm, as this converts
    A LogNorm converts the input to a masked array, masking for X <= 0
    By using a LogNorm, this functionality is also tested.

    This test covers all plotting modes that use the same pipeline
    (inherit from Collection).
    """
    x_0 = np.arange(25, dtype='float32').reshape(5, 5) % 5
    x_1 = np.arange(25, dtype='float32').reshape(5, 5).T % 5
    x_2 = np.arange(25, dtype='float32').reshape(5, 5) % 6

    fig, axes = plt.subplots(3, 5)

    axes[0, 0].pcolormesh(x_1)
    axes[0, 1].pcolormesh((x_0, x_1), cmap='BiPeak')
    axes[0, 2].pcolormesh((x_0, x_1, x_2), cmap='3VarAddA')
    axes[0, 3].pcolormesh((x_0, x_1), cmap='BiPeak')
    axes[0, 4].pcolormesh((x_0, x_1, x_2), cmap='3VarAddA')

    vmin = 1
    vmax = 3
    axes[1, 0].pcolormesh(x_1, vmin=vmin, vmax=vmax)
    axes[1, 1].pcolormesh((x_0, x_1), cmap='BiPeak', vmin=vmin, vmax=vmax)
    axes[1, 2].pcolormesh((x_0, x_1, x_2), cmap='3VarAddA',
                          vmin=vmin, vmax=vmax)
    axes[1, 3].pcolormesh((x_0, x_1), cmap='BiPeak',
                          vmin=(None, vmin), vmax=(None, vmax))
    axes[1, 4].pcolormesh((x_0, x_1, x_2), cmap='3VarAddA',
                          vmin=(None, vmin, None), vmax=(None, vmax, None))

    n = mcolors.LogNorm(vmin=1, vmax=5)
    axes[2, 0].pcolormesh(x_1, norm=n)
    axes[2, 1].pcolormesh((x_0, x_1), cmap='BiPeak', norm=n)
    axes[2, 2].pcolormesh((x_0, x_1, x_2), cmap='3VarAddA', norm=n)
    axes[2, 3].pcolormesh((x_0, x_1), cmap='BiPeak', norm=(None, n))
    axes[2, 4].pcolormesh((x_0, x_1, x_2), cmap='3VarAddA',
                          norm=(None, n, None))

    remove_ticks_and_titles(fig)


@image_comparison(["multivariate_imshow_alpha.png"])
def test_multivariate_imshow_alpha():
    """
    Check that the the alpha keyword works for pcolormesh
    """
    x_0 = np.arange(25, dtype='float32').reshape(5, 5) % 5
    x_1 = np.arange(25, dtype='float32').reshape(5, 5).T % 5
    x_2 = np.arange(25, dtype='float32').reshape(5, 5) % 6

    fig, axes = plt.subplots(2, 3)

    # interpolation='nearest' to reduce size of baseline image
    axes[0, 0].imshow(x_1, interpolation='nearest',  alpha=0.5)
    axes[0, 1].imshow((x_0, x_1), interpolation='nearest', cmap='BiPeak', alpha=0.5)
    axes[0, 2].imshow((x_0, x_1, x_2), interpolation='nearest',
                      cmap='3VarAddA', alpha=0.5)

    al = np.arange(25, dtype='float32').reshape(5, 5)[::-1].T % 6 / 5

    axes[1, 0].imshow(x_1, interpolation='nearest',  alpha=al)
    axes[1, 1].imshow((x_0, x_1), interpolation='nearest', cmap='BiPeak', alpha=al)
    axes[1, 2].imshow((x_0, x_1, x_2), interpolation='nearest',
                      cmap='3VarAddA', alpha=al)

    remove_ticks_and_titles(fig)


@image_comparison(["multivariate_imshow_norm.png"])
def test_multivariate_imshow_norm():
    """
    Test vmin, vmax and norm
    Norm is checked via a LogNorm.
    A LogNorm converts the input to a masked array, masking for X <= 0
    By using a LogNorm, this functionality is also tested.
    """
    x_0 = np.arange(25, dtype='float32').reshape(5, 5) % 5
    x_1 = np.arange(25, dtype='float32').reshape(5, 5).T % 5
    x_2 = np.arange(25, dtype='float32').reshape(5, 5) % 6

    fig, axes = plt.subplots(3, 5, dpi=10)

    # interpolation='nearest' to reduce size of baseline image and
    # removes ambiguity when using masked array (from LogNorm)
    axes[0, 0].imshow(x_1, interpolation='nearest')
    axes[0, 1].imshow((x_0, x_1), interpolation='nearest', cmap='BiPeak')
    axes[0, 2].imshow((x_0, x_1, x_2), interpolation='nearest', cmap='3VarAddA')
    axes[0, 3].imshow((x_0, x_1), interpolation='nearest', cmap='BiPeak')
    axes[0, 4].imshow((x_0, x_1, x_2), interpolation='nearest', cmap='3VarAddA')

    vmin = 1
    vmax = 3
    axes[1, 0].imshow(x_1, interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[1, 1].imshow((x_0, x_1), interpolation='nearest', cmap='BiPeak',
                      vmin=vmin, vmax=vmax)
    axes[1, 2].imshow((x_0, x_1, x_2), interpolation='nearest', cmap='3VarAddA',
                          vmin=vmin, vmax=vmax)
    axes[1, 3].imshow((x_0, x_1), interpolation='nearest', cmap='BiPeak',
                          vmin=(None, vmin), vmax=(None, vmax))
    axes[1, 4].imshow((x_0, x_1, x_2), interpolation='nearest', cmap='3VarAddA',
                          vmin=(None, vmin, None), vmax=(None, vmax, None))

    n = mcolors.LogNorm(vmin=1, vmax=5)
    axes[2, 0].imshow(x_1, interpolation='nearest', norm=n)
    axes[2, 1].imshow((x_0, x_1), interpolation='nearest', cmap='BiPeak', norm=n)
    axes[2, 2].imshow((x_0, x_1, x_2), interpolation='nearest', cmap='3VarAddA', norm=n)
    axes[2, 3].imshow((x_0, x_1), interpolation='nearest', cmap='BiPeak',
                      norm=(None, n))
    axes[2, 4].imshow((x_0, x_1, x_2), interpolation='nearest', cmap='3VarAddA',
                          norm=(None, n, None))

    remove_ticks_and_titles(fig)
