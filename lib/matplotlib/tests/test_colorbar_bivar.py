import numpy as np
import pytest

import matplotlib as mpl
import matplotlib.colorbar_bivar as mcolorbar_bivar
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison


@pytest.mark.parametrize('use_gridspec', [True, False])
@image_comparison(['bivar_cbar_locationing.png',
                   ], style='mpl20',
                  remove_text=True, savefig_kwarg={'dpi': 40}, tol=0.05)
def test_colorbar_bivar_location(use_gridspec):
    data = (np.arange(12).reshape((3, 4)) % 4,
            np.arange(12).reshape((3, 4)) % 3)
    # -------------------
    locations = ['left', 'right', 'top', 'bottom']
    fig, axes = plt.subplots(2, 2)
    for i, ax in enumerate(axes.ravel()):
        mim = ax.imshow(data, cmap='BiOrangeBlue')
        fig.colorbar_bivar(mim, location=locations[i], use_gridspec=use_gridspec)


@image_comparison(['bivar_cbar_locationing_constrained.png',
                   ], style='mpl20',
                  remove_text=True, savefig_kwarg={'dpi': 40}, tol=0.05)
def test_colorbar_bivar_location_constrained():
    data = (np.arange(12).reshape((3, 4)) % 4,
            np.arange(12).reshape((3, 4)) % 3)
    # -------------------
    locations = ['left', 'right', 'top', 'bottom']
    fig, axes = plt.subplots(2, 2, constrained_layout='constrained')
    for i, ax in enumerate(axes.ravel()):
        mim = ax.imshow(data, cmap='BiOrangeBlue')
        fig.colorbar_bivar(mim, location=locations[i])


@pytest.mark.parametrize('use_gridspec', [True, False])
@image_comparison(['bivar_cbar_sharing.png',
                   ], style='mpl20',
                  remove_text=True, savefig_kwarg={'dpi': 40}, tol=0.05)
def test_colorbar_bivar_sharing(use_gridspec):
    data = (np.arange(12).reshape((3, 4)) % 4,
            np.arange(12).reshape((3, 4)) % 3)
    # -------------------
    plt.figure()
    ax1 = plt.subplot(211, anchor='NE', aspect='equal')
    plt.imshow(data, cmap='BiOrangeBlue')
    ax2 = plt.subplot(223)
    plt.imshow(data, cmap='BiOrangeBlue')
    ax3 = plt.subplot(224)
    plt.imshow(data, cmap='BiOrangeBlue')

    plt.colorbar_bivar(ax=[ax2, ax3, ax1], location='right', pad=0.0, shrink=0.5,
                       panchor=False, use_gridspec=use_gridspec)
    plt.colorbar_bivar(ax=[ax2, ax3, ax1], location='left', shrink=0.5,
                       panchor=False, use_gridspec=use_gridspec)
    plt.colorbar_bivar(ax=[ax1], location='bottom', panchor=False,
                       anchor=(0.8, 0.5), shrink=0.6, use_gridspec=use_gridspec)


@pytest.mark.parametrize('constrained', [False, True],
                         ids=['standard', 'constrained'])
def test_bivar_cbar_single_ax_panchor_east(constrained):
    fig, ax = plt.subplots(constrained_layout=constrained)
    ax.set_anchor('N')
    assert ax.get_anchor() == 'N'
    mp = ax.imshow([[[0, 1], [1, 2]], [[2, 1], [0, 1]]], cmap='BiOrangeBlue')
    fig.colorbar_bivar(mp, panchor='E')
    assert ax.get_anchor() == 'E'


def test_bivar_cbar_set_xylabel():
    fig, axes = plt.subplots(1, 2)
    mp = axes[0].imshow([[[0, 1], [1, 2]], [[2, 1], [0, 1]]], cmap='BiOrangeBlue')
    cb = fig.colorbar_bivar(mp, cax=axes[1])
    cb.set_ylabel('y')
    cb.set_xlabel('x')
    assert axes[1].get_ylabel() == 'y'
    assert axes[1].get_xlabel() == 'x'
    assert cb.xaxis is axes[1].xaxis
    assert cb.yaxis is axes[1].yaxis


def test_bivar_cbar_log_no_scale():
    fig, axes = plt.subplots(1, 2)
    mp = axes[0].imshow([[[100, 1], [10, 1]], [[0.5, 0], [0.3, 1]]],
                        cmap='BiOrangeBlue',
                        norm=['log', mcolors.NoNorm()],
                        )
    assert axes[1].get_yscale() == 'linear'
    assert axes[1].get_xscale() == 'linear'
    fig.colorbar_bivar(mp, cax=axes[1])
    assert axes[1].get_yscale() == 'log'
    assert axes[1].get_xscale() == 'function'


def test_bivar_cbar_change_vmin_vmax():
    fig, axes = plt.subplots(1, 2)
    mp = axes[0].imshow([[[100, 1], [10, 1]], [[0.5, 0], [0.3, 1]]],
                        cmap='BiOrangeBlue',
                        )
    fig.colorbar_bivar(mp, cax=axes[1])
    assert np.all(axes[1].get_ylim() == np.array([1, 100]))
    assert np.all(axes[1].get_xlim() == np.array([0, 1]))
    mp.colorizer.norm.vmin = [-1, -2]
    mp.colorizer.norm.vmax = [3, 5]
    assert np.all(axes[1].get_ylim() == np.array([-1, 3]))
    assert np.all(axes[1].get_xlim() == np.array([-2, 5]))


def test_bivar_cbar_change_norms():
    fig, axes = plt.subplots(1, 2)
    mp = axes[0].imshow([[[100, 1], [10, 1]], [[0.5, 0.2], [0.3, 1]]],
                        cmap='BiOrangeBlue',
                        )
    fig.colorbar_bivar(mp, cax=axes[1])
    assert axes[1].get_yscale() == 'linear'
    assert axes[1].get_xscale() == 'linear'
    mp.colorizer.norm = ['log', mcolors.NoNorm()]
    assert axes[1].get_yscale() == 'log'
    assert axes[1].get_xscale() == 'function'


def test_bivar_cbar_anchor():
    # right
    fig, ax = plt.subplots(figsize=(6, 2))
    mp = ax.imshow([[[0, 1], [1, 2]], [[2, 1], [0, 1]]], cmap='BiOrangeBlue')
    anchor = (1, 0.3)
    shrink = 0.3
    cbar = fig.colorbar_bivar(mp, anchor=anchor, shrink=shrink)

    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (y1 - y0) * anchor[1] + y0
    np.testing.assert_allclose(
            [cy1, cy0],
            [y1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + y0 * shrink])

    # left
    fig, ax = plt.subplots(figsize=(6, 2))
    mp = ax.imshow([[[0, 1], [1, 2]], [[2, 1], [0, 1]]], cmap='BiOrangeBlue')
    anchor = (1, 0.7)
    shrink = 0.3
    cbar = fig.colorbar_bivar(mp, anchor=anchor, shrink=shrink, location='left')

    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (y1 - y0) * anchor[1] + y0
    np.testing.assert_allclose(
            [cy1, cy0],
            [y1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + y0 * shrink])

    # top
    fig, ax = plt.subplots(figsize=(2, 6))
    mp = ax.imshow([[[0, 1], [1, 2]], [[2, 1], [0, 1]]], cmap='BiOrangeBlue')
    anchor = (0.3, 1)
    shrink = 0.3
    cbar = fig.colorbar_bivar(mp, anchor=anchor, shrink=shrink, location='top')

    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (x1 - x0) * anchor[0] + x0
    np.testing.assert_allclose(
            [cx1, cx0],
            [x1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + x0 * shrink])

    # bottom
    fig, ax = plt.subplots(figsize=(2, 6))
    mp = ax.imshow([[[0, 1], [1, 2]], [[2, 1], [0, 1]]], cmap='BiOrangeBlue')
    anchor = (0.3, 1)
    shrink = 0.3
    cbar = fig.colorbar_bivar(mp, anchor=anchor, shrink=shrink, location='bottom')

    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (x1 - x0) * anchor[0] + x0
    np.testing.assert_allclose(
            [cx1, cx0],
            [x1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + x0 * shrink])


def test_bivar_cbar_aspect():
    fig, ax = plt.subplots(1, 1)
    mp = ax.imshow([[[100, 1], [10, 1]], [[0.5, 0.2], [0.3, 1]]],
                   cmap='BiOrangeBlue',
                   )
    cbar = fig.colorbar_bivar(mp, aspect=0.3)
    assert cbar.ax.get_box_aspect() == 0.3
    assert cbar.ax._colorbar_info["aspect"] == 0.3
    assert cbar.aspect == 0.3
    cbar.aspect = 4
    assert cbar.ax.get_box_aspect() == 4
    assert cbar.ax._colorbar_info["aspect"] == 4
    assert cbar.aspect == 4


@pytest.mark.parametrize('location', ['left', 'right', 'top', 'bottom'])
@pytest.mark.parametrize('layout', [None, 'constrained'])
def test_bivar_cbar_aspect_kept_by_layout(location, layout):
    fig, ax = plt.subplots(layout=layout)
    mp = ax.imshow([[[100, 1], [10, 1]], [[0.5, 0.2], [0.3, 1]]],
                   cmap='BiOrangeBlue',
                   )
    cbar = fig.colorbar_bivar(mp, location=location, aspect=0.5)
    fig.draw_without_rendering()
    assert cbar.aspect == 0.5
    cbar.aspect = 3
    fig.draw_without_rendering()
    assert cbar.aspect == 3


# If we decide in the future to disallow calling colorbar() on the "wrong" figure,
# just delete this test.
def test_bivar_cbar_wrong_figure():
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    mp = ax0.imshow([[[100, 1], [10, 1]], [[0.5, 0.2], [0.3, 1]]],
                    cmap='BiOrangeBlue',
                    )
    with pytest.warns(UserWarning, match="different Figure"):
        fig1.colorbar_bivar(mp)


@pytest.mark.parametrize('use_gridspec', [True, False])
@pytest.mark.parametrize('nested_gridspecs', [True, False])
def test_bivar_cbar_remove_from_figure(nested_gridspecs, use_gridspec):
    """Test `remove` with the specified ``use_gridspec`` setting."""
    fig = plt.figure()
    if nested_gridspecs:
        gs = fig.add_gridspec(2, 2)[1, 1].subgridspec(2, 2)
        ax = fig.add_subplot(gs[1, 1])
    else:
        ax = fig.add_subplot()
    mp = ax.pcolormesh([[[100, 1], [10, 1]], [[0.5, 0.2], [0.3, 1]]],
                       cmap='BiOrangeBlue',
                       )
    pre_position = ax.get_position()
    cb = fig.colorbar_bivar(mp, use_gridspec=use_gridspec)
    fig.subplots_adjust()
    cb.remove()
    fig.subplots_adjust()
    post_position = ax.get_position()
    assert (pre_position.get_points() == post_position.get_points()).all()


def test_bivar_cbar_remove_with_no_mappable():
    fig, ax = plt.subplots()
    norm = mpl.colors.MultiNorm(['linear', 'linear'])
    ca = mpl.colorizer.Colorizer('BiOrangeBlue', norm)
    cb = mcolorbar_bivar.BivarColorbar(ax, ca)
    cb.remove()


def test_bivar_cbar_ticklocations():
    norm = mpl.colors.MultiNorm(['linear', 'linear'])
    ca = mpl.colorizer.Colorizer('BiOrangeBlue', norm)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='ticklocations must be a tuple of'):
        cbar = fig.colorbar_bivar(ca, cax=ax, ticklocations=['left'])
    cbar = fig.colorbar_bivar(ca, cax=ax, ticklocations=['left', 'top'])
    assert cbar.ax.yaxis.get_label_position() == 'left'
    assert cbar.ax.xaxis.get_label_position() == 'top'


def test_wrong_kind_colorbar_bivar():
    fig, ax = plt.subplots(1, 1)
    mp = ax.imshow([[[100, 1], [10, 1]], [[0.5, 0.2], [0.3, 1]]],
                   cmap='BiOrangeBlue',
                   )
    with pytest.raises(ValueError, match='used together with a scalar colormap'):
        fig.colorbar(mp, ax=ax)

    fig, ax = plt.subplots(1, 1)
    mp = ax.imshow([[100, 1], [10, 1]])
    with pytest.raises(ValueError, match='bivariate colorbar can only be used '):
        fig.colorbar_bivar(mp, ax=ax)


def test_colorbar_bivar_set_get_view():
    # maybe not the best way to test this, this is normally used
    # interacitively ._get_view() and ._set_view() are normally
    # used in interactive mode
    fig, axes = plt.subplots(1, 1)
    mp = axes.imshow([[[100, 1], [10, 1]], [[0.5, 0.2], [0.3, 1]]],
                     cmap='BiOrangeBlue',
                     )
    cb = fig.colorbar_bivar(mp, ax=axes)
    # limits are x: (0.2, 1), y: (1, 100)
    view = cb._get_view()
    view = np.array(view) + 1
    cb._set_view(view)
    # limits are x: (1.2, 2), y: (2, 101)
    assert mp.colorizer.norm.norms[1].vmin == 1.2
    assert mp.colorizer.norm.norms[1].vmax == 2
    assert mp.colorizer.norm.norms[0].vmin == 2
    assert mp.colorizer.norm.norms[0].vmax == 101


def test_colorbar_bivar_no_ax():
    norm = mpl.colors.MultiNorm(['linear', 'linear'])
    ca = mpl.colorizer.Colorizer('BiOrangeBlue', norm)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='Unable to determine Axes'):
        fig.colorbar_bivar(ca)


def test_colorbar_bivar_custom_norm():
    """
    This tests setting the correct scale for norms that do not have a
    ._scale property.

    In the normal case, colorbar_bivar uses the ._scale property
    of each norm to set the transform on each axis.
    """
    class CustomHalfNorm(mcolors.Normalize):
        def __init__(self):
            super().__init__()

        @property
        def vmin(self):
            return 0

        @vmin.setter
        def vmin(self, val):
            ...

        @property
        def vmax(self):
            return 1

        @vmax.setter
        def vmax(self, val):
            ...

        @property
        def clip(self):
            return False

        @clip.setter
        def clip(self, val):
            ...

        def __call__(self, value, clip=None):
            return value / 2

        def inverse(self, value):
            return 2 * value

        def autoscale(self, A):
            pass

        def autoscale_None(self, A):
            pass

        def scaled(self):
            return True

        @property
        def n_components(self):
            return 1

    norm = mpl.colors.MultiNorm([CustomHalfNorm(), CustomHalfNorm()])
    ca = mpl.colorizer.Colorizer('BiOrangeBlue', norm)
    fig, ax = plt.subplots()
    fig.colorbar_bivar(ca, cax=ax)
    assert ax.get_yscale() == 'function'
    assert ax.get_xscale() == 'function'


def test_colorbar_bivar_not_via_fig():
    # This test makes a colorbar without calling
    # fig.colorbar_bivar() and without a ColorizingArtist
    norm = mpl.colors.MultiNorm(['linear', 'linear'])
    ca = mpl.colorizer.Colorizer('BiOrangeBlue', norm)
    fig, ax = plt.subplots()
    cb = mcolorbar_bivar.BivarColorbar(ax, ca)
    assert cb._colorizer is ca


def test_bivar_cbar_cla_unpatches_ax():
    norm = mpl.colors.MultiNorm(['linear', 'linear'])
    ca = mpl.colorizer.Colorizer('BiOrangeBlue', norm)
    fig, ax = plt.subplots()
    cb = mcolorbar_bivar.BivarColorbar(ax, ca)
    patched = ['cla', '_get_view', '_set_view', '_set_view_from_bbox', 'drag_pan']
    assert all(name in vars(ax) for name in patched)
    ax.cla()
    assert not any(name in vars(ax) for name in patched)
    # ax is read-only
    with pytest.raises(AttributeError):
        cb.ax = ax


@image_comparison(['bivar_cbar_not_rasterized.png',
                   ], style='mpl20',
                  remove_text=True, savefig_kwarg={'dpi': 40}, tol=0.05)
def test_bivar_cbar_not_rasterized():
    cm = mpl.bivar_colormaps['BiOrangeBlue'].resampled((5, 3))
    fig, axes = plt.subplots(1, 1)
    mp = axes.imshow([[[100, 1], [10, 1]], [[0.5, 0.2], [0.3, 1]]],
                     cmap=cm,
                     interpolation='nearest'
                     )
    fig.colorbar_bivar(mp)
