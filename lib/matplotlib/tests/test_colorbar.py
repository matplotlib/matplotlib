import numpy as np
import pytest

from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm, PowerNorm
from matplotlib.cm import get_cmap
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import LogLocator, LogFormatter


def _get_cmap_norms():
    """
    Define a colormap and appropriate norms for each of the four
    possible settings of the extend keyword.

    Helper function for _colorbar_extension_shape and
    colorbar_extension_length.
    """
    # Create a color map and specify the levels it represents.
    cmap = get_cmap("RdBu", lut=5)
    clevs = [-5., -2.5, -.5, .5, 1.5, 3.5]
    # Define norms for the color maps.
    norms = dict()
    norms['neither'] = BoundaryNorm(clevs, len(clevs) - 1)
    norms['min'] = BoundaryNorm([-10] + clevs[1:], len(clevs) - 1)
    norms['max'] = BoundaryNorm(clevs[:-1] + [10], len(clevs) - 1)
    norms['both'] = BoundaryNorm([-10] + clevs[1:-1] + [10], len(clevs) - 1)
    return cmap, norms


def _colorbar_extension_shape(spacing):
    '''
    Produce 4 colorbars with rectangular extensions for either uniform
    or proportional spacing.

    Helper function for test_colorbar_extension_shape.
    '''
    # Get a colormap and appropriate norms for each extension type.
    cmap, norms = _get_cmap_norms()
    # Create a figure and adjust whitespace for subplots.
    fig = plt.figure()
    fig.subplots_adjust(hspace=4)
    for i, extension_type in enumerate(('neither', 'min', 'max', 'both')):
        # Get the appropriate norm and use it to get colorbar boundaries.
        norm = norms[extension_type]
        boundaries = values = norm.boundaries
        # Create a subplot.
        cax = fig.add_subplot(4, 1, i + 1)
        # Generate the colorbar.
        cb = ColorbarBase(cax, cmap=cmap, norm=norm,
                boundaries=boundaries, values=values,
                extend=extension_type, extendrect=True,
                orientation='horizontal', spacing=spacing)
        # Turn off text and ticks.
        cax.tick_params(left=False, labelleft=False,
                        bottom=False, labelbottom=False)
    # Return the figure to the caller.
    return fig


def _colorbar_extension_length(spacing):
    '''
    Produce 12 colorbars with variable length extensions for either
    uniform or proportional spacing.

    Helper function for test_colorbar_extension_length.
    '''
    # Get a colormap and appropriate norms for each extension type.
    cmap, norms = _get_cmap_norms()
    # Create a figure and adjust whitespace for subplots.
    fig = plt.figure()
    fig.subplots_adjust(hspace=.6)
    for i, extension_type in enumerate(('neither', 'min', 'max', 'both')):
        # Get the appropriate norm and use it to get colorbar boundaries.
        norm = norms[extension_type]
        boundaries = values = norm.boundaries
        for j, extendfrac in enumerate((None, 'auto', 0.1)):
            # Create a subplot.
            cax = fig.add_subplot(12, 1, i*3 + j + 1)
            # Generate the colorbar.
            ColorbarBase(cax, cmap=cmap, norm=norm,
                         boundaries=boundaries, values=values,
                         extend=extension_type, extendfrac=extendfrac,
                         orientation='horizontal', spacing=spacing)
            # Turn off text and ticks.
            cax.tick_params(left=False, labelleft=False,
                            bottom=False, labelbottom=False)
    # Return the figure to the caller.
    return fig


@image_comparison(
        baseline_images=['colorbar_extensions_shape_uniform',
                         'colorbar_extensions_shape_proportional'],
        extensions=['png'])
def test_colorbar_extension_shape():
    '''Test rectangular colorbar extensions.'''
    # Create figures for uniform and proportionally spaced colorbars.
    _colorbar_extension_shape('uniform')
    _colorbar_extension_shape('proportional')


@image_comparison(baseline_images=['colorbar_extensions_uniform',
                                   'colorbar_extensions_proportional'],
                  extensions=['png'])
def test_colorbar_extension_length():
    '''Test variable length colorbar extensions.'''
    # Create figures for uniform and proportionally spaced colorbars.
    _colorbar_extension_length('uniform')
    _colorbar_extension_length('proportional')


@image_comparison(baseline_images=['cbar_with_orientation',
                                   'cbar_locationing',
                                   'double_cbar',
                                   'cbar_sharing',
                                   ],
                  extensions=['png'], remove_text=True,
                  savefig_kwarg={'dpi': 40})
def test_colorbar_positioning():
    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]

    # -------------------
    plt.figure()
    plt.contourf(data, levels=levels)
    plt.colorbar(orientation='horizontal', use_gridspec=False)

    locations = ['left', 'right', 'top', 'bottom']
    plt.figure()
    for i, location in enumerate(locations):
        plt.subplot(2, 2, i + 1)
        plt.contourf(data, levels=levels)
        plt.colorbar(location=location, use_gridspec=False)

    # -------------------
    plt.figure()
    # make some other data (random integers)
    data_2nd = np.array([[2, 3, 2, 3], [1.5, 2, 2, 3], [2, 3, 3, 4]])
    # make the random data expand to the shape of the main data
    data_2nd = np.repeat(np.repeat(data_2nd, 10, axis=1), 10, axis=0)

    color_mappable = plt.contourf(data, levels=levels, extend='both')
    # test extend frac here
    hatch_mappable = plt.contourf(data_2nd, levels=[1, 2, 3], colors='none',
                                  hatches=['/', 'o', '+'], extend='max')
    plt.contour(hatch_mappable, colors='black')

    plt.colorbar(color_mappable, location='left', label='variable 1',
                 use_gridspec=False)
    plt.colorbar(hatch_mappable, location='right', label='variable 2',
                 use_gridspec=False)

    # -------------------
    plt.figure()
    ax1 = plt.subplot(211, anchor='NE', aspect='equal')
    plt.contourf(data, levels=levels)
    ax2 = plt.subplot(223)
    plt.contourf(data, levels=levels)
    ax3 = plt.subplot(224)
    plt.contourf(data, levels=levels)

    plt.colorbar(ax=[ax2, ax3, ax1], location='right', pad=0.0, shrink=0.5,
                 panchor=False, use_gridspec=False)
    plt.colorbar(ax=[ax2, ax3, ax1], location='left', shrink=0.5,
                 panchor=False, use_gridspec=False)
    plt.colorbar(ax=[ax1], location='bottom', panchor=False,
                 anchor=(0.8, 0.5), shrink=0.6, use_gridspec=False)


@image_comparison(baseline_images=['cbar_with_subplots_adjust'],
                  extensions=['png'], remove_text=True,
                  savefig_kwarg={'dpi': 40})
def test_gridspec_make_colorbar():
    plt.figure()
    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]

    plt.subplot(121)
    plt.contourf(data, levels=levels)
    plt.colorbar(use_gridspec=True, orientation='vertical')

    plt.subplot(122)
    plt.contourf(data, levels=levels)
    plt.colorbar(use_gridspec=True, orientation='horizontal')

    plt.subplots_adjust(top=0.95, right=0.95, bottom=0.2, hspace=0.25)


@image_comparison(baseline_images=['colorbar_single_scatter'],
                  extensions=['png'], remove_text=True,
                  savefig_kwarg={'dpi': 40})
def test_colorbar_single_scatter():
    # Issue #2642: if a path collection has only one entry,
    # the norm scaling within the colorbar must ensure a
    # finite range, otherwise a zero denominator will occur in _locate.
    plt.figure()
    x = np.arange(4)
    y = x.copy()
    z = np.ma.masked_greater(np.arange(50, 54), 50)
    cmap = plt.get_cmap('jet', 16)
    cs = plt.scatter(x, y, z, c=z, cmap=cmap)
    plt.colorbar(cs)


@pytest.mark.parametrize('use_gridspec', [False, True],
                         ids=['no gridspec', 'with gridspec'])
def test_remove_from_figure(use_gridspec):
    """
    Test `remove_from_figure` with the specified ``use_gridspec`` setting
    """
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2], [3, 4], cmap="spring")
    sc.set_array(np.array([5, 6]))
    pre_figbox = np.array(ax.figbox)
    cb = fig.colorbar(sc, use_gridspec=use_gridspec)
    fig.subplots_adjust()
    cb.remove()
    fig.subplots_adjust()
    post_figbox = np.array(ax.figbox)
    assert (pre_figbox == post_figbox).all()


def test_colorbarbase():
    # smoke test from #3805
    ax = plt.gca()
    ColorbarBase(ax, plt.cm.bone)


@image_comparison(
    baseline_images=['colorbar_closed_patch'],
    remove_text=True)
def test_colorbar_closed_patch():
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_axes([0.05, 0.85, 0.9, 0.1])
    ax2 = fig.add_axes([0.1, 0.65, 0.75, 0.1])
    ax3 = fig.add_axes([0.05, 0.45, 0.9, 0.1])
    ax4 = fig.add_axes([0.05, 0.25, 0.9, 0.1])
    ax5 = fig.add_axes([0.05, 0.05, 0.9, 0.1])

    cmap = get_cmap("RdBu", lut=5)

    im = ax1.pcolormesh(np.linspace(0, 10, 16).reshape((4, 4)), cmap=cmap)
    values = np.linspace(0, 10, 5)

    with rc_context({'axes.linewidth': 16}):
        plt.colorbar(im, cax=ax2, cmap=cmap, orientation='horizontal',
                     extend='both', extendfrac=0.5, values=values)
        plt.colorbar(im, cax=ax3, cmap=cmap, orientation='horizontal',
                     extend='both', values=values)
        plt.colorbar(im, cax=ax4, cmap=cmap, orientation='horizontal',
                     extend='both', extendrect=True, values=values)
        plt.colorbar(im, cax=ax5, cmap=cmap, orientation='horizontal',
                     extend='neither', values=values)


def test_colorbar_ticks():
    # test fix for #5673
    fig, ax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    X, Y = np.meshgrid(x, y)
    Z = X * Y
    clevs = np.array([-12, -5, 0, 5, 12], dtype=float)
    colors = ['r', 'g', 'b', 'c']
    cs = ax.contourf(X, Y, Z, clevs, colors=colors)
    cbar = fig.colorbar(cs, ax=ax, extend='neither',
                        orientation='horizontal', ticks=clevs)
    assert len(cbar.ax.xaxis.get_ticklocs()) == len(clevs)


def test_colorbar_minorticks_on_off():
    # test for github issue #11510 and PR #11584
    np.random.seed(seed=12345)
    data = np.random.randn(20, 20)
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots()
        # purposefully setting vmin and vmax to odd fractions
        # so as to check for the correct locations of the minor ticks
        im = ax.pcolormesh(data, vmin=-2.3, vmax=3.3)

        cbar = fig.colorbar(im, extend='both')
        cbar.minorticks_on()
        correct_minorticklocs = np.array([-2.2, -1.8, -1.6, -1.4, -1.2, -0.8,
                                          -0.6, -0.4, -0.2, 0.2, 0.4, 0.6,
                                           0.8, 1.2, 1.4, 1.6, 1.8, 2.2, 2.4,
                                           2.6, 2.8, 3.2])
        # testing after minorticks_on()
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_minorticklocs(),
                                       correct_minorticklocs)
        cbar.minorticks_off()
        # testing after minorticks_off()
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_minorticklocs(),
                                       np.array([]))

        im.set_clim(vmin=-1.2, vmax=1.2)
        cbar.minorticks_on()
        correct_minorticklocs = np.array([-1.2, -1.1, -0.9, -0.8, -0.7, -0.6,
                                          -0.4, -0.3, -0.2, -0.1,  0.1, 0.2,
                                           0.3,  0.4,  0.6,  0.7,  0.8,  0.9,
                                           1.1,  1.2])
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_minorticklocs(),
                                       correct_minorticklocs)


def test_colorbar_autoticks():
    # Test new autotick modes. Needs to be classic because
    # non-classic doesn't go this route.
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots(2, 1)
        x = np.arange(-3.0, 4.001)
        y = np.arange(-4.0, 3.001)
        X, Y = np.meshgrid(x, y)
        Z = X * Y
        pcm = ax[0].pcolormesh(X, Y, Z)
        cbar = fig.colorbar(pcm, ax=ax[0], extend='both',
                            orientation='vertical')

        pcm = ax[1].pcolormesh(X, Y, Z)
        cbar2 = fig.colorbar(pcm, ax=ax[1], extend='both',
                            orientation='vertical', shrink=0.4)
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_ticklocs(),
                np.arange(-10, 11., 5.))
        np.testing.assert_almost_equal(cbar2.ax.yaxis.get_ticklocs(),
                np.arange(-10, 11., 10.))


def test_colorbar_autotickslog():
    # Test new autotick modes...
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots(2, 1)
        x = np.arange(-3.0, 4.001)
        y = np.arange(-4.0, 3.001)
        X, Y = np.meshgrid(x, y)
        Z = X * Y
        pcm = ax[0].pcolormesh(X, Y, 10**Z, norm=LogNorm())
        cbar = fig.colorbar(pcm, ax=ax[0], extend='both',
                            orientation='vertical')

        pcm = ax[1].pcolormesh(X, Y, 10**Z, norm=LogNorm())
        cbar2 = fig.colorbar(pcm, ax=ax[1], extend='both',
                            orientation='vertical', shrink=0.4)
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_ticklocs(),
                10**np.arange(-12, 12.2, 4.))
        np.testing.assert_almost_equal(cbar2.ax.yaxis.get_ticklocs(),
                10**np.arange(-12, 13., 12.))


def test_colorbar_get_ticks():
    # test feature for #5792
    plt.figure()
    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]

    plt.subplot()
    plt.contourf(data, levels=levels)

    # testing getter for user set ticks
    userTicks = plt.colorbar(ticks=[0, 600, 1200])
    assert userTicks.get_ticks().tolist() == [0, 600, 1200]

    # testing for getter after calling set_ticks
    userTicks.set_ticks([600, 700, 800])
    assert userTicks.get_ticks().tolist() == [600, 700, 800]

    # testing for getter after calling set_ticks with some ticks out of bounds
    userTicks.set_ticks([600, 1300, 1400, 1500])
    assert userTicks.get_ticks().tolist() == [600]

    # testing getter when no ticks are assigned
    defTicks = plt.colorbar(orientation='horizontal')
    assert defTicks.get_ticks().tolist() == levels


def test_colorbar_lognorm_extension():
    # Test that colorbar with lognorm is extended correctly
    f, ax = plt.subplots()
    cb = ColorbarBase(ax, norm=LogNorm(vmin=0.1, vmax=1000.0),
                      orientation='vertical', extend='both')
    assert cb._values[0] >= 0.0


def test_colorbar_powernorm_extension():
    # Test that colorbar with powernorm is extended correctly
    f, ax = plt.subplots()
    cb = ColorbarBase(ax, norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0),
                      orientation='vertical', extend='both')
    assert cb._values[0] >= 0.0


def test_colorbar_axes_kw():
    # test fix for #8493: This does only test, that axes-related keywords pass
    # and do not raise an exception.
    plt.figure()
    plt.imshow(([[1, 2], [3, 4]]))
    plt.colorbar(orientation='horizontal', fraction=0.2, pad=0.2, shrink=0.5,
                 aspect=10, anchor=(0., 0.), panchor=(0., 1.))


def test_colorbar_log_minortick_labels():
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots()
        pcm = ax.imshow([[10000, 50000]], norm=LogNorm())
        cb = fig.colorbar(pcm)
        fig.canvas.draw()
        lb = cb.ax.yaxis.get_ticklabels(which='both')
        expected = [r'$\mathdefault{10^{4}}$',
                    r'$\mathdefault{2\times10^{4}}$',
                    r'$\mathdefault{3\times10^{4}}$',
                    r'$\mathdefault{4\times10^{4}}$']
        for l, exp in zip(lb, expected):
            assert l.get_text() == exp


def test_colorbar_renorm():
    x, y = np.ogrid[-4:4:31j, -4:4:31j]
    z = 120000*np.exp(-x**2 - y**2)

    fig, ax = plt.subplots()
    im = ax.imshow(z)
    cbar = fig.colorbar(im)

    norm = LogNorm(z.min(), z.max())
    im.set_norm(norm)
    cbar.set_norm(norm)
    cbar.locator = LogLocator()
    cbar.formatter = LogFormatter()
    cbar.update_normal(im)
    assert np.isclose(cbar.vmin, z.min())

    norm = LogNorm(z.min() * 1000, z.max() * 1000)
    im.set_norm(norm)
    cbar.set_norm(norm)
    cbar.update_normal(im)
    assert np.isclose(cbar.vmin, z.min() * 1000)
    assert np.isclose(cbar.vmax, z.max() * 1000)


def test_colorbar_get_ticks():
    with rc_context({'_internal.classic_mode': False}):

        fig, ax = plt. subplots()
        np.random.seed(19680801)
        pc = ax.pcolormesh(np.random.rand(30, 30))
        cb = fig.colorbar(pc)
        np.testing.assert_allclose(cb.get_ticks(), [0.2, 0.4, 0.6, 0.8])
