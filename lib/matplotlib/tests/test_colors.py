from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from nose.tools import assert_raises, assert_equal

import numpy as np
from numpy.testing.utils import assert_array_equal, assert_array_almost_equal

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison, cleanup


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
    LogNorm ignored clip, now it has the same
    behavior as Normalize, e.g., values > vmax are bigger than 1
    without clip, with clip they are 1.
    """
    ln = mcolors.LogNorm(clip=True, vmax=5)
    assert_array_equal(ln([1, 6]), [0, 1.0])


def test_PowerNorm():
    a = np.array([0, 0.5, 1, 1.5], dtype=np.float)
    pnorm = mcolors.PowerNorm(1)
    norm = mcolors.Normalize()
    assert_array_almost_equal(norm(a), pnorm(a))

    a = np.array([-0.5, 0, 2, 4, 8], dtype=np.float)
    expected = [0, 0, 1/16, 1/4, 1]
    pnorm = mcolors.PowerNorm(2, vmin=0, vmax=8)
    assert_array_almost_equal(pnorm(a), expected)
    assert_equal(pnorm(a[0]), expected[0])
    assert_equal(pnorm(a[2]), expected[2])
    assert_array_almost_equal(a[1:], pnorm.inverse(pnorm(a))[1:])

    # Clip = True
    a = np.array([-0.5, 0, 1, 8, 16], dtype=np.float)
    expected = [0, 0, 0, 1, 1]
    pnorm = mcolors.PowerNorm(2, vmin=2, vmax=8, clip=True)
    assert_array_almost_equal(pnorm(a), expected)
    assert_equal(pnorm(a[0]), expected[0])
    assert_equal(pnorm(a[-1]), expected[-1])

    # Clip = True at call time
    a = np.array([-0.5, 0, 1, 8, 16], dtype=np.float)
    expected = [0, 0, 0, 1, 1]
    pnorm = mcolors.PowerNorm(2, vmin=2, vmax=8, clip=False)
    assert_array_almost_equal(pnorm(a, clip=True), expected)
    assert_equal(pnorm(a[0], clip=True), expected[0])
    assert_equal(pnorm(a[-1], clip=True), expected[-1])


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
    expected = [0., 0.53980074, 0.826991, 1.02758204]
    assert_array_almost_equal(normed_vals, expected)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)

    # Ensure that specifying vmin returns the same result as above
    norm = mcolors.SymLogNorm(3, vmin=-30, vmax=5, linscale=1.2)
    normed_vals = norm(vals)
    assert_array_almost_equal(normed_vals, expected)


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


@image_comparison(baseline_images=['levels_and_colors'],
                  extensions=['png'])
def test_cmap_and_norm_from_levels_and_colors():
    data = np.linspace(-2, 4, 49).reshape(7, 7)
    levels = [-1, 2, 2.5, 3]
    colors = ['red', 'green', 'blue', 'yellow', 'black']
    extend = 'both'
    cmap, norm = mcolors.from_levels_and_colors(levels, colors, extend=extend)

    ax = plt.axes()
    m = plt.pcolormesh(data, cmap=cmap, norm=norm)
    plt.colorbar(m)

    # Hide the axes labels (but not the colorbar ones, as they are useful)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_visible(False)


def test_cmap_and_norm_from_levels_and_colors2():
    levels = [-1, 2, 2.5, 3]
    colors = ['red', (0, 1, 0), 'blue', (0.5, 0.5, 0.5), (0.0, 0.0, 0.0, 1.0)]
    clr = mcolors.colorConverter.to_rgba_array(colors)
    bad = (0.1, 0.1, 0.1, 0.1)
    no_color = (0.0, 0.0, 0.0, 0.0)
    masked_value = 'masked_value'

    # Define the test values which are of interest.
    # Note: levels are lev[i] <= v < lev[i+1]
    tests = [('both', None, {-2: clr[0],
                             -1: clr[1],
                             2: clr[2],
                             2.25: clr[2],
                             3: clr[4],
                             3.5: clr[4],
                             masked_value: bad}),

             ('min', -1, {-2: clr[0],
                          -1: clr[1],
                          2: clr[2],
                          2.25: clr[2],
                          3: no_color,
                          3.5: no_color,
                          masked_value: bad}),

             ('max', -1, {-2: no_color,
                          -1: clr[0],
                          2: clr[1],
                          2.25: clr[1],
                          3: clr[3],
                          3.5: clr[3],
                          masked_value: bad}),

             ('neither', -2, {-2: no_color,
                              -1: clr[0],
                              2: clr[1],
                              2.25: clr[1],
                              3: no_color,
                              3.5: no_color,
                              masked_value: bad}),
             ]

    for extend, i1, cases in tests:
        cmap, norm = mcolors.from_levels_and_colors(levels, colors[0:i1],
                                                    extend=extend)
        cmap.set_bad(bad)
        for d_val, expected_color in cases.items():
            if d_val == masked_value:
                d_val = np.ma.array([1], mask=True)
            else:
                d_val = [d_val]
            assert_array_equal(expected_color, cmap(norm(d_val))[0],
                               'Wih extend={0!r} and data '
                               'value={1!r}'.format(extend, d_val))

    assert_raises(ValueError, mcolors.from_levels_and_colors, levels, colors)


def test_rgb_hsv_round_trip():
    for a_shape in [(500, 500, 3), (500, 3), (1, 3), (3,)]:
        np.random.seed(0)
        tt = np.random.random(a_shape)
        assert_array_almost_equal(tt,
            mcolors.hsv_to_rgb(mcolors.rgb_to_hsv(tt)))
        assert_array_almost_equal(tt,
            mcolors.rgb_to_hsv(mcolors.hsv_to_rgb(tt)))


@cleanup
def test_autoscale_masked():
    # Test for #2336. Previously fully masked data would trigger a ValueError.
    data = np.ma.masked_all((12, 20))
    plt.pcolor(data)
    plt.draw()


def test_colors_no_float():
    # Gray must be a string to distinguish 3-4 grays from RGB or RGBA.

    def gray_from_float_rgb():
        return mcolors.colorConverter.to_rgb(0.4)

    def gray_from_float_rgba():
        return mcolors.colorConverter.to_rgba(0.4)

    assert_raises(ValueError, gray_from_float_rgb)
    assert_raises(ValueError, gray_from_float_rgba)


def test_light_source_shading_color_range():
    # see also
    #http://matplotlib.org/examples/pylab_examples/shading_example.html

    from matplotlib.colors import LightSource
    from matplotlib.colors import Normalize

    refinput = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    norm = Normalize(vmin=0, vmax=50)
    ls = LightSource(azdeg=0, altdeg=65)
    testoutput = ls.shade(refinput, plt.cm.jet, norm=norm)
    refoutput = np.array([
        [[0., 0., 0.58912656, 1.],
        [0., 0., 0.67825312, 1.],
        [0., 0., 0.76737968, 1.],
        [0., 0., 0.85650624, 1.]],
        [[0., 0., 0.9456328, 1.],
        [0., 0., 1., 1.],
        [0., 0.04901961, 1., 1.],
        [0., 0.12745098, 1., 1.]],
        [[0., 0.22156863, 1., 1.],
        [0., 0.3, 1., 1.],
        [0., 0.37843137, 1., 1.],
        [0., 0.45686275, 1., 1.]]
        ])
    assert_array_almost_equal(refoutput, testoutput)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
