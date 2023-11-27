import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import (image_comparison,
                                           remove_ticks_and_titles)
import matplotlib.colors as mcolors
import matplotlib as mpl
import pytest
import re

from io import BytesIO
from PIL import Image
import base64


@image_comparison(["bivariate_visualizations.png"])
def test_bivariate_visualizations():
    x_0 = np.arange(25, dtype='float32').reshape(5, 5) % 5
    x_1 = np.arange(25, dtype='float32').reshape(5, 5).T % 5

    fig, axes = plt.subplots(1, 6, figsize=(10, 2))

    axes[0].imshow((x_0, x_1), cmap='BiPeak', interpolation='nearest')
    axes[1].matshow((x_0, x_1), cmap='BiPeak')
    axes[2].pcolor((x_0, x_1), cmap='BiPeak')
    axes[3].pcolormesh((x_0, x_1), cmap='BiPeak')

    x = np.arange(5)
    y = np.arange(5)
    X, Y = np.meshgrid(x, y)
    axes[4].pcolormesh(X, Y, (x_0, x_1), cmap='BiPeak')

    patches = [
        mpl.patches.Wedge((.3, .7), .1, 0, 360),             # Full circle
        mpl.patches.Wedge((.7, .8), .2, 0, 360, width=0.05),  # Full ring
        mpl.patches.Wedge((.8, .3), .2, 0, 45),              # Full sector
        mpl.patches.Wedge((.8, .3), .2, 22.5, 90, width=0.10),  # Ring sector
    ]
    colors_0 = np.arange(len(patches)) // 2
    colors_1 = np.arange(len(patches)) % 2
    p = mpl.collections.PatchCollection(patches, cmap='BiPeak', alpha=0.5)
    p.set_array((colors_0, colors_1))
    axes[5].add_collection(p)

    remove_ticks_and_titles(fig)


@image_comparison(["multivariate_visualizations.png"])
def test_multivariate_visualizations():
    x_0 = np.arange(25, dtype='float32').reshape(5, 5) % 5
    x_1 = np.arange(25, dtype='float32').reshape(5, 5).T % 5
    x_2 = np.arange(25, dtype='float32').reshape(5, 5) % 6

    fig, axes = plt.subplots(1, 6, figsize=(10, 2))

    axes[0].imshow((x_0, x_1, x_2), cmap='3VarAddA', interpolation='nearest')
    axes[1].matshow((x_0, x_1, x_2), cmap='3VarAddA')
    axes[2].pcolor((x_0, x_1, x_2), cmap='3VarAddA')
    axes[3].pcolormesh((x_0, x_1, x_2), cmap='3VarAddA')

    x = np.arange(5)
    y = np.arange(5)
    X, Y = np.meshgrid(x, y)
    axes[4].pcolormesh(X, Y, (x_0, x_1, x_2), cmap='3VarAddA')

    patches = [
        mpl.patches.Wedge((.3, .7), .1, 0, 360),             # Full circle
        mpl.patches.Wedge((.7, .8), .2, 0, 360, width=0.05),  # Full ring
        mpl.patches.Wedge((.8, .3), .2, 0, 45),              # Full sector
        mpl.patches.Wedge((.8, .3), .2, 22.5, 90, width=0.10),  # Ring sector
    ]
    colors_0 = np.arange(len(patches)) // 2
    colors_1 = np.arange(len(patches)) % 2
    colors_2 = np.arange(len(patches)) % 3
    p = mpl.collections.PatchCollection(patches, cmap='3VarAddA', alpha=0.5)
    p.set_array((colors_0, colors_1, colors_2))
    axes[5].add_collection(p)

    remove_ticks_and_titles(fig)


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


@image_comparison(["bivariate_cmap_shapes.png"])
def test_bivariate_cmap_shapes():
    x_0 = np.arange(100, dtype='float32').reshape(10, 10) % 10
    x_1 = np.arange(100, dtype='float32').reshape(10, 10).T % 10

    fig, axes = plt.subplots(1, 4, figsize=(10, 2))

    # shape = square
    axes[0].imshow((x_0, x_1), cmap='BiPeak', vmin=1, vmax=8, interpolation='nearest')
    # shape = cone
    axes[1].imshow((x_0, x_1), cmap='BiCone', vmin=0.5, vmax=8.5,
                   interpolation='nearest')

    # shape = ignore
    cmap = mpl.bivar_colormaps['BiCone']
    cmap.shape = 'ignore'
    axes[2].imshow((x_0, x_1), cmap=cmap, vmin=1, vmax=8, interpolation='nearest')

    # shape = circleignore
    cmap = mpl.bivar_colormaps['BiCone']
    cmap.shape = 'circleignore'
    axes[3].imshow((x_0, x_1), cmap=cmap, vmin=0.5, vmax=8.5, interpolation='nearest')
    remove_ticks_and_titles(fig)


@image_comparison(["multivariate_figimage.png"])
def test_multivariate_figimage():
    fig = plt.figure(figsize=(2, 2), dpi=100)
    x, y = np.ix_(np.arange(100) / 100.0, np.arange(100) / 100)
    z = np.sin(x**2 + y**2 - x*y)
    c = np.sin(20*x**2 + 50*y**2)
    img = np.stack((z, c))

    fig.figimage(img, xo=0, yo=0, origin='lower', cmap='BiPeak')
    fig.figimage(img[:, ::-1, :], xo=0, yo=100, origin='lower', cmap='BiPeak')
    fig.figimage(img[:, :, ::-1], xo=100, yo=0, origin='lower', cmap='BiPeak')
    fig.figimage(img[:, ::-1, ::-1], xo=100, yo=100, origin='lower', cmap='BiPeak')


@image_comparison(["multivar_cmap_call.png"])
def test_multivar_cmap_call():
    """
    This evaluates manual calls to a bivariate colormap
    The figure exists because implementing an image comparison
    is easier than anumeraical comparisons for mulitdimensional arrays
    """
    x_0 = np.arange(100, dtype='float32').reshape(10, 10) % 10
    x_1 = np.arange(100, dtype='float32').reshape(10, 10).T % 10

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))

    cmap = mpl.multivar_colormaps['2VarAddA']

    # call with 1D
    im = cmap((x_0[0]/9, x_1[::-1, 0]/9))
    axes[0].scatter(np.arange(10), np.arange(10), c=im)

    # call with 2D
    im = cmap((x_0/9, x_1/9))
    axes[1].imshow(im, interpolation='nearest')

    # call with 3D array
    im = cmap(((x_0/9, x_0/9),
               (x_1/9, x_1/9)))
    axes[2].imshow(im.reshape((20, 10, 4)), interpolation='nearest')

    # call with constant alpha, and data of type int
    im = cmap((x_0.astype('int')*25, x_1.astype('int')*25), alpha=0.5)
    axes[3].imshow(im, interpolation='nearest')

    # call with variable alpha
    im = cmap((x_0/9, x_1/9), alpha=(x_0/9)**2, bytes=True)
    axes[4].imshow(im, interpolation='nearest')

    remove_ticks_and_titles(fig)


def test_multivar_cmap_call_tuple():
    cmap = mpl.multivar_colormaps['2VarAddA']
    assert_array_equal(cmap((0.0, 0.0)), (0, 0, 0, 1))
    assert_array_equal(cmap((1.0, 1.0)), (1, 1, 1, 1))

    cmap = mpl.multivar_colormaps['2VarSubA']
    assert_array_equal(cmap((0.0, 0.0)), (1, 1, 1, 1))
    assert_allclose(cmap((1.0, 1.0)), (0, 0, 0, 1), atol=0.1)


@image_comparison(["bivariate_cmap_call.png"])
def test_bivariate_cmap_call():
    """
    This evaluates manual calls to a bivariate colormap
    The figure exists because implementing an image comparison
    is easier than anumeraical comparisons for mulitdimensional arrays
    """
    x_0 = np.arange(100, dtype='float32').reshape(10, 10) % 10
    x_1 = np.arange(100, dtype='float32').reshape(10, 10).T % 10

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))

    cmap = mpl.bivar_colormaps['BiCone']

    # call with 1D
    im = cmap((x_0[0]/9, x_1[::-1, 0]/9))
    axes[0].scatter(np.arange(10), np.arange(10), c=im)

    # call with 2D
    im = cmap((x_0/9, x_1/9))
    axes[1].imshow(im, interpolation='nearest')

    # call with 3D array
    im = cmap(((x_0/9, x_0/9),
               (x_1/9, x_1/9)))
    axes[2].imshow(im.reshape((20, 10, 4)), interpolation='nearest')

    # call with constant alpha, and data of type int
    im = cmap((x_0.astype('int')*25, x_1.astype('int')*25), alpha=0.5)
    axes[3].imshow(im, interpolation='nearest')

    # call with variable alpha
    im = cmap((x_0/9, x_1/9), alpha=(x_0/9)**2, bytes=True)
    axes[4].imshow(im, interpolation='nearest')

    remove_ticks_and_titles(fig)


def test_bivar_cmap_call_tuple():
    cmap = mpl.bivar_colormaps['BiOrangeBlue']
    assert_allclose(cmap((1.0, 1.0)), (1, 1, 1, 1), atol=0.01)
    assert_allclose(cmap((0.0, 0.0)), (0, 0, 0, 1), atol=0.1)


@image_comparison(["bivar_cmap_from_image.png"])
def test_bivar_cmap_from_image():
    """
    This tests the creation and use of a bivariate colormap
    generated from an image
    """
    # create bivariate colormap
    im = np.ones((10, 12, 4))
    im[:, :, 0] = np.arange(10)[:, np.newaxis]/10
    im[:, :, 1] = np.arange(12)[np.newaxis, :]/12
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(im, interpolation='nearest')

    # use bivariate colormap
    data_0 = np.arange(12).reshape((3, 4))
    data_1 = np.arange(12).reshape((4, 3)).T
    cmap = mpl.colors.BivarColormapFromImage(im, 'custom')
    axes[1].imshow((data_0, data_1), cmap=cmap,
                   interpolation='nearest')

    remove_ticks_and_titles(fig)


def test_wrong_multivar_clim_shape():
    fig, ax = plt.subplots()
    im = np.arange(24).reshape((2, 3, 4))
    with pytest.raises(ValueError, match="Unable to map the input for vmin"):
        ax.imshow(im, cmap='BiPeak', vmin=(None, None, None))
    with pytest.raises(ValueError, match="Unable to map the input for vmax"):
        ax.imshow(im, cmap='BiPeak', vmax=(None, None, None))


def test_wrong_multivar_norm_shape():
    fig, ax = plt.subplots()
    im = np.arange(24).reshape((2, 3, 4))
    with pytest.raises(ValueError, match="Unable to map the input for norm"):
        ax.imshow(im, cmap='BiPeak', norm=(None, None, None))


def test_wrong_multivar_data_shape():
    fig, ax = plt.subplots()
    im = np.arange(12).reshape((1, 3, 4))
    with pytest.raises(ValueError, match="the data must have a first dimension 2"):
        ax.imshow(im, cmap='BiPeak')
    im = np.arange(12).reshape((3, 4))
    with pytest.raises(ValueError, match="the data must have a first dimension 2"):
        ax.imshow(im, cmap='BiPeak')


def test_missing_multivar_cmap_imshow():
    fig, ax = plt.subplots()
    im = np.arange(200).reshape((2, 10, 10))
    with pytest.raises(TypeError,
                       match=("a valid colormap must be explicitly declared"
                              + ", for example cmap='BiOrangeBlue'")):
        ax.imshow(im)
    im = np.arange(300).reshape((3, 10, 10))
    with pytest.raises(TypeError,
                       match=("multivariate colormap must be explicitly declared"
                              + ", for example cmap='3VarAddA")):
        ax.imshow(im)
    im = np.arange(1000).reshape((10, 10, 10))
    with pytest.raises(TypeError,
                       match=re.escape("Invalid shape (10, 10, 10) for image data")):
        ax.imshow(im)


def test_setting_A_on_vectormappable():
    # correct use
    vm = mpl.cm.VectorMappable(cmap='3VarAddA')
    data = np.arange(3*25).reshape((3, 5, 5))
    vm._A = data

    # attempting to set wrong shape of data
    with pytest.raises(ValueError, match=re.escape(
      "For the selected colormap the data must have a first dimension 3"
                                                   )):
        data = np.arange(2*25).reshape((2, 5, 5))
        vm._A = data

    # wrong use -> set components independently
    # error raised on get
    vm = mpl.cm.VectorMappable(cmap='3VarAddA')
    data = np.arange(3*25).reshape((3, 5, 5))
    vm.scalars[0]._A = data[0]
    with pytest.raises(AttributeError, match=re.escape(
      "_A on has only been indedependently set on ScalarMappables, which is unsupported"
                                                       )):
        _ = vm._A

    # wrong use -> A set on the VectorMappable and separately on the ScalarMappable
    # error raised on get
    vm = mpl.cm.VectorMappable(cmap='3VarAddA')
    vm._A = data
    vm.scalars[0]._A = np.copy(data[0])
    with pytest.raises(AttributeError, match=re.escape(
      "_A on the ScalarMappables are not views of _multivar_A on the VectorMappable"
                                                       )):
        _ = vm._A


def test_setting_norm_on_vectormappable():
    # correct use
    vm = mpl.cm.VectorMappable(cmap='3VarAddA')
    vm.set_norm('linear')
    vm.set_norm(['linear', 'log', 'asinh'])

    # attempting to set wrong shape of norm
    with pytest.raises(ValueError, match=re.escape(
      "Unable to map the input for norm (('None', 'None')) to 3 variables"
                                                   )):
        vm.set_norm(('None', 'None'))


def test_setting_clim_on_vectormappable():
    # correct use
    vm = mpl.cm.VectorMappable(cmap='3VarAddA')
    vm.set_clim(0, 1)
    vm.set_clim([0, 0, 0], [1, 2, 3])
    # attempting to set wrong shape of vmin/vmax
    with pytest.raises(ValueError, match=re.escape(
      "Unable to map the input for vmin ([0, 0]) to 3 variables"
                                                   )):
        vm.set_clim(vmin=[0, 0])
    with pytest.raises(ValueError, match=re.escape(
      "Unable to map the input for vmax ([1, 2]) to 3 variables"
                                                   )):
        vm.set_clim(vmax=[1, 2])


def test_bivariate_repr_png():
    cmap = mpl.bivar_colormaps['BiCone']
    png = cmap._repr_png_()
    assert len(png) > 0
    img = Image.open(BytesIO(png))
    assert img.width > 0
    assert img.height > 0
    assert 'Title' in img.text
    assert 'Description' in img.text
    assert 'Author' in img.text
    assert 'Software' in img.text


def test_bivariate_repr_html():
    cmap = mpl.bivar_colormaps['BiCone']
    html = cmap._repr_html_()
    assert len(html) > 0
    png = cmap._repr_png_()
    assert base64.b64encode(png).decode('ascii') in html
    assert cmap.name in html
    assert html.startswith('<div')
    assert html.endswith('</div>')


def test_multivariate_repr_html():
    cmap = mpl.multivar_colormaps['4VarAddA']
    html = cmap._repr_html_()
    assert len(html) > 0
    for c in cmap.colormaps:
        png = c._repr_png_()
        assert base64.b64encode(png).decode('ascii') in html
    assert cmap.name in html
    assert html.startswith('<div')
    assert html.endswith('</div>')
