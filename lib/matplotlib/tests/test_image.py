from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
import io
import os

from nose.plugins.attrib import attr

import numpy as np

from matplotlib.testing.decorators import (image_comparison,
                                           knownfailureif, cleanup)
from matplotlib.image import BboxImage, imread, NonUniformImage
from matplotlib.transforms import Bbox
from matplotlib import rcParams, rc_context
import matplotlib.pyplot as plt

from numpy.testing import assert_array_equal

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@image_comparison(baseline_images=['image_interps'])
def test_image_interps():
    'make the basic nearest, bilinear and bicubic interps'
    X = np.arange(100)
    X = X.reshape(5, 20)

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.imshow(X, interpolation='nearest')
    ax1.set_title('three interpolations')
    ax1.set_ylabel('nearest')

    ax2 = fig.add_subplot(312)
    ax2.imshow(X, interpolation='bilinear')
    ax2.set_ylabel('bilinear')

    ax3 = fig.add_subplot(313)
    ax3.imshow(X, interpolation='bicubic')
    ax3.set_ylabel('bicubic')

@image_comparison(baseline_images=['interp_nearest_vs_none'],
                  extensions=['pdf', 'svg'], remove_text=True)
def test_interp_nearest_vs_none():
    'Test the effect of "nearest" and "none" interpolation'
    # Setting dpi to something really small makes the difference very
    # visible. This works fine with pdf, since the dpi setting doesn't
    # affect anything but images, but the agg output becomes unusably
    # small.
    rcParams['savefig.dpi'] = 3
    X = np.array([[[218, 165, 32], [122, 103, 238]],
                  [[127, 255, 0], [255, 99, 71]]], dtype=np.uint8)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(X, interpolation='none')
    ax1.set_title('interpolation none')
    ax2 = fig.add_subplot(122)
    ax2.imshow(X, interpolation='nearest')
    ax2.set_title('interpolation nearest')


@image_comparison(baseline_images=['figimage-0', 'figimage-1'], extensions=['png'])
def test_figimage():
    'test the figimage method'

    for suppressComposite in False, True:
        fig = plt.figure(figsize=(2,2), dpi=100)
        fig.suppressComposite = suppressComposite
        x,y = np.ix_(np.arange(100.0)/100.0, np.arange(100.0)/100.0)
        z = np.sin(x**2 + y**2 - x*y)
        c = np.sin(20*x**2 + 50*y**2)
        img = z + c/5

        fig.figimage(img, xo=0, yo=0, origin='lower')
        fig.figimage(img[::-1,:], xo=0, yo=100, origin='lower')
        fig.figimage(img[:,::-1], xo=100, yo=0, origin='lower')
        fig.figimage(img[::-1,::-1], xo=100, yo=100, origin='lower')

@cleanup
def test_image_python_io():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1,2,3])
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    plt.imread(buffer)

@knownfailureif(not HAS_PIL)
def test_imread_pil_uint16():
    img = plt.imread(os.path.join(os.path.dirname(__file__),
                     'baseline_images', 'test_image', 'uint16.tif'))
    assert (img.dtype == np.uint16)
    assert np.sum(img) == 134184960

# def test_image_unicode_io():
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot([1,2,3])
#     fname = u"\u0a3a\u0a3a.png"
#     fig.savefig(fname)
#     plt.imread(fname)
#     os.remove(fname)

def test_imsave():
    # The goal here is that the user can specify an output logical DPI
    # for the image, but this will not actually add any extra pixels
    # to the image, it will merely be used for metadata purposes.

    # So we do the traditional case (dpi == 1), and the new case (dpi
    # == 100) and read the resulting PNG files back in and make sure
    # the data is 100% identical.
    from numpy import random
    random.seed(1)
    data = random.rand(256, 128)

    buff_dpi1 = io.BytesIO()
    plt.imsave(buff_dpi1, data, dpi=1)

    buff_dpi100 = io.BytesIO()
    plt.imsave(buff_dpi100, data, dpi=100)

    buff_dpi1.seek(0)
    arr_dpi1 = plt.imread(buff_dpi1)

    buff_dpi100.seek(0)
    arr_dpi100 = plt.imread(buff_dpi100)

    assert arr_dpi1.shape == (256, 128, 4)
    assert arr_dpi100.shape == (256, 128, 4)

    assert_array_equal(arr_dpi1, arr_dpi100)

def test_imsave_color_alpha():
    # Test that imsave accept arrays with ndim=3 where the third dimension is
    # color and alpha without raising any exceptions, and that the data is
    # acceptably preserved through a save/read roundtrip.
    from numpy import random
    random.seed(1)
    data = random.rand(256, 128, 4)

    buff = io.BytesIO()
    plt.imsave(buff, data)

    buff.seek(0)
    arr_buf = plt.imread(buff)

    # Recreate the float -> uint8 -> float32 conversion of the data
    data = (255*data).astype('uint8').astype('float32')/255
    # Wherever alpha values were rounded down to 0, the rgb values all get set
    # to 0 during imsave (this is reasonable behaviour).
    # Recreate that here:
    for j in range(3):
        data[data[:, :, 3] == 0, j] = 1

    assert_array_equal(data, arr_buf)


@cleanup
def test_cursor_data():
    from matplotlib.backend_bases import MouseEvent

    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(100).reshape(10, 10), origin='upper')

    x, y = 4, 4
    xdisp, ydisp = ax.transData.transform_point([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    z = im.get_cursor_data(event)
    assert z == 44, "Did not get 44, got %d" % z

    # Now try for a point outside the image
    # Tests issue #4957
    x, y = 10.1, 4
    xdisp, ydisp = ax.transData.transform_point([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    z = im.get_cursor_data(event)
    assert z is None, "Did not get None, got %d" % z

    # Hmm, something is wrong here... I get 0, not None...
    # But, this works further down in the tests with extents flipped
    #x, y = 0.1, -0.1
    #xdisp, ydisp = ax.transData.transform_point([x, y])
    #event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    #z = im.get_cursor_data(event)
    #assert z is None, "Did not get None, got %d" % z

    ax.clear()
    # Now try with the extents flipped.
    im = ax.imshow(np.arange(100).reshape(10, 10), origin='lower')

    x, y = 4, 4
    xdisp, ydisp = ax.transData.transform_point([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    z = im.get_cursor_data(event)
    assert z == 44, "Did not get 44, got %d" % z

    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(100).reshape(10, 10), extent=[0, 0.5, 0, 0.5])

    x, y = 0.25, 0.25
    xdisp, ydisp = ax.transData.transform_point([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    z = im.get_cursor_data(event)
    assert z == 55, "Did not get 55, got %d" % z

    # Now try for a point outside the image
    # Tests issue #4957
    x, y = 0.75, 0.25
    xdisp, ydisp = ax.transData.transform_point([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    z = im.get_cursor_data(event)
    assert z is None, "Did not get None, got %d" % z

    x, y = 0.01, -0.01
    xdisp, ydisp = ax.transData.transform_point([x, y])

    event = MouseEvent('motion_notify_event', fig.canvas, xdisp, ydisp)
    z = im.get_cursor_data(event)
    assert z is None, "Did not get None, got %d" % z


@image_comparison(baseline_images=['image_clip'])
def test_image_clip():
    from math import pi

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='hammer')

    d = [[1,2],[3,4]]

    im = ax.imshow(d, extent=(-pi,pi,-pi/2,pi/2))

@image_comparison(baseline_images=['image_cliprect'])
def test_image_cliprect():
    import matplotlib.patches as patches

    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = [[1,2],[3,4]]

    im = ax.imshow(d, extent=(0,5,0,5))

    rect = patches.Rectangle(xy=(1,1), width=2, height=2, transform=im.axes.transData)
    im.set_clip_path(rect)

@image_comparison(baseline_images=['imshow'], remove_text=True)
def test_imshow():
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    arr = np.arange(100).reshape((10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(arr, interpolation="bilinear", extent=(1,2,1,2))
    ax.set_xlim(0,3)
    ax.set_ylim(0,3)

@image_comparison(baseline_images=['no_interpolation_origin'], remove_text=True)
def test_no_interpolation_origin():
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.imshow(np.arange(100).reshape((2, 50)), origin="lower", interpolation='none')

    ax = fig.add_subplot(212)
    ax.imshow(np.arange(100).reshape((2, 50)), interpolation='none')

@image_comparison(baseline_images=['image_shift'], remove_text=True,
                  extensions=['pdf', 'svg'])
def test_image_shift():
    from matplotlib.colors import LogNorm

    imgData = [[1.0/(x) + 1.0/(y) for x in range(1,100)] for y in range(1,100)]
    tMin=734717.945208
    tMax=734717.946366

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(imgData, norm=LogNorm(), interpolation='none',
              extent=(tMin, tMax, 1, 100))
    ax.set_aspect('auto')

@cleanup
def test_image_edges():
    f = plt.figure(figsize=[1, 1])
    ax = f.add_axes([0, 0, 1, 1], frameon=False)

    data = np.tile(np.arange(12), 15).reshape(20, 9)

    im = ax.imshow(data, origin='upper',
                   extent=[-10, 10, -10, 10], interpolation='none',
                   cmap='gray'
                   )

    x = y = 2
    ax.set_xlim([-x, x])
    ax.set_ylim([-y, y])

    ax.set_xticks([])
    ax.set_yticks([])

    buf = io.BytesIO()
    f.savefig(buf, facecolor=(0, 1, 0))

    buf.seek(0)

    im = plt.imread(buf)
    r, g, b, a = sum(im[:, 0])
    r, g, b, a = sum(im[:, -1])

    assert g != 100, 'Expected a non-green edge - but sadly, it was.'

@image_comparison(baseline_images=['image_composite_background'], remove_text=True)
def test_image_composite_background():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    arr = np.arange(12).reshape(4, 3)
    ax.imshow(arr, extent=[0, 2, 15, 0])
    ax.imshow(arr, extent=[4, 6, 15, 0])
    ax.set_facecolor((1, 0, 0, 0.5))
    ax.set_xlim([0, 12])

@image_comparison(baseline_images=['image_composite_alpha'], remove_text=True)
def test_image_composite_alpha():
    """
    Tests that the alpha value is recognized and correctly applied in the
    process of compositing images together.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    arr = np.zeros((11, 21, 4))
    arr[:, :, 0] = 1
    arr[:, :, 3] = np.concatenate((np.arange(0, 1.1, 0.1), np.arange(0, 1, 0.1)[::-1]))
    arr2 = np.zeros((21, 11, 4))
    arr2[:, :, 0] = 1
    arr2[:, :, 1] = 1
    arr2[:, :, 3] = np.concatenate((np.arange(0, 1.1, 0.1), np.arange(0, 1, 0.1)[::-1]))[:, np.newaxis]
    ax.imshow(arr, extent=[1, 2, 5, 0], alpha=0.3)
    ax.imshow(arr, extent=[2, 3, 5, 0], alpha=0.6)
    ax.imshow(arr, extent=[3, 4, 5, 0])
    ax.imshow(arr2, extent=[0, 5, 1, 2])
    ax.imshow(arr2, extent=[0, 5, 2, 3], alpha=0.6)
    ax.imshow(arr2, extent=[0, 5, 3, 4], alpha=0.3)
    ax.set_facecolor((0, 0.5, 0, 1))
    ax.set_xlim([0, 5])
    ax.set_ylim([5, 0])


@image_comparison(baseline_images=['rasterize_10dpi'], extensions=['pdf','svg'], remove_text=True)
def test_rasterize_dpi():
    # This test should check rasterized rendering with high output resolution.
    # It plots a rasterized line and a normal image with implot. So it will catch
    # when images end up in the wrong place in case of non-standard dpi setting.
    # Instead of high-res rasterization i use low-res.  Therefore the fact that the
    # resolution is non-standard is is easily checked by image_comparison.
    import numpy as np
    import matplotlib.pyplot as plt

    img = np.asarray([[1, 2], [3, 4]])

    fig, axes = plt.subplots(1, 3, figsize = (3, 1))

    axes[0].imshow(img)

    axes[1].plot([0,1],[0,1], linewidth=20., rasterized=True)
    axes[1].set(xlim = (0,1), ylim = (-1, 2))

    axes[2].plot([0,1],[0,1], linewidth=20.)
    axes[2].set(xlim = (0,1), ylim = (-1, 2))

    # Low-dpi PDF rasterization errors prevent proper image comparison tests.
    # Hide detailed structures like the axes spines.
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    rcParams['savefig.dpi'] = 10


@image_comparison(baseline_images=['bbox_image_inverted'],
                  extensions=['png', 'pdf'])
def test_bbox_image_inverted():
    # This is just used to produce an image to feed to BboxImage
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot([1, 2, 3])

    im_buffer = io.BytesIO()
    fig.savefig(im_buffer)
    im_buffer.seek(0)
    image = imread(im_buffer)

    bbox_im = BboxImage(Bbox([[100, 100], [0, 0]]))
    bbox_im.set_data(image)
    axes.add_artist(bbox_im)


@cleanup
def test_get_window_extent_for_AxisImage():
    # Create a figure of known size (1000x1000 pixels), place an image
    # object at a given location and check that get_window_extent()
    # returns the correct bounding box values (in pixels).

    im = np.array([[0.25, 0.75, 1.0, 0.75], [0.1, 0.65, 0.5, 0.4], \
        [0.6, 0.3, 0.0, 0.2], [0.7, 0.9, 0.4, 0.6]])
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.subplot()
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    im_obj = ax.imshow(im, extent=[0.4, 0.7, 0.2, 0.9], interpolation='nearest')

    fig.canvas.draw()
    renderer = fig.canvas.renderer
    im_bbox = im_obj.get_window_extent(renderer)

    assert_array_equal(im_bbox.get_points(), [[400, 200], [700, 900]])


@image_comparison(baseline_images=['zoom_and_clip_upper_origin'],
                  remove_text=True,
                  extensions=['png'])
def test_zoom_and_clip_upper_origin():
    image = np.arange(100)
    image = image.reshape((10, 10))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.set_ylim(2.0, -0.5)
    ax.set_xlim(-0.5, 2.0)


@cleanup
def test_nonuniformimage_setcmap():
    ax = plt.gca()
    im = NonUniformImage(ax)
    im.set_cmap('Blues')


@cleanup
def test_nonuniformimage_setnorm():
    ax = plt.gca()
    im = NonUniformImage(ax)
    im.set_norm(plt.Normalize())

@knownfailureif(not HAS_PIL)
@cleanup
def test_jpeg_alpha():
    plt.figure(figsize=(1, 1), dpi=300)
    # Create an image that is all black, with a gradient from 0-1 in
    # the alpha channel from left to right.
    im = np.zeros((300, 300, 4), dtype=np.float)
    im[..., 3] = np.linspace(0.0, 1.0, 300)

    plt.figimage(im)

    buff = io.BytesIO()
    with rc_context({'savefig.facecolor': 'red'}):
        plt.savefig(buff, transparent=True, format='jpg', dpi=300)

    buff.seek(0)
    image = Image.open(buff)

    # If this fails, there will be only one color (all black). If this
    # is working, we should have all 256 shades of grey represented.
    print("num colors: ", len(image.getcolors(256)))
    assert len(image.getcolors(256)) >= 175 and len(image.getcolors(256)) <= 185
    # The fully transparent part should be red, not white or black
    # or anything else
    print("corner pixel: ", image.getpixel((0, 0)))
    assert image.getpixel((0, 0)) == (254, 0, 0)


@cleanup
def test_minimized_rasterized():
    # This ensures that the rasterized content in the colorbars is
    # only as thick as the colorbar, and doesn't extend to other parts
    # of the image.  See #5814.  While the original bug exists only
    # in Postscript, the best way to detect it is to generate SVG
    # and then parse the output to make sure the two colorbar images
    # are the same size.
    from xml.etree import ElementTree

    np.random.seed(0)
    data = np.random.rand(10, 10)

    fig, ax = plt.subplots(1, 2)
    p1 = ax[0].pcolormesh(data)
    p2 = ax[1].pcolormesh(data)

    plt.colorbar(p1, ax=ax[0])
    plt.colorbar(p2, ax=ax[1])

    buff = io.BytesIO()
    plt.savefig(buff, format='svg')

    buff = io.BytesIO(buff.getvalue())
    tree = ElementTree.parse(buff)
    width = None
    for image in tree.iter('image'):
        if width is None:
            width = image['width']
        else:
            if image['width'] != width:
                assert False


@attr('network')
def test_load_from_url():
    req = six.moves.urllib.request.urlopen(
        "http://matplotlib.org/_static/logo_sidebar_horiz.png")
    Z = plt.imread(req)


if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
