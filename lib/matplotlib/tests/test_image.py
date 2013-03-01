from __future__ import print_function
import numpy as np

from matplotlib.testing.decorators import image_comparison, knownfailureif, cleanup
from matplotlib import rcParams
import matplotlib.pyplot as plt
from nose.tools import assert_raises
from numpy.testing import assert_array_equal

import io
import os

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


@image_comparison(baseline_images=['figimage-0', 'figimage-1'], extensions=['png'], tol=1.5e-3)
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
    # The goal is to test that imsave will accept arrays with ndim=3 where
    # the third dimension is color and alpha without raising any exceptions
    from numpy import random
    random.seed(1)
    data = random.rand(256, 128, 4)

    buff = io.BytesIO()
    plt.imsave(buff, data)

    buff.seek(0)
    arr_buf = plt.imread(buff)

    assert arr_buf.shape == data.shape

    # Unfortunately, the AGG process "flattens" the RGBA data
    # into an equivalent RGB data with no transparency. So we
    # Can't directly compare the arrays like we could in some
    # other imsave tests.

@image_comparison(baseline_images=['image_clip'])
def test_image_clip():
    from math import pi

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='hammer')

    d = [[1,2],[3,4]]

    im = ax.imshow(d, extent=(-pi,pi,-pi/2,pi/2))

@image_comparison(baseline_images=['imshow'], tol=1.5e-3, remove_text=True)
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


if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
