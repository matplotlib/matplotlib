import numpy as np

from matplotlib.testing.decorators import image_comparison, knownfailureif, cleanup
from matplotlib import rcParams
import matplotlib.pyplot as plt
from nose.tools import assert_raises
from numpy.testing import assert_array_equal

import cStringIO
import os

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

@image_comparison(baseline_images=['interp_nearest_vs_none'], extensions=['pdf', 'svg'])
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
    buffer = cStringIO.StringIO()
    fig.savefig(buffer)
    buffer.seek(0)
    plt.imread(buffer)

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

    buff_dpi1 = cStringIO.StringIO()
    plt.imsave(buff_dpi1, data, dpi=1)

    buff_dpi100 = cStringIO.StringIO()
    plt.imsave(buff_dpi100, data, dpi=100)

    buff_dpi1.seek(0)
    arr_dpi1 = plt.imread(buff_dpi1)

    buff_dpi100.seek(0)
    arr_dpi100 = plt.imread(buff_dpi100)

    assert arr_dpi1.shape == (256, 128, 4)
    assert arr_dpi100.shape == (256, 128, 4)

    assert_array_equal(arr_dpi1, arr_dpi100)

@image_comparison(baseline_images=['image_clip'])
def test_image_clip():
    from math import pi

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='hammer')

    d = [[1,2],[3,4]]

    im = ax.imshow(d, extent=(-pi,pi,-pi/2,pi/2))

@image_comparison(baseline_images=['imshow'], tol=1.5e-3)
def test_imshow():
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    arr = np.arange(100).reshape((10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(arr, interpolation="bilinear", extent=(1,2,1,2))
    ax.set_xlim(0,3)
    ax.set_ylim(0,3)

if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
