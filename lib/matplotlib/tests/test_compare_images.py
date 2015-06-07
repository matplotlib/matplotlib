from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import os
import shutil

from nose.tools import assert_equal, assert_not_equal, assert_almost_equal

from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import _image_directories


baseline_dir, result_dir = _image_directories(lambda: 'dummy func')


# Tests of the image comparison algorithm.
def image_comparison_expect_rms(im1, im2, tol, expect_rms):
    """Compare two images, expecting a particular RMS error.

    im1 and im2 are filenames relative to the baseline_dir directory.

    tol is the tolerance to pass to compare_images.

    expect_rms is the expected RMS value, or None. If None, the test will
    succeed if compare_images succeeds. Otherwise, the test will succeed if
    compare_images fails and returns an RMS error almost equal to this value.
    """
    im1 = os.path.join(baseline_dir, im1)
    im2_src = os.path.join(baseline_dir, im2)
    im2 = os.path.join(result_dir, im2)
    # Move im2 from baseline_dir to result_dir. This will ensure that
    # compare_images writes the diff file to result_dir, instead of trying to
    # write to the (possibly read-only) baseline_dir.
    shutil.copyfile(im2_src, im2)
    results = compare_images(im1, im2, tol=tol, in_decorator=True)

    if expect_rms is None:
        assert_equal(None, results)
    else:
        assert_not_equal(None, results)
        assert_almost_equal(expect_rms, results['rms'], places=4)


def test_image_compare_basic():
    #: Test comparison of an image and the same image with minor differences.

    # This expects the images to compare equal under normal tolerance, and have
    # a small RMS.
    im1 = 'basn3p02.png'
    im2 = 'basn3p02-minorchange.png'
    image_comparison_expect_rms(im1, im2, tol=10, expect_rms=None)

    # Now test with no tolerance.
    image_comparison_expect_rms(im1, im2, tol=0, expect_rms=6.50646)


def test_image_compare_1px_offset():
    #: Test comparison with an image that is shifted by 1px in the X axis.
    im1 = 'basn3p02.png'
    im2 = 'basn3p02-1px-offset.png'
    image_comparison_expect_rms(im1, im2, tol=0, expect_rms=90.15611)


def test_image_compare_half_1px_offset():
    #: Test comparison with an image with half the pixels shifted by 1px in
    #: the X axis.
    im1 = 'basn3p02.png'
    im2 = 'basn3p02-half-1px-offset.png'
    image_comparison_expect_rms(im1, im2, tol=0, expect_rms=63.75)


def test_image_compare_scrambled():
    #: Test comparison of an image and the same image scrambled.

    # This expects the images to compare completely different, with a very
    # large RMS.
    # Note: The image has been scrambled in a specific way, by having each
    # color component of each pixel randomly placed somewhere in the image. It
    # contains exactly the same number of pixels of each color value of R, G
    # and B, but in a totally different position.
    im1 = 'basn3p02.png'
    im2 = 'basn3p02-scrambled.png'
    # Test with no tolerance to make sure that we pick up even a very small RMS
    # error.
    image_comparison_expect_rms(im1, im2, tol=0, expect_rms=172.63582)


def test_image_compare_shade_difference():
    #: Test comparison of an image and a slightly brighter image.
    # The two images are solid color, with the second image being exactly 1
    # color value brighter.
    # This expects the images to compare equal under normal tolerance, and have
    # an RMS of exactly 1.
    im1 = 'all127.png'
    im2 = 'all128.png'
    image_comparison_expect_rms(im1, im2, tol=0, expect_rms=1.0)

    # Now test the reverse comparison.
    image_comparison_expect_rms(im2, im1, tol=0, expect_rms=1.0)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
