from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure

from matplotlib.testing.decorators import _image_directories, cleanup
from matplotlib import pyplot
import numpy


def compare_bytes_with_image(func, baseline_image, actual_image_as_bytes,
                             tolerance=0):
        baseline_dir, result_dir = _image_directories(func)

        actual_fname = os.path.join(result_dir, baseline_image) + '.png'
        expected_fname = os.path.join(baseline_dir, baseline_image) + '.png'
        with open(actual_fname, 'wb') as f:
            f.write(actual_image_as_bytes)

        err = compare_images(expected_fname, actual_fname, tolerance,
                             in_decorator=True)

        if not os.path.exists(expected_fname):
            raise ImageComparisonFailure(
                'image does not exist: %s' % expected_fname)

        if err:
            raise ImageComparisonFailure(
                'images not close: %(actual)s vs. %(expected)s '
                '(RMS %(rms).3f)' % err)


@cleanup
def test_return_bytes():
    fig = pyplot.figure()
    axis = fig.add_subplot(111)
    axis.plot([1,2,3,4,5,6,7,8], [1,4,9,2,4,3,6,8], 'b')
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    image_as_bytes = fig.savefig(None, format='png')

    compare_bytes_with_image(test_return_bytes, 'plot_to_png_bytes',
                             image_as_bytes)
    pyplot.close(fig)


@cleanup
def test_pyplot_return_bytes():
    fig = pyplot.figure()
    pyplot.plot([1,2,3,4,5,6,7,8], [1,4,9,2,4,3,6,8], 'r')
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    image_as_bytes = pyplot.savefig(None, format='png')

    compare_bytes_with_image(test_pyplot_return_bytes,
                             'pyplot_plot_to_png_bytes',
                             image_as_bytes)
    pyplot.close(fig)
