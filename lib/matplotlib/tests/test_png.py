from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import glob
import os

import numpy as np

from matplotlib.testing.decorators import image_comparison
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import sys
on_win = (sys.platform == 'win32')


@image_comparison(baseline_images=['pngsuite'], extensions=['png'],
                  tol=0.01 if on_win else 0)
def test_pngsuite():
    dirname = os.path.join(
        os.path.dirname(__file__),
        'baseline_images',
        'pngsuite')
    files = glob.glob(os.path.join(dirname, 'basn*.png'))
    files.sort()

    fig = plt.figure(figsize=(len(files), 2))

    for i, fname in enumerate(files):
        data = plt.imread(fname)
        cmap = None  # use default colormap
        if data.ndim == 2:
            # keep grayscale images gray
            cmap = cm.gray
        plt.imshow(data, extent=[i, i + 1, 0, 1], cmap=cmap)

    plt.gca().patch.set_facecolor("#ddffff")
    plt.gca().set_xlim(0, len(files))


def test_imread_png_uint16():
    from matplotlib import _png
    img = _png.read_png_int(os.path.join(os.path.dirname(__file__),
                                     'baseline_images/test_png/uint16.png'))

    assert (img.dtype == np.uint16)
    assert np.sum(img.flatten()) == 134184960
