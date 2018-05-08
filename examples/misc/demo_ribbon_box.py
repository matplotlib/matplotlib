"""
===============
Demo Ribbon Box
===============

"""

import numpy as np

from matplotlib import cbook, colors as mcolors
from matplotlib.image import BboxImage
import matplotlib.pyplot as plt


class RibbonBox:

    original_image = plt.imread(
        cbook.get_sample_data("Minduka_Present_Blue_Pack.png"))
    cut_location = 70
    b_and_h = original_image[:, :, 2:3]
    color = original_image[:, :, 2:3] - original_image[:, :, 0:1]
    alpha = original_image[:, :, 3:4]
    nx = original_image.shape[1]

    def __init__(self, color):
        rgb = mcolors.to_rgba(color)[:3]
        self.im = np.dstack(
            [self.b_and_h - self.color * (1 - np.array(rgb)), self.alpha])

    def get_stretched_image(self, stretch_factor):
        stretch_factor = max(stretch_factor, 1)
        ny, nx, nch = self.im.shape
        ny2 = int(ny*stretch_factor)
        return np.vstack(
            [self.im[:self.cut_location],
             np.broadcast_to(
                 self.im[self.cut_location], (ny2 - ny, nx, nch)),
             self.im[self.cut_location:]])


class RibbonBoxImage(BboxImage):
    zorder = 1

    def __init__(self, bbox, color, **kwargs):
        super().__init__(bbox, **kwargs)
        self._ribbonbox = RibbonBox(color)

    def draw(self, renderer, *args, **kwargs):
        bbox = self.get_window_extent(renderer)
        stretch_factor = bbox.height / bbox.width

        ny = int(stretch_factor*self._ribbonbox.nx)
        if self.get_array() is None or self.get_array().shape[0] != ny:
            arr = self._ribbonbox.get_stretched_image(stretch_factor)
            self.set_array(arr)

        super().draw(renderer, *args, **kwargs)


if True:
    from matplotlib.transforms import Bbox, TransformedBbox
    from matplotlib.ticker import ScalarFormatter

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    fig, ax = plt.subplots()

    years = np.arange(2004, 2009)
    box_colors = [(0.8, 0.2, 0.2),
                  (0.2, 0.8, 0.2),
                  (0.2, 0.2, 0.8),
                  (0.7, 0.5, 0.8),
                  (0.3, 0.8, 0.7),
                  ]
    heights = np.random.random(years.shape) * 7000 + 3000

    fmt = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(fmt)

    for year, h, bc in zip(years, heights, box_colors):
        bbox0 = Bbox.from_extents(year - 0.4, 0., year + 0.4, h)
        bbox = TransformedBbox(bbox0, ax.transData)
        rb_patch = RibbonBoxImage(bbox, bc, interpolation="bicubic")

        ax.add_artist(rb_patch)

        ax.annotate(r"%d" % (int(h/100.)*100),
                    (year, h), va="bottom", ha="center")

    patch_gradient = BboxImage(ax.bbox, interpolation="bicubic", zorder=0.1)
    gradient = np.zeros((2, 2, 4))
    gradient[:, :, :3] = [1, 1, 0.]
    gradient[:, :, 3] = [[0.1, 0.3], [0.3, 0.5]]  # alpha channel
    patch_gradient.set_array(gradient)
    ax.add_artist(patch_gradient)

    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, 10000)

    plt.show()
