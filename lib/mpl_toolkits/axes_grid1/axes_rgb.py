from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import numpy as np
from .axes_divider import make_axes_locatable, Size, locatable_axes_factory

def make_rgb_axes(ax, pad=0.01, axes_class=None, add_all=True):
    """
    pad : fraction of the axes height.
    """

    divider = make_axes_locatable(ax)

    pad_size = Size.Fraction(pad, Size.AxesY(ax))

    xsize = Size.Fraction((1.-2.*pad)/3., Size.AxesX(ax))
    ysize = Size.Fraction((1.-2.*pad)/3., Size.AxesY(ax))

    divider.set_horizontal([Size.AxesX(ax), pad_size, xsize])
    divider.set_vertical([ysize, pad_size, ysize, pad_size, ysize])

    ax.set_axes_locator(divider.new_locator(0, 0, ny1=-1))

    ax_rgb = []
    if axes_class is None:
        try:
            axes_class = locatable_axes_factory(ax._axes_class)
        except AttributeError:
            axes_class = locatable_axes_factory(type(ax))

    for ny in [4, 2, 0]:
        ax1 = axes_class(ax.get_figure(),
                         ax.get_position(original=True),
                         sharex=ax, sharey=ax)
        locator = divider.new_locator(nx=2, ny=ny)
        ax1.set_axes_locator(locator)
        for t in ax1.yaxis.get_ticklabels() + ax1.xaxis.get_ticklabels():
            t.set_visible(False)
        try:
            for axis in ax1.axis.values():
                axis.major_ticklabels.set_visible(False)
        except AttributeError:
            pass

        ax_rgb.append(ax1)

    if add_all:
        fig = ax.get_figure()
        for ax1 in ax_rgb:
            fig.add_axes(ax1)

    return ax_rgb

#import matplotlib.axes as maxes


def imshow_rgb(ax, r, g, b, **kwargs):
    ny, nx = r.shape
    R = np.zeros([ny, nx, 3], dtype="d")
    R[:,:,0] = r
    G = np.zeros_like(R)
    G[:,:,1] = g
    B = np.zeros_like(R)
    B[:,:,2] = b

    RGB = R + G + B

    im_rgb = ax.imshow(RGB, **kwargs)

    return im_rgb


from .mpl_axes import Axes

class RGBAxesBase(object):

    def __init__(self, *kl, **kwargs):
        pad = kwargs.pop("pad", 0.0)
        add_all = kwargs.pop("add_all", True)
        axes_class = kwargs.pop("axes_class", None)




        if axes_class is None:
            axes_class = self._defaultAxesClass

        ax = axes_class(*kl, **kwargs)

        divider = make_axes_locatable(ax)

        pad_size = Size.Fraction(pad, Size.AxesY(ax))

        xsize = Size.Fraction((1.-2.*pad)/3., Size.AxesX(ax))
        ysize = Size.Fraction((1.-2.*pad)/3., Size.AxesY(ax))

        divider.set_horizontal([Size.AxesX(ax), pad_size, xsize])
        divider.set_vertical([ysize, pad_size, ysize, pad_size, ysize])

        ax.set_axes_locator(divider.new_locator(0, 0, ny1=-1))

        ax_rgb = []
        for ny in [4, 2, 0]:
            ax1 = axes_class(ax.get_figure(),
                             ax.get_position(original=True),
                             sharex=ax, sharey=ax, **kwargs)
            locator = divider.new_locator(nx=2, ny=ny)
            ax1.set_axes_locator(locator)
            ax1.axis[:].toggle(ticklabels=False)
            #for t in ax1.yaxis.get_ticklabels() + ax1.xaxis.get_ticklabels():
            #    t.set_visible(False)
            #if hasattr(ax1, "_axislines"):
            #    for axisline in ax1._axislines.values():
            #        axisline.major_ticklabels.set_visible(False)
            ax_rgb.append(ax1)

        self.RGB = ax
        self.R, self.G, self.B = ax_rgb

        if add_all:
            fig = ax.get_figure()
            fig.add_axes(ax)
            self.add_RGB_to_figure()

        self._config_axes()

    def _config_axes(self):
        for ax1 in [self.RGB, self.R, self.G, self.B]:
            #for sp1 in ax1.spines.values():
            #    sp1.set_color("w")
            ax1.axis[:].line.set_color("w")
            ax1.axis[:].major_ticks.set_mec("w")
            # for tick in ax1.xaxis.get_major_ticks() + ax1.yaxis.get_major_ticks():
            #     tick.tick1line.set_mec("w")
            #     tick.tick2line.set_mec("w")



    def add_RGB_to_figure(self):
        self.RGB.get_figure().add_axes(self.R)
        self.RGB.get_figure().add_axes(self.G)
        self.RGB.get_figure().add_axes(self.B)

    def imshow_rgb(self, r, g, b, **kwargs):
        ny, nx = r.shape
        R = np.zeros([ny, nx, 3], dtype="d")
        R[:,:,0] = r
        G = np.zeros_like(R)
        G[:,:,1] = g
        B = np.zeros_like(R)
        B[:,:,2] = b

        RGB = R + G + B

        im_rgb = self.RGB.imshow(RGB, **kwargs)
        im_r = self.R.imshow(R, **kwargs)
        im_g = self.G.imshow(G, **kwargs)
        im_b = self.B.imshow(B, **kwargs)

        return im_rgb, im_r, im_g, im_b


class RGBAxes(RGBAxesBase):
    _defaultAxesClass = Axes
