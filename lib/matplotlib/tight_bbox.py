"""
This module is to support *bbox_inches* option in savefig command.
"""

import warnings
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D



def adjust_bbox(fig, format, bbox_inches):
    """
    Temporarily adjust the figure so that only the specified area
    (bbox_inches) is saved.

    It modifies fig.bbox, fig.bbox_inches,
    fig.transFigure._boxout, and fig.patch.  While the figure size
    changes, the scale of the original figure is conserved.  A
    function which restores the original values are returned.
    """

    origBbox = fig.bbox
    origBboxInches = fig.bbox_inches
    _boxout = fig.transFigure._boxout

    asp_list = []
    locator_list = []
    for ax in fig.axes:
        pos = ax.get_position(original=False).frozen()
        locator_list.append(ax.get_axes_locator())
        asp_list.append(ax.get_aspect())

        def _l(a, r, pos=pos): return pos
        ax.set_axes_locator(_l)
        ax.set_aspect("auto")



    def restore_bbox():

        for ax, asp, loc in zip(fig.axes, asp_list, locator_list):
            ax.set_aspect(asp)
            ax.set_axes_locator(loc)

        fig.bbox = origBbox
        fig.bbox_inches = origBboxInches
        fig.transFigure._boxout = _boxout
        fig.transFigure.invalidate()
        fig.patch.set_bounds(0, 0, 1, 1)

    adjust_bbox_handler = _adjust_bbox_handler_d.get(format)
    if adjust_bbox_handler is not None:
        adjust_bbox_handler(fig, bbox_inches)
        return restore_bbox
    else:
        warnings.warn("bbox_inches option for %s backend is not implemented yet." % (format))
        return None


def adjust_bbox_png(fig, bbox_inches):
    """
    adjust_bbox for png (Agg) format
    """

    tr = fig.dpi_scale_trans

    _bbox = TransformedBbox(bbox_inches,
                            tr)
    x0, y0 = _bbox.x0, _bbox.y0
    fig.bbox_inches = Bbox.from_bounds(0, 0,
                                       bbox_inches.width,
                                       bbox_inches.height)

    x0, y0 = _bbox.x0, _bbox.y0
    w1, h1 = fig.bbox.width, fig.bbox.height
    fig.transFigure._boxout = Bbox.from_bounds(-x0, -y0,
                                                       w1, h1)
    fig.transFigure.invalidate()

    fig.bbox = TransformedBbox(fig.bbox_inches, tr)

    fig.patch.set_bounds(x0/w1, y0/h1,
                         fig.bbox.width/w1, fig.bbox.height/h1)


def adjust_bbox_pdf(fig, bbox_inches):
    """
    adjust_bbox for pdf & eps format
    """

    tr = Affine2D().scale(72)

    _bbox = TransformedBbox(bbox_inches, tr)

    fig.bbox_inches = Bbox.from_bounds(0, 0,
                                       bbox_inches.width,
                                       bbox_inches.height)
    x0, y0 = _bbox.x0, _bbox.y0
    f = 72. / fig.dpi
    w1, h1 = fig.bbox.width*f, fig.bbox.height*f
    fig.transFigure._boxout = Bbox.from_bounds(-x0, -y0,
                                                       w1, h1)
    fig.transFigure.invalidate()

    fig.bbox = TransformedBbox(fig.bbox_inches, tr)

    fig.patch.set_bounds(x0/w1, y0/h1,
                         fig.bbox.width/w1, fig.bbox.height/h1)


def process_figure_for_rasterizing(figure,
                                   bbox_inches_restore, mode):
    
    """
    This need to be called when figure dpi changes during the drawing
    (e.g., rasterizing). It recovers the bbox and re-adjust it with
    the new dpi.
    """

    bbox_inches, restore_bbox = bbox_inches_restore
    restore_bbox()
    r = adjust_bbox(figure, mode,
                    bbox_inches)

    return bbox_inches, r


_adjust_bbox_handler_d = {}
for format in ["png", "raw", "rgba", "jpg", "jpeg", "tiff"]:
    _adjust_bbox_handler_d[format] = adjust_bbox_png
for format in ["pdf", "eps", "svg", "svgz"]:
    _adjust_bbox_handler_d[format] = adjust_bbox_pdf
