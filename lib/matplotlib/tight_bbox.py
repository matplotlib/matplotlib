"""
Helper module for the *bbox_inches* parameter in `.Figure.savefig`.
"""

import contextlib

from matplotlib.cbook import _setattr_cm
from matplotlib.transforms import Bbox, TransformedBbox, Affine2D


def adjust_bbox(fig, bbox_inches, fixed_dpi=None):
    """
    Temporarily adjust the figure so that only the specified area
    (bbox_inches) is saved.

    It modifies fig.bbox, fig.bbox_inches,
    fig.transFigure._boxout, and fig.patch.  While the figure size
    changes, the scale of the original figure is conserved.  A
    function which restores the original values are returned.
    """
    def no_op_apply_aspect(position=None):
        return

    stack = contextlib.ExitStack()

    stack.callback(fig.set_tight_layout, fig.get_tight_layout())
    fig.set_tight_layout(False)

    for ax in fig.axes:
        pos = ax.get_position(original=False).frozen()

        def _l(a, r, pos=pos):
            return pos

        stack.callback(ax.set_axes_locator, ax.get_axes_locator())
        ax.set_axes_locator(_l)

        # override the method that enforces the aspect ratio
        # on the Axes
        stack.enter_context(_setattr_cm(ax, apply_aspect=no_op_apply_aspect))

    if fixed_dpi is not None:
        tr = Affine2D().scale(fixed_dpi)
        dpi_scale = fixed_dpi / fig.dpi
    else:
        tr = Affine2D().scale(fig.dpi)
        dpi_scale = 1.

    _bbox = TransformedBbox(bbox_inches, tr)

    stack.enter_context(
        _setattr_cm(fig, bbox_inches=Bbox.from_bounds(
            0, 0, bbox_inches.width, bbox_inches.height)))
    x0, y0 = _bbox.x0, _bbox.y0
    w1, h1 = fig.bbox.width * dpi_scale, fig.bbox.height * dpi_scale
    stack.enter_context(
        _setattr_cm(fig.transFigure,
                    _boxout=Bbox.from_bounds(-x0, -y0, w1, h1)))
    fig.transFigure.invalidate()
    stack.callback(fig.transFigure.invalidate)

    stack.enter_context(
        _setattr_cm(fig, bbox=TransformedBbox(fig.bbox_inches, tr)))

    stack.callback(fig.patch.set_bounds, 0, 0, 1, 1)
    fig.patch.set_bounds(x0 / w1, y0 / h1,
                         fig.bbox.width / w1, fig.bbox.height / h1)

    return stack.close


def process_figure_for_rasterizing(fig, bbox_inches_restore, fixed_dpi=None):
    """
    A function that needs to be called when figure dpi changes during the
    drawing (e.g., rasterizing).  It recovers the bbox and re-adjust it with
    the new dpi.
    """

    bbox_inches, restore_bbox = bbox_inches_restore
    restore_bbox()
    r = adjust_bbox(fig, bbox_inches, fixed_dpi)

    return bbox_inches, r
