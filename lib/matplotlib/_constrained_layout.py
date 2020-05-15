"""
Adjust subplot layouts so that there are no overlapping axes or axes
decorations.  All axes decorations are dealt with (labels, ticks, titles,
ticklabels) and some dependent artists are also dealt with (colorbar, suptitle,
legend).

Layout is done via `~matplotlib.gridspec`, with one constraint per gridspec,
so it is possible to have overlapping axes if the gridspecs overlap (i.e.
using `~matplotlib.gridspec.GridSpecFromSubplotSpec`).  Axes placed using
``figure.subplots()`` or ``figure.add_subplots()`` will participate in the
layout.  Axes manually placed via ``figure.add_axes()`` will not.

See Tutorial: :doc:`/tutorials/intermediate/constrainedlayout_guide`
"""

# Development Notes:

# What gets a layoutbox:
#  - figure
#    - gridspec
#      - subplotspec
#        EITHER:
#         - axes + pos for the axes (i.e. the total area taken by axis and
#            the actual "position" argument that needs to be sent to
#             ax.set_position.)
#           - The axes layout box will also encompass the legend, and that is
#             how legends get included (axes legends, not figure legends)
#         - colorbars are siblings of the axes if they are single-axes
#           colorbars
#        OR:
#         - a gridspec can be inside a subplotspec.
#           - subplotspec
#           EITHER:
#            - axes...
#           OR:
#            - gridspec... with arbitrary nesting...
#      - colorbars are siblings of the subplotspecs if they are multi-axes
#        colorbars.
#   - suptitle:
#      - right now suptitles are just stacked atop everything else in figure.
#        Could imagine suptitles being gridspec suptitles, but not implemented
#
#   Todo:    AnchoredOffsetbox connected to gridspecs or axes.  This would
#        be more general way to add extra-axes annotations.

import logging

import numpy as np

import matplotlib.figure as mfigure
import matplotlib.cbook as cbook
import matplotlib._layoutgrid as layoutgrid
from matplotlib.transforms import (Bbox, TransformedBbox, ScaledTranslation,
                                   IdentityTransform)

_log = logging.getLogger(__name__)


def _spans_overlap(span0, span1):
    return span0.start in span1 or span1.start in span0


######################################################
def do_constrained_layout(fig, renderer, h_pad, w_pad,
                          hspace=None, wspace=None):
    """
    Do the constrained_layout.  Called at draw time in
     ``figure.constrained_layout()``

    Parameters
    ----------
    fig : Figure
      is the ``figure`` instance to do the layout in.

    renderer : Renderer
      the renderer to use.

     h_pad, w_pad : float
       are in figure-normalized units, and are a padding around the axes
       elements.

     hspace, wspace : float
        are in fractions of the subplot sizes.

    """

    # Steps:
    #
    # 1. get a list of unique gridspecs in this figure.  Each gridspec will be
    # constrained separately.
    # 2. Check for gaps in the gridspecs.  i.e. if not every axes slot in the
    # gridspec has been filled.  If empty, add a ghost axis that is made so
    # that it cannot be seen (though visible=True).  This is needed to make
    # a blank spot in the layout.
    # 3. Compare the tight_bbox of each axes to its `position`, and assume that
    # the difference is the space needed by the elements around the edge of
    # the axes (decorations) like the title, ticklabels, x-labels, etc.  This
    # can include legends who overspill the axes boundaries.
    # 4. Constrain gridspec elements to line up:
    #     a) if colnum0 != colnumC, the two subplotspecs are stacked next to
    #     each other, with the appropriate order.
    #     b) if colnum0 == colnumC, line up the left or right side of the
    #     _poslayoutbox (depending if it is the min or max num that is equal).
    #     c) do the same for rows...
    # 5. The above doesn't constrain relative sizes of the _poslayoutboxes
    # at all, and indeed zero-size is a solution that the solver often finds
    # more convenient than expanding the sizes.  Right now the solution is to
    # compare subplotspec sizes (i.e. drowsC and drows0) and constrain the
    # larger _poslayoutbox to be larger than the ratio of the sizes. i.e. if
    # drows0 > drowsC, then ax._poslayoutbox > axc._poslayoutbox*drowsC/drows0.
    # This works fine *if* the decorations are similar between the axes.
    # If the larger subplotspec has much larger axes decorations, then the
    # constraint above is incorrect.
    #
    # We need the greater than in the above, in general, rather than an equals
    # sign.  Consider the case of the left column having 2 rows, and the right
    # column having 1 row.  We want the top and bottom of the _poslayoutboxes
    # to line up. So that means if there are decorations on the left column
    # axes they will be smaller than half as large as the right hand axis.
    #
    # This can break down if the decoration size for the right hand axis (the
    # margins) is very large.  There must be a math way to check for this case.

    invTransFig = fig.transPanel.inverted().transform_bbox

    # list of unique gridspecs that contain child axes:
    gss = set()
    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec'):
            gs = ax.get_subplotspec().get_gridspec()
            if gs._layoutgrid is not None:
                gss.add(gs)
    gss = list(gss)
    if len(gss) == 0:
        cbook._warn_external('There are no gridspecs with layoutboxes. '
                             'Possibly did not call parent GridSpec with the'
                             ' figure= keyword')

    #if fig._layoutbox.constrained_layout_called < 1:
    #    for gs in gss:
            # fill in any empty gridspec slots w/ ghost axes...
    #       _make_ghost_gridspec_slots(fig, gs)

    for _ in range(2):
        # do the algorithm twice.  This has to be done because decorators
        # change size after the first re-position (i.e. x/yticklabels get
        # larger/smaller).  This second reposition tends to be much milder,
        # so doing twice makes things work OK.
        _make_layout_margins(fig, renderer, h_pad=h_pad, w_pad=w_pad,
                             hspace=hspace, wspace=wspace)

        _make_margin_suptitles(fig, renderer, h_pad=h_pad, w_pad=w_pad)

        fig._layoutgrid.update_variables()
        if not _check_ok(fig):
            _reset_margins(fig)
            return

        for ax in fig.axes:
            _reposition_axes(ax, renderer, h_pad=h_pad, w_pad=w_pad)
        _reset_margins(fig)

def _check_ok(fig):
    """
    check that no axes have collapsed to zero size.  If they have
    stops....
    """
    for panel in fig.panels:
        ok = _check_ok(panel)
        if not ok:
            return False

    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec'):
            gs = ax.get_subplotspec().get_gridspec()
            lg = gs._layoutgrid

            for i in range(gs.nrows):
                for j in range(gs.ncols):
                    bb = lg.get_inner_bbox(i, j)
                    if bb.width <= 0 or bb.height <= 0:
                        return False
    return True

def _make_margin_suptitles(fig, renderer, *, w_pad=0, h_pad=0):
    ## just fig for now...
    for panel in fig.panels:
        _make_margin_suptitles(panel, renderer, w_pad=w_pad, h_pad=h_pad)
    invTransFig = fig.transPanel.inverted().transform_bbox

    if fig._suptitle is not None:
        bbox = invTransFig(fig._suptitle.get_tightbbox(renderer))
        fig._layoutgrid.edit_margin_min('top', bbox.height + 2 * h_pad)
    if 0:
        if fig._supxlabel is not None:
            bbox = invTransFig(fig._supxlabel.get_tightbbox(renderer))
            fig._layoutgrid.edit_margin_min('bottom', bbox.height + 2 * h_pad)

        if fig._supylabel is not None:
            bbox = invTransFig(fig._supylabel.get_tightbbox(renderer))
            fig._layoutgrid.edit_margin_min('left', bbox.width + 2 * w_pad)

def _make_layout_margins(fig, renderer, *, w_pad=0, h_pad=0,
                         hspace=0, wspace=0):
    """
    For each axes, make a margin between the *pos* layoutbox and the
    *axes* layoutbox be a minimum size that can accommodate the
    decorations on the axis.
    """

    for panel in fig.panels:
        _make_layout_margins(panel, renderer, w_pad=w_pad, h_pad=h_pad,
                             hspace=hspace, wspace=wspace)

    for ax in [a for a in fig.axes if hasattr(a, 'get_subplotspec')]:

        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()
        nrows, ncols = gs.get_geometry()

        if gs._layoutgrid is None:
            return

        pos, bbox = _get_pos_and_bbox(ax, renderer)
        # this can go wrong:
        if not (np.isfinite(bbox.width) and np.isfinite(bbox.height)):
            # just abort, this is likely a bad set of coordinates that
            # is transitory...
            return
        margin = {}

        margin['left'] = -bbox.x0 + pos.x0 + w_pad
        if ss.colspan.start > 0:
            margin['left'] += wspace / ncols  # interior padding
        margin['right'] = bbox.x1 - pos.x1 + w_pad
        if ss.colspan.stop < ncols:
            margin['right'] += wspace / ncols

        # remember that rows are ordered from top:
        margin['bottom'] = pos.y0 - bbox.y0 + h_pad
        if ss.rowspan.start > 0:
            margin['bottom'] += hspace / nrows
        margin['top'] = -pos.y1 + bbox.y1 + h_pad
        if ss.rowspan.stop < nrows:
            margin['top'] += hspace / nrows

        # increase margin for colorbars...
        for cbax in ax._colorbars:
            cbp_rspan, cbp_cspan = _get_cb_parent_spans(cbax)
            loc = cbax._colorbar_info['location']
            if loc == 'right':
                if cbp_cspan.stop == ss.colspan.stop:
                    cbpos, cbbbox = _get_pos_and_bbox(cbax, renderer)
                    margin[loc] += cbbbox.width + w_pad * 2
                    print(cbbbox.width )
            elif loc == 'left':
                if cbp_cspan.start == ss.colspan.start:
                    cbpos, cbbbox = _get_pos_and_bbox(cbax, renderer)
                    margin[loc] += cbbbox.width + w_pad * 2
            elif loc == 'top':
                if cbp_rspan.start == ss.rowspan.start:
                    cbpos, cbbbox = _get_pos_and_bbox(cbax, renderer)
                    margin[loc] += cbbbox.height + h_pad * 2
            else:
                if cbp_rspan.stop == ss.rowspan.stop:
                    cbpos, cbbbox = _get_pos_and_bbox(cbax, renderer)
                    margin[loc] += cbbbox.height + h_pad * 2

        gs._layoutgrid.edit_margin_min('left', margin['left'], ss.colspan[0])
        gs._layoutgrid.edit_margin_min('right', margin['right'], ss.colspan[-1])

        gs._layoutgrid.edit_margin_min('top', margin['top'], ss.rowspan[0])
        gs._layoutgrid.edit_margin_min('bottom', margin['bottom'], ss.rowspan[-1])

def _get_cb_parent_spans(cbax):
    """
    Make margins for a big colorbar that has more than one parent...
    """
    rowspan = range(100, -100, -100)
    colspan = range(100, -100, -100)
    for parent in cbax._colorbar_info['parents']:
        ss = parent.get_subplotspec()
        print(ss.rowspan, rowspan)

        mn = min([ss.rowspan.start, rowspan.start])
        mx = max([ss.rowspan.stop, rowspan.stop])
        rowspan = range(mn, mx)
        colspan = range(min([ss.colspan.start, colspan.start]),
                        max([ss.colspan.stop, colspan.stop]))

    return rowspan, colspan


def _get_pos_and_bbox(ax, renderer):

    fig = ax.figure
    invTransFig = fig.transFigure.inverted()

    pos = ax.get_position(original=True)
    # pos is in panel co-ords, but we need in figure for the layout
    pos = pos.transformed(fig.transPanel+invTransFig)
    try:
        tightbbox = ax.get_tightbbox(renderer=renderer, for_layout_only=True)
    except TypeError:
        tightbbox = ax.get_tightbbox(renderer=renderer)

    if tightbbox is None:
        bbox = pos
    else:
        bbox = invTransFig.transform_bbox(tightbbox)
    return pos, bbox

def _reposition_axes(ax, renderer, *, w_pad=0, h_pad=0):
    """
    For each axes, make a margin between the *pos* layoutbox and the
    *axes* layoutbox be a minimum size that can accommodate the
    decorations on the axis.
    """
    if not hasattr(ax, 'get_subplotspec'):
        return

    fig = ax.figure
    # grid bbox is in Figure co-ordinates, but we specify in panel
    # co-ordinates...
    trans = fig.transFigure + fig.transPanel.inverted()

    ss = ax.get_subplotspec()
    gs = ss.get_gridspec()
    if gs._layoutgrid is None:
        return

    bbox = gs._layoutgrid.get_inner_bbox(rows=ss.rowspan, cols=ss.colspan)

    bboxouter = gs._layoutgrid.get_outer_bbox(rows=ss.rowspan,
                                              cols=ss.colspan)

    pos = ax.get_position(original=True)
    # transform from figure to panel for set_position:
    newbbox = trans.transform_bbox(bbox)
    ax._set_position(newbbox)

    # move the colorbars:

    # we need to keep track of some stuff if there is more than
    # one colorbar.
    oldw ={'right': 0, 'left': 0}
    oldh = {'bottom': 0, 'top': 0}
    for nn, cbax in enumerate(ax._colorbars):
        if ax == cbax._colorbar_info['parents'][0]:
            oldw, oldh = _reposition_colorbar(cbax, renderer,
                                            w_pad, h_pad,
                                            oldw, oldh)


def _reposition_colorbar(cbax, renderer, w_pad, h_pad, oldw, oldh):
    parents = cbax._colorbar_info['parents']
    gs = parents[0].get_gridspec()
    fig = cbax.figure
    trans = fig.transFigure + fig.transPanel.inverted()


    cb_rspans, cb_cspans = _get_cb_parent_spans(cbax)
    bboxouter = gs._layoutgrid.get_outer_bbox(rows=cb_rspans,
                                              cols=cb_cspans)
    pb = gs._layoutgrid.get_inner_bbox(rows=cb_rspans,
                                         cols=cb_cspans)
    location = cbax._colorbar_info['location']
    anchor = cbax._colorbar_info['anchor']
    fraction = cbax._colorbar_info['fraction']
    aspect = cbax._colorbar_info['aspect']
    shrink = cbax._colorbar_info['shrink']
    pad = cbax._colorbar_info['pad']
    parent_anchor = cbax._colorbar_info['panchor']

    cbpos, cbbbox = _get_pos_and_bbox(cbax, renderer)

    # Colorbar gets put at extreme edge of outer bbox of the subplotspec
    # It needs to be moved in by: 1) a pad 2) its "margin" 3) by
    # any colorbars already added at this location:
    if location in ('left', 'right'):
        pbcb = pb.shrunk(fraction, shrink).anchored(anchor, pb)
        if location is 'right':
            margin = cbbbox.x1 - cbpos.x1  # decorators on CB
            dx = bboxouter.x1 - pbcb.x1 - w_pad - oldw['right'] - margin
            oldw['right'] += cbbbox.width + 2 * w_pad
            pbcb = pbcb.translated(dx, 0)
        else:
            margin = cbpos.x0 - cbbbox.x0
            dx = bboxouter.x0 - pbcb.x0 + w_pad + oldw['left'] + margin
            oldw['left'] += cbbbox.width + 2 * w_pad
            pbcb = pbcb.translated(dx, 0)
    else:
        pbcb = pb.shrunk(shrink, fraction).anchored(anchor, pb)
        if location is 'top':
            margin = cbbbox.y1 - cbpos.y1
            dy = bboxouter.y1 - pbcb.y1 - h_pad - oldh['top'] - margin
            oldh['top'] += cbbbox.height + 2 * h_pad
            pbcb = pbcb.translated(0, dy)
        else:
            margin = cbpos.y0 - cbbbox.y0
            dy = bboxouter.y0 - pbcb.y0 + h_pad + oldh['bottom'] + margin
            oldh['bottom'] += cbbbox.height + 2 * h_pad
            pbcb = pbcb.translated(0, dy)
    pbcb = trans.transform_bbox(pbcb)
    cbax._set_position(pbcb)
    cbax.set_aspect(aspect, anchor=anchor, adjustable='box')
    return oldw, oldh

def _reset_margins(fig):
    """
    These need to be reset after each draw, otherwise the largest margin
    is sticky
    """
    for span in fig.panels:
        _reset_margins(span)
    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec'):
            ss = ax.get_subplotspec()
            gs = ss.get_gridspec()
            gs._layoutgrid.reset_margins()
    fig._layoutgrid.reset_margins()

