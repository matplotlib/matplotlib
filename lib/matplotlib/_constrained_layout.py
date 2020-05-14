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
from matplotlib.transforms import Bbox, TransformedBbox

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
        for ax in fig.axes:
            _log.debug(ax)
            _make_layout_margins(ax, renderer, h_pad, w_pad)

        fig._layoutgrid.update_variables()
        for ax in fig.axes:
            _reposition_axes(ax)
        for ax in fig.axes:
            _reset_margins(ax)


def _make_layout_margins(ax, renderer, h_pad, w_pad):
    """
    For each axes, make a margin between the *pos* layoutbox and the
    *axes* layoutbox be a minimum size that can accommodate the
    decorations on the axis.
    """
    if not hasattr(ax, 'get_subplotspec'):
        return
    fig = ax.figure
    invTransFig = fig.transFigure.inverted().transform_bbox

    ss = ax.get_subplotspec()
    gs = ss.get_gridspec()
    if gs._layoutgrid is None:
        return

    pos = ax.get_position(original=True)
    # pos is in panel co-ords, but we need in figure for the layout
    pos = invTransFig(pos.transformed(fig.transPanel))

    try:
        tightbbox = ax.get_tightbbox(renderer=renderer, for_layout_only=True)
    except TypeError:
        tightbbox = ax.get_tightbbox(renderer=renderer)

    if tightbbox is None:
        bbox = pos
    else:
        bbox = invTransFig(tightbbox)

    # this can go wrong:
    if not (np.isfinite(bbox.width) and np.isfinite(bbox.height)):
        # just abort, this is likely a bad set of coordinates that
        # is transitory...
        return

    width = -bbox.x0 + pos.x0 + w_pad
    gs._layoutgrid.edit_margin_min('left', width, ss.colspan[0])
    width = bbox.x1 - pos.x1 + w_pad
    gs._layoutgrid.edit_margin_min('right', width, ss.colspan[-1])

    # remember that rows are ordered from top:
    height = pos.y0 - bbox.y0 + h_pad
    gs._layoutgrid.edit_margin_min('bottom', height, ss.rowspan[-1])
    height = -pos.y1 + bbox.y1 + h_pad
    gs._layoutgrid.edit_margin_min('top', height, ss.rowspan[0])

    # check if we have a colorbar:
    print('colorbars!', ax._colorbars)
    print(gs._layoutgrid)
    _log.debug('left %f', (-bbox.x0 + pos.x0 + w_pad))
    _log.debug('right %f', (bbox.x1 - pos.x1 + w_pad))
    _log.debug('bottom %f', (-bbox.y0 + pos.y0 + h_pad))
    _log.debug('bbox.y0 %f', bbox.y0)
    _log.debug('pos.y0 %f', pos.y0)

def _reposition_axes(ax):
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
    newbbox = trans.transform_bbox(bbox)
    ax._set_position(newbbox)

def _reset_margins(ax):
    """
    These need to be reset after each draw, otherwise the largest margin
    is sticky
    """
    if not hasattr(ax, 'get_subplotspec'):
        return
    ss = ax.get_subplotspec()
    gs = ss.get_gridspec()
    gs._layoutgrid.reset_margins()



######### Old
def _position_suptitles(fig, h_pad, w_pad):
    # not really position suptitle,s, but deal with everything at the "figure" level

    for childf in fig.panels:
        _position_suptitles(childf, h_pad, w_pad)

    suptitle = fig._suptitle
    if suptitle is not None:
        bbox = suptitle._layoutbox.get_bbox()
        bbox = bbox.transformed(fig.transFigure)
        bbox = bbox.transformed(fig.transPanel.inverted())
        suptitle.set_y(bbox.extents[3] - h_pad)

    figlb = fig._layoutbox
    h = 0.0
    for child in figlb.children:
        h += child.height
    c = (h == figlb.height - 2 * h_pad)
    figlb.solver.addConstraint(c | 1e8)

def _resize_suptitles(fig, renderer, h_pad, w_pad):

    for childf in fig.panels:
        _resize_suptitles(childf, renderer, h_pad, w_pad)

    invTransFig = fig.transFigure.inverted().transform_bbox

    suptitle = fig._suptitle

    do_suptitle = (suptitle is not None and
                   suptitle._layoutbox is not None and
                   suptitle.get_in_layout())
    if do_suptitle:
        bbox = invTransFig(
            suptitle.get_window_extent(renderer=renderer))
        height = bbox.height
        print('height', bbox.height)
        if np.isfinite(height):
            # reserve at top of figure include an h_pad above and below
            suptitle._layoutbox.edit_height(height + h_pad * 2)
            suptitle._layoutbox.edit_width(bbox.width)

    else:
        if suptitle is not None and suptitle._layoutbox is not None:
            suptitle._layoutbox.edit_height(0)
            suptitle._layoutbox.edit_width(0)


def layoutcolorbarsingle(ax, cax, shrink, aspect, location, pad=0.05):
    """
    Do the layout for a colorbar, to not overly pollute colorbar.py

    *pad* is in fraction of the original axis size.
    """
    axlb = ax._layoutbox
    axpos = ax._poslayoutbox
    axsslb = ax.get_subplotspec()._layoutbox
    lb = layoutbox.LayoutBox(
            parent=axsslb,
            name=axsslb.name + '.cbar',
            artist=cax)

    if location in ('left', 'right'):
        lbpos = layoutbox.LayoutBox(
                parent=lb,
                name=lb.name + '.pos',
                tightwidth=False,
                pos=True,
                subplot=False,
                artist=cax)

        if location == 'right':
            # arrange to right of parent axis
            layoutbox.hstack([axlb, lb], padding=pad * axlb.width,
                             strength='strong')
        else:
            layoutbox.hstack([lb, axlb], padding=pad * axlb.width)
        # constrain the height and center...
        layoutbox.match_heights([axpos, lbpos], [1, shrink])
        layoutbox.align([axpos, lbpos], 'v_center')
        # set the width of the pos box
        lbpos.constrain_width(shrink * axpos.height * (1/aspect),
                              strength='strong')
    elif location in ('bottom', 'top'):
        lbpos = layoutbox.LayoutBox(
                parent=lb,
                name=lb.name + '.pos',
                tightheight=True,
                pos=True,
                subplot=False,
                artist=cax)

        if location == 'bottom':
            layoutbox.vstack([axlb, lb], padding=pad * axlb.height)
        else:
            layoutbox.vstack([lb, axlb], padding=pad * axlb.height)
        # constrain the height and center...
        layoutbox.match_widths([axpos, lbpos],
                               [1, shrink], strength='strong')
        layoutbox.align([axpos, lbpos], 'h_center')
        # set the height of the pos box
        lbpos.constrain_height(axpos.width * aspect * shrink,
                                strength='medium')

    return lb, lbpos


def _getmaxminrowcolumn(axs):
    """
    Find axes covering the first and last rows and columns of a list of axes.
    """
    startrow = startcol = np.inf
    stoprow = stopcol = -np.inf
    startax_row = startax_col = stopax_row = stopax_col = None
    for ax in axs:
        subspec = ax.get_subplotspec()
        if subspec.rowspan.start < startrow:
            startrow = subspec.rowspan.start
            startax_row = ax
        if subspec.rowspan.stop > stoprow:
            stoprow = subspec.rowspan.stop
            stopax_row = ax
        if subspec.colspan.start < startcol:
            startcol = subspec.colspan.start
            startax_col = ax
        if subspec.colspan.stop > stopcol:
            stopcol = subspec.colspan.stop
            stopax_col = ax
    return (startrow, stoprow - 1, startax_row, stopax_row,
            startcol, stopcol - 1, startax_col, stopax_col)


def layoutcolorbargridspec(parents, cax, shrink, aspect, location, pad=0.05):
    """
    Do the layout for a colorbar, to not overly pollute colorbar.py

    *pad* is in fraction of the original axis size.
    """

    gs = parents[0].get_subplotspec().get_gridspec()
    # parent layout box....
    gslb = gs._layoutbox

    lb = layoutbox.LayoutBox(parent=gslb.parent,
                             name=gslb.parent.name + '.cbar',
                             artist=cax)
    # figure out the row and column extent of the parents.
    (minrow, maxrow, minax_row, maxax_row,
     mincol, maxcol, minax_col, maxax_col) = _getmaxminrowcolumn(parents)

    if location in ('left', 'right'):
        lbpos = layoutbox.LayoutBox(
                parent=lb,
                name=lb.name + '.pos',
                tightwidth=False,
                pos=True,
                subplot=False,
                artist=cax)
        for ax in parents:
            if location == 'right':
                order = [ax._layoutbox, lb]
            else:
                order = [lb, ax._layoutbox]
            layoutbox.hstack(order, padding=pad * gslb.width,
                             strength='strong')
        # constrain the height and center...
        # This isn't quite right.  We'd like the colorbar
        # pos to line up w/ the axes poss, not the size of the
        # gs.

        # Horizontal Layout: need to check all the axes in this gridspec
        for ch in gslb.children:
            subspec = ch.artist
            if location == 'right':
                if subspec.colspan.stop - 1 <= maxcol:
                    order = [subspec._layoutbox, lb]
                    # arrange to right of the parents
                elif subspec.colspan.start > maxcol:
                    order = [lb, subspec._layoutbox]
            elif location == 'left':
                if subspec.colspan.start >= mincol:
                    order = [lb, subspec._layoutbox]
                elif subspec.colspan.stop - 1 < mincol:
                    order = [subspec._layoutbox, lb]
            layoutbox.hstack(order, padding=pad * gslb.width,
                             strength='strong')

        # Vertical layout:
        maxposlb = minax_row._poslayoutbox
        minposlb = maxax_row._poslayoutbox
        # now we want the height of the colorbar pos to be
        # set by the top and bottom of the min/max axes...
        # bottom            top
        #     b             t
        # h = (top-bottom)*shrink
        # b = bottom + (top-bottom - h) / 2.
        lbpos.constrain_height(
                (maxposlb.top - minposlb.bottom) *
                shrink, strength='strong')
        lbpos.constrain_bottom(
                (maxposlb.top - minposlb.bottom) *
                (1 - shrink)/2 + minposlb.bottom,
                strength='strong')

        # set the width of the pos box
        lbpos.constrain_width(lbpos.height * (shrink / aspect),
                              strength='strong')
    elif location in ('bottom', 'top'):
        lbpos = layoutbox.LayoutBox(
                parent=lb,
                name=lb.name + '.pos',
                tightheight=True,
                pos=True,
                subplot=False,
                artist=cax)

        for ax in parents:
            if location == 'bottom':
                order = [ax._layoutbox, lb]
            else:
                order = [lb, ax._layoutbox]
            layoutbox.vstack(order, padding=pad * gslb.width,
                             strength='strong')

        # Vertical Layout: need to check all the axes in this gridspec
        for ch in gslb.children:
            subspec = ch.artist
            if location == 'bottom':
                if subspec.rowspan.stop - 1 <= minrow:
                    order = [subspec._layoutbox, lb]
                elif subspec.rowspan.start > maxrow:
                    order = [lb, subspec._layoutbox]
            elif location == 'top':
                if subspec.rowspan.stop - 1 < minrow:
                    order = [subspec._layoutbox, lb]
                elif subspec.rowspan.start >= maxrow:
                    order = [lb, subspec._layoutbox]
            layoutbox.vstack(order, padding=pad * gslb.width,
                             strength='strong')

        # Do horizontal layout...
        maxposlb = maxax_col._poslayoutbox
        minposlb = minax_col._poslayoutbox
        lbpos.constrain_width((maxposlb.right - minposlb.left) *
                              shrink)
        lbpos.constrain_left(
                (maxposlb.right - minposlb.left) *
                (1-shrink)/2 + minposlb.left)
        # set the height of the pos box
        lbpos.constrain_height(lbpos.width * shrink * aspect,
                               strength='medium')

    return lb, lbpos
