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

import matplotlib.cbook as cbook
import matplotlib._layoutbox as layoutbox

_log = logging.getLogger(__name__)


def _spans_overlap(span0, span1):
    return span0.start in span1 or span1.start in span0


def _axes_all_finite_sized(fig):
    """Return whether all axes in the figure have a finite width and height."""
    for ax in fig.axes:
        if ax._layoutbox is not None:
            newpos = ax._poslayoutbox.get_rect()
            if newpos[2] <= 0 or newpos[3] <= 0:
                return False
    return True


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

    invTransFig = fig.transFigure.inverted().transform_bbox

    # list of unique gridspecs that contain child axes:
    gss = set()
    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec'):
            gs = ax.get_subplotspec().get_gridspec()
            if gs._layoutbox is not None:
                gss.add(gs)
    if len(gss) == 0:
        cbook._warn_external('There are no gridspecs with layoutboxes. '
                             'Possibly did not call parent GridSpec with the'
                             ' figure= keyword')

    if fig._layoutbox.constrained_layout_called < 1:
        for gs in gss:
            # fill in any empty gridspec slots w/ ghost axes...
            _make_ghost_gridspec_slots(fig, gs)

    for _ in range(2):
        # do the algorithm twice.  This has to be done because decorators
        # change size after the first re-position (i.e. x/yticklabels get
        # larger/smaller).  This second reposition tends to be much milder,
        # so doing twice makes things work OK.
        for ax in fig.axes:
            _log.debug(ax._layoutbox)
            if ax._layoutbox is not None:
                # make margins for each layout box based on the size of
                # the decorators.
                _make_layout_margins(ax, renderer, h_pad, w_pad)

        # do layout for suptitle.
        suptitle = fig._suptitle
        do_suptitle = (suptitle is not None and
                       suptitle._layoutbox is not None and
                       suptitle.get_in_layout())
        if do_suptitle:
            bbox = invTransFig(
                suptitle.get_window_extent(renderer=renderer))
            height = bbox.height
            if np.isfinite(height):
                # reserve at top of figure include an h_pad above and below
                suptitle._layoutbox.edit_height(height + h_pad * 2)

        # OK, the above lines up ax._poslayoutbox with ax._layoutbox
        # now we need to
        #   1) arrange the subplotspecs.  We do it at this level because
        #      the subplotspecs are meant to contain other dependent axes
        #      like colorbars or legends.
        #   2) line up the right and left side of the ax._poslayoutbox
        #      that have the same subplotspec maxes.

        if fig._layoutbox.constrained_layout_called < 1:
            # arrange the subplotspecs...  This is all done relative to each
            # other.  Some subplotspecs contain axes, and others contain
            # gridspecs the ones that contain gridspecs are a set proportion
            # of their parent gridspec.  The ones that contain axes are
            # not so constrained.
            figlb = fig._layoutbox
            for child in figlb.children:
                if child._is_gridspec_layoutbox():
                    # This routine makes all the subplot spec containers
                    # have the correct arrangement.  It just stacks the
                    # subplot layoutboxes in the correct order...
                    _arrange_subplotspecs(child, hspace=hspace, wspace=wspace)

            for gs in gss:
                _align_spines(fig, gs)

        fig._layoutbox.constrained_layout_called += 1
        fig._layoutbox.update_variables()

        # check if any axes collapsed to zero.  If not, don't change positions:
        if _axes_all_finite_sized(fig):
            # Now set the position of the axes...
            for ax in fig.axes:
                if ax._layoutbox is not None:
                    newpos = ax._poslayoutbox.get_rect()
                    # Now set the new position.
                    # ax.set_position will zero out the layout for
                    # this axis, allowing users to hard-code the position,
                    # so this does the same w/o zeroing layout.
                    ax._set_position(newpos, which='original')
            if do_suptitle:
                newpos = suptitle._layoutbox.get_rect()
                suptitle.set_y(1.0 - h_pad)
            else:
                if suptitle is not None and suptitle._layoutbox is not None:
                    suptitle._layoutbox.edit_height(0)
        else:
            cbook._warn_external('constrained_layout not applied.  At least '
                                 'one axes collapsed to zero width or height.')


def _make_ghost_gridspec_slots(fig, gs):
    """
    Check for unoccupied gridspec slots and make ghost axes for these
    slots...  Do for each gs separately.  This is a pretty big kludge
    but shouldn't have too much ill effect.  The worst is that
    someone querying the figure will wonder why there are more
    axes than they thought.
    """
    nrows, ncols = gs.get_geometry()
    hassubplotspec = np.zeros(nrows * ncols, dtype=bool)
    axs = []
    for ax in fig.axes:
        if (hasattr(ax, 'get_subplotspec')
                and ax._layoutbox is not None
                and ax.get_subplotspec().get_gridspec() == gs):
            axs += [ax]
    for ax in axs:
        ss0 = ax.get_subplotspec()
        hassubplotspec[ss0.num1:(ss0.num2 + 1)] = True
    for nn, hss in enumerate(hassubplotspec):
        if not hss:
            # this gridspec slot doesn't have an axis so we
            # make a "ghost".
            ax = fig.add_subplot(gs[nn])
            ax.set_visible(False)


def _make_layout_margins(ax, renderer, h_pad, w_pad):
    """
    For each axes, make a margin between the *pos* layoutbox and the
    *axes* layoutbox be a minimum size that can accommodate the
    decorations on the axis.
    """
    fig = ax.figure
    invTransFig = fig.transFigure.inverted().transform_bbox
    pos = ax.get_position(original=True)
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
    # use stored h_pad if it exists
    h_padt = ax._poslayoutbox.h_pad
    if h_padt is None:
        h_padt = h_pad
    w_padt = ax._poslayoutbox.w_pad
    if w_padt is None:
        w_padt = w_pad
    ax._poslayoutbox.edit_left_margin_min(-bbox.x0 + pos.x0 + w_padt)
    ax._poslayoutbox.edit_right_margin_min(bbox.x1 - pos.x1 + w_padt)
    ax._poslayoutbox.edit_bottom_margin_min(-bbox.y0 + pos.y0 + h_padt)
    ax._poslayoutbox.edit_top_margin_min(bbox.y1-pos.y1+h_padt)
    _log.debug('left %f', (-bbox.x0 + pos.x0 + w_pad))
    _log.debug('right %f', (bbox.x1 - pos.x1 + w_pad))
    _log.debug('bottom %f', (-bbox.y0 + pos.y0 + h_padt))
    _log.debug('bbox.y0 %f', bbox.y0)
    _log.debug('pos.y0 %f', pos.y0)
    # Sometimes its possible for the solver to collapse
    # rather than expand axes, so they all have zero height
    # or width.  This stops that...  It *should* have been
    # taken into account w/ pref_width...
    if fig._layoutbox.constrained_layout_called < 1:
        ax._poslayoutbox.constrain_height_min(20, strength='weak')
        ax._poslayoutbox.constrain_width_min(20, strength='weak')
        ax._layoutbox.constrain_height_min(20, strength='weak')
        ax._layoutbox.constrain_width_min(20, strength='weak')
        ax._poslayoutbox.constrain_top_margin(0, strength='weak')
        ax._poslayoutbox.constrain_bottom_margin(0, strength='weak')
        ax._poslayoutbox.constrain_right_margin(0, strength='weak')
        ax._poslayoutbox.constrain_left_margin(0, strength='weak')


def _align_spines(fig, gs):
    """
    - Align right/left and bottom/top spines of appropriate subplots.
    - Compare size of subplotspec including height and width ratios
       and make sure that the axes spines are at least as large
       as they should be.
    """
    # for each gridspec...
    nrows, ncols = gs.get_geometry()
    width_ratios = gs.get_width_ratios()
    height_ratios = gs.get_height_ratios()
    if width_ratios is None:
        width_ratios = np.ones(ncols)
    if height_ratios is None:
        height_ratios = np.ones(nrows)

    # get axes in this gridspec....
    axs = [ax for ax in fig.axes
           if (hasattr(ax, 'get_subplotspec')
               and ax._layoutbox is not None
               and ax.get_subplotspec().get_gridspec() == gs)]
    rowspans = []
    colspans = []
    heights = []
    widths = []

    for ax in axs:
        ss0 = ax.get_subplotspec()
        rowspan = ss0.rowspan
        colspan = ss0.colspan
        rowspans.append(rowspan)
        colspans.append(colspan)
        heights.append(sum(height_ratios[rowspan.start:rowspan.stop]))
        widths.append(sum(width_ratios[colspan.start:colspan.stop]))

    for idx0, ax0 in enumerate(axs):
        # Compare ax to all other axs:  If the subplotspecs start (/stop) at
        # the same column, then line up their left (/right) sides; likewise
        # for rows/top/bottom.
        rowspan0 = rowspans[idx0]
        colspan0 = colspans[idx0]
        height0 = heights[idx0]
        width0 = widths[idx0]
        alignleft = False
        alignright = False
        alignbot = False
        aligntop = False
        alignheight = False
        alignwidth = False
        for idx1 in range(idx0 + 1, len(axs)):
            ax1 = axs[idx1]
            rowspan1 = rowspans[idx1]
            colspan1 = colspans[idx1]
            width1 = widths[idx1]
            height1 = heights[idx1]
            # Horizontally align axes spines if they have the same min or max:
            if not alignleft and colspan0.start == colspan1.start:
                _log.debug('same start columns; line up layoutbox lefts')
                layoutbox.align([ax0._poslayoutbox, ax1._poslayoutbox],
                                'left')
                alignleft = True
            if not alignright and colspan0.stop == colspan1.stop:
                _log.debug('same stop columns; line up layoutbox rights')
                layoutbox.align([ax0._poslayoutbox, ax1._poslayoutbox],
                                'right')
                alignright = True
            # Vertically align axes spines if they have the same min or max:
            if not aligntop and rowspan0.start == rowspan1.start:
                _log.debug('same start rows; line up layoutbox tops')
                layoutbox.align([ax0._poslayoutbox, ax1._poslayoutbox],
                                'top')
                aligntop = True
            if not alignbot and rowspan0.stop == rowspan1.stop:
                _log.debug('same stop rows; line up layoutbox bottoms')
                layoutbox.align([ax0._poslayoutbox, ax1._poslayoutbox],
                                'bottom')
                alignbot = True

            # Now we make the widths and heights of position boxes
            # similar. (i.e the spine locations)
            # This allows vertically stacked subplots to have different sizes
            # if they occupy different amounts of the gridspec, e.g. if
            #   gs = gridspec.GridSpec(3, 1)
            #   ax0 = gs[0, :]
            #   ax1 = gs[1:, :]
            # then len(rowspan0) = 1, and len(rowspan1) = 2,
            # and ax1 should be at least twice as large as ax0.
            # But it can be more than twice as large because
            # it needs less room for the labeling.

            # For heights, do it if the subplots share a column.
            if not alignheight and len(rowspan0) == len(rowspan1):
                ax0._poslayoutbox.constrain_height(
                    ax1._poslayoutbox.height * height0 / height1)
                alignheight = True
            elif _spans_overlap(colspan0, colspan1):
                if height0 > height1:
                    ax0._poslayoutbox.constrain_height_min(
                        ax1._poslayoutbox.height * height0 / height1)
                    # these constraints stop the smaller axes from
                    # being allowed to go to zero height...
                    ax1._poslayoutbox.constrain_height_min(
                        ax0._poslayoutbox.height * height1 / (height0*1.8))
                elif height0 < height1:
                    ax1._poslayoutbox.constrain_height_min(
                        ax0._poslayoutbox.height * height1 / height0)
                    ax0._poslayoutbox.constrain_height_min(
                        ax0._poslayoutbox.height * height0 / (height1*1.8))
            # For widths, do it if the subplots share a row.
            if not alignwidth and len(colspan0) == len(colspan1):
                ax0._poslayoutbox.constrain_width(
                    ax1._poslayoutbox.width * width0 / width1)
                alignwidth = True
            elif _spans_overlap(rowspan0, rowspan1):
                if width0 > width1:
                    ax0._poslayoutbox.constrain_width_min(
                        ax1._poslayoutbox.width * width0 / width1)
                    ax1._poslayoutbox.constrain_width_min(
                        ax0._poslayoutbox.width * width1 / (width0*1.8))
                elif width0 < width1:
                    ax1._poslayoutbox.constrain_width_min(
                        ax0._poslayoutbox.width * width1 / width0)
                    ax0._poslayoutbox.constrain_width_min(
                        ax1._poslayoutbox.width * width0 / (width1*1.8))


def _arrange_subplotspecs(gs, hspace=0, wspace=0):
    """Recursively arrange the subplotspec children of the given gridspec."""
    sschildren = []
    for child in gs.children:
        if child._is_subplotspec_layoutbox():
            for child2 in child.children:
                # check for gridspec children...
                if child2._is_gridspec_layoutbox():
                    _arrange_subplotspecs(child2, hspace=hspace, wspace=wspace)
            sschildren += [child]
    # now arrange the subplots...
    for child0 in sschildren:
        ss0 = child0.artist
        nrows, ncols = ss0.get_gridspec().get_geometry()
        rowspan0 = ss0.rowspan
        colspan0 = ss0.colspan
        sschildren = sschildren[1:]
        for child1 in sschildren:
            ss1 = child1.artist
            rowspan1 = ss1.rowspan
            colspan1 = ss1.colspan
            # OK, this tells us the relative layout of child0 with child1.
            pad = wspace / ncols
            if colspan0.stop <= colspan1.start:
                layoutbox.hstack([ss0._layoutbox, ss1._layoutbox], padding=pad)
            if colspan1.stop <= colspan0.start:
                layoutbox.hstack([ss1._layoutbox, ss0._layoutbox], padding=pad)
            # vertical alignment
            pad = hspace / nrows
            if rowspan0.stop <= rowspan1.start:
                layoutbox.vstack([ss0._layoutbox, ss1._layoutbox], padding=pad)
            if rowspan1.stop <= rowspan0.start:
                layoutbox.vstack([ss1._layoutbox, ss0._layoutbox], padding=pad)


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
