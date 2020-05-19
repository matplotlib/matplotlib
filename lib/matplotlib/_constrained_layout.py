"""
Adjust subplot layouts so that there are no overlapping axes or axes
decorations.  All axes decorations are dealt with (labels, ticks, titles,
ticklabels) and some dependent artists are also dealt with (colorbar,
suptitle).

Layout is done via `~matplotlib.gridspec`, with one constraint per gridspec,
so it is possible to have overlapping axes if the gridspecs overlap (i.e.
using `~matplotlib.gridspec.GridSpecFromSubplotSpec`).  Axes placed using
``figure.subplots()`` or ``figure.add_subplots()`` will participate in the
layout.  Axes manually placed via ``figure.add_axes()`` will not.

See Tutorial: :doc:`/tutorials/intermediate/constrainedlayout_guide`
"""

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
        are fraction of the figure to dedicate to space between the
        axes.
    """

    # list of unique gridspecs that contain child axes:
    gss = set()
    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec'):
            gs = ax.get_subplotspec().get_gridspec()
            if gs._layoutgrid is not None:
                gss.add(gs)
    gss = list(gss)
    if len(gss) == 0:
        cbook._warn_external('There are no gridspecs with layoutgrids. '
                             'Possibly did not call parent GridSpec with the'
                             ' "figure" keyword')

    for _ in range(2):
        print('Doing algo...')
        # do the algorithm twice.  This has to be done because decorations
        # change size after the first re-position (i.e. x/yticklabels get
        # larger/smaller).  This second reposition tends to be much milder,
        # so doing twice makes things work OK.

        # make margins for all the axes and subpanels in the
        # figure.
        _make_layout_margins(fig, renderer, h_pad=h_pad, w_pad=w_pad,
                             hspace=hspace, wspace=wspace)

        _make_margin_suptitles(fig, renderer, h_pad=h_pad, w_pad=w_pad)

        # goofy thing
        _match_submerged_margins(fig)

        # update all the variables in the layout.
        fig._layoutgrid.update_variables()

        if _check_ok(fig):  # check nothing collapsed to zero
            _reposition_axes(fig, renderer, h_pad=h_pad, w_pad=w_pad,
                             hspace=hspace, wspace=wspace)
        else:
            cbook._warn_external('constrained_layout not applied because '
                                 'axes sizes collapsed to zero.  Try making '
                                 'figure larger or axes decorations smaller.')

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
            if lg is not None:
                for i in range(gs.nrows):
                    for j in range(gs.ncols):
                        bb = lg.get_inner_bbox(i, j)
                        if bb.width <= 0 or bb.height <= 0:
                            return False
    return True


def _make_layout_margins(fig, renderer, *, w_pad=0, h_pad=0,
                         hspace=0, wspace=0):
    """
    For each axes, make a margin between the *pos* layoutbox and the
    *axes* layoutbox be a minimum size that can accommodate the
    decorations on the axis.
    """
    for panel in fig.panels:  # recursively make child panel margins
        _make_layout_margins(panel, renderer, w_pad=w_pad, h_pad=h_pad,
                             hspace=hspace, wspace=wspace)

    for ax in [a for a in fig.axes if hasattr(a, 'get_subplotspec')]:

        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()
        nrows, ncols = gs.get_geometry()

        if gs._layoutgrid is None:
            return

        pos, bbox = _get_pos_and_bbox(ax, renderer)

        # the margin is the distance between the bounding box of the axes
        # and its position.  Then we add the fixed padding (w_pad) and
        # the interior padding (wspace)
        margin = {}
        margin['left'] = -bbox.x0 + pos.x0 + w_pad
        if ss.colspan.start > 0:
            margin['left'] += wspace / ncols / 2  # interior padding
        margin['right'] = bbox.x1 - pos.x1 + w_pad
        if ss.colspan.stop < ncols:
            margin['right'] += wspace / ncols / 2

        # remember that rows are ordered from top:
        margin['bottom'] = pos.y0 - bbox.y0 + h_pad
        if ss.rowspan.stop < nrows:
            margin['bottom'] += hspace / nrows / 2
        margin['top'] = -pos.y1 + bbox.y1 + h_pad
        if ss.rowspan.start > 0:
            margin['top'] += hspace / nrows / 2

        # increase margin for colorbars...
        for cbax in ax._colorbars:
            # colorbars can be child of more than one subplot spec:
            cbp_rspan, cbp_cspan = _get_cb_parent_spans(cbax)
            loc = cbax._colorbar_info['location']
            if loc in ['right', 'left']:
                pass
            if loc == 'right':
                if cbp_cspan.stop == ss.colspan.stop:
                    # only increase if the colorbar is on the right edge
                    cbpos, cbbbox = _get_pos_and_bbox(cbax, renderer)
                    margin[loc] += cbbbox.width + w_pad * 2
            elif loc == 'left':
                if cbp_cspan.start == ss.colspan.start:
                    # only increase if the colorbar is on the left edge
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

        # pass the new margins down to the layout grid for the solution...
        gs._layoutgrid.edit_margin_min('left', margin['left'],
                                       ss.colspan[0])
        gs._layoutgrid.edit_margin_min('right', margin['right'],
                                       ss.colspan[-1])

        gs._layoutgrid.edit_margin_min('top', margin['top'],
                                       ss.rowspan[0])
        gs._layoutgrid.edit_margin_min('bottom', margin['bottom'],
                                       ss.rowspan[-1])


def _make_margin_suptitles(fig, renderer, *, w_pad=0, h_pad=0):
    # Figure out how large the suptitle is and make the
    # top level figure margin larger.
    for panel in fig.panels:
        _make_margin_suptitles(panel, renderer, w_pad=w_pad, h_pad=h_pad)
    invTransFig = fig.transPanel.inverted().transform_bbox
    pan2fig = fig.transPanel + fig.transFigure.inverted()
    w_pad, h_pad = pan2fig.transform((w_pad, h_pad))

    if fig._suptitle is not None and fig._suptitle.get_in_layout():
        bbox = invTransFig(fig._suptitle.get_tightbbox(renderer))
        p = fig._suptitle.get_position()
        fig._suptitle.set_position((p[0], 1-h_pad))
        fig._layoutgrid.edit_margin_min('top', bbox.height +  2 * h_pad)

    if 0:
        if fig._supxlabel is not None:
            bbox = invTransFig(fig._supxlabel.get_tightbbox(renderer))
            fig._layoutgrid.edit_margin_min('bottom', bbox.height + 2 * h_pad)

        if fig._supylabel is not None:
            bbox = invTransFig(fig._supylabel.get_tightbbox(renderer))
            fig._layoutgrid.edit_margin_min('left', bbox.width + 2 * w_pad)


def _match_submerged_margins(fig):
    """
    Make the margins that are submerged inside an axes the same
    size.

    This allows axes that span two columns (or rows) that are offset
    from one another to have the same size.

    i.e. if in row 0, the axes is at columns 0, 1 and for row 1,
    the axes is at columns 1 and 2, then the right margin at row 0
    needs to be the same size as the right margin at row 1 and
    the left margin for rows 1 and 2 should be the same.

    See test_constrained_layout::test_constrained_layout12 for an example.
    """

    for panel in fig.panels:
        _match_submerged_margins(panel)

    axs = [a for a in fig._localaxes if hasattr(a, 'get_subplotspec')]
    for ax1 in axs:
        ss1 = ax1.get_subplotspec()
        gs1 = ss1.get_gridspec()
        lg1 = gs1._layoutgrid
        if lg1 is not None:
            # interior columns:
            nc = len(ss1.colspan)
            if  nc > 1:
                maxsubl = np.max(
                    lg1.margin_vals['left'][ss1.colspan[1:]])
                maxsubr = np.max(
                    lg1.margin_vals['right'][ss1.colspan[:-1]])

                for ax2 in axs:
                    ss2 = ax2.get_subplotspec()
                    gs2 = ss2.get_gridspec()
                    lg2 = gs2._layoutgrid
                    if lg2 is not None:
                        nc = len(ss2.colspan)
                        if nc > 1:
                            maxsubl2 = np.max(
                                lg2.margin_vals['left'][ss2.colspan[1:]])
                            if maxsubl2 > maxsubl:
                                maxsubl = maxsubl2
                            maxsubr2 = np.max(
                                lg2.margin_vals['right'][ss2.colspan[:-1]])
                            if maxsubr2 > maxsubr:
                                maxsubr = maxsubr2
                for i in ss1.colspan[1:]:
                    lg1.edit_margin_min('left', maxsubl, col=i)
                for i in ss1.colspan[:-1]:
                    lg1.edit_margin_min('right', maxsubr, col=i)

            # interior rows:
            nc = len(ss1.rowspan)
            if  nc > 1:
                maxsubt = np.max(
                    lg1.margin_vals['top'][ss1.rowspan[1:]])
                maxsubb = np.max(
                    lg1.margin_vals['bottom'][ss1.rowspan[:-1]])

                for ax2 in axs:
                    ss2 = ax2.get_subplotspec()
                    gs2 = ss2.get_gridspec()
                    lg2 = gs2._layoutgrid
                    if lg2 is not None:
                        nc = len(ss2.rowspan)
                        if nc > 1:
                            maxsubt2 = np.max(
                                lg2.margin_vals['top'][ss2.rowspan[1:]])
                            if maxsubt2 > maxsubt:
                                maxsubt = maxsubt2
                            maxsubb2 = np.max(
                                lg2.margin_vals['bottom'][ss2.rowspan[:-1]])
                            if maxsubb2 > maxsubb:
                                maxsubb = maxsubb2
                for i in ss1.rowspan[1:]:
                    lg1.edit_margin_min('top', maxsubt, col=i)
                for i in ss1.rowspan[:-1]:
                    lg1.edit_margin_min('bottom', maxsubb, col=i)

def _get_cb_parent_spans(cbax):
    """
    Figure out which subplotspecs this colorbar belongs to
    """
    rowspan = range(100, -100, -100)
    colspan = range(100, -100, -100)
    for parent in cbax._colorbar_info['parents']:
        ss = parent.get_subplotspec()
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
        bbox = tightbbox.transformed(invTransFig)
    return pos, bbox

def _reposition_axes(fig, renderer, *, w_pad=0, h_pad=0, hspace=0, wspace=0):
    """
    For each axes, make a margin between the *pos* layoutbox and the
    *axes* layoutbox be a minimum size that can accommodate the
    decorations on the axis.
    """
    trans = fig.transFigure + fig.transPanel.inverted()

    for panel in fig.panels:
        inner = panel._parent._layoutgrid.get_inner_bbox()
        inner = trans.transform_bbox(inner)
        outer = panel._parent._layoutgrid.get_outer_bbox()
        outer = trans.transform_bbox(outer)
        left = inner.x0 - outer.x0
        right = outer.x1 - inner.x1
        bottom = inner.y0 - outer.y0
        top = outer.y1 - inner.y1

        panel._redo_transform_rel_fig(margins=(left, bottom, right, top))

        _reposition_axes(panel, renderer,
                         w_pad=w_pad, h_pad=h_pad,
                         wspace=wspace, hspace=hspace)


    for ax in [a for a in fig._localaxes if hasattr(a, 'get_subplotspec')]:

        # grid bbox is in Figure co-ordinates, but we specify in panel
        # co-ordinates...
        ss = ax.get_subplotspec()
        gs = ss.get_gridspec()
        if gs._layoutgrid is None:
            return

        bbox = gs._layoutgrid.get_inner_bbox(rows=ss.rowspan, cols=ss.colspan)

        bboxouter = gs._layoutgrid.get_outer_bbox(rows=ss.rowspan,
                                                  cols=ss.colspan)

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
                        w_pad=w_pad, h_pad=h_pad,
                        wspace=wspace, hspace=hspace, oldw=oldw, oldh=oldh)


def _reposition_colorbar(cbax, renderer, *, w_pad=0, h_pad=0, hspace=0,
                         wspace=0, oldw=0, oldh=0):

    parents = cbax._colorbar_info['parents']
    gs = parents[0].get_gridspec()
    ncols, nrows = gs.ncols, gs.nrows
    fig = cbax.figure
    trans = fig.transFigure + fig.transPanel.inverted()

    cb_rspans, cb_cspans = _get_cb_parent_spans(cbax)
    bboxouter = gs._layoutgrid.get_outer_bbox(rows=cb_rspans, cols=cb_cspans)
    pb = gs._layoutgrid.get_inner_bbox(rows=cb_rspans, cols=cb_cspans)
    location = cbax._colorbar_info['location']
    anchor = cbax._colorbar_info['anchor']
    fraction = cbax._colorbar_info['fraction']
    aspect = cbax._colorbar_info['aspect']
    shrink = cbax._colorbar_info['shrink']
    pad = cbax._colorbar_info['pad']
    parent_anchor = cbax._colorbar_info['panchor']

    cbpos, cbbbox = _get_pos_and_bbox(cbax, renderer)

    if cb_cspans.stop == ncols:
        wspace = 0
    if cb_rspans.start == 0:
        hspace = 0

    # Colorbar gets put at extreme edge of outer bbox of the subplotspec
    # It needs to be moved in by: 1) a pad 2) its "margin" 3) by
    # any colorbars already added at this location:
    if location in ('left', 'right'):
        pbcb = pb.shrunk(fraction, shrink).anchored(anchor, pb)
        if location == 'right':
            margin = cbbbox.x1 - cbpos.x1  # decorators on CB
            dx = bboxouter.x1 - pbcb.x1
            dx = dx - w_pad - oldw['right'] - margin - wspace / ncols / 2
            oldw['right'] += cbbbox.width + 2 * w_pad
            pbcb = pbcb.translated(dx, 0)
        else:
            margin = cbpos.x0 - cbbbox.x0
            dx = bboxouter.x0 - pbcb.x0
            dx = dx + w_pad + oldw['left'] + margin + wspace / ncols / 2
            oldw['left'] += cbbbox.width + 2 * w_pad
            pbcb = pbcb.translated(dx, 0)
    else:
        pbcb = pb.shrunk(shrink, fraction).anchored(anchor, pb)
        if location == 'top':
            margin = cbbbox.y1 - cbpos.y1
            dy = bboxouter.y1 - pbcb.y1
            dy = dy - h_pad - oldh['top'] - margin - hspace / nrows / 2
            oldh['top'] += cbbbox.height + 2 * h_pad
            pbcb = pbcb.translated(0, dy)
        else:
            margin = cbpos.y0 - cbbbox.y0
            dy = bboxouter.y0 - pbcb.y0
            dy = dy + h_pad + oldh['bottom'] + margin + hspace / nrows / 2
            oldh['bottom'] += cbbbox.height + 2 * h_pad
            pbcb = pbcb.translated(0, dy)

    pbcb = trans.transform_bbox(pbcb)
    cbax.set_transform(fig.transPanel)
    cbax._set_position(pbcb)
    cbax.set_aspect(aspect, anchor=anchor, adjustable='box')
    return oldw, oldh

def _reset_margins(fig):
    """
    Reset the margins in the layoutboxes of fig.

    These need to be reset to zero before they are recalculated
    so we can
    """
    for span in fig.panels:
        _reset_margins(span)
    for ax in fig.axes:
        if hasattr(ax, 'get_subplotspec'):
            ss = ax.get_subplotspec()
            gs = ss.get_gridspec()
            if gs._layoutgrid is not None:
                gs._layoutgrid.reset_margins()
    fig._layoutgrid.reset_margins()
