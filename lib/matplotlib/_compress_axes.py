import logging

import numpy as np

"""
This code attemprs to compress axes if they have excessive space between
axes, usually because the axes have fixed aspect ratios.
"""

def compress_axes(fig, *, bboxes=None, w_pad=0.05, h_pad=0.05):
    """
    Utility that will attempt to compress axes on a figure together.

    w_pad, h_pad are inches and are the half distance to the next
    axes in width and height respectively.
    """

    w, h = fig.get_size_inches()
    w_pad = w_pad / w * 2
    h_pad = h_pad / h * 2
    print('compress', w_pad, h_pad)

    renderer = fig.canvas.get_renderer()
    gss = set()
    invTransFig = fig.transFigure.inverted().transform_bbox

    if bboxes is None:
        bboxes = dict()
        for ax in fig.axes:
            bboxes[ax] = invTransFig(ax.get_tightbbox(renderer))

    colorbars = []
    for ax in fig.axes:
        if hasattr(ax, '_colorbar_info'):
            colorbars += [ax]
        elif hasattr(ax, 'get_subplotspec'):
            gs = ax.get_subplotspec().get_gridspec()
            gss.add(gs)
            for cba in ax._colorbars:
                # get the bbox including the colorbar for this axis
                if cba._colorbar_info['location'] == 'right':
                    bboxes[ax].x1 = bboxes[cba].x1
                if cba._colorbar_info['location'] == 'left':
                    bboxes[ax].x0 = bboxes[cba].x0
                if cba._colorbar_info['location'] == 'top':
                    bboxes[ax].y1 = bboxes[cba].y1
                if cba._colorbar_info['location'] == 'bottom':
                    bboxes[ax].y0 = bboxes[cba].y0

    # we placed everything, but what if there are huge gaps...
    for gs in gss:
        axs = [ax for ax in fig.axes
               if (hasattr(ax, 'get_subplotspec')
                   and ax.get_subplotspec().get_gridspec() == gs)]
        nrows, ncols = gs.get_geometry()
        # get widths:
        dxs = np.zeros((nrows, ncols))
        dys = np.zeros((nrows, ncols))
        margxs = np.zeros((nrows, ncols))
        margys = np.zeros((nrows, ncols))

        for i in range(nrows):
            for j in range(ncols):
                for ax in axs:
                    ss = ax.get_subplotspec()
                    if (i in ss.rowspan) and (j in ss.colspan):
                        di = ss.colspan[-1] - ss.colspan[0] + 1
                        dj = ss.rowspan[-1] - ss.rowspan[0] + 1
                        dxs[i, j] = bboxes[ax].bounds[2] / di
                        if ss.colspan[-1] < ncols - 1:
                            dxs[i, j] = dxs[i, j] + w_pad / di
                        dys[i, j] = bboxes[ax].bounds[3] / dj
                        if ss.rowspan[0] > 0 :
                            dys[i, j] = dys[i, j] + h_pad / dj
                        orpos = ax.get_position(original=True)
                        margxs[i, j] = bboxes[ax].x0 - orpos.x0
                        margys[i, j] = bboxes[ax].y0 - orpos.y0

        margxs = np.flipud(margxs)
        margys = np.flipud(margys)
        dys = np.flipud(dys)
        dxs = np.flipud(dxs)

        ddxs = np.max(dxs, axis=0)
        ddys = np.max(dys, axis=1)
        dx = np.sum(ddxs)
        dy = np.sum(ddys)
        x1 = y1 = -np.Inf
        x0 = y0 = np.Inf

        if (dx < dy) and (dx < 0.9):
            print('compress x')
            margx = np.min(margxs, axis=0)
            # Squish x!
            extra = (1 - dx) / 2
            for ax in axs:
                ss = ax.get_subplotspec()
                orpos = ax.get_position(original=True)
                x = extra
                for j in range(0, ss.colspan[0]):
                    x += ddxs[j]
                deltax = -orpos.x0 + x - margx[ss.colspan[0]]
                orpos.x1 = orpos.x1 + deltax
                orpos.x0 = orpos.x0 + deltax
                # keep track of new bbox edges for placing colorbars
                if bboxes[ax].x1 + deltax > x1:
                    x1 = bboxes[ax].x1 + deltax
                if bboxes[ax].x0 + deltax < x0:
                    x0 = bboxes[ax].x0 + deltax
                bboxes[ax].x0 = bboxes[ax].x0 + deltax
                bboxes[ax].x1 = bboxes[ax].x1 + deltax
                # Now set the new position.
                ax._set_position(orpos, which='original')
                # shift any colorbars belongig to this axis
                for cba in ax._colorbars:
                    pos = cba.get_position(original=True)
                    if cba._colorbar_info['location'] in ['bottom', 'top']:
                        # shrink to make same size as active...
                        posac = ax.get_position(original=False)
                        dx = (1 - cba._colorbar_info['shrink']) * (posac.x1 -
                                posac.x0) / 2
                        pos.x0 = posac.x0 + dx
                        pos.x1 = posac.x1 - dx
                    else:
                        pos.x0 = pos.x0 + deltax
                        pos.x1 = pos.x1 + deltax
                    cba._set_position(pos, which='original')
                    colorbars.remove(cba)
            for cb in colorbars:
                # shift any colorbars belonging to the gridspec.
                pos = cb.get_position(original=True)
                bbox = bboxes[cb]
                if cb._colorbar_info['location'] == 'right':
                    marg = bbox.x0 - pos.x0
                    x = x1 + marg + w_pad
                    pos.x1 = pos.x1 - pos.x0 + x
                    pos.x0 = x
                elif cb._colorbar_info['location'] == 'left':
                    marg = bbox.x1 - pos.x1
                    # left edge:
                    x = x0 - marg - w_pad
                    _dx = pos.x1 - pos.x0
                    pos.x1 = x - marg
                    pos.x0 = x - marg - _dx
                else:
                    marg = bbox.x0 - pos.x0
                    pos.x0 = x0 - marg
                    marg = bbox.x1 - pos.x1
                    pos.x1 = x1 - marg
                cb._set_position(pos, which='original')

        if (dx > dy) and (dy < 0.9):
            print('compress y')
            margy = np.min(margys, axis=1)
            # Squish y!
            extra = (1 - dy) / 2
            for ax in axs:
                ss = ax.get_subplotspec()
                orpos = ax.get_position(original=True)
                y = extra
                for j in range(0, nrows - ss.rowspan[-1] - 1):
                    y += ddys[j]
                deltay = -orpos.y0 + y - margy[nrows - ss.rowspan[-1] - 1]
                orpos.y1 = orpos.y1 + deltay
                orpos.y0 = orpos.y0 + deltay
                ax._set_position(orpos, which='original')
                # keep track of new bbox edges for placing colorbars
                if bboxes[ax].y1 + deltay > y1:
                    y1 = bboxes[ax].y1 + deltay
                if bboxes[ax].y0 + deltay < y0:
                    y0 = bboxes[ax].y0 + deltay
                bboxes[ax].y0 = bboxes[ax].y0 + deltay
                bboxes[ax].y1 = bboxes[ax].y1 + deltay
                # shift any colorbars belongig to this axis
                for cba in ax._colorbars:
                    pos = cba.get_position(original=True)
                    if cba._colorbar_info['location'] in ['right', 'left']:
                        # shrink to make same size as active...
                        posac = ax.get_position(original=False)
                        dy = (1 - cba._colorbar_info['shrink']) * (posac.y1 -
                                posac.y0) / 2
                        pos.y0 = posac.y0 + dy
                        pos.y1 = posac.y1 - dy
                    else:
                        pos.y0 = pos.y0 + deltay
                        pos.y1 = pos.y1 + deltay
                    cba._set_position(pos, which='original')
                    colorbars.remove(cba)
            for cb in colorbars:
                # shift any colorbars belonging to the gridspec.
                pos = cb.get_position(original=True)
                bbox = bboxes[cb]
                if cb._colorbar_info['location'] == 'top':
                    marg = bbox.y0 - pos.y0
                    y = y1 + marg + h_pad
                    pos.y1 = pos.y1 - pos.y0 + y
                    pos.y0 = y
                elif cb._colorbar_info['location'] == 'bottom':
                    marg = bbox.y1 - pos.y1
                    # left edge:
                    y = y0 - marg - h_pad
                    _dy = pos.y1 - pos.y0
                    pos.y1 = y - marg
                    pos.y0 = y - marg - _dy
                else:
                    marg = bbox.y0 - pos.y0
                    pos.y0 = y0 - marg
                    marg = bbox.y1 - pos.y1
                    pos.y1 = y1 - marg
                cb._set_position(pos, which='original')
            # need to do suptitles:
            suptitle = fig._suptitle
            do_suptitle = (suptitle is not None and
                           suptitle._layoutbox is not None and
                           suptitle.get_in_layout())
            if do_suptitle:
                x, y = suptitle.get_position()
                bbox = invTransFig(suptitle.get_window_extent(renderer))
                marg = y - bbox.y0
                suptitle.set_y(y1 + marg)
