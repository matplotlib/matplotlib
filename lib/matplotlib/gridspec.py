"""
:mod:`~matplotlib.gridspec` is a module which specifies the location
of the subplot in the figure.

    ``GridSpec``
        specifies the geometry of the grid that a subplot will be
        placed. The number of rows and number of columns of the grid
        need to be set. Optionally, the subplot layout parameters
        (e.g., left, right, etc.) can be tuned.

    ``SubplotSpec``
        specifies the location of the subplot in the given *GridSpec*.


"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import zip

import matplotlib
rcParams = matplotlib.rcParams

import matplotlib.transforms as mtransforms

import numpy as np
import warnings

class GridSpecBase(object):
    """
    A base class of GridSpec that specifies the geometry of the grid
    that a subplot will be placed.
    """

    def __init__(self, nrows, ncols,
                 height_ratios=None, width_ratios=None):
        """
        The number of rows and number of columns of the grid need to
        be set. Optionally, the ratio of heights and widths of rows and
        columns can be specified.
        """
        #self.figure = figure
        self._nrows , self._ncols = nrows, ncols

        self.set_height_ratios(height_ratios)
        self.set_width_ratios(width_ratios)

    def get_geometry(self):
        'get the geometry of the grid, e.g., 2,3'
        return self._nrows, self._ncols

    def get_subplot_params(self, fig=None):
        pass

    def new_subplotspec(self, loc, rowspan=1, colspan=1):
        """
        create and return a SuplotSpec instance.
        """
        loc1, loc2 = loc
        subplotspec = self[loc1:loc1+rowspan, loc2:loc2+colspan]
        return subplotspec

    def set_width_ratios(self, width_ratios):
        if width_ratios is not None and len(width_ratios) != self._ncols:
            raise ValueError('Expected the given number of width ratios to '
                             'match the number of columns of the grid')
        self._col_width_ratios = width_ratios

    def get_width_ratios(self):
        return self._col_width_ratios

    def set_height_ratios(self, height_ratios):
        if height_ratios is not None and len(height_ratios) != self._nrows:
            raise ValueError('Expected the given number of height ratios to '
                             'match the number of rows of the grid')
        self._row_height_ratios = height_ratios

    def get_height_ratios(self):
        return self._row_height_ratios

    def get_grid_positions(self, fig):
        """
        return lists of bottom and top position of rows, left and
        right positions of columns.
        """
        nrows, ncols = self.get_geometry()

        subplot_params = self.get_subplot_params(fig)
        left = subplot_params.left
        right = subplot_params.right
        bottom = subplot_params.bottom
        top = subplot_params.top
        wspace = subplot_params.wspace
        hspace = subplot_params.hspace
        totWidth = right-left
        totHeight = top-bottom

        # calculate accumulated heights of columns
        cellH = totHeight/(nrows + hspace*(nrows-1))
        sepH = hspace*cellH

        if self._row_height_ratios is not None:
            netHeight = cellH * nrows
            tr = float(sum(self._row_height_ratios))
            cellHeights = [netHeight*r/tr for r in self._row_height_ratios]
        else:
            cellHeights = [cellH] * nrows

        sepHeights = [0] + ([sepH] * (nrows-1))
        cellHs = np.add.accumulate(np.ravel(list(zip(sepHeights, cellHeights))))


        # calculate accumulated widths of rows
        cellW = totWidth/(ncols + wspace*(ncols-1))
        sepW = wspace*cellW

        if self._col_width_ratios is not None:
            netWidth = cellW * ncols
            tr = float(sum(self._col_width_ratios))
            cellWidths = [netWidth*r/tr for r in self._col_width_ratios]
        else:
            cellWidths = [cellW] * ncols

        sepWidths = [0] + ([sepW] * (ncols-1))
        cellWs = np.add.accumulate(np.ravel(list(zip(sepWidths, cellWidths))))



        figTops = [top - cellHs[2*rowNum] for rowNum in range(nrows)]
        figBottoms = [top - cellHs[2*rowNum+1] for rowNum in range(nrows)]
        figLefts = [left + cellWs[2*colNum] for colNum in range(ncols)]
        figRights = [left + cellWs[2*colNum+1] for colNum in range(ncols)]


        return figBottoms, figTops, figLefts, figRights

    def __getitem__(self, key):
        """
        create and return a SuplotSpec instance.
        """
        nrows, ncols = self.get_geometry()
        total = nrows*ncols

        if isinstance(key, tuple):
            try:
                k1, k2 = key
            except ValueError:
                raise ValueError("unrecognized subplot spec")

            if isinstance(k1, slice):
                row1, row2, _ = k1.indices(nrows)
            else:
                if k1 < 0:
                    k1 += nrows
                if k1 >= nrows or k1 < 0 :
                    raise IndexError("index out of range")
                row1, row2 = k1, k1+1


            if isinstance(k2, slice):
                col1, col2, _ = k2.indices(ncols)
            else:
                if k2 < 0:
                    k2 += ncols
                if k2 >= ncols or k2 < 0 :
                    raise IndexError("index out of range")
                col1, col2 = k2, k2+1


            num1 = row1*ncols + col1
            num2 = (row2-1)*ncols + (col2-1)

        # single key
        else:
            if isinstance(key, slice):
                num1, num2, _ = key.indices(total)
                num2 -= 1
            else:
                if key < 0:
                    key += total
                if key >= total or key < 0 :
                    raise IndexError("index out of range")
                num1, num2 = key, None


        return SubplotSpec(self, num1, num2)


class GridSpec(GridSpecBase):
    """
    A class that specifies the geometry of the grid that a subplot
    will be placed. The location of grid is determined by similar way
    as the SubplotParams.
    """

    def __init__(self, nrows, ncols,
                 left=None, bottom=None, right=None, top=None,
                 wspace=None, hspace=None,
                 width_ratios=None, height_ratios=None):
        """
        The number of rows and number of columns of the
        grid need to be set. Optionally, the subplot layout parameters
        (e.g., left, right, etc.) can be tuned.
        """
        #self.figure = figure
        self.left=left
        self.bottom=bottom
        self.right=right
        self.top=top
        self.wspace=wspace
        self.hspace=hspace

        GridSpecBase.__init__(self, nrows, ncols,
                              width_ratios=width_ratios,
                              height_ratios=height_ratios)
        #self.set_width_ratios(width_ratios)
        #self.set_height_ratios(height_ratios)


    _AllowedKeys = ["left", "bottom", "right", "top", "wspace", "hspace"]

    def update(self, **kwargs):
        """
        Update the current values.  If any kwarg is None, default to
        the current value, if set, otherwise to rc.
        """

        for k, v in six.iteritems(kwargs):
            if k in self._AllowedKeys:
                setattr(self, k, v)
            else:
                raise AttributeError("%s is unknown keyword" % (k,))


        from matplotlib import _pylab_helpers
        from matplotlib.axes import SubplotBase
        for figmanager in six.itervalues(_pylab_helpers.Gcf.figs):
            for ax in figmanager.canvas.figure.axes:
                # copied from Figure.subplots_adjust
                if not isinstance(ax, SubplotBase):
                    # Check if sharing a subplots axis
                    if ax._sharex is not None and isinstance(ax._sharex, SubplotBase):
                        if ax._sharex.get_subplotspec().get_gridspec() == self:
                            ax._sharex.update_params()
                            ax.set_position(ax._sharex.figbox)
                    elif ax._sharey is not None and isinstance(ax._sharey,SubplotBase):
                        if ax._sharey.get_subplotspec().get_gridspec() == self:
                            ax._sharey.update_params()
                            ax.set_position(ax._sharey.figbox)
                else:
                    ss = ax.get_subplotspec().get_topmost_subplotspec()
                    if ss.get_gridspec() == self:
                        ax.update_params()
                        ax.set_position(ax.figbox)



    def get_subplot_params(self, fig=None):
        """
        return a dictionary of subplot layout parameters. The default
        parameters are from rcParams unless a figure attribute is set.
        """
        from matplotlib.figure import SubplotParams
        import copy
        if fig is None:
            kw = dict([(k, rcParams["figure.subplot."+k]) \
                       for k in self._AllowedKeys])
            subplotpars = SubplotParams(**kw)
        else:
            subplotpars = copy.copy(fig.subplotpars)

        update_kw = dict([(k, getattr(self, k)) for k in self._AllowedKeys])
        subplotpars.update(**update_kw)

        return subplotpars

    def locally_modified_subplot_params(self):
        return [k for k in self._AllowedKeys if getattr(self, k)]


    def tight_layout(self, fig, renderer=None, pad=1.08, h_pad=None, w_pad=None, rect=None):
        """
        Adjust subplot parameters to give specified padding.

        Parameters:

        pad : float
            padding between the figure edge and the edges of subplots, as a fraction of the font-size.
        h_pad, w_pad : float
            padding (height/width) between edges of adjacent subplots.
            Defaults to `pad_inches`.
        rect : if rect is given, it is interpreted as a rectangle
            (left, bottom, right, top) in the normalized figure
            coordinate that the whole subplots area (including
            labels) will fit into. Default is (0, 0, 1, 1).
        """

        from .tight_layout import (get_subplotspec_list,
                                   get_tight_layout_figure,
                                   get_renderer)

        subplotspec_list = get_subplotspec_list(fig.axes, grid_spec=self)
        if None in subplotspec_list:
            warnings.warn("This figure includes Axes that are not "
                          "compatible with tight_layout, so its "
                          "results might be incorrect.")

        if renderer is None:
            renderer = get_renderer(fig)

        kwargs = get_tight_layout_figure(fig, fig.axes, subplotspec_list,
                                         renderer,
                                         pad=pad, h_pad=h_pad, w_pad=w_pad,
                                         rect=rect,
                                         )

        self.update(**kwargs)


class GridSpecFromSubplotSpec(GridSpecBase):
    """
    GridSpec whose subplot layout parameters are inherited from the
    location specified by a given SubplotSpec.
    """
    def __init__(self, nrows, ncols,
                 subplot_spec,
                 wspace=None, hspace=None,
                 height_ratios=None, width_ratios=None):
        """
        The number of rows and number of columns of the grid need to
        be set. An instance of SubplotSpec is also needed to be set
        from which the layout parameters will be inherited. The wspace
        and hspace of the layout can be optionally specified or the
        default values (from the figure or rcParams) will be used.
        """
        self._wspace=wspace
        self._hspace=hspace

        self._subplot_spec = subplot_spec

        GridSpecBase.__init__(self, nrows, ncols,
                              width_ratios=width_ratios,
                              height_ratios=height_ratios)


    def get_subplot_params(self, fig=None):
        """
        return a dictionary of subplot layout parameters.
        """

        if fig is None:
            hspace = rcParams["figure.subplot.hspace"]
            wspace = rcParams["figure.subplot.wspace"]
        else:
            hspace = fig.subplotpars.hspace
            wspace = fig.subplotpars.wspace

        if self._hspace is not None:
            hspace = self._hspace

        if self._wspace is not None:
            wspace = self._wspace

        figbox = self._subplot_spec.get_position(fig, return_all=False)

        left, bottom, right, top = figbox.extents

        from matplotlib.figure import SubplotParams
        sp = SubplotParams(left=left,
                           right=right,
                           bottom=bottom,
                           top=top,
                           wspace=wspace,
                           hspace=hspace)

        return sp


    def get_topmost_subplotspec(self):
        'get the topmost SubplotSpec instance associated with the subplot'
        return self._subplot_spec.get_topmost_subplotspec()


class SubplotSpec(object):
    """
    specifies the location of the subplot in the given *GridSpec*.
    """

    def __init__(self, gridspec, num1, num2=None):
        """
        The subplot will occupy the num1-th cell of the given
        gridspec.  If num2 is provided, the subplot will span between
        num1-th cell and num2-th cell.

        The index stars from 0.
        """

        rows, cols = gridspec.get_geometry()
        total = rows*cols

        self._gridspec = gridspec
        self.num1 = num1
        self.num2 = num2

    def get_gridspec(self):
        return self._gridspec


    def get_geometry(self):
        """
        get the subplot geometry, e.g., 2,2,3. Unlike SuplorParams,
        index is 0-based
        """
        rows, cols = self.get_gridspec().get_geometry()
        return rows, cols, self.num1, self.num2


    def get_position(self, fig, return_all=False):
        """
        update the subplot position from fig.subplotpars
        """

        gridspec = self.get_gridspec()
        nrows, ncols = gridspec.get_geometry()

        figBottoms, figTops, figLefts, figRights = \
                    gridspec.get_grid_positions(fig)


        rowNum, colNum =  divmod(self.num1, ncols)
        figBottom = figBottoms[rowNum]
        figTop = figTops[rowNum]
        figLeft = figLefts[colNum]
        figRight = figRights[colNum]

        if self.num2 is not None:

            rowNum2, colNum2 =  divmod(self.num2, ncols)
            figBottom2 = figBottoms[rowNum2]
            figTop2 = figTops[rowNum2]
            figLeft2 = figLefts[colNum2]
            figRight2 = figRights[colNum2]

            figBottom = min(figBottom, figBottom2)
            figLeft = min(figLeft, figLeft2)
            figTop = max(figTop, figTop2)
            figRight = max(figRight, figRight2)

        figbox = mtransforms.Bbox.from_extents(figLeft, figBottom,
                                               figRight, figTop)


        if return_all:
            return figbox, rowNum, colNum, nrows, ncols
        else:
            return figbox


    def get_topmost_subplotspec(self):
        'get the topmost SubplotSpec instance associated with the subplot'
        gridspec = self.get_gridspec()
        if hasattr(gridspec, "get_topmost_subplotspec"):
            return gridspec.get_topmost_subplotspec()
        else:
            return self

    def __eq__(self, other):
        # check to make sure other has the attributes
        # we need to do the comparison
        if not (hasattr(other, '_gridspec') and
                hasattr(other, 'num1') and
                hasattr(other, 'num2')):
            return False
        return all((self._gridspec == other._gridspec,
                    self.num1 == other.num1,
                    self.num2 == other.num2))

    def __hash__(self):
        return (hash(self._gridspec) ^
                hash(self.num1) ^
                hash(self.num2))
