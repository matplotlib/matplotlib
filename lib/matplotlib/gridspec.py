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

from __future__ import division

import matplotlib
rcParams = matplotlib.rcParams

import matplotlib.transforms as mtransforms


class GridSpec(object):
    """
    A class that specifies the geometry of the grid that a subplot
    will be placed. 
    """
    def __init__(self, nrows, ncols,
                 left=None, bottom=None, right=None, top=None,
                 wspace=None, hspace=None):
        """
        The number of rows and number of columns of the
        grid need to be set. Optionally, the subplot layout parameters
        (e.g., left, right, etc.) can be tuned.
        """
        #self.figure = figure
        self._nrows , self._ncols = nrows, ncols
        self.left=left
        self.bottom=bottom
        self.right=right
        self.top=top
        self.wspace=wspace
        self.hspace=hspace

    def get_geometry(self):
        'get the geometry of the grid, eg 2,3'
        return self._nrows, self._ncols

    _AllowedKeys = ["left", "bottom", "right", "top", "wspace", "hspace"]

    def update(self, **kwargs):
        """
        Update the current values.  If any kwarg is None, default to
        the current value, if set, otherwise to rc.
        """

        for k, v in kwargs.items():
            if k in self._AllowedKeys:
                setattr(self, k, v)
            else:
                raise AttributeError("%s is unknown keyword" % (k,))


        from matplotlib import _pylab_helpers
        from matplotlib.axes import SubplotBase
        for figmanager in _pylab_helpers.Gcf.figs.values():
            for ax in figmanager.canvas.figure.axes:
                # copied from Figure.subplots_adjust
                if not isinstance(ax, SubplotBase):
                    # Check if sharing a subplots axis
                    if ax._sharex is not None and isinstance(ax._sharex, SubplotBase):
                        ax._sharex.update_params()
                        ax.set_position(ax._sharex.figbox)
                    elif ax._sharey is not None and isinstance(ax._sharey,SubplotBase):
                        ax._sharey.update_params()
                        ax.set_position(ax._sharey.figbox)
                else:
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


    def new_subplotspec(self, loc, rowspan=1, colspan=1):
        """
        create and return a SuplotSpec instance.
        """
        loc1, loc2 = loc
        subplotspec = self[loc1:loc1+rowspan, loc2:loc2+colspan]
        return subplotspec


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
                row1, row2 = k1, k1+1


            if isinstance(k2, slice):
                col1, col2, _ = k2.indices(ncols)
            else:
                if k2 < 0:
                    k2 += ncols
                col1, col2 = k2, k2+1


            num1 = row1*nrows + col1
            num2 = (row2-1)*nrows + (col2-1)

        # single key
        else:
            if isinstance(key, slice):
                num1, num2, _ = key.indices(total)
                num2 -= 1
            else:
                if key < 0:
                    key += total
                num1, num2 = key, None


        return SubplotSpec(self, num1, num2)


class GridSpecFromSubplotSpec(GridSpec):
    """
    GridSpec whose subplot layout parameters are inherited from the
    location specified by a given SubplotSpec.
    """
    def __init__(self, nrows, ncols,
                 subplot_spec,
                 wspace=None, hspace=None):
        """
        The number of rows and number of columns of the grid need to
        be set. An instance of SubplotSpec is also need to be set from
        which the layout parameters will be inheirted. The wspace and
        hspace of the layout can be optionally specified or the
        default values (from the figure or rcParams) will be used.
        """
        self._nrows , self._ncols = nrows, ncols
        self._wspace=wspace
        self._hspace=hspace

        self._subplot_spec = subplot_spec

    def get_subplot_params(self, fig=None):

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
        get the subplot geometry, eg 2,2,3. Unlike SuplorParams,
        index is 0-based
        """
        rows, cols = self.get_gridspec().get_geometry()
        return rows, cols, self.num1, self.num2


    def get_position(self, fig, return_all=False):
        """
        update the subplot position from fig.subplotpars
        """

        gridspec = self.get_gridspec()
        rows, cols = gridspec.get_geometry()

        subplot_params = gridspec.get_subplot_params(fig)
        left = subplot_params.left
        right = subplot_params.right
        bottom = subplot_params.bottom
        top = subplot_params.top
        wspace = subplot_params.wspace
        hspace = subplot_params.hspace
        totWidth = right-left
        totHeight = top-bottom

        figH = totHeight/(rows + hspace*(rows-1))
        sepH = hspace*figH

        figW = totWidth/(cols + wspace*(cols-1))
        sepW = wspace*figW

        rowNum, colNum =  divmod(self.num1, cols)
        figBottom = top - (rowNum+1)*figH - rowNum*sepH
        figLeft = left + colNum*(figW + sepW)
        figTop = figBottom + figH
        figRight = figLeft + figW

        if self.num2 is not None:

            rowNum2, colNum2 =  divmod(self.num2, cols)
            figBottom2 = top - (rowNum2+1)*figH - rowNum2*sepH
            figLeft2 = left + colNum2*(figW + sepW)
            figTop2 = figBottom2 + figH
            figRight2 = figLeft2 + figW

            figBottom = min(figBottom, figBottom2)
            figLeft = min(figLeft, figLeft2)
            figTop = max(figTop, figTop2)
            figRight = max(figRight, figRight2)

        figbox = mtransforms.Bbox.from_extents(figLeft, figBottom,
                                               figRight, figTop)


        if return_all:
            return figbox, rowNum, colNum, rows, cols
        else:
            return figbox


