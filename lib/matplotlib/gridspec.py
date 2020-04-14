r"""
:mod:`~matplotlib.gridspec` contains classes that help to layout multiple
`~axes.Axes` in a grid-like pattern within a figure.

The `GridSpec` specifies the overall grid structure. Individual cells within
the grid are referenced by `SubplotSpec`\s.

See the tutorial :ref:`sphx_glr_tutorials_intermediate_gridspec.py` for a
comprehensive usage guide.
"""

import copy
import logging

import numpy as np

import matplotlib as mpl
from matplotlib import _pylab_helpers, cbook, tight_layout, rcParams
from matplotlib.transforms import Bbox
import matplotlib._layoutbox as layoutbox

_log = logging.getLogger(__name__)


class GridSpecBase:
    """
    A base class of GridSpec that specifies the geometry of the grid
    that a subplot will be placed.
    """

    def __init__(self, nrows, ncols, height_ratios=None, width_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            The number of rows and columns of the grid.
        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.
        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each column gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
        """
        self._nrows, self._ncols = nrows, ncols
        self.set_height_ratios(height_ratios)
        self.set_width_ratios(width_ratios)

    def __repr__(self):
        height_arg = (', height_ratios=%r' % (self._row_height_ratios,)
                      if self._row_height_ratios is not None else '')
        width_arg = (', width_ratios=%r' % (self._col_width_ratios,)
                     if self._col_width_ratios is not None else '')
        return '{clsname}({nrows}, {ncols}{optionals})'.format(
            clsname=self.__class__.__name__,
            nrows=self._nrows,
            ncols=self._ncols,
            optionals=height_arg + width_arg,
            )

    nrows = property(lambda self: self._nrows,
                     doc="The number of rows in the grid.")
    ncols = property(lambda self: self._ncols,
                     doc="The number of columns in the grid.")

    def get_geometry(self):
        """
        Return a tuple containing the number of rows and columns in the grid.
        """
        return self._nrows, self._ncols

    def get_subplot_params(self, figure=None):
        # Must be implemented in subclasses
        pass

    def new_subplotspec(self, loc, rowspan=1, colspan=1):
        """
        Create and return a `.SubplotSpec` instance.

        Parameters
        ----------
        loc : (int, int)
            The position of the subplot in the grid as
            ``(row_index, column_index)``.
        rowspan, colspan : int, default: 1
            The number of rows and columns the subplot should span in the grid.
        """
        loc1, loc2 = loc
        subplotspec = self[loc1:loc1+rowspan, loc2:loc2+colspan]
        return subplotspec

    def set_width_ratios(self, width_ratios):
        """
        Set the relative widths of the columns.

        *width_ratios* must be of length *ncols*. Each column gets a relative
        width of ``width_ratios[i] / sum(width_ratios)``.
        """
        if width_ratios is not None and len(width_ratios) != self._ncols:
            raise ValueError('Expected the given number of width ratios to '
                             'match the number of columns of the grid')
        self._col_width_ratios = width_ratios

    def get_width_ratios(self):
        """
        Return the width ratios.

        This is *None* if no width ratios have been set explicitly.
        """
        return self._col_width_ratios

    def set_height_ratios(self, height_ratios):
        """
        Set the relative heights of the rows.

        *height_ratios* must be of length *nrows*. Each row gets a relative
        height of ``height_ratios[i] / sum(height_ratios)``.
        """
        if height_ratios is not None and len(height_ratios) != self._nrows:
            raise ValueError('Expected the given number of height ratios to '
                             'match the number of rows of the grid')
        self._row_height_ratios = height_ratios

    def get_height_ratios(self):
        """
        Return the height ratios.

        This is *None* if no height ratios have been set explicitly.
        """
        return self._row_height_ratios

    def get_grid_positions(self, fig, raw=False):
        """
        Return the positions of the grid cells in figure coordinates.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure the grid should be applied to. The subplot parameters
            (margins and spacing between subplots) are taken from *fig*.
        raw : bool, default: False
            If *True*, the subplot parameters of the figure are not taken
            into account. The grid spans the range [0, 1] in both directions
            without margins and there is no space between grid cells. This is
            used for constrained_layout.

        Returns
        -------
        bottoms, tops, lefts, rights : array
            The bottom, top, left, right positions of the grid cells in
            figure coordinates.
        """
        nrows, ncols = self.get_geometry()

        if raw:
            left = 0.
            right = 1.
            bottom = 0.
            top = 1.
            wspace = 0.
            hspace = 0.
        else:
            subplot_params = self.get_subplot_params(fig)
            left = subplot_params.left
            right = subplot_params.right
            bottom = subplot_params.bottom
            top = subplot_params.top
            wspace = subplot_params.wspace
            hspace = subplot_params.hspace
        tot_width = right - left
        tot_height = top - bottom

        # calculate accumulated heights of columns
        cell_h = tot_height / (nrows + hspace*(nrows-1))
        sep_h = hspace * cell_h
        if self._row_height_ratios is not None:
            norm = cell_h * nrows / sum(self._row_height_ratios)
            cell_heights = [r * norm for r in self._row_height_ratios]
        else:
            cell_heights = [cell_h] * nrows
        sep_heights = [0] + ([sep_h] * (nrows-1))
        cell_hs = np.cumsum(np.column_stack([sep_heights, cell_heights]).flat)

        # calculate accumulated widths of rows
        cell_w = tot_width / (ncols + wspace*(ncols-1))
        sep_w = wspace * cell_w
        if self._col_width_ratios is not None:
            norm = cell_w * ncols / sum(self._col_width_ratios)
            cell_widths = [r * norm for r in self._col_width_ratios]
        else:
            cell_widths = [cell_w] * ncols
        sep_widths = [0] + ([sep_w] * (ncols-1))
        cell_ws = np.cumsum(np.column_stack([sep_widths, cell_widths]).flat)

        fig_tops, fig_bottoms = (top - cell_hs).reshape((-1, 2)).T
        fig_lefts, fig_rights = (left + cell_ws).reshape((-1, 2)).T
        return fig_bottoms, fig_tops, fig_lefts, fig_rights

    def __getitem__(self, key):
        """Create and return a `.SubplotSpec` instance."""
        nrows, ncols = self.get_geometry()

        def _normalize(key, size, axis):  # Includes last index.
            orig_key = key
            if isinstance(key, slice):
                start, stop, _ = key.indices(size)
                if stop > start:
                    return start, stop - 1
                raise IndexError("GridSpec slice would result in no space "
                                 "allocated for subplot")
            else:
                if key < 0:
                    key = key + size
                if 0 <= key < size:
                    return key, key
                elif axis is not None:
                    raise IndexError(f"index {orig_key} is out of bounds for "
                                     f"axis {axis} with size {size}")
                else:  # flat index
                    raise IndexError(f"index {orig_key} is out of bounds for "
                                     f"GridSpec with size {size}")

        if isinstance(key, tuple):
            try:
                k1, k2 = key
            except ValueError:
                raise ValueError("unrecognized subplot spec")
            num1, num2 = np.ravel_multi_index(
                [_normalize(k1, nrows, 0), _normalize(k2, ncols, 1)],
                (nrows, ncols))
        else:  # Single key
            num1, num2 = _normalize(key, nrows * ncols, None)

        return SubplotSpec(self, num1, num2)


class GridSpec(GridSpecBase):
    """
    A grid layout to place subplots within a figure.

    The location of the grid cells is determined in a similar way to
    `~.figure.SubplotParams` using *left*, *right*, *top*, *bottom*, *wspace*
    and *hspace*.
    """
    def __init__(self, nrows, ncols, figure=None,
                 left=None, bottom=None, right=None, top=None,
                 wspace=None, hspace=None,
                 width_ratios=None, height_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            The number of rows and columns of the grid.

        figure : `~.figure.Figure`, optional
            Only used for constrained layout to create a proper layoutbox.

        left, right, top, bottom : float, optional
            Extent of the subplots as a fraction of figure width or height.
            Left cannot be larger than right, and bottom cannot be larger than
            top. If not given, the values will be inferred from a figure or
            rcParams at draw time. See also `GridSpec.get_subplot_params`.

        wspace : float, optional
            The amount of width reserved for space between subplots,
            expressed as a fraction of the average axis width.
            If not given, the values will be inferred from a figure or
            rcParams when necessary. See also `GridSpec.get_subplot_params`.

        hspace : float, optional
            The amount of height reserved for space between subplots,
            expressed as a fraction of the average axis height.
            If not given, the values will be inferred from a figure or
            rcParams when necessary. See also `GridSpec.get_subplot_params`.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each column gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.

        """
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        self.wspace = wspace
        self.hspace = hspace
        self.figure = figure

        GridSpecBase.__init__(self, nrows, ncols,
                              width_ratios=width_ratios,
                              height_ratios=height_ratios)

        if self.figure is None or not self.figure.get_constrained_layout():
            self._layoutbox = None
        else:
            self.figure.init_layoutbox()
            self._layoutbox = layoutbox.LayoutBox(
                parent=self.figure._layoutbox,
                name='gridspec' + layoutbox.seq_id(),
                artist=self)
        # by default the layoutbox for a gridspec will fill a figure.
        # but this can change below if the gridspec is created from a
        # subplotspec. (GridSpecFromSubplotSpec)

    _AllowedKeys = ["left", "bottom", "right", "top", "wspace", "hspace"]

    def __getstate__(self):
        state = self.__dict__
        try:
            state.pop('_layoutbox')
        except KeyError:
            pass
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        # layoutboxes don't survive pickling...
        self._layoutbox = None

    def update(self, **kwargs):
        """
        Update the subplot parameters of the grid.

        Parameters that are not explicitly given are not changed. Setting a
        parameter to *None* resets it to :rc:`figure.subplot.*`.

        Parameters
        ----------
        left, right, top, bottom : float or None, optional
            Extent of the subplots as a fraction of figure width or height.
        wspace, hspace : float, optional
            Spacing between the subplots as a fraction of the average subplot
            width / height.
        """
        for k, v in kwargs.items():
            if k in self._AllowedKeys:
                setattr(self, k, v)
            else:
                raise AttributeError(f"{k} is an unknown keyword")
        for figmanager in _pylab_helpers.Gcf.figs.values():
            for ax in figmanager.canvas.figure.axes:
                # copied from Figure.subplots_adjust
                if not isinstance(ax, mpl.axes.SubplotBase):
                    # Check if sharing a subplots axis
                    if isinstance(ax._sharex, mpl.axes.SubplotBase):
                        if ax._sharex.get_subplotspec().get_gridspec() == self:
                            ax._sharex.update_params()
                            ax._set_position(ax._sharex.figbox)
                    elif isinstance(ax._sharey, mpl.axes.SubplotBase):
                        if ax._sharey.get_subplotspec().get_gridspec() == self:
                            ax._sharey.update_params()
                            ax._set_position(ax._sharey.figbox)
                else:
                    ss = ax.get_subplotspec().get_topmost_subplotspec()
                    if ss.get_gridspec() == self:
                        ax.update_params()
                        ax._set_position(ax.figbox)

    def get_subplot_params(self, figure=None):
        """
        Return the `~.SubplotParams` for the GridSpec.

        In order of precedence the values are taken from

        - non-*None* attributes of the GridSpec
        - the provided *figure*
        - :rc:`figure.subplot.*`
        """
        if figure is None:
            kw = {k: rcParams["figure.subplot."+k] for k in self._AllowedKeys}
            subplotpars = mpl.figure.SubplotParams(**kw)
        else:
            subplotpars = copy.copy(figure.subplotpars)

        subplotpars.update(**{k: getattr(self, k) for k in self._AllowedKeys})

        return subplotpars

    def locally_modified_subplot_params(self):
        """
        Return a list of the names of the subplot parameters explicitly set
        in the GridSpec.

        This is a subset of the attributes of `.SubplotParams`.
        """
        return [k for k in self._AllowedKeys if getattr(self, k)]

    def tight_layout(self, figure, renderer=None,
                     pad=1.08, h_pad=None, w_pad=None, rect=None):
        """
        Adjust subplot parameters to give specified padding.

        Parameters
        ----------
        pad : float
            Padding between the figure edge and the edges of subplots, as a
            fraction of the font-size.
        h_pad, w_pad : float, optional
            Padding (height/width) between edges of adjacent subplots.
            Defaults to *pad*.
        rect : tuple of 4 floats, optional
            (left, bottom, right, top) rectangle in normalized figure
            coordinates that the whole subplots area (including labels) will
            fit into.  Default is (0, 0, 1, 1).
        """

        subplotspec_list = tight_layout.get_subplotspec_list(
            figure.axes, grid_spec=self)
        if None in subplotspec_list:
            cbook._warn_external("This figure includes Axes that are not "
                                 "compatible with tight_layout, so results "
                                 "might be incorrect.")

        if renderer is None:
            renderer = tight_layout.get_renderer(figure)

        kwargs = tight_layout.get_tight_layout_figure(
            figure, figure.axes, subplotspec_list, renderer,
            pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
        if kwargs:
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
        self._wspace = wspace
        self._hspace = hspace
        self._subplot_spec = subplot_spec
        GridSpecBase.__init__(self, nrows, ncols,
                              width_ratios=width_ratios,
                              height_ratios=height_ratios)
        # do the layoutboxes
        subspeclb = subplot_spec._layoutbox
        if subspeclb is None:
            self._layoutbox = None
        else:
            # OK, this is needed to divide the figure.
            self._layoutbox = subspeclb.layout_from_subplotspec(
                    subplot_spec,
                    name=subspeclb.name + '.gridspec' + layoutbox.seq_id(),
                    artist=self)

    def get_subplot_params(self, figure=None):
        """Return a dictionary of subplot layout parameters.
        """
        hspace = (self._hspace if self._hspace is not None
                  else figure.subplotpars.hspace if figure is not None
                  else rcParams["figure.subplot.hspace"])
        wspace = (self._wspace if self._wspace is not None
                  else figure.subplotpars.wspace if figure is not None
                  else rcParams["figure.subplot.wspace"])

        figbox = self._subplot_spec.get_position(figure)
        left, bottom, right, top = figbox.extents

        return mpl.figure.SubplotParams(left=left, right=right,
                                        bottom=bottom, top=top,
                                        wspace=wspace, hspace=hspace)

    def get_topmost_subplotspec(self):
        """
        Return the topmost `.SubplotSpec` instance associated with the subplot.
        """
        return self._subplot_spec.get_topmost_subplotspec()


class SubplotSpec:
    """
    Specifies the location of a subplot in a `GridSpec`.

    .. note::

        Likely, you'll never instantiate a `SubplotSpec` yourself. Instead you
        will typically obtain one from a `GridSpec` using item-access.

    Parameters
    ----------
    gridspec : `~matplotlib.gridspec.GridSpec`
        The GridSpec, which the subplot is referencing.
    num1, num2 : int
        The subplot will occupy the num1-th cell of the given
        gridspec.  If num2 is provided, the subplot will span between
        num1-th cell and num2-th cell *inclusive*.

        The index starts from 0.
    """
    def __init__(self, gridspec, num1, num2=None):
        self._gridspec = gridspec
        self.num1 = num1
        self.num2 = num2
        if gridspec._layoutbox is not None:
            glb = gridspec._layoutbox
            # So note that here we don't assign any layout yet,
            # just make the layoutbox that will contain all items
            # associated w/ this axis.  This can include other axes like
            # a colorbar or a legend.
            self._layoutbox = layoutbox.LayoutBox(
                    parent=glb,
                    name=glb.name + '.ss' + layoutbox.seq_id(),
                    artist=self)
        else:
            self._layoutbox = None

    # num2 is a property only to handle the case where it is None and someone
    # mutates num1.

    @property
    def num2(self):
        return self.num1 if self._num2 is None else self._num2

    @num2.setter
    def num2(self, value):
        self._num2 = value

    def __getstate__(self):
        state = self.__dict__
        try:
            state.pop('_layoutbox')
        except KeyError:
            pass
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        # layoutboxes don't survive pickling...
        self._layoutbox = None

    def get_gridspec(self):
        return self._gridspec

    def get_geometry(self):
        """
        Return the subplot geometry as tuple ``(n_rows, n_cols, start, stop)``.

        The indices *start* and *stop* define the range of the subplot within
        the `GridSpec`. *stop* is inclusive (i.e. for a single cell
        ``start == stop``).
        """
        rows, cols = self.get_gridspec().get_geometry()
        return rows, cols, self.num1, self.num2

    def get_rows_columns(self):
        """
        Return the subplot row and column numbers as a tuple
        ``(n_rows, n_cols, row_start, row_stop, col_start, col_stop)``.
        """
        gridspec = self.get_gridspec()
        nrows, ncols = gridspec.get_geometry()
        row_start, col_start = divmod(self.num1, ncols)
        row_stop, col_stop = divmod(self.num2, ncols)
        return nrows, ncols, row_start, row_stop, col_start, col_stop

    @property
    def rowspan(self):
        """The rows spanned by this subplot, as a `range` object."""
        ncols = self.get_gridspec().ncols
        return range(self.num1 // ncols, self.num2 // ncols + 1)

    @property
    def colspan(self):
        """The columns spanned by this subplot, as a `range` object."""
        ncols = self.get_gridspec().ncols
        return range(self.num1 % ncols, self.num2 % ncols + 1)

    def get_position(self, figure, return_all=False):
        """
        Update the subplot position from ``figure.subplotpars``.
        """
        gridspec = self.get_gridspec()
        nrows, ncols = gridspec.get_geometry()
        rows, cols = np.unravel_index([self.num1, self.num2], (nrows, ncols))
        fig_bottoms, fig_tops, fig_lefts, fig_rights = \
            gridspec.get_grid_positions(figure)

        fig_bottom = fig_bottoms[rows].min()
        fig_top = fig_tops[rows].max()
        fig_left = fig_lefts[cols].min()
        fig_right = fig_rights[cols].max()
        figbox = Bbox.from_extents(fig_left, fig_bottom, fig_right, fig_top)

        if return_all:
            return figbox, rows[0], cols[0], nrows, ncols
        else:
            return figbox

    def get_topmost_subplotspec(self):
        """
        Return the topmost `SubplotSpec` instance associated with the subplot.
        """
        gridspec = self.get_gridspec()
        if hasattr(gridspec, "get_topmost_subplotspec"):
            return gridspec.get_topmost_subplotspec()
        else:
            return self

    def __eq__(self, other):
        """
        Two SubplotSpecs are considered equal if they refer to the same
        position(s) in the same `GridSpec`.
        """
        # other may not even have the attributes we are checking.
        return ((self._gridspec, self.num1, self.num2)
                == (getattr(other, "_gridspec", object()),
                    getattr(other, "num1", object()),
                    getattr(other, "num2", object())))

    def __hash__(self):
        return hash((self._gridspec, self.num1, self.num2))

    def subgridspec(self, nrows, ncols, **kwargs):
        """
        Create a GridSpec within this subplot.

        The created `.GridSpecFromSubplotSpec` will have this `SubplotSpec` as
        a parent.

        Parameters
        ----------
        nrows : int
            Number of rows in grid.

        ncols : int
            Number or columns in grid.

        Returns
        -------
        gridspec : `.GridSpecFromSubplotSpec`

        Other Parameters
        ----------------
        **kwargs
            All other parameters are passed to `.GridSpecFromSubplotSpec`.

        See Also
        --------
        matplotlib.pyplot.subplots

        Examples
        --------
        Adding three subplots in the space occupied by a single subplot::

            fig = plt.figure()
            gs0 = fig.add_gridspec(3, 1)
            ax1 = fig.add_subplot(gs0[0])
            ax2 = fig.add_subplot(gs0[1])
            gssub = gs0[2].subgridspec(1, 3)
            for i in range(3):
                fig.add_subplot(gssub[0, i])
        """
        return GridSpecFromSubplotSpec(nrows, ncols, self, **kwargs)
