"""
Colorbars are a visualization of the mapping from scalar values to colors.
In Matplotlib they are drawn into a dedicated `~.axes.Axes`.

.. note::
   Colorbars are typically created through `.Figure.colorbar` or its pyplot
   wrapper `.pyplot.colorbar`, which use `.make_axes` and `.Colorbar`
   internally.

   As an end-user, you most likely won't have to call the methods or
   instantiate the classes in this module explicitly.

:class:`ColorbarBase`
    The base class with full colorbar drawing functionality.
    It can be used as-is to make a colorbar for a given colormap;
    a mappable object (e.g., image) is not needed.

:class:`Colorbar`
    On top of `.ColorbarBase` this connects the colorbar with a
    `.ScalarMappable` such as an image or contour plot.

:func:`make_axes`
    Create an `~.axes.Axes` suitable for a colorbar. This functions can be
    used with figures containing a single axes or with freely placed axes.

:func:`make_axes_gridspec`
    Create a `~.SubplotBase` suitable for a colorbar. This function should
    be used for adding a colorbar to a `.GridSpec`.
"""

import copy
import logging
import textwrap

import numpy as np

import matplotlib as mpl
from matplotlib import _api, collections, cm, colors, contour, ticker
from matplotlib.axes._base import _TransformedBoundsLocator
from matplotlib.axes._axes import Axes
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import docstring

_log = logging.getLogger(__name__)

_make_axes_param_doc = """
location : None or {'left', 'right', 'top', 'bottom'}
    The location, relative to the parent axes, where the colorbar axes
    is created.  It also determines the *orientation* of the colorbar
    (colorbars on the left and right are vertical, colorbars at the top
    and bottom are horizontal).  If None, the location will come from the
    *orientation* if it is set (vertical colorbars on the right, horizontal
    ones at the bottom), or default to 'right' if *orientation* is unset.
orientation : None or {'vertical', 'horizontal'}
    The orientation of the colorbar.  It is preferable to set the *location*
    of the colorbar, as that also determines the *orientation*; passing
    incompatible values for *location* and *orientation* raises an exception.
fraction : float, default: 0.15
    Fraction of original axes to use for colorbar.
shrink : float, default: 1.0
    Fraction by which to multiply the size of the colorbar.
aspect : float, default: 20
    Ratio of long to short dimensions.
"""
_make_axes_other_param_doc = """
pad : float, default: 0.05 if vertical, 0.15 if horizontal
    Fraction of original axes between colorbar and new image axes.
anchor : (float, float), optional
    The anchor point of the colorbar axes.
    Defaults to (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal.
panchor : (float, float), or *False*, optional
    The anchor point of the colorbar parent axes. If *False*, the parent
    axes' anchor will be unchanged.
    Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.
"""

_colormap_kw_doc = """

    ============  ====================================================
    Property      Description
    ============  ====================================================
    *extend*      {'neither', 'both', 'min', 'max'}
                  If not 'neither', make pointed end(s) for out-of-
                  range values.  These are set for a given colormap
                  using the colormap set_under and set_over methods.
    *extendfrac*  {*None*, 'auto', length, lengths}
                  If set to *None*, both the minimum and maximum
                  triangular colorbar extensions with have a length of
                  5% of the interior colorbar length (this is the
                  default setting). If set to 'auto', makes the
                  triangular colorbar extensions the same lengths as
                  the interior boxes (when *spacing* is set to
                  'uniform') or the same lengths as the respective
                  adjacent interior boxes (when *spacing* is set to
                  'proportional'). If a scalar, indicates the length
                  of both the minimum and maximum triangular colorbar
                  extensions as a fraction of the interior colorbar
                  length. A two-element sequence of fractions may also
                  be given, indicating the lengths of the minimum and
                  maximum colorbar extensions respectively as a
                  fraction of the interior colorbar length.
    *extendrect*  bool
                  If *False* the minimum and maximum colorbar extensions
                  will be triangular (the default). If *True* the
                  extensions will be rectangular.
    *spacing*     {'uniform', 'proportional'}
                  Uniform spacing gives each discrete color the same
                  space; proportional makes the space proportional to
                  the data interval.
    *ticks*       *None* or list of ticks or Locator
                  If None, ticks are determined automatically from the
                  input.
    *format*      None or str or Formatter
                  If None, `~.ticker.ScalarFormatter` is used.
                  If a format string is given, e.g., '%.3f', that is used.
                  An alternative `~.ticker.Formatter` may be given instead.
    *drawedges*   bool
                  Whether to draw lines at color boundaries.
    *label*       str
                  The label on the colorbar's long axis.
    ============  ====================================================

    The following will probably be useful only in the context of
    indexed colors (that is, when the mappable has norm=NoNorm()),
    or other unusual circumstances.

    ============   ===================================================
    Property       Description
    ============   ===================================================
    *boundaries*   None or a sequence
    *values*       None or a sequence which must be of length 1 less
                   than the sequence of *boundaries*. For each region
                   delimited by adjacent entries in *boundaries*, the
                   colormapped to the corresponding value in values
                   will be used.
    ============   ===================================================

"""

docstring.interpd.update(colorbar_doc="""
Add a colorbar to a plot.

Parameters
----------
mappable
    The `matplotlib.cm.ScalarMappable` (i.e., `~matplotlib.image.AxesImage`,
    `~matplotlib.contour.ContourSet`, etc.) described by this colorbar.
    This argument is mandatory for the `.Figure.colorbar` method but optional
    for the `.pyplot.colorbar` function, which sets the default to the current
    image.

    Note that one can create a `.ScalarMappable` "on-the-fly" to generate
    colorbars not attached to a previously drawn artist, e.g. ::

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

cax : `~matplotlib.axes.Axes`, optional
    Axes into which the colorbar will be drawn.

ax : `~matplotlib.axes.Axes`, list of Axes, optional
    One or more parent axes from which space for a new colorbar axes will be
    stolen, if *cax* is None.  This has no effect if *cax* is set.

use_gridspec : bool, optional
    If *cax* is ``None``, a new *cax* is created as an instance of Axes.  If
    *ax* is an instance of Subplot and *use_gridspec* is ``True``, *cax* is
    created as an instance of Subplot using the :mod:`~.gridspec` module.

Returns
-------
colorbar : `~matplotlib.colorbar.Colorbar`
    See also its base class, `~matplotlib.colorbar.ColorbarBase`.

Notes
-----
Additional keyword arguments are of two kinds:

  axes properties:
%s
%s
  colorbar properties:
%s

If *mappable* is a `~.contour.ContourSet`, its *extend* kwarg is included
automatically.

The *shrink* kwarg provides a simple way to scale the colorbar with respect
to the axes. Note that if *cax* is specified, it determines the size of the
colorbar and *shrink* and *aspect* kwargs are ignored.

For more precise control, you can manually specify the positions of
the axes objects in which the mappable and the colorbar are drawn.  In
this case, do not use any of the axes properties kwargs.

It is known that some vector graphics viewers (svg and pdf) renders white gaps
between segments of the colorbar.  This is due to bugs in the viewers, not
Matplotlib.  As a workaround, the colorbar can be rendered with overlapping
segments::

    cbar = colorbar()
    cbar.solids.set_edgecolor("face")
    draw()

However this has negative consequences in other circumstances, e.g. with
semi-transparent images (alpha < 1) and colorbar extensions; therefore, this
workaround is not used by default (see issue #1188).
""" % (textwrap.indent(_make_axes_param_doc, "    "),
       textwrap.indent(_make_axes_other_param_doc, "    "),
       _colormap_kw_doc))

# Deprecated since 3.4.
colorbar_doc = docstring.interpd.params["colorbar_doc"]
colormap_kw_doc = _colormap_kw_doc
make_axes_kw_doc = _make_axes_param_doc + _make_axes_other_param_doc


def _set_ticks_on_axis_warn(*args, **kw):
    # a top level function which gets put in at the axes'
    # set_xticks and set_yticks by ColorbarBase.__init__.
    _api.warn_external("Use the colorbar set_ticks() method instead.")


class ColorbarAxes(Axes):
    """
    ColorbarAxes packages two axes, a parent axes that takes care of
    positioning the axes, and an inset_axes that takes care of the drawing,
    labels, ticks, etc. The inset axes is used as a way to properly
    position the extensions (triangles or rectangles) that are used to indicate
    over/under colors.

    Users should not normally instantiate this class, but it is the class
    returned by ``cbar = fig.colorbar(im); cax = cbar.ax``.
    """
    def __init__(self, parent, userax=True):
        """
        Parameters
        ----------
        parent : Axes
            Axes that specifies the position of the colorbar.
        userax : boolean
            True if the user passed `.Figure.colorbar` the axes manually.
        """

        if userax:
            # copy position:
            fig = parent.figure
            outer_ax = fig.add_axes(parent.get_position())
            # copy the locator if one exists:
            outer_ax._axes_locator = parent._axes_locator
            # if the parent is a child of another axes, swap these...
            if (parent._axes is not None and
                    parent in parent._axes.child_axes):
                parent._axes.add_child_axes(outer_ax)
                outer_ax._axes.child_axes.remove(parent)
            else:
                parent.remove()
        else:
            outer_ax = parent

        inner_ax = outer_ax.inset_axes([0, 0, 1, 1])
        self.__dict__.update(inner_ax.__dict__)

        self.outer_ax = outer_ax
        self.inner_ax = inner_ax
        self.outer_ax.xaxis.set_visible(False)
        self.outer_ax.yaxis.set_visible(False)
        self.outer_ax.set_facecolor('none')
        self.outer_ax.tick_params = self.inner_ax.tick_params
        self.outer_ax.set_xticks = self.inner_ax.set_xticks
        self.outer_ax.set_yticks = self.inner_ax.set_yticks
        for attr in ["get_position", "set_position", "set_aspect"]:
            setattr(self, attr, getattr(self.outer_ax, attr))
        if userax:
            # point the parent's methods all at this axes...
            parent.__dict__ = self.__dict__

    def _set_inner_bounds(self, bounds):
        """
        Change the inset_axes location...
        """
        self.inner_ax._axes_locator = _TransformedBoundsLocator(
            bounds, self.outer_ax.transAxes)


class _ColorbarSpine(mspines.Spine):
    def __init__(self, axes):
        self._ax = axes
        super().__init__(axes, 'colorbar',
                         mpath.Path(np.empty((0, 2)), closed=True))
        mpatches.Patch.set_transform(self, axes.outer_ax.transAxes)

    def get_window_extent(self, renderer=None):
        # This Spine has no Axis associated with it, and doesn't need to adjust
        # its location, so we can directly get the window extent from the
        # super-super-class.
        return mpatches.Patch.get_window_extent(self, renderer=renderer)

    def set_xy(self, xy):
        self._path = mpath.Path(xy, closed=True)
        self._xy = xy
        self.stale = True

    def draw(self, renderer):
        ret = mpatches.Patch.draw(self, renderer)
        self.stale = False
        return ret


class ColorbarBase:
    r"""
    Draw a colorbar in an existing axes.

    There are only some rare cases in which you would work directly with a
    `.ColorbarBase` as an end-user. Typically, colorbars are used
    with `.ScalarMappable`\s such as an `.AxesImage` generated via
    `~.axes.Axes.imshow`. For these cases you will use `.Colorbar` and
    likely create it via `.pyplot.colorbar` or `.Figure.colorbar`.

    The main application of using a `.ColorbarBase` explicitly is drawing
    colorbars that are not associated with other elements in the figure, e.g.
    when showing a colormap by itself.

    If the *cmap* kwarg is given but *boundaries* and *values* are left as
    None, then the colormap will be displayed on a 0-1 scale. To show the
    under- and over-value colors, specify the *norm* as::

        norm=colors.Normalize(clip=False)

    To show the colors versus index instead of on the 0-1 scale,
    use::

        norm=colors.NoNorm()

    Useful public methods are :meth:`set_label` and :meth:`add_lines`.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.
    lines : list
        A list of `.LineCollection` (empty if no lines were drawn).
    dividers : `.LineCollection`
        A LineCollection (empty if *drawedges* is ``False``).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` instance in which the colorbar is drawn.
    cmap : `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
        The colormap to use.
    norm : `~matplotlib.colors.Normalize`

    alpha : float
        The colorbar transparency between 0 (transparent) and 1 (opaque).

    values

    boundaries

    orientation : {'vertical', 'horizontal'}

    ticklocation : {'auto', 'left', 'right', 'top', 'bottom'}

    extend : {'neither', 'both', 'min', 'max'}

    spacing : {'uniform', 'proportional'}

    ticks : `~matplotlib.ticker.Locator` or array-like of float

    format : str or `~matplotlib.ticker.Formatter`

    drawedges : bool

    filled : bool

    extendfrac

    extendrec

    label : str

    userax : boolean
        Whether the user created the axes or not.  Default True
    """

    n_rasterize = 50  # rasterize solids if number of colors >= n_rasterize

    @_api.make_keyword_only("3.3", "cmap")
    def __init__(self, ax, cmap=None,
                 norm=None,
                 alpha=None,
                 values=None,
                 boundaries=None,
                 orientation='vertical',
                 ticklocation='auto',
                 extend=None,
                 spacing='uniform',  # uniform or proportional
                 ticks=None,
                 format=None,
                 drawedges=False,
                 filled=True,
                 extendfrac=None,
                 extendrect=False,
                 label='',
                 userax=False,
                 ):
        _api.check_isinstance([colors.Colormap, None], cmap=cmap)
        _api.check_in_list(
            ['vertical', 'horizontal'], orientation=orientation)
        _api.check_in_list(
            ['auto', 'left', 'right', 'top', 'bottom'],
            ticklocation=ticklocation)
        _api.check_in_list(
            ['uniform', 'proportional'], spacing=spacing)

        # wrap the axes so that it can be positioned as an inset axes:
        ax = ColorbarAxes(ax, userax=userax)
        self.ax = ax
        ax.set(navigate=False)

        if cmap is None:
            cmap = cm.get_cmap()
        if norm is None:
            norm = colors.Normalize()
        if extend is None:
            if hasattr(norm, 'extend'):
                extend = norm.extend
            else:
                extend = 'neither'
        self.alpha = alpha
        self.cmap = cmap
        self.norm = norm
        self.values = values
        self.boundaries = boundaries
        self.extend = extend
        self._inside = _api.check_getitem(
            {'neither': slice(0, None), 'both': slice(1, -1),
             'min': slice(1, None), 'max': slice(0, -1)},
            extend=extend)
        self.spacing = spacing
        self.orientation = orientation
        self.drawedges = drawedges
        self.filled = filled
        self.extendfrac = extendfrac
        self.extendrect = extendrect
        self.solids = None
        self.solids_patches = []
        self.lines = []

        for spine in self.ax.spines.values():
            spine.set_visible(False)
        for spine in self.ax.outer_ax.spines.values():
            spine.set_visible(False)
        self.outline = self.ax.spines['outline'] = _ColorbarSpine(self.ax)

        self.patch = mpatches.Polygon(
            np.empty((0, 2)),
            color=mpl.rcParams['axes.facecolor'], linewidth=0.01, zorder=-1)
        ax.add_artist(self.patch)

        self.dividers = collections.LineCollection(
            [],
            colors=[mpl.rcParams['axes.edgecolor']],
            linewidths=[0.5 * mpl.rcParams['axes.linewidth']])
        self.ax.add_collection(self.dividers)

        self.locator = None
        self.formatter = None
        self.__scale = None  # linear, log10 for now.  Hopefully more?

        if ticklocation == 'auto':
            ticklocation = 'bottom' if orientation == 'horizontal' else 'right'
        self.ticklocation = ticklocation

        self.set_label(label)
        self._reset_locator_formatter_scale()

        if np.iterable(ticks):
            self.locator = ticker.FixedLocator(ticks, nbins=len(ticks))
        else:
            self.locator = ticks    # Handle default in _ticker()

        if isinstance(format, str):
            self.formatter = ticker.FormatStrFormatter(format)
        else:
            self.formatter = format  # Assume it is a Formatter or None
        self.draw_all()

    def draw_all(self):
        """
        Calculate any free parameters based on the current cmap and norm,
        and do all the drawing.
        """
        if self.orientation == 'vertical':
            if mpl.rcParams['ytick.minor.visible']:
                self.minorticks_on()
        else:
            if mpl.rcParams['xtick.minor.visible']:
                self.minorticks_on()
        self._long_axis().set(label_position=self.ticklocation,
                              ticks_position=self.ticklocation)
        self._short_axis().set_ticks([])
        self._short_axis().set_ticks([], minor=True)

        # Set self._boundaries and self._values, including extensions.
        # self._boundaries are the edges of each square of color, and
        # self._values are the value to map into the norm to get the
        # color:
        self._process_values()
        # Set self.vmin and self.vmax to first and last boundary, excluding
        # extensions:
        self.vmin, self.vmax = self._boundaries[self._inside][[0, -1]]
        # Compute the X/Y mesh.
        X, Y, extendlen = self._mesh()
        # draw the extend triangles, and shrink the inner axes to accomodate.
        # also adds the outline path to self.outline spine:
        self._do_extends(extendlen)

        self.ax.set_xlim(self.vmin, self.vmax)
        self.ax.set_ylim(self.vmin, self.vmax)

        # set up the tick locators and formatters.  A bit complicated because
        # boundary norms + uniform spacing requires a manual locator.
        self.update_ticks()

        if self.filled:
            ind = np.arange(len(self._values))
            if self._extend_lower():
                ind = ind[1:]
            if self._extend_upper():
                ind = ind[:-1]
            self._add_solids(X, Y, self._values[ind, np.newaxis])

    def _add_solids(self, X, Y, C):
        """Draw the colors; optionally add separators."""
        # Cleanup previously set artists.
        if self.solids is not None:
            self.solids.remove()
        for solid in self.solids_patches:
            solid.remove()
        # Add new artist(s), based on mappable type.  Use individual patches if
        # hatching is needed, pcolormesh otherwise.
        mappable = getattr(self, 'mappable', None)
        if (isinstance(mappable, contour.ContourSet)
                and any(hatch is not None for hatch in mappable.hatches)):
            self._add_solids_patches(X, Y, C, mappable)
        else:
            self.solids = self.ax.pcolormesh(
                X, Y, C, cmap=self.cmap, norm=self.norm, alpha=self.alpha,
                edgecolors='none', shading='flat')
            if not self.drawedges:
                if len(self._y) >= self.n_rasterize:
                    self.solids.set_rasterized(True)
        self.dividers.set_segments(
            np.dstack([X, Y])[1:-1] if self.drawedges else [])

    def _add_solids_patches(self, X, Y, C, mappable):
        hatches = mappable.hatches * len(C)  # Have enough hatches.
        patches = []
        for i in range(len(X) - 1):
            xy = np.array([[X[i, 0], Y[i, 0]],
                           [X[i, 1], Y[i, 0]],
                           [X[i + 1, 1], Y[i + 1, 0]],
                           [X[i + 1, 0], Y[i + 1, 1]]])
            patch = mpatches.PathPatch(mpath.Path(xy),
                                       facecolor=self.cmap(self.norm(C[i][0])),
                                       hatch=hatches[i], linewidth=0,
                                       antialiased=False, alpha=self.alpha)
            self.ax.add_patch(patch)
            patches.append(patch)
        self.solids_patches = patches

    def _do_extends(self, extendlen):
        """
        Make adjustments of the inner axes for the extend triangles (or
        rectanges) and add them as patches.
        """
        # extend lengths are fraction of the *inner* part of colorbar,
        # not the total colorbar:
        elower = extendlen[0] if self._extend_lower() else 0
        eupper = extendlen[1] if self._extend_upper() else 0
        total_len = eupper + elower + 1
        elower = elower / total_len
        eupper = eupper / total_len
        inner_length = 1 / total_len

        # make the inner axes smaller to make room for the extend rectangle
        top = elower + inner_length

        # xyout is the outline of the colorbar including the extend patches:
        if not self.extendrect:
            # triangle:
            xyout = np.array([[0, elower], [0.5, 0], [1, elower],
                              [1, top], [0.5, 1], [0, top], [0, elower]])
        else:
            # rectangle:
            xyout = np.array([[0, elower], [0, 0], [1, 0], [1, elower],
                              [1, top], [1, 1], [0, 1], [0, top],
                              [0, elower]])

        bounds = np.array([0.0, elower, 1.0, inner_length])
        if self.orientation == 'horizontal':
            bounds = bounds[[1, 0, 3, 2]]
            xyout = xyout[:, ::-1]
        self.ax._set_inner_bounds(bounds)

        # xyout is the path for the spine:
        self.outline.set_xy(xyout)
        if not self.filled:
            return

        # Make extend triangles or rectangles filled patches.  These are
        # defined in the outer parent axes' co-ordinates:
        mappable = getattr(self, 'mappable', None)
        if (isinstance(mappable, contour.ContourSet)
                and any(hatch is not None for hatch in mappable.hatches)):
            hatches = mappable.hatches
        else:
            hatches = [None]

        if self._extend_lower():
            if not self.extendrect:
                # triangle
                xy = np.array([[0.5, 0], [1, elower], [0, elower]])
            else:
                # rectangle
                xy = np.array([[0, 0], [1., 0], [1, elower], [0, elower]])
            if self.orientation == 'horizontal':
                xy = xy[:, ::-1]
            # add the patch
            color = self.cmap(self.norm(self._values[0]))
            patch = mpatches.PathPatch(
                mpath.Path(xy), facecolor=color, linewidth=0,
                antialiased=False, transform=self.ax.outer_ax.transAxes,
                hatch=hatches[0])
            self.ax.outer_ax.add_patch(patch)
        if self._extend_upper():
            if not self.extendrect:
                # triangle
                xy = np.array([[0.5, 1], [1, 1-eupper], [0, 1-eupper]])
            else:
                # rectangle
                xy = np.array([[0, 1], [1, 1], [1, 1-eupper], [0, 1-eupper]])
            if self.orientation == 'horizontal':
                xy = xy[:, ::-1]
            # add the patch
            color = self.cmap(self.norm(self._values[-1]))
            patch = mpatches.PathPatch(
                mpath.Path(xy), facecolor=color,
                linewidth=0, antialiased=False,
                transform=self.ax.outer_ax.transAxes, hatch=hatches[-1])
            self.ax.outer_ax.add_patch(patch)
        return

    def add_lines(self, levels, colors, linewidths, erase=True):
        """
        Draw lines on the colorbar.

        The lines are appended to the list :attr:`lines`.

        Parameters
        ----------
        levels : array-like
            The positions of the lines.
        colors : color or list of colors
            Either a single color applying to all lines or one color value for
            each line.
        linewidths : float or array-like
            Either a single linewidth applying to all lines or one linewidth
            for each line.
        erase : bool, default: True
            Whether to remove any previously added lines.
        """
        y = self._locate(levels)
        rtol = (self._y[-1] - self._y[0]) * 1e-10
        igood = (y < self._y[-1] + rtol) & (y > self._y[0] - rtol)
        y = y[igood]
        if np.iterable(colors):
            colors = np.asarray(colors)[igood]
        if np.iterable(linewidths):
            linewidths = np.asarray(linewidths)[igood]
        X, Y = np.meshgrid([self._y[0], self._y[-1]], y)
        if self.orientation == 'vertical':
            xy = np.stack([X, Y], axis=-1)
        else:
            xy = np.stack([Y, X], axis=-1)
        col = collections.LineCollection(xy, linewidths=linewidths,
                                         colors=colors)

        if erase and self.lines:
            for lc in self.lines:
                lc.remove()
            self.lines = []
        self.lines.append(col)

        # make a clip path that is just a linewidth bigger than the axes...
        fac = np.max(linewidths) / 72
        xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        inches = self.ax.get_figure().dpi_scale_trans
        # do in inches:
        xy = inches.inverted().transform(self.ax.transAxes.transform(xy))
        xy[[0, 1, 4], 1] -= fac
        xy[[2, 3], 1] += fac
        # back to axes units...
        xy = self.ax.transAxes.inverted().transform(inches.transform(xy))
        if self.orientation == 'horizontal':
            xy = xy.T
        col.set_clip_path(mpath.Path(xy, closed=True),
                          self.ax.transAxes)
        self.ax.add_collection(col)
        self.stale = True

    def update_ticks(self):
        """
        Setup the ticks and ticklabels. This should not be needed by users.
        """
        ax = self.ax
        # Get the locator and formatter; defaults to self.locator if not None.
        self._get_ticker_locator_formatter()
        self._long_axis().set_major_locator(self.locator)
        self._long_axis().set_minor_locator(self.minorlocator)
        self._long_axis().set_major_formatter(self.formatter)

    def _get_ticker_locator_formatter(self):
        """
        Return the ``locator`` and ``formatter`` of the colorbar.

        If they have not been defined (i.e. are *None*), the formatter and
        locator are retrieved from the axis, or from the value of the
        boundaries for a boundary norm.

        Called by update_ticks...
        """
        locator = self.locator
        formatter = self.formatter
        minorlocator = self.minorlocator
        if isinstance(self.norm, colors.BoundaryNorm):
            b = self.norm.boundaries
            if locator is None:
                locator = ticker.FixedLocator(b, nbins=10)
        elif self.boundaries is not None:
            b = self._boundaries[self._inside]
            if locator is None:
                locator = ticker.FixedLocator(b, nbins=10)
        else:  # most cases:
            if locator is None:
                # we haven't set the locator explicitly, so use the default
                # for this axis:
                locator = self._long_axis().get_major_locator()
            if minorlocator is None:
                minorlocator = self._long_axis().get_minor_locator()
            if isinstance(self.norm, colors.NoNorm):
                # default locator:
                nv = len(self._values)
                base = 1 + int(nv / 10)
                locator = ticker.IndexLocator(base=base, offset=0)

        if minorlocator is None:
            minorlocator = ticker.NullLocator()

        if formatter is None:
            formatter = self._long_axis().get_major_formatter()

        self.locator = locator
        self.formatter = formatter
        self.minorlocator = minorlocator
        _log.debug('locator: %r', locator)

    @_api.delete_parameter("3.5", "update_ticks")
    def set_ticks(self, ticks, update_ticks=True, *, minor=False):
        """
        Set tick locations.

        Parameters
        ----------
        ticks : array-like or `~matplotlib.ticker.Locator` or None
            The tick positions can be hard-coded by an array of values; or
            they can be defined by a `.Locator`. Setting to *None* reverts
            to using a default locator.
        
        minor : boolean, default: False
            If True, set the minor ticks

        update_ticks : bool, default: True
            As of 3.5 this has no effect.

        """
        self._long_axis().set_ticks(ticks, minor=minor)
        self.stale = True

    def get_ticks(self, minor=False):
        """
        Return the ticks as a list of locations.

        Parameters
        ----------
        minor : boolean, default: False
            if True return the minor ticks.
        """
        if minor:
            return self._long_axis().get_minorticklocs()
        else:
            return self._long_axis().get_majorticklocs()

    @_api.delete_parameter("3.5", "update_ticks")
    def set_ticklabels(self, ticklabels, update_ticks=True, **kwargs):
        """
        Set tick labels.

        Parameters
        ----------
        ticklabels : sequence of str or of `.Text`
            Texts for labeling each tick location in the sequence set by
            `.Axis.set_ticks`; the number of labels must match the number of
            locations.

        update_ticks : bool, default: True
            This keyword argument is ignored and will be be removed.
            Deprecated
        """
        self.ax._long_axis().set_ticklabels(self, **kwargs)
        self.stale = True

    def minorticks_on(self):
        """
        Turn on colorbar minor ticks.
        """
        self.ax.minorticks_on()
        self.minorlocator = self._long_axis().get_minor_locator()
        self._short_axis().set_minor_locator(ticker.NullLocator())

    def minorticks_off(self):
        """Turn the minor ticks of the colorbar off."""
        self.minorlocator = ticker.NullLocator()
        self._long_axis().set_minor_locator(self.minorlocator)

    def set_label(self, label, *, loc=None, **kwargs):
        """
        Add a label to the long axis of the colorbar.

        Parameters
        ----------
        label : str
            The label text.
        loc : str, optional
            The location of the label.

            - For horizontal orientation one of {'left', 'center', 'right'}
            - For vertical orientation one of {'bottom', 'center', 'top'}

            Defaults to :rc:`xaxis.labellocation` or :rc:`yaxis.labellocation`
            depending on the orientation.
        **kwargs
            Keyword arguments are passed to `~.Axes.set_xlabel` /
            `~.Axes.set_ylabel`.
            Supported keywords are *labelpad* and `.Text` properties.
        """
        if self.orientation == "vertical":
            self.ax.set_ylabel(label, loc=loc, **kwargs)
        else:
            self.ax.set_xlabel(label, loc=loc, **kwargs)
        self.stale = True

    def set_alpha(self, alpha):
        """Set the transparency between 0 (transparent) and 1 (opaque)."""
        self.alpha = alpha

    def remove(self):
        """Remove this colorbar from the figure."""
        self.ax.inner_ax.remove()
        self.ax.outer_ax.remove()

    def _ticker(self, locator, formatter):
        """
        Return the sequence of ticks (colorbar data locations),
        ticklabels (strings), and the corresponding offset string.
        """
        if isinstance(self.norm, colors.NoNorm) and self.boundaries is None:
            intv = self._values[0], self._values[-1]
        else:
            intv = self.vmin, self.vmax
        locator.create_dummy_axis(minpos=intv[0])
        locator.axis.set_view_interval(*intv)
        locator.axis.set_data_interval(*intv)
        formatter.set_axis(locator.axis)

        b = np.array(locator())
        if isinstance(locator, ticker.LogLocator):
            eps = 1e-10
            b = b[(b <= intv[1] * (1 + eps)) & (b >= intv[0] * (1 - eps))]
        else:
            eps = (intv[1] - intv[0]) * 1e-10
            b = b[(b <= intv[1] + eps) & (b >= intv[0] - eps)]
        ticks = self._locate(b)
        ticklabels = formatter.format_ticks(b)
        offset_string = formatter.get_offset()
        return ticks, ticklabels, offset_string

    def _process_values(self):
        """
        Set `_boundaries` and `_values` based on the self.boundaries and
        self.values if not None, or based on the size of the colormap and
        the vmin/vmax of the norm.
        """
        if self.values is not None:
            # set self._boundaries from the values...
            self._values = np.array(self.values)
            if self.boundaries is None:
                # bracket values by 1/2 dv:
                b = np.zeros(len(self.values) + 1)
                b[1:-1] = 0.5 * (self._values[:-1] + self._values[1:])
                b[0] = 2.0 * b[1] - b[2]
                b[-1] = 2.0 * b[-2] - b[-3]
                self._boundaries = b
                return
            self._boundaries = np.array(self.boundaries)
            return

        # otherwise values are set from the boundaries
        if isinstance(self.norm, colors.BoundaryNorm):
            b = self.norm.boundaries
        else:
            # otherwise make the boundaries from the size of the cmap:
            N = self.cmap.N + 1
            b, _ = self._uniform_y(N)
        # add extra boundaries if needed:
        if self._extend_lower():
            b = np.hstack((b[0] - 1, b))
        if self._extend_upper():
            b = np.hstack((b, b[-1] + 1))

        # transform from 0-1 to vmin-vmax:
        if not self.norm.scaled():
            self.norm.vmin = 0
            self.norm.vmax = 1
        self.norm.vmin, self.norm.vmax = mtransforms.nonsingular(
            self.norm.vmin, self.norm.vmax, expander=0.1)
        if not isinstance(self.norm, colors.BoundaryNorm):
            b = self.norm.inverse(b)

        self._boundaries = np.asarray(b, dtype=float)
        self._values = 0.5 * (self._boundaries[:-1] + self._boundaries[1:])
        if isinstance(self.norm, colors.NoNorm):
            self._values = (self._values + 0.00001).astype(np.int16)

    def _mesh(self):
        """
        Return the coordinate arrays for the colorbar pcolormesh/patches.

        These are scaled between vmin and vmax, and already handle colorbar
        orientation.
        """
        # copy the norm and change the vmin and vmax to the vmin and
        # vmax of the colorbar, not the norm.  This allows the situation
        # where the colormap has a narrower range than the colorbar, to
        # accommodate extra contours:
        norm = copy.copy(self.norm)
        norm.vmin = self.vmin
        norm.vmax = self.vmax
        x = np.array([0.0, 1.0])
        y, extendlen = self._proportional_y()
        # invert:
        if (isinstance(norm, (colors.BoundaryNorm, colors.NoNorm)) or
                (self.__scale == 'manual')):
            # if a norm doesn't have a named scale, or we are not using a norm:
            dv = self.vmax - self.vmin
            x = x * dv + self.vmin
            y = y * dv + self.vmin
        else:
            y = norm.inverse(y)
            x = norm.inverse(x)
        self._y = y
        X, Y = np.meshgrid(x, y)
        if self.orientation == 'vertical':
            return (X, Y, extendlen)
        else:
            return (Y, X, extendlen)

    def _forward_boundaries(self, x):
        b = self._boundaries
        y = np.interp(x, b, np.linspace(0, b[-1], len(b)))
        eps = (b[-1] - b[0]) * 1e-6
        y[x < b[0]-eps] = -1
        y[x > b[-1]+eps] = 2
        return y

    def _inverse_boundaries(self, x):
        b = self._boundaries
        return np.interp(x, np.linspace(0, b[-1], len(b)), b)

    def _reset_locator_formatter_scale(self):
        """
        Reset the locator et al to defaults.  Any user-hardcoded changes
        need to be re-entered if this gets called (either at init, or when
        the mappable normal gets changed: Colorbar.update_normal)
        """
        self._process_values()
        self.locator = None
        self.minorlocator = None
        self.formatter = None
        if ((self.spacing == 'uniform') and
            ((self.boundaries is not None) or
              isinstance(self.norm, colors.BoundaryNorm))):
            funcs = (self._forward_boundaries, self._inverse_boundaries)
            self.ax.set_xscale('function', functions=funcs)
            self.ax.set_yscale('function', functions=funcs)
            self.__scale = 'function'
        elif hasattr(self.norm, '_scale') and (self.norm._scale is not None):
            self.ax.set_xscale(self.norm._scale)
            self.ax.set_yscale(self.norm._scale)
            self.__scale = self.norm._scale.name
        else:
            self.ax.set_xscale('linear')
            self.ax.set_yscale('linear')
            if type(self.norm) is colors.Normalize:
                self.__scale = 'linear'
            else:
                self.__scale = 'manual'

    def _locate(self, x):
        """
        Given a set of color data values, return their
        corresponding colorbar data coordinates.
        """
        if isinstance(self.norm, (colors.NoNorm, colors.BoundaryNorm)):
            b = self._boundaries
            xn = x
        else:
            # Do calculations using normalized coordinates so
            # as to make the interpolation more accurate.
            b = self.norm(self._boundaries, clip=False).filled()
            xn = self.norm(x, clip=False).filled()

        bunique = b[self._inside]
        yunique = self._y

        z = np.interp(xn, bunique, yunique)
        return z

    # trivial helpers

    def _uniform_y(self, N):
        """
        Return colorbar data coordinates for *N* uniformly
        spaced boundaries, plus extension lengths if required.
        """
        automin = automax = 1. / (N - 1.)
        extendlength = self._get_extension_lengths(self.extendfrac,
                                                   automin, automax,
                                                   default=0.05)
        y = np.linspace(0, 1, N)
        return y, extendlength

    def _proportional_y(self):
        """
        Return colorbar data coordinates for the boundaries of
        a proportional colorbar, plus extension lengths if required:
        """
        if isinstance(self.norm, colors.BoundaryNorm):
            y = (self._boundaries - self._boundaries[0])
            y = y / (self._boundaries[-1] - self._boundaries[0])
            # need yscaled the same as the axes scale to get
            # the extend lengths.
            if self.spacing == 'uniform':
                yscaled = self._forward_boundaries(self._boundaries)
            else:
                yscaled = y
        else:
            y = self.norm(self._boundaries.copy())
            y = np.ma.filled(y, np.nan)
            # the norm and the scale should be the same...
            yscaled = y
        y = y[self._inside]
        yscaled = yscaled[self._inside]
        # normalize from 0..1:
        norm = colors.Normalize(y[0], y[-1])
        y = np.ma.filled(norm(y), np.nan)
        norm = colors.Normalize(yscaled[0], yscaled[-1])
        yscaled = np.ma.filled(norm(yscaled), np.nan)
        # make the lower and upper extend lengths proportional to the lengths
        # of the first and last boundary spacing (if extendfrac='auto'):
        automin = yscaled[1] - yscaled[0]
        automax = yscaled[-1] - yscaled[-2]
        extendlength = [0, 0]
        if self._extend_lower() or self._extend_upper():
            extendlength = self._get_extension_lengths(
                    self.extendfrac, automin, automax, default=0.05)
        return y, extendlength

    def _get_extension_lengths(self, frac, automin, automax, default=0.05):
        """
        Return the lengths of colorbar extensions.

        This is a helper method for _uniform_y and _proportional_y.
        """
        # Set the default value.
        extendlength = np.array([default, default])
        if isinstance(frac, str):
            _api.check_in_list(['auto'], extendfrac=frac.lower())
            # Use the provided values when 'auto' is required.
            extendlength[:] = [automin, automax]
        elif frac is not None:
            try:
                # Try to set min and max extension fractions directly.
                extendlength[:] = frac
                # If frac is a sequence containing None then NaN may
                # be encountered. This is an error.
                if np.isnan(extendlength).any():
                    raise ValueError()
            except (TypeError, ValueError) as err:
                # Raise an error on encountering an invalid value for frac.
                raise ValueError('invalid value for extendfrac') from err
        return extendlength

    def _extend_lower(self):
        """Return whether the lower limit is open ended."""
        return self.extend in ('both', 'min')

    def _extend_upper(self):
        """Return whether the upper limit is open ended."""
        return self.extend in ('both', 'max')

    def _long_axis(self):
        """Return the long axis"""
        if self.orientation == 'vertical':
            return self.ax.yaxis
        return self.ax.xaxis

    def _short_axis(self):
        """Return the short axis"""
        if self.orientation == 'vertical':
            return self.ax.xaxis
        return self.ax.yaxis


def _add_disjoint_kwargs(d, **kwargs):
    """
    Update dict *d* with entries in *kwargs*, which must be absent from *d*.
    """
    for k, v in kwargs.items():
        if k in d:
            _api.warn_deprecated(
                "3.3", message=f"The {k!r} parameter to Colorbar has no "
                "effect because it is overridden by the mappable; it is "
                "deprecated since %(since)s and will be removed %(removal)s.")
        d[k] = v


class Colorbar(ColorbarBase):
    """
    This class connects a `ColorbarBase` to a `~.cm.ScalarMappable`
    such as an `~.image.AxesImage` generated via `~.axes.Axes.imshow`.

    .. note::
        This class is not intended to be instantiated directly; instead, use
        `.Figure.colorbar` or `.pyplot.colorbar` to create a colorbar.
    """

    def __init__(self, ax, mappable, **kwargs):
        # Ensure the given mappable's norm has appropriate vmin and vmax set
        # even if mappable.draw has not yet been called.
        if mappable.get_array() is not None:
            mappable.autoscale_None()

        self.mappable = mappable
        _add_disjoint_kwargs(kwargs, cmap=mappable.cmap, norm=mappable.norm)

        if isinstance(mappable, contour.ContourSet):
            cs = mappable
            _add_disjoint_kwargs(
                kwargs,
                alpha=cs.get_alpha(),
                boundaries=cs._levels,
                values=cs.cvalues,
                extend=cs.extend,
                filled=cs.filled,
            )
            kwargs.setdefault(
                'ticks', ticker.FixedLocator(cs.levels, nbins=10))
            super().__init__(ax, **kwargs)
            if not cs.filled:
                self.add_lines(cs)
        else:
            if getattr(mappable.cmap, 'colorbar_extend', False) is not False:
                kwargs.setdefault('extend', mappable.cmap.colorbar_extend)
            if isinstance(mappable, martist.Artist):
                _add_disjoint_kwargs(kwargs, alpha=mappable.get_alpha())
            super().__init__(ax, **kwargs)

        mappable.colorbar = self
        mappable.colorbar_cid = mappable.callbacksSM.connect(
            'changed', self.update_normal)

    @_api.deprecated("3.3", alternative="update_normal")
    def on_mappable_changed(self, mappable):
        """
        Update this colorbar to match the mappable's properties.

        Typically this is automatically registered as an event handler
        by :func:`colorbar_factory` and should not be called manually.
        """
        _log.debug('colorbar mappable changed')
        self.update_normal(mappable)

    def add_lines(self, CS, erase=True):
        """
        Add the lines from a non-filled `~.contour.ContourSet` to the colorbar.

        Parameters
        ----------
        CS : `~.contour.ContourSet`
            The line positions are taken from the ContourSet levels. The
            ContourSet must not be filled.
        erase : bool, default: True
            Whether to remove any previously added lines.
        """
        if not isinstance(CS, contour.ContourSet) or CS.filled:
            raise ValueError('add_lines is only for a ContourSet of lines')
        tcolors = [c[0] for c in CS.tcolors]
        tlinewidths = [t[0] for t in CS.tlinewidths]
        # Wishlist: Make colorbar lines auto-follow changes in contour lines.
        super().add_lines(CS.levels, tcolors, tlinewidths, erase=erase)

    def update_normal(self, mappable):
        """
        Update solid patches, lines, etc.

        This is meant to be called when the norm of the image or contour plot
        to which this colorbar belongs changes.

        If the norm on the mappable is different than before, this resets the
        locator and formatter for the axis, so if these have been customized,
        they will need to be customized again.  However, if the norm only
        changes values of *vmin*, *vmax* or *cmap* then the old formatter
        and locator will be preserved.
        """
        _log.debug('colorbar update normal %r %r', mappable.norm, self.norm)
        self.mappable = mappable
        self.set_alpha(mappable.get_alpha())
        self.cmap = mappable.cmap
        if mappable.norm != self.norm:
            self.norm = mappable.norm
            self._reset_locator_formatter_scale()

        self.draw_all()
        if isinstance(self.mappable, contour.ContourSet):
            CS = self.mappable
            if not CS.filled:
                self.add_lines(CS)
        self.stale = True

    @_api.deprecated("3.3", alternative="update_normal")
    def update_bruteforce(self, mappable):
        """
        Destroy and rebuild the colorbar.  This is
        intended to become obsolete, and will probably be
        deprecated and then removed.  It is not called when
        the pyplot.colorbar function or the Figure.colorbar
        method are used to create the colorbar.
        """
        # We are using an ugly brute-force method: clearing and
        # redrawing the whole thing.  The problem is that if any
        # properties have been changed by methods other than the
        # colorbar methods, those changes will be lost.
        self.ax.cla()
        self.locator = None
        self.formatter = None

        # clearing the axes will delete outline, patch, solids, and lines:
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.outline = self.ax.spines['outline'] = _ColorbarSpine(self.ax)
        self.patch = mpatches.Polygon(
            np.empty((0, 2)),
            color=mpl.rcParams['axes.facecolor'], linewidth=0.01, zorder=-1)
        self.ax.add_artist(self.patch)
        self.solids = None
        self.lines = []
        self.update_normal(mappable)
        self.draw_all()
        if isinstance(self.mappable, contour.ContourSet):
            CS = self.mappable
            if not CS.filled:
                self.add_lines(CS)
            #if self.lines is not None:
            #    tcolors = [c[0] for c in CS.tcolors]
            #    self.lines.set_color(tcolors)
        #Fixme? Recalculate boundaries, ticks if vmin, vmax have changed.
        #Fixme: Some refactoring may be needed; we should not
        # be recalculating everything if there was a simple alpha
        # change.

    def remove(self):
        """
        Remove this colorbar from the figure.

        If the colorbar was created with ``use_gridspec=True`` the previous
        gridspec is restored.
        """
        super().remove()
        self.mappable.callbacksSM.disconnect(self.mappable.colorbar_cid)
        self.mappable.colorbar = None
        self.mappable.colorbar_cid = None

        try:
            ax = self.mappable.axes
        except AttributeError:
            return

        try:
            gs = ax.get_subplotspec().get_gridspec()
            subplotspec = gs.get_topmost_subplotspec()
        except AttributeError:
            # use_gridspec was False
            pos = ax.get_position(original=True)
            ax._set_position(pos)
        else:
            # use_gridspec was True
            ax.set_subplotspec(subplotspec)


def _normalize_location_orientation(location, orientation):
    if location is None:
        location = _api.check_getitem(
            {None: "right", "vertical": "right", "horizontal": "bottom"},
            orientation=orientation)
    loc_settings = _api.check_getitem({
        "left":   {"location": "left", "orientation": "vertical",
                   "anchor": (1.0, 0.5), "panchor": (0.0, 0.5), "pad": 0.10},
        "right":  {"location": "right", "orientation": "vertical",
                   "anchor": (0.0, 0.5), "panchor": (1.0, 0.5), "pad": 0.05},
        "top":    {"location": "top", "orientation": "horizontal",
                   "anchor": (0.5, 0.0), "panchor": (0.5, 1.0), "pad": 0.05},
        "bottom": {"location": "bottom", "orientation": "horizontal",
                   "anchor": (0.5, 1.0), "panchor": (0.5, 0.0), "pad": 0.15},
    }, location=location)
    if orientation is not None and orientation != loc_settings["orientation"]:
        # Allow the user to pass both if they are consistent.
        raise TypeError("location and orientation are mutually exclusive")
    return loc_settings


@docstring.Substitution(_make_axes_param_doc, _make_axes_other_param_doc)
def make_axes(parents, location=None, orientation=None, fraction=0.15,
              shrink=1.0, aspect=20, **kw):
    """
    Create an `~.axes.Axes` suitable for a colorbar.

    The axes is placed in the figure of the *parents* axes, by resizing and
    repositioning *parents*.

    Parameters
    ----------
    parents : `~.axes.Axes` or list of `~.axes.Axes`
        The Axes to use as parents for placing the colorbar.
    %s

    Returns
    -------
    cax : `~.axes.Axes`
        The child axes.
    kw : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.

    Other Parameters
    ----------------
    %s
    """
    loc_settings = _normalize_location_orientation(location, orientation)
    # put appropriate values into the kw dict for passing back to
    # the Colorbar class
    kw['orientation'] = loc_settings['orientation']
    location = kw['ticklocation'] = loc_settings['location']

    anchor = kw.pop('anchor', loc_settings['anchor'])
    panchor = kw.pop('panchor', loc_settings['panchor'])

    # turn parents into a list if it is not already. We do this w/ np
    # because `plt.subplots` can return an ndarray and is natural to
    # pass to `colorbar`.
    parents = np.atleast_1d(parents).ravel()
    fig = parents[0].get_figure()

    pad0 = 0.05 if fig.get_constrained_layout() else loc_settings['pad']
    pad = kw.pop('pad', pad0)

    if not all(fig is ax.get_figure() for ax in parents):
        raise ValueError('Unable to create a colorbar axes as not all '
                         'parents share the same figure.')

    # take a bounding box around all of the given axes
    parents_bbox = mtransforms.Bbox.union(
        [ax.get_position(original=True).frozen() for ax in parents])

    pb = parents_bbox
    if location in ('left', 'right'):
        if location == 'left':
            pbcb, _, pb1 = pb.splitx(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splitx(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(1.0, shrink).anchored(anchor, pbcb)
    else:
        if location == 'bottom':
            pbcb, _, pb1 = pb.splity(fraction, fraction + pad)
        else:
            pb1, _, pbcb = pb.splity(1 - fraction - pad, 1 - fraction)
        pbcb = pbcb.shrunk(shrink, 1.0).anchored(anchor, pbcb)

        # define the aspect ratio in terms of y's per x rather than x's per y
        aspect = 1.0 / aspect

    # define a transform which takes us from old axes coordinates to
    # new axes coordinates
    shrinking_trans = mtransforms.BboxTransform(parents_bbox, pb1)

    # transform each of the axes in parents using the new transform
    for ax in parents:
        new_posn = shrinking_trans.transform(ax.get_position(original=True))
        new_posn = mtransforms.Bbox(new_posn)
        ax._set_position(new_posn)
        if panchor is not False:
            ax.set_anchor(panchor)

    cax = fig.add_axes(pbcb, label="<colorbar>")
    for a in parents:
        # tell the parent it has a colorbar
        a._colorbars += [cax]
    cax._colorbar_info = dict(
        location=location,
        parents=parents,
        shrink=shrink,
        anchor=anchor,
        panchor=panchor,
        fraction=fraction,
        aspect=aspect,
        pad=pad)
    # and we need to set the aspect ratio by hand...
    cax.set_aspect(aspect, anchor=anchor, adjustable='box')

    return cax, kw


@docstring.Substitution(_make_axes_param_doc, _make_axes_other_param_doc)
def make_axes_gridspec(parent, *, location=None, orientation=None,
                       fraction=0.15, shrink=1.0, aspect=20, **kw):
    """
    Create a `~.SubplotBase` suitable for a colorbar.

    The axes is placed in the figure of the *parent* axes, by resizing and
    repositioning *parent*.

    This function is similar to `.make_axes`. Primary differences are

    - `.make_axes_gridspec` should only be used with a `.SubplotBase` parent.

    - `.make_axes` creates an `~.axes.Axes`; `.make_axes_gridspec` creates a
      `.SubplotBase`.

    - `.make_axes` updates the position of the parent.  `.make_axes_gridspec`
      replaces the ``grid_spec`` attribute of the parent with a new one.

    While this function is meant to be compatible with `.make_axes`,
    there could be some minor differences.

    Parameters
    ----------
    parent : `~.axes.Axes`
        The Axes to use as parent for placing the colorbar.
    %s

    Returns
    -------
    cax : `~.axes.SubplotBase`
        The child axes.
    kw : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.

    Other Parameters
    ----------------
    %s
    """

    loc_settings = _normalize_location_orientation(location, orientation)
    kw['orientation'] = loc_settings['orientation']
    location = kw['ticklocation'] = loc_settings['location']

    anchor = kw.pop('anchor', loc_settings['anchor'])
    panchor = kw.pop('panchor', loc_settings['panchor'])
    pad = kw.pop('pad', loc_settings["pad"])
    wh_space = 2 * pad / (1 - pad)

    if location in ('left', 'right'):
        # for shrinking
        height_ratios = [
                (1-anchor[1])*(1-shrink), shrink, anchor[1]*(1-shrink)]

        if location == 'left':
            gs = parent.get_subplotspec().subgridspec(
                    1, 2, wspace=wh_space,
                    width_ratios=[fraction, 1-fraction-pad])
            ss_main = gs[1]
            ss_cb = gs[0].subgridspec(
                    3, 1, hspace=0, height_ratios=height_ratios)[1]
        else:
            gs = parent.get_subplotspec().subgridspec(
                    1, 2, wspace=wh_space,
                    width_ratios=[1-fraction-pad, fraction])
            ss_main = gs[0]
            ss_cb = gs[1].subgridspec(
                    3, 1, hspace=0, height_ratios=height_ratios)[1]
    else:
        # for shrinking
        width_ratios = [
                anchor[0]*(1-shrink), shrink, (1-anchor[0])*(1-shrink)]

        if location == 'bottom':
            gs = parent.get_subplotspec().subgridspec(
                    2, 1, hspace=wh_space,
                    height_ratios=[1-fraction-pad, fraction])
            ss_main = gs[0]
            ss_cb = gs[1].subgridspec(
                    1, 3, wspace=0, width_ratios=width_ratios)[1]
            aspect = 1 / aspect
        else:
            gs = parent.get_subplotspec().subgridspec(
                    2, 1, hspace=wh_space,
                    height_ratios=[fraction, 1-fraction-pad])
            ss_main = gs[1]
            ss_cb = gs[0].subgridspec(
                    1, 3, wspace=0, width_ratios=width_ratios)[1]
            aspect = 1 / aspect

    parent.set_subplotspec(ss_main)
    parent.set_anchor(panchor)

    fig = parent.get_figure()
    cax = fig.add_subplot(ss_cb, label="<colorbar>")
    cax.set_aspect(aspect, anchor=loc_settings["anchor"], adjustable='box')
    return cax, kw


@_api.deprecated("3.4", alternative="Colorbar")
class ColorbarPatch(Colorbar):
    pass


@_api.deprecated("3.4", alternative="Colorbar")
def colorbar_factory(cax, mappable, **kwargs):
    """
    Create a colorbar on the given axes for the given mappable.

    .. note::
        This is a low-level function to turn an existing axes into a colorbar
        axes.  Typically, you'll want to use `~.Figure.colorbar` instead, which
        automatically handles creation and placement of a suitable axes as
        well.

    Parameters
    ----------
    cax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` to turn into a colorbar.
    mappable : `~matplotlib.cm.ScalarMappable`
        The mappable to be described by the colorbar.
    **kwargs
        Keyword arguments are passed to the respective colorbar class.

    Returns
    -------
    `.Colorbar`
        The created colorbar instance.
    """
    return Colorbar(cax, mappable, **kwargs)
