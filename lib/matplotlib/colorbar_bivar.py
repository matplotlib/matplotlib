"""
Colorbars for bivariate colormaps.

.. admonition:: Provisional status of colorbar_bivar

    The ``colorbar_bivar`` module and its contents are considered provisional
    and may change at any time without a deprecation period.

A bivariate colorbar (`BivarColorbar`) shows a `.BivarColormap` as a 2D image,
with one data variable along each axis. It is typically created with
`.Figure.colorbar_bivar`.

The Axes of these colorbars are created and placed using the same machinery as
regular colorbars (`.colorbar.make_axes` and `.colorbar.make_axes_gridspec`),
so they take part in figure layout in the same way.
"""

import numpy as np

from matplotlib import _api, _docstring
from matplotlib import colorbar as mcolorbar
from matplotlib import colorizer as mcolorizer


_docstring.interpd.register(
    _make_bivar_axes_kw_doc="""
location : None or {'left', 'right', 'top', 'bottom'}
    The location, relative to the parent Axes, where the colorbar Axes
    is created. Also determines the position of the ticks and labels,
    which will favour being away from the parent axes.

fraction : float, default: 0.15
    Fraction of original Axes to use for colorbar.

shrink : float, default: 1.0
    Fraction by which to multiply the size of the colorbar.

aspect : float, default: 1.0
    Ratio of height to width of the colorbar.

pad : float, default: 0.05 if left or right, 0.15 if top or bottom
    Fraction of original Axes between colorbar and new image Axes.

anchor : (float, float), optional
    The anchor point of the colorbar Axes.
    Defaults to (0.0, 0.5) if left or right; (0.5, 1.0) if top or bottom.

panchor : (float, float), or *False*, optional
    The anchor point of the colorbar parent Axes. If *False*, the parent
    axes' anchor will be unchanged.
    Defaults to (1.0, 0.5) if left or right; (0.5, 0.0) if top or bottom.
""",
    _bivar_colorbar_kw_doc="""
ticklocations : tuple describing the ticklocation of the y and x axis
    The first element must be {'auto', 'left', 'right'}
    The second element must be {'auto', 'top', 'bottom''}.
    If 'auto', a reasonable default is chosen based on *location*.
""")


class BivarColorbar:
    r"""
    Draw a bivariate colorbar in an existing Axes.

    Typically, bivariate colorbars are created using `.Figure.colorbar_bivar`
    and associated with `.ColorizingArtist`\s (such as an
    `.AxesImage` generated via `~.axes.Axes.imshow`).

    BivarColorbar uses a regular Axes as underlying Artist.
    Unlike `.Colorbar`, `BivarColorbar` does not support
    customizing the ticks via a high-level API. Instead, customize ticks
    directly on the underlying Axes ``BivarColorbar.ax``.
    """

    _n_rasterize = 256  # rasterize solids if number of colors > _n_rasterize

    def __init__(
        self, ax, mappable,
        *,
        location=None,
        ticklocations=('auto', 'auto'),
        aspect=1.0,
    ):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance in which the colorbar is drawn.

        mappable : `.ColorizingArtist` or `.Colorizer`
            The mappable whose colormap and norm will be used.

        location : None or {'left', 'right', 'top', 'bottom'}
            The location of the bivariate colorbar relative to the Axes.
            If not None, this information is used to determine
            'auto' *ticklocations*.

        Other Parameters
        ----------------
        ticklocations : tuple describing the ticklocation of the y and x axis
            The first element must be {'auto', 'left', 'right'}
            The second element must be {'auto', 'top', 'bottom''}.
            If 'auto', a reasonable default is chosen based on *location*.
        """

        self._ax = ax
        if isinstance(mappable, mcolorizer.Colorizer):
            mappable = mcolorizer.ColorizingArtist(mappable)

        self._mappable = mappable
        self.aspect = aspect

        mappable.colorbar = self
        mappable.colorbar_cid = mappable.callbacks.connect(
            'changed', self.update_normal)

        yticklocation, xticklocation = _resolve_bivar_ticklocations(
            ticklocations, location)
        self.ax.yaxis.set(label_position=yticklocation,
                          ticks_position=yticklocation)
        self.ax.xaxis.set(label_position=xticklocation,
                          ticks_position=xticklocation)

        self._image = None
        self.update_normal()  # also calls _draw_all()
        self.ax._colorbar = self
        self._patch_ax_methods()

    @property
    def ax(self):
        """The `~.axes.Axes` in which the colorbar is drawn."""
        return self._ax

    @property
    def _colorizer(self):
        # Not stored, so it follows if the colorizer of the mappable is replaced.
        return self._mappable.colorizer

    @property
    def aspect(self):
        """The aspect ratio (height / width) of the colorbar."""
        return self.ax.get_box_aspect()

    @aspect.setter
    def aspect(self, aspect):
        aspect = float(aspect)
        self.ax.set_box_aspect(aspect)
        if hasattr(self.ax, "_colorbar_info"):
            colorbar_info = self.ax._colorbar_info
            # Layout managers treat colorbars at the top or bottom as horizontal
            # and invert their aspect, so store it inverted for those locations.
            if colorbar_info["location"] in ('top', 'bottom'):
                colorbar_info["aspect"] = 1 / aspect
            else:
                colorbar_info["aspect"] = aspect

    def _patch_ax_methods(self):
        """Override Axes for colorbar-specific behavior."""
        self.ax.cla = self._cbar_cla
        # interactive methods
        self.ax._get_view = self._get_view
        self.ax._set_view = self._set_view
        self.ax._set_view_from_bbox = self._set_view_from_bbox
        self.ax.drag_pan = self._drag_pan

    def _unpatch_ax_methods(self):
        """Undo override Axes for colorbar-specific behavior."""
        # Note: del works because we added these methods as
        # instance attributes, hiding the class-level methods.
        # Deleting the attributes uncovers the methods.
        del self.ax.cla
        # interactive methods
        del self.ax._get_view
        del self.ax._set_view
        del self.ax._set_view_from_bbox
        del self.ax.drag_pan

    def _draw_all(self):
        """
        Calculate any free parameters based on the current cmap and norm,
        and do all the drawing.
        """

        # transform from 0-1 to vmin-vmax:
        if self._mappable.get_array() is not None:
            self._mappable.autoscale_None()
        if not self._colorizer.norm.scaled():
            # If we still aren't scaled after autoscaling, use 0, 1 as default
            self._set_view([0, 1, 0, 1])
        n, m = self._colorizer.cmap.N, self._colorizer.cmap.M
        x = self._colorizer.norm.norms[1].inverse(np.linspace(0, 1, m + 1))
        y = self._colorizer.norm.norms[0].inverse(np.linspace(0, 1, n + 1))
        X, Y = np.meshgrid(x, y)

        if self._image is not None:
            self._image.remove()
        self._image = self.ax.pcolormesh(
                X, Y, self._colorizer.cmap.lut,
                rasterized=n * m > self._n_rasterize,
                edgecolors='none', shading='flat')
        # Apply norm scaling (supports LogNorm etc.)
        if getattr(self._colorizer.norm.norms[0], '_scale', None):
            # use the norm's scale (if it exists and is not None):
            self.ax.set_yscale(self._colorizer.norm.norms[0]._scale)
        else:
            # fallback for custom norms, or NoNorm()
            self.ax.set_yscale(
                'function',
                functions=(
                    self._colorizer.norm.norms[0],
                    self._colorizer.norm.norms[0].inverse
                )
            )

        if getattr(self._colorizer.norm.norms[1], '_scale', None):
            # use the norm's scale (if it exists and is not None):
            self.ax.set_xscale(self._colorizer.norm.norms[1]._scale)
        else:
            # fallback for custom norms, or NoNorm()
            self.ax.set_xscale(
                'function',
                functions=(
                    self._colorizer.norm.norms[1],
                    self._colorizer.norm.norms[1].inverse
                )
            )
        # Manually set limits (image is in Axes coordinates)
        extent = [
            self._colorizer.norm.norms[1].vmin,
            self._colorizer.norm.norms[1].vmax,
            self._colorizer.norm.norms[0].vmin,
            self._colorizer.norm.norms[0].vmax,
        ]
        self.ax.set_ylim(extent[2:4])
        self.ax.set_xlim(extent[0:2])

    def set_xlabel(self, label):
        self.ax.set_xlabel(label)

    def set_ylabel(self, label):
        self.ax.set_ylabel(label)

    @property
    def xaxis(self):
        return self.ax.xaxis

    @property
    def yaxis(self):
        return self.ax.yaxis

    def update_normal(self, mappable=None):
        """Redraw the colorbar after the norm or colormap of the mappable changed."""
        # This method is the callback in
        #     mappable.colorbar_cid = mappable.callbacks.connect(
        #         'changed', self.update_normal)
        # which requires a callable that accepts the mappable as an argument.
        # _draw_all() takes no arguments, so it cannot be connected directly.
        self._draw_all()

    def remove(self):
        """
        Remove this colorbar from the figure.

        If the colorbar was created with ``use_gridspec=True`` the previous
        gridspec is restored.
        """
        if hasattr(self.ax, '_colorbar_info'):
            parents = self.ax._colorbar_info['parents']
            for a in parents:
                if self.ax in a._colorbars:
                    a._colorbars.remove(self.ax)
        self.ax.remove()
        self._mappable.callbacks.disconnect(self._mappable.colorbar_cid)
        self._mappable.colorbar = None
        self._mappable.colorbar_cid = None

        ax = self._mappable.axes
        if ax is None:
            return
        try:
            subplotspec = self.ax.get_subplotspec().get_gridspec()._subplot_spec
        except AttributeError:  # use_gridspec was False
            pos = ax.get_position(original=True)
            ax._set_position(pos)
        else:  # use_gridspec was True
            ax.set_subplotspec(subplotspec)

    def _get_view(self):
        ynorm, xnorm = self._colorizer.norm.norms
        return ynorm.vmin, ynorm.vmax, xnorm.vmin, xnorm.vmax

    def _set_view(self, view):
        ynorm, xnorm = self._colorizer.norm.norms
        if (view[0] != ynorm.vmin
           or view[1] != ynorm.vmax
           or view[2] != xnorm.vmin
           or view[3] != xnorm.vmax):
            with self._colorizer.norm.callbacks.blocked():
                ynorm.vmin, ynorm.vmax, xnorm.vmin, xnorm.vmax = view
            self._colorizer.norm._changed()

    def _set_view_from_bbox(self, bbox, direction='in',
                            mode=None, twinx=False, twiny=False):
        new_xbound, new_ybound = self.ax._prepare_view_from_bbox(
            bbox, direction=direction, mode=mode, twinx=twinx, twiny=twiny)
        ynorm, xnorm = self._colorizer.norm.norms
        ynorm.vmin, ynorm.vmax = new_ybound
        xnorm.vmin, xnorm.vmax = new_xbound

    def _drag_pan(self, button, key, x, y):
        points = self.ax._get_pan_points(button, key, x, y)
        if points is not None:
            ynorm, xnorm = self._colorizer.norm.norms
            xnorm.vmin, xnorm.vmax = points[:, 0]
            ynorm.vmin, ynorm.vmax = points[:, 1]

    def _cbar_cla(self):
        """Function to clear the interactive colorbar state."""
        self._unpatch_ax_methods()
        # We now restore the old cla() back and can call it directly
        self.ax.cla()


def _resolve_bivar_ticklocations(ticklocations, location):
    """
    Return the (y, x) tick locations of a bivariate colorbar.

    'auto' entries in *ticklocations* are replaced by a default based on
    *location*.
    """
    if len(ticklocations) != 2:
        raise ValueError("ticklocations must be a tuple of length 2")
    yticklocation, xticklocation = ticklocations
    _api.check_in_list(['auto', 'left', 'right'], ticklocation=yticklocation)
    _api.check_in_list(['auto', 'top', 'bottom'], ticklocation=xticklocation)
    if yticklocation == 'auto':
        yticklocation = _api.getitem_checked(
            {None: "right", "left": "left", "right": "right",
             "top": "left", "bottom": "left"}, location=location)
    if xticklocation == 'auto':
        xticklocation = _api.getitem_checked(
            {None: "bottom", "left": "bottom", "right": "bottom",
             "top": "top", "bottom": "bottom"}, location=location)
    return yticklocation, xticklocation


def _to_bivar_kwargs(kwargs, aspect):
    """
    Convert the keyword arguments returned by `.colorbar.make_axes` for use
    with `BivarColorbar`.

    A bivariate colorbar has no orientation and its tick locations are
    derived from its *location*.
    """
    kwargs.pop('orientation')
    kwargs['location'] = kwargs.pop('ticklocation')
    kwargs['aspect'] = aspect
    return kwargs


@_docstring.interpd
def make_axes(parents, location=None, fraction=0.15,
              shrink=1.0, aspect=1.0, **kwargs):
    """
    Create an `~.axes.Axes` suitable for a bivariate colorbar.

    The Axes is placed in the figure of the *parents* Axes, by resizing and
    repositioning *parents*. See `.colorbar.make_axes`.

    Parameters
    ----------
    parents : `~matplotlib.axes.Axes` or iterable or `numpy.ndarray` of `~.axes.Axes`
        The Axes to use as parents for placing the colorbar.
    %(_make_bivar_axes_kw_doc)s

    Returns
    -------
    cax : `~matplotlib.axes.Axes`
        The child Axes.
    kwargs : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.
    """
    cax, kwargs = mcolorbar.make_axes(parents, location=location, fraction=fraction,
                                      shrink=shrink, aspect=aspect, **kwargs)
    return cax, _to_bivar_kwargs(kwargs, aspect)


@_docstring.interpd
def make_axes_gridspec(parent, *, location=None,
                       fraction=0.15, shrink=1.0, aspect=1.0, **kwargs):
    """
    Create an `~.axes.Axes` suitable for a bivariate colorbar.

    The Axes is placed in the figure of the *parent* Axes, by resizing and
    repositioning *parent*. See `.colorbar.make_axes_gridspec`.

    Parameters
    ----------
    parent : `~matplotlib.axes.Axes`
        The Axes to use as parent for placing the colorbar.
    %(_make_bivar_axes_kw_doc)s

    Returns
    -------
    cax : `~matplotlib.axes.Axes`
        The child Axes.
    kwargs : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.
    """
    cax, kwargs = mcolorbar.make_axes_gridspec(
        parent, location=location, fraction=fraction, shrink=shrink,
        aspect=aspect, **kwargs)
    return cax, _to_bivar_kwargs(kwargs, aspect)
