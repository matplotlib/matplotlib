"""
Colorbars for multivariate colormaps.

.. admonition:: Provisional status of colorbar_multivar

    The ``colorbar_multivar`` module and its contents are considered provisional
    and may change at any time without a deprecation period.

A multivariate colorbar (`MultivarColorbar`) shows a `.MultivarColormap` as one
regular `.Colorbar` per variable. It is typically created with
`.Figure.colorbar_multivar`.

The space for the colorbars is taken from the parent Axes by
`.colorbar.make_axes`, as for a single colorbar, and then split into one Axes
per variable.
"""

from collections.abc import Sequence

import numpy as np

from matplotlib import _docstring
from matplotlib import colorbar as mcolorbar
from matplotlib import colorizer as mcolorizer


_docstring.interpd.register(
    _make_multivar_axes_kw_doc="""
location : None or {'left', 'right', 'top', 'bottom'}
    The location, relative to the parent Axes, where the colorbar Axes
    are created.  It also determines the *orientation* of the colorbars
    (colorbars on the left and right are vertical, colorbars at the top
    and bottom are horizontal).  If None, the location will come from the
    *orientation* if it is set (vertical colorbars on the right, horizontal
    ones at the bottom), or default to 'right' if *orientation* is unset.

orientation : None or {'vertical', 'horizontal'}
    The orientation of the colorbars.  It is preferable to set the *location*
    of the colorbars, as that also determines the *orientation*; passing
    incompatible values for *location* and *orientation* raises an exception.

fraction : float, default: 0.15
    Fraction of original Axes to use for each row or column of colorbars.

shrink : float, default: 1.0
    Fraction by which to multiply the size of the colorbars.

aspect : float, default: 20
    Ratio of long to short dimensions of the block of colorbars along the
    long axis.

pad : float, default: 0.05 if vertical, 0.15 if horizontal
    Fraction of original Axes between colorbars and new image Axes.

anchor : (float, float), optional
    The anchor point of the colorbar Axes.
    Defaults to (0.0, 0.5) if vertical; (0.5, 1.0) if horizontal.

panchor : (float, float), or *False*, optional
    The anchor point of the colorbars parent Axes. If *False*, the parent
    axes' anchor will be unchanged.
    Defaults to (1.0, 0.5) if vertical; (0.5, 0.0) if horizontal.

major_pad : float, default: 0.2
    Fraction of the long axis used for spacing between the colorbars.

minor_pad : float, default: 0.6
    Fraction of the short axis used for spacing between the colorbars.
""")


class MultivarColorbar(Sequence):
    r"""
    Draw a multivariate colorbar in existing Axes.

    Typically, multivariate colorbars are created using `.Figure.colorbar_multivar`
    and associated with `.ColorizingArtist`\s (such as an
    `.AxesImage` generated via `~.axes.Axes.imshow`).

    A MultivarColorbar is a sequence of regular `.Colorbar` objects, one per
    variable, which can be accessed by index.
    """

    def __init__(self, axes, mappable, **kwargs):
        """
        Parameters
        ----------
        axes : list of `~matplotlib.axes.Axes`
            The `~.axes.Axes` instances in which the colorbars are drawn, one
            per variable of the colormap.

        mappable : `.ColorizingArtist` or `.Colorizer`
            The mappable whose colormap and norm will be used.

        **kwargs
            Passed on to each `.Colorbar`.
        """
        if isinstance(mappable, mcolorizer.Colorizer):
            mappable = mcolorizer.ColorizingArtist(mappable)

        self._mappable = mappable
        self._axes = list(axes)
        cmap = self._colorizer.cmap
        norm = self._colorizer.norm
        self._colorbars = [mcolorbar.Colorbar(self._axes[i],
                                              norm=norm.norms[i],
                                              cmap=cmap[i],
                                              **kwargs,
                                              ) for i in range(cmap.n_variates)]

        mappable.colorbar = self
        mappable.colorbar_cid = mappable.callbacks.connect(
            'changed', self.update_normal)

    @property
    def axes(self):
        """The `~.axes.Axes` in which the colorbars are drawn."""
        return self._axes

    @property
    def _colorizer(self):
        # Not stored, so it follows if the colorizer of the mappable is replaced.
        return self._mappable.colorizer

    def _set_colorbar_info(self, colorbar_info):
        """Store the layout information created by `make_axes`."""
        self._colorbar_info = colorbar_info

    def update_normal(self, mappable=None):
        """Redraw the colorbars after the norm or colormap of the mappable changed."""
        # This method is the callback in
        #     mappable.colorbar_cid = mappable.callbacks.connect(
        #         'changed', self.update_normal)
        # which requires a callable that accepts the mappable as an argument.
        for colorbar in self._colorbars:
            colorbar.update_normal()

    def remove(self):
        """
        Remove the colorbars from the figure.

        If the colorbars were created by `.Figure.colorbar_multivar`, the
        position of the parent Axes is restored.
        """
        if hasattr(self, '_colorbar_info'):
            for parent in self._colorbar_info['parents']:
                if self in parent._colorbars:
                    parent._colorbars.remove(self)

        for ax in self._axes:
            ax.remove()

        self._mappable.callbacks.disconnect(self._mappable.colorbar_cid)
        self._mappable.colorbar = None
        self._mappable.colorbar_cid = None

        ax = getattr(self._mappable, 'axes', None)
        if ax is None:
            return
        try:
            subplotspec = ax.get_subplotspec().get_gridspec()._subplot_spec
        except AttributeError:  # The parent was not placed with a gridspec.
            pos = ax.get_position(original=True)
            ax._set_position(pos)
        else:
            ax.set_subplotspec(subplotspec)

    def __getitem__(self, index):
        return self._colorbars[index]

    def __len__(self):
        return len(self._colorbars)


def _get_grid_shape(n_variates, n_major):
    """
    Return the number of colorbars along the long axis (*n_major*) and the
    number of rows or columns of colorbars (*n_minor*).
    """
    if n_major == -1:
        return n_variates, 1
    if n_major == 0:
        raise ValueError("n_major cannot be zero")
    n_minor = -(-n_variates // n_major)  # ceil division
    return n_major, n_minor


def _subdivide_bbox(bbox, n_major, n_minor, orientation,
                    major_pad=0.1, minor_pad=0.6):
    """
    Split *bbox* into a grid of *n_major* x *n_minor* colorbar bboxes.

    The bboxes are ordered starting at the top left, along the long axis
    first.
    """
    major_width = (1 - major_pad) / n_major
    major_space = major_pad / (n_major - 1) if n_major > 1 else 0
    major_split = np.empty(2 * (n_major - 1))
    major_split[0::2] = major_width
    major_split[1::2] = major_space
    major_split = np.cumsum(major_split)

    minor_width = (1 - minor_pad) / n_minor
    minor_space = minor_pad / (n_minor - 1) if n_minor > 1 else 0
    minor_split = np.empty(2 * (n_minor - 1))
    minor_split[0::2] = minor_width
    minor_split[1::2] = minor_space
    minor_split = np.cumsum(minor_split)

    sub_bboxes = []
    vertical = orientation == "vertical"
    bboxes = (bbox.splitx(*minor_split) if vertical
              else bbox.splity(*minor_split)[::-1])
    for bbox_i in bboxes[::2]:  # every other bbox is a space
        bboxes_j = (bbox_i.splity(*major_split)[::-1] if vertical
                    else bbox_i.splitx(*major_split))
        sub_bboxes.extend(bboxes_j[::2])
    return sub_bboxes


@_docstring.interpd
def make_axes(parents, n_variates, n_major=-1, location=None,
              orientation=None, fraction=0.15, shrink=1.0, aspect=20,
              major_pad=0.2, minor_pad=0.6, **kwargs):
    """
    Create `~.axes.Axes` suitable for a multivariate colorbar.

    The Axes are placed in the figure of the *parents* Axes, by resizing and
    repositioning *parents*. Space is taken as for a single colorbar by
    `.colorbar.make_axes`, and then split into one Axes per variable.

    Parameters
    ----------
    parents : `~matplotlib.axes.Axes` or iterable or `numpy.ndarray` of `~.axes.Axes`
        The Axes to use as parents for placing the colorbars.

    n_variates : int
        The number of colorbars to make.

    n_major : int, default: -1
        Number of colorbars along the long axis of the colorbars.
        If -1, all colorbars are placed sequentially along the long axis.
    %(_make_multivar_axes_kw_doc)s

    Returns
    -------
    caxes : list of `~matplotlib.axes.Axes`
        The child Axes.
    kwargs : dict
        The reduced keyword dictionary to be passed when creating the colorbar
        instance.
    colorbar_info : dict
        Information about the layout of the colorbars.
    """
    n_major, n_minor = _get_grid_shape(n_variates, n_major)
    loc_settings = mcolorbar._normalize_location_orientation(location, orientation)
    location = loc_settings['location']
    orientation = loc_settings['orientation']
    if n_minor > 1:
        # With several rows or columns, keep the block next to the parents.
        kwargs.setdefault('anchor', {'left': (1, 0.5), 'right': (0, 0.5),
                                     'top': (0.5, 0), 'bottom': (0.5, 1)}[location])
    aspect = aspect / n_major

    # Take the space for the whole block as for a single colorbar ...
    block, kwargs = mcolorbar.make_axes(
        parents, location=location, fraction=fraction * n_minor, shrink=shrink,
        aspect=aspect, **kwargs)
    colorbar_info = block._colorbar_info
    for parent in colorbar_info['parents']:
        parent._colorbars.remove(block)
    fig = block.get_figure(root=False)
    block_bbox = block.get_position(original=True)
    block.remove()

    # ... and split it into one Axes per variable.
    box_aspect = 1 / aspect if orientation == 'horizontal' else aspect
    caxes = []
    for bbox in _subdivide_bbox(block_bbox, n_major, n_minor, orientation,
                                major_pad=major_pad,
                                minor_pad=minor_pad)[:n_variates]:
        cax = fig.add_axes(bbox, label="<colorbar>")
        cax.set_anchor(colorbar_info['anchor'])
        cax.set_box_aspect(box_aspect)
        cax.set_aspect('auto')
        caxes.append(cax)

    colorbar_info.update(n_major=n_major, n_minor=n_minor, major_pad=major_pad,
                         minor_pad=minor_pad, orientation=orientation)
    return caxes, kwargs, colorbar_info
