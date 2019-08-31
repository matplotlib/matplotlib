"""
Builtin colormaps, colormap handling utilities, and the `ScalarMappable` mixin.

.. seealso::

  :doc:`/gallery/color/colormap_reference` for a list of builtin colormaps.

  :doc:`/tutorials/colors/colormap-manipulation` for examples of how to
  make colormaps.

  :doc:`/tutorials/colors/colormaps` an in-depth discussion of
  choosing colormaps.

  :doc:`/tutorials/colors/colormapnorms` for more details about data
  normalization.
"""

import functools

import numpy as np
from numpy import ma

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed


def _reverser(f, x):  # Deprecated, remove this at the same time as revcmap.
    return f(1 - x)  # Toplevel helper for revcmap ensuring cmap picklability.


@cbook.deprecated("3.2", alternative="Colormap.reversed()")
def revcmap(data):
    """Can only handle specification *data* in dictionary format."""
    data_r = {}
    for key, val in data.items():
        if callable(val):
            # Return a partial object so that the result is picklable.
            valnew = functools.partial(_reverser, val)
        else:
            # Flip x and exchange the y values facing x = 0 and x = 1.
            valnew = [(1.0 - x, y1, y0) for x, y0, y1 in reversed(val)]
        data_r[key] = valnew
    return data_r


LUTSIZE = mpl.rcParams['image.lut']


def _gen_cmap_d():
    """
    Generate a dict mapping standard colormap names to standard colormaps, as
    well as the reversed colormaps.
    """
    cmap_d = {**cmaps_listed}
    for name, spec in datad.items():
        cmap_d[name] = (  # Precache the cmaps at a fixed lutsize..
            colors.LinearSegmentedColormap(name, spec, LUTSIZE)
            if 'red' in spec else
            colors.ListedColormap(spec['listed'], name)
            if 'listed' in spec else
            colors.LinearSegmentedColormap.from_list(name, spec, LUTSIZE))
    # Generate reversed cmaps.
    for cmap in list(cmap_d.values()):
        rmap = cmap.reversed()
        cmap_d[rmap.name] = rmap
    return cmap_d


cmap_d = _gen_cmap_d()
locals().update(cmap_d)


# Continue with definitions ...


def register_cmap(name=None, cmap=None, data=None, lut=None):
    """
    Add a colormap to the set recognized by :func:`get_cmap`.

    It can be used in two ways::

        register_cmap(name='swirly', cmap=swirly_cmap)

        register_cmap(name='choppy', data=choppydata, lut=128)

    In the first case, *cmap* must be a :class:`matplotlib.colors.Colormap`
    instance.  The *name* is optional; if absent, the name will
    be the :attr:`~matplotlib.colors.Colormap.name` attribute of the *cmap*.

    In the second case, the three arguments are passed to
    the :class:`~matplotlib.colors.LinearSegmentedColormap` initializer,
    and the resulting colormap is registered.
    """
    cbook._check_isinstance((str, None), name=name)
    if name is None:
        try:
            name = cmap.name
        except AttributeError:
            raise ValueError("Arguments must include a name or a Colormap")
    if isinstance(cmap, colors.Colormap):
        cmap_d[name] = cmap
        return
    # For the remainder, let exceptions propagate.
    if lut is None:
        lut = mpl.rcParams['image.lut']
    cmap = colors.LinearSegmentedColormap(name, data, lut)
    cmap_d[name] = cmap


def get_cmap(name=None, lut=None):
    """
    Get a colormap instance, defaulting to rc values if *name* is None.

    Colormaps added with :func:`register_cmap` take precedence over
    built-in colormaps.

    Parameters
    ----------
    name : `matplotlib.colors.Colormap` or str or None, default: None
        If a `Colormap` instance, it will be returned.  Otherwise, the name of
        a colormap known to Matplotlib, which will be resampled by *lut*.  The
        default, None, means :rc:`image.cmap`.
    lut : int or None, default: None
        If *name* is not already a Colormap instance and *lut* is not None, the
        colormap will be resampled to have *lut* entries in the lookup table.
    """
    if name is None:
        name = mpl.rcParams['image.cmap']
    if isinstance(name, colors.Colormap):
        return name
    cbook._check_in_list(sorted(cmap_d), name=name)
    if lut is None:
        return cmap_d[name]
    else:
        return cmap_d[name]._resample(lut)


class ScalarMappable:
    """
    This is a mixin class to support scalar data to RGBA mapping.
    The ScalarMappable makes use of data normalization before returning
    RGBA colors from the given colormap.

    """
    def __init__(self, norm=None, cmap=None):
        """

        Parameters
        ----------
        norm : :class:`matplotlib.colors.Normalize` instance
            The normalizing object which scales data, typically into the
            interval ``[0, 1]``.
            If *None*, *norm* defaults to a *colors.Normalize* object which
            initializes its scaling based on the first data processed.
        cmap : str or :class:`~matplotlib.colors.Colormap` instance
            The colormap used to map normalized data values to RGBA colors.
        """

        self.callbacksSM = cbook.CallbackRegistry()

        if cmap is None:
            cmap = get_cmap()
        if norm is None:
            norm = colors.Normalize()

        self._A = None
        #: The Normalization instance of this ScalarMappable.
        self.norm = norm
        #: The Colormap instance of this ScalarMappable.
        self.cmap = get_cmap(cmap)
        #: The last colorbar associated with this ScalarMappable. May be None.
        self.colorbar = None
        self.update_dict = {'array': False}

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """
        Return a normalized rgba array corresponding to *x*.

        In the normal case, *x* is a 1-D or 2-D sequence of scalars, and
        the corresponding ndarray of rgba values will be returned,
        based on the norm and colormap set for this ScalarMappable.

        There is one special case, for handling images that are already
        rgb or rgba, such as might have been read from an image file.
        If *x* is an ndarray with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an rgb or rgba array, and no mapping will be done.
        The array can be uint8, or it can be floating point with
        values in the 0-1 range; otherwise a ValueError will be raised.
        If it is a masked array, the mask will be ignored.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the pre-existing alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.

        In either case, if *bytes* is *False* (default), the rgba
        array will be floats in the 0-1 range; if it is *True*,
        the returned rgba array will be uint8 in the 0 to 255 range.

        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).

        """
        # First check for special case, image input:
        try:
            if x.ndim == 3:
                if x.shape[2] == 3:
                    if alpha is None:
                        alpha = 1
                    if x.dtype == np.uint8:
                        alpha = np.uint8(alpha * 255)
                    m, n = x.shape[:2]
                    xx = np.empty(shape=(m, n, 4), dtype=x.dtype)
                    xx[:, :, :3] = x
                    xx[:, :, 3] = alpha
                elif x.shape[2] == 4:
                    xx = x
                else:
                    raise ValueError("third dimension must be 3 or 4")
                if xx.dtype.kind == 'f':
                    if norm and (xx.max() > 1 or xx.min() < 0):
                        raise ValueError("Floating point image RGB values "
                                         "must be in the 0..1 range.")
                    if bytes:
                        xx = (xx * 255).astype(np.uint8)
                elif xx.dtype == np.uint8:
                    if not bytes:
                        xx = xx.astype(np.float32) / 255
                else:
                    raise ValueError("Image RGB array must be uint8 or "
                                     "floating point; found %s" % xx.dtype)
                return xx
        except AttributeError:
            # e.g., x is not an ndarray; so try mapping it
            pass

        # This is the normal case, mapping a scalar array:
        x = ma.asarray(x)
        if norm:
            x = self.norm(x)
        rgba = self.cmap(x, alpha=alpha, bytes=bytes)
        return rgba

    def set_array(self, A):
        """Set the image array from numpy array *A*.

        Parameters
        ----------
        A : ndarray
        """
        self._A = A
        self.update_dict['array'] = True

    def get_array(self):
        'Return the array'
        return self._A

    def get_cmap(self):
        'return the colormap'
        return self.cmap

    def get_clim(self):
        'return the min, max of the color limits for image scaling'
        return self.norm.vmin, self.norm.vmax

    def set_clim(self, vmin=None, vmax=None):
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             The limits may also be passed as a tuple (*vmin*, *vmax*) as a
             single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
        if vmax is None:
            try:
                vmin, vmax = vmin
            except (TypeError, ValueError):
                pass
        if vmin is not None:
            self.norm.vmin = colors._sanitize_extrema(vmin)
        if vmax is not None:
            self.norm.vmax = colors._sanitize_extrema(vmax)
        self.changed()

    def get_alpha(self):
        """
        Returns
        -------
        alpha : float
            Always returns 1.
        """
        # This method is intended to be overridden by Artist sub-classes
        return 1.

    def set_cmap(self, cmap):
        """
        set the colormap for luminance data

        Parameters
        ----------
        cmap : colormap or registered colormap name
        """
        cmap = get_cmap(cmap)
        self.cmap = cmap
        self.changed()

    def set_norm(self, norm):
        """Set the normalization instance.

        Parameters
        ----------
        norm : `.Normalize`

        Notes
        -----
        If there are any colorbars using the mappable for this norm, setting
        the norm of the mappable will reset the norm, locator, and formatters
        on the colorbar to default.

        """
        cbook._check_isinstance((colors.Normalize, None), norm=norm)
        if norm is None:
            norm = colors.Normalize()
        self.norm = norm
        self.changed()

    def autoscale(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        self.norm.autoscale(self._A)
        self.changed()

    def autoscale_None(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        self.norm.autoscale_None(self._A)
        self.changed()

    def add_checker(self, checker):
        """
        Add an entry to a dictionary of boolean flags
        that are set to True when the mappable is changed.
        """
        self.update_dict[checker] = False

    def check_update(self, checker):
        """
        If mappable has changed since the last check,
        return True; else return False
        """
        if self.update_dict[checker]:
            self.update_dict[checker] = False
            return True
        return False

    def changed(self):
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal
        """
        self.callbacksSM.process('changed', self)

        for key in self.update_dict:
            self.update_dict[key] = True
        self.stale = True
