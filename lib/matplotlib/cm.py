"""
Builtin colormaps, colormap handling utilities, and the `VectorMappable` and
`ScalarMappable` mixin.

.. seealso::

  :doc:`/gallery/color/colormap_reference` for a list of builtin colormaps.

  :ref:`colormap-manipulation` for examples of how to make
  colormaps.

  :ref:`colormaps` an in-depth discussion of choosing
  colormaps.

  :ref:`colormapnorms` for more details about data normalization.
"""

from collections.abc import Mapping
import functools

import numpy as np
from numpy import ma

import matplotlib as mpl
from matplotlib import _api, colors, cbook, scale
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed


_LUTSIZE = mpl.rcParams['image.lut']


def _gen_cmap_registry():
    """
    Generate a dict mapping standard colormap names to standard colormaps, as
    well as the reversed colormaps.
    """
    cmap_d = {**cmaps_listed}
    for name, spec in datad.items():
        cmap_d[name] = (  # Precache the cmaps at a fixed lutsize..
            colors.LinearSegmentedColormap(name, spec, _LUTSIZE)
            if 'red' in spec else
            colors.ListedColormap(spec['listed'], name)
            if 'listed' in spec else
            colors.LinearSegmentedColormap.from_list(name, spec, _LUTSIZE))

    # Register colormap aliases for gray and grey.
    aliases = {
        # alias -> original name
        'grey': 'gray',
        'gist_grey': 'gist_gray',
        'gist_yerg': 'gist_yarg',
        'Grays': 'Greys',
    }
    for alias, original_name in aliases.items():
        cmap = cmap_d[original_name].copy()
        cmap.name = alias
        cmap_d[alias] = cmap

    # Generate reversed cmaps.
    for cmap in list(cmap_d.values()):
        rmap = cmap.reversed()
        cmap_d[rmap.name] = rmap
    return cmap_d


class ColormapRegistry(Mapping):
    r"""
    Container for colormaps that are known to Matplotlib by name.

    The universal registry instance is `matplotlib.colormaps`. There should be
    no need for users to instantiate `.ColormapRegistry` themselves.

    Read access uses a dict-like interface mapping names to `.Colormap`\s::

        import matplotlib as mpl
        cmap = mpl.colormaps['viridis']

    Returned `.Colormap`\s are copies, so that their modification does not
    change the global definition of the colormap.

    Additional colormaps can be added via `.ColormapRegistry.register`::

        mpl.colormaps.register(my_colormap)

    To get a list of all registered colormaps, you can do::

        from matplotlib import colormaps
        list(colormaps)
    """
    def __init__(self, cmaps):
        self._cmaps = cmaps
        self._builtin_cmaps = tuple(cmaps)

    def __getitem__(self, item):
        try:
            return self._cmaps[item].copy()
        except KeyError:
            raise KeyError(f"{item!r} is not a known colormap name") from None

    def __iter__(self):
        return iter(self._cmaps)

    def __len__(self):
        return len(self._cmaps)

    def __str__(self):
        return ('ColormapRegistry; available colormaps:\n' +
                ', '.join(f"'{name}'" for name in self))

    def __call__(self):
        """
        Return a list of the registered colormap names.

        This exists only for backward-compatibility in `.pyplot` which had a
        ``plt.colormaps()`` method. The recommended way to get this list is
        now ``list(colormaps)``.
        """
        return list(self)

    def register(self, cmap, *, name=None, force=False):
        """
        Register a new colormap.

        The colormap name can then be used as a string argument to any ``cmap``
        parameter in Matplotlib. It is also available in ``pyplot.get_cmap``.

        The colormap registry stores a copy of the given colormap, so that
        future changes to the original colormap instance do not affect the
        registered colormap. Think of this as the registry taking a snapshot
        of the colormap at registration.

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap
            The colormap to register.

        name : str, optional
            The name for the colormap. If not given, ``cmap.name`` is used.

        force : bool, default: False
            If False, a ValueError is raised if trying to overwrite an already
            registered name. True supports overwriting registered colormaps
            other than the builtin colormaps.
        """
        _api.check_isinstance(colors.Colormap, cmap=cmap)

        name = name or cmap.name
        if name in self:
            if not force:
                # don't allow registering an already existing cmap
                # unless explicitly asked to
                raise ValueError(
                    f'A colormap named "{name}" is already registered.')
            elif name in self._builtin_cmaps:
                # We don't allow overriding a builtin.
                raise ValueError("Re-registering the builtin cmap "
                                 f"{name!r} is not allowed.")

            # Warn that we are updating an already existing colormap
            _api.warn_external(f"Overwriting the cmap {name!r} "
                               "that was already in the registry.")

        self._cmaps[name] = cmap.copy()
        # Someone may set the extremes of a builtin colormap and want to register it
        # with a different name for future lookups. The object would still have the
        # builtin name, so we should update it to the registered name
        if self._cmaps[name].name != name:
            self._cmaps[name].name = name

    def unregister(self, name):
        """
        Remove a colormap from the registry.

        You cannot remove built-in colormaps.

        If the named colormap is not registered, returns with no error, raises
        if you try to de-register a default colormap.

        .. warning::

            Colormap names are currently a shared namespace that may be used
            by multiple packages. Use `unregister` only if you know you
            have registered that name before. In particular, do not
            unregister just in case to clean the name before registering a
            new colormap.

        Parameters
        ----------
        name : str
            The name of the colormap to be removed.

        Raises
        ------
        ValueError
            If you try to remove a default built-in colormap.
        """
        if name in self._builtin_cmaps:
            raise ValueError(f"cannot unregister {name!r} which is a builtin "
                             "colormap.")
        self._cmaps.pop(name, None)

    def get_cmap(self, cmap):
        """
        Return a color map specified through *cmap*.

        Parameters
        ----------
        cmap : str or `~matplotlib.colors.Colormap` or None

            - if a `.Colormap`, return it
            - if a string, look it up in ``mpl.colormaps``
            - if None, return the Colormap defined in :rc:`image.cmap`

        Returns
        -------
        Colormap
        """
        # get the default color map
        if cmap is None:
            return self[mpl.rcParams["image.cmap"]]

        # if the user passed in a Colormap, simply return it
        if isinstance(cmap, colors.Colormap):
            return cmap
        if isinstance(cmap, str):
            _api.check_in_list(sorted(_colormaps), cmap=cmap)
            # otherwise, it must be a string so look it up
            return self[cmap]
        raise TypeError(
            'get_cmap expects None or an instance of a str or Colormap . ' +
            f'you passed {cmap!r} of type {type(cmap)}'
        )


# public access to the colormaps should be via `matplotlib.colormaps`. For now,
# we still create the registry here, but that should stay an implementation
# detail.
_colormaps = ColormapRegistry(_gen_cmap_registry())
globals().update(_colormaps)

_multivar_colormaps = ColormapRegistry({})
globals().update(_multivar_colormaps)

_bivar_colormaps = ColormapRegistry({})
globals().update(_bivar_colormaps)


# This is an exact copy of pyplot.get_cmap(). It was removed in 3.9, but apparently
# caused more user trouble than expected. Re-added for 3.9.1 and extended the
# deprecation period for two additional minor releases.
@_api.deprecated(
    '3.7',
    removal='3.11',
    alternative="``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()``"
                " or ``pyplot.get_cmap()``"
    )
def get_cmap(name=None, lut=None):
    """
    Get a colormap instance, defaulting to rc values if *name* is None.

    Parameters
    ----------
    name : `~matplotlib.colors.Colormap` or str or None, default: None
        If a `.Colormap` instance, it will be returned. Otherwise, the name of
        a colormap known to Matplotlib, which will be resampled by *lut*. The
        default, None, means :rc:`image.cmap`.
    lut : int or None, default: None
        If *name* is not already a Colormap instance and *lut* is not None, the
        colormap will be resampled to have *lut* entries in the lookup table.

    Returns
    -------
    Colormap
    """
    if name is None:
        name = mpl.rcParams['image.cmap']
    if isinstance(name, colors.Colormap):
        return name
    _api.check_in_list(sorted(_colormaps), name=name)
    if lut is None:
        return _colormaps[name]
    else:
        return _colormaps[name].resampled(lut)


def _auto_norm_from_scale(scale_cls):
    """
    Automatically generate a norm class from *scale_cls*.

    This differs from `.colors.make_norm_from_scale` in the following points:

    - This function is not a class decorator, but directly returns a norm class
      (as if decorating `.Normalize`).
    - The scale is automatically constructed with ``nonpositive="mask"``, if it
      supports such a parameter, to work around the difference in defaults
      between standard scales (which use "clip") and norms (which use "mask").

    Note that ``make_norm_from_scale`` caches the generated norm classes
    (not the instances) and reuses them for later calls.  For example,
    ``type(_auto_norm_from_scale("log")) == LogNorm``.
    """
    # Actually try to construct an instance, to verify whether
    # ``nonpositive="mask"`` is supported.
    try:
        norm = colors.make_norm_from_scale(
            functools.partial(scale_cls, nonpositive="mask"))(
            colors.Normalize)()
    except TypeError:
        norm = colors.make_norm_from_scale(scale_cls)(
            colors.Normalize)()
    return type(norm)


class VectorMappable:
    """
    A mixin class to map one or multiple sets of scalar data to RGBA.

    The VectorMappable applies data normalization before returning RGBA colors
    from the given `~matplotlib.colors.Colormap`, `~matplotlib.colors.BivarColormap`,
    or `~matplotlib.colors.MultivarColormap`.
    """

    def __init__(self, norm=None, cmap=None):
        """
        Parameters
        ----------
        norm : `.Normalize` (or subclass thereof) or str or None
            The normalizing object which scales data, typically into the
            interval ``[0, 1]``.
            If a `str`, a `.Normalize` subclass is dynamically generated based
            on the scale with the corresponding name.
            If *None*, *norm* defaults to a *colors.Normalize* object which
            initializes its scaling based on the first data processed.
        cmap : str or `~matplotlib.colors.Colormap`
            The colormap used to map normalized data values to RGBA colors.
        """
        self._A = None
        self.cmap = None
        self.set_cmap(cmap)

        self._id_norm = [None] * self.cmap.n_variates
        self._norm = [None] * self.cmap.n_variates
        self.set_norm(norm)  # The Normalize instance of this VectorMappable.
        #: The last colorbar associated with this VectorMappable. May be None.
        self.colorbar = None
        self.callbacks = cbook.CallbackRegistry(signals=["changed"])

    def _scale_norm(self, norm, vmin, vmax):
        """
        Helper for initial scaling.

        Used by public functions that create a VectorMappable and support
        parameters *vmin*, *vmax* and *norm*. This makes sure that a *norm*
        will take precedence over *vmin*, *vmax*.

        Note that this method does not set the norm.
        """
        norm = _ensure_multivariate_norm(self.cmap.n_variates, norm)
        vmin, vmax = _ensure_multivariate_clim(self.cmap.n_variates, vmin, vmax)
        for i, _ in enumerate(norm):
            if vmin[i] is not None or vmax[i] is not None:
                if isinstance(norm[i], colors.Normalize):
                    raise ValueError(
                        "Passing a Normalize instance simultaneously with "
                        "vmin/vmax is not supported.  Please pass vmin/vmax "
                        "directly to the norm when creating it.")
                self._set_clim(i, vmin[i], vmax[i])

        # always resolve the autoscaling so we have concrete limits
        # rather than deferring to draw time.
        self.autoscale_None()

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """
        Return a normalized RGBA array corresponding to *x*.

        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding `~numpy.ndarray` of RGBA values will be returned,
        based on the norm and colormap set for this VectorMappable.

        There is one special case, for handling images that are already
        RGB or RGBA, such as might have been read from an image file.
        If *x* is an `~numpy.ndarray` with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an RGB or RGBA array, and no mapping will be done.
        The array can be `~numpy.uint8`, or it can be floats with
        values in the 0-1 range; otherwise a ValueError will be raised.
        Any NaNs or masked elements will be set to 0 alpha.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the preexisting alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.

        In either case, if *bytes* is *False* (default), the RGBA
        array will be floats in the 0-1 range; if it is *True*,
        the returned RGBA array will be `~numpy.uint8` in the 0 to 255 range.

        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).

        """
        # First check for special case, image input:
        if self.cmap.n_variates == 1:
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
                        raise ValueError("Third dimension must be 3 or 4")
                    if xx.dtype.kind == 'f':
                        # If any of R, G, B, or A is nan, set to 0
                        if np.any(nans := np.isnan(x)):
                            if x.shape[2] == 4:
                                xx = xx.copy()
                            xx[np.any(nans, axis=2), :] = 0

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
                    # Account for any masked entries in the original array
                    # If any of R, G, B, or A are masked for an entry, we set alpha to 0
                    if np.ma.is_masked(x):
                        xx[np.any(np.ma.getmaskarray(x), axis=2), 3] = 0
                    return xx
            except AttributeError:
                # e.g., x is not an ndarray; so try mapping it
                pass

            # This is the normal case, mapping a scalar array:
            x = ma.asarray(x)
            if norm:
                x = self._norm[0](x)
            rgba = self.cmap(x, alpha=alpha, bytes=bytes)
        else:  # multivariate
            x = _ensure_multivariate_data(self.cmap.n_variates, x)
            x = _iterable_variates_in_data(x)
            if isinstance(self.cmap, colors.BivarColormap):
                if norm:
                    normed_0 = self._norm[0](x[0])
                    normed_1 = self._norm[1](x[1])
                else:
                    normed_0 = x[0].copy()
                    normed_1 = x[1].copy()
                rgba = self.cmap((normed_0, normed_1), alpha=alpha, bytes=bytes)
            else:  # i.e. isinstance(self._cmaps, colors.MultivarColormap)
                if norm:
                    x = [norm(xx) for norm, xx in zip(self._norm, x)]
                rgba = self.cmap(x, alpha=alpha, bytes=bytes)
        return rgba

    def set_array(self, A):
        """
        Set the value array from array-like *A*.

        Parameters
        ----------
        A : array-like or None
            The values that are mapped to colors.

            The base class `.VectorMappable` does not make any assumptions on
            the dimensionality and shape of the value array *A*.
        """
        if A is None:
            self._A = None
            return
        A = _ensure_multivariate_data(self.cmap.n_variates, A)

        A = cbook.safe_masked_invalid(A, copy=True)
        if not np.can_cast(A.dtype, float, "same_kind"):
            if A.dtype.fields is None:
                raise TypeError(f"Image data of dtype {A.dtype} cannot be "
                                f"converted to float")
            else:
                for key in A.dtype.fields:
                    if not np.can_cast(A[key].dtype, float, "same_kind"):
                        raise TypeError(f"Image data of dtype {A.dtype} cannot be "
                                        f"converted to a sequence of float")
        self._A = A
        for AA, norm in zip(_iterable_variates_in_data(A), self._norm):
            if not norm.scaled():
                norm.autoscale_None(AA)

    def get_array(self):
        """
        Return the array of values, that are mapped to colors.

        The base class `.VectorMappable` does not make any assumptions on
        the dimensionality and shape of the array.
        """
        return self._A

    def get_cmap(self):
        """Return the `.Colormap` instance."""
        return self.cmap

    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
        if self.cmap.n_variates == 1:
            return self._norm[0].vmin, self._norm[0].vmax
        return [n.vmin for n in self._norm], [n.vmax for n in self._norm]

    def _set_clim(self, i, vmin, vmax):
        if vmax is None:
            try:
                vmin, vmax = vmin
            except (TypeError, ValueError):
                pass
        if vmin is not None:
            self._norm[i].vmin = colors._sanitize_extrema(vmin)
        if vmax is not None:
            self._norm[i].vmax = colors._sanitize_extrema(vmax)

    def set_clim(self, vmin=None, vmax=None):
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             For scalar data, the limits may also be passed as a
             tuple (*vmin*, *vmax*) as a single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        if self.cmap.n_variates == 1:
            try:
                vmin, vmax = vmin
            except (TypeError, ValueError):
                pass
        vmin, vmax = _ensure_multivariate_clim(self.cmap.n_variates, vmin, vmax)
        for i, _ in enumerate(self._norm):
            self._set_clim(i, vmin[i], vmax[i])

    def get_alpha(self):
        """
        Returns
        -------
        float
            Always returns 1.
        """
        # This method is intended to be overridden by Artist sub-classes
        return 1.

    def set_cmap(self, cmap):
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        in_init = self.cmap is None

        self.cmap = _ensure_cmap(cmap)
        if not in_init:
            self.changed()  # Things are not set up properly yet.

    @property
    def norm(self):
        if self.cmap.n_variates == 1:
            return self._norm[0]
        return self._norm

    @norm.setter
    def norm(self, norm):
        self.set_norm(norm)

    def set_norm(self, norm):
        """
        Set the normalization instance.

        Parameters
        ----------
        norm : `.Normalize` or str or None

        Notes
        -----
        If there are any colorbars using the mappable for this norm, setting
        the norm of the mappable will reset the norm, locator, and formatters
        on the colorbar to default.
        """

        norm = _ensure_multivariate_norm(self.cmap.n_variates, norm)

        changed = False
        for i, n in enumerate(norm):
            _api.check_isinstance((colors.Normalize, str, None), norm=n)
            if n is None:
                n = colors.Normalize()
            elif isinstance(n, str):
                try:
                    scale_cls = scale._scale_mapping[n]
                except KeyError:
                    raise ValueError(
                        "Invalid norm str name; the following values are "
                        f"supported: {', '.join(scale._scale_mapping)}"
                    ) from None
                n = _auto_norm_from_scale(scale_cls)()

            if n is self._norm[i]:
                continue

            if self._norm[i] is not None:
                # Remove the current callback and connect to the new one
                self._norm[i].callbacks.disconnect(self._id_norm[i])
                # emit changed if we are changing norm
                # do not emit during initialization (self.norm[i] is None)
                changed = True
            self._norm[i] = n
            self._id_norm[i] = self._norm[i].callbacks.connect('changed',
                                                               self.changed)
        if changed:
            self.changed()

    def autoscale(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        for n, a in zip(self._norm, _iterable_variates_in_data(self._A)):
            n.autoscale(a)

    def autoscale_None(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        if self.cmap.n_variates == 1:
            self._norm[0].autoscale_None(self._A)
        else:
            for n, a in zip(self._norm, _iterable_variates_in_data(self._A)):
                n.autoscale_None(a)

    def changed(self):
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        """
        self.callbacks.process('changed', self)
        self.stale = True

    def _parse_multivariate_data(self, data):
        """
        Parse data to a dtype with self.cmap.n_variates.

        Input data of shape (n_variates, n, m) is converted to an array of shape
        (n, m) with data type np.dtype(f'{data.dtype}, ' * n_variates)

        Complex data is returned as a view with dtype np.dtype('float64, float64')
        or np.dtype('float32, float32')

        If n_variates is 1 and data is not of type np.ndarray (i.e. PIL.Image),
        the data is returned unchanged.

        If data is None, the function returns None

        Parameters
        ----------
        data : np.ndarray, PIL.Image or None

        Returns
        -------
            np.ndarray, PIL.Image or None
        """

        return _ensure_multivariate_data(self.cmap.n_variates, data)


class ScalarMappable(VectorMappable):
    """
    A mixin class to map one or scalar data to RGBA.

    The VectorMappable applies data normalization before returning RGBA colors
    from the given `~matplotlib.colors.Colormap`
    """

    def __init__(self, norm=None, cmap=None):
        super().__init__(norm=norm, cmap=cmap)

    def set_cmap(self, cmap):
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        in_init = self.cmap is None

        self.cmap = _ensure_cmap(cmap, accept_multivariate=False)
        if not in_init:
            self.changed()  # Things are not set up properly yet.


# The docstrings here must be generic enough to apply to all relevant methods.
mpl._docstring.interpd.update(
    cmap_doc="""\
cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
    The Colormap instance or registered colormap name used to map scalar data
    to colors.""",
    norm_doc="""\
norm : str or `~matplotlib.colors.Normalize`, optional
    The normalization method used to scale scalar data to the [0, 1] range
    before mapping to colors using *cmap*. By default, a linear scaling is
    used, mapping the lowest value to 0 and the highest to 1.

    If given, this can be one of the following:

    - An instance of `.Normalize` or one of its subclasses
      (see :ref:`colormapnorms`).
    - A scale name, i.e. one of "linear", "log", "symlog", "logit", etc.  For a
      list of available scales, call `matplotlib.scale.get_scale_names()`.
      In that case, a suitable `.Normalize` subclass is dynamically generated
      and instantiated.""",
    vmin_vmax_doc="""\
vmin, vmax : float, optional
    When using scalar data and no explicit *norm*, *vmin* and *vmax* define
    the data range that the colormap covers. By default, the colormap covers
    the complete value range of the supplied data. It is an error to use
    *vmin*/*vmax* when a *norm* instance is given (but using a `str` *norm*
    name together with *vmin*/*vmax* is acceptable).""",
)


def _ensure_cmap(cmap, accept_multivariate=True):
    """
    For internal use to preserve type stability of errors, and
    to ensure that we have a `~matplotlib.colors.Colormap`,
    `~matplotlib.colors.MultivarColormap` or
    `~matplotlib.colors.BivarColormap` object.
    This is necessary in order to know the number of variates.

    objects, strings in mpl.colormaps, or None.

    Parameters
    ----------
    cmap : None, str, Colormap

        - if a `~matplotlib.colors.Colormap`,
          `~matplotlib.colors.MultivarColormap` or
          `~matplotlib.colors.BivarColormap`,
          return it
        - if a string, look it up in three corresponding databases
          when not found: raise an error based on the expected shape
        - if None, look up the default color map in mpl.colormaps

    accept_multivariate : bool, default True

        - if False, accept only Colormap, string in mpl.colormaps or None

    Returns
    -------
    Colormap, MultivarColormap or BivarColormap
    """
    if not accept_multivariate:
        if isinstance(cmap, colors.Colormap):
            return cmap
        cmap_name = cmap if cmap is not None else mpl.rcParams["image.cmap"]
        # use check_in_list to ensure type stability of the exception raised by
        # the internal usage of this (ValueError vs KeyError)
        if cmap_name not in _colormaps:
            _api.check_in_list(sorted(_colormaps), cmap=cmap_name)

    if isinstance(cmap, colors.ColormapBase):
        return cmap

    cmap_name = cmap if cmap is not None else mpl.rcParams["image.cmap"]
    if cmap_name in mpl.colormaps:
        return mpl.colormaps[cmap_name]
    if cmap_name in mpl.multivar_colormaps:
        return mpl.multivar_colormaps[cmap_name]
    if cmap_name in mpl.bivar_colormaps:
        return mpl.bivar_colormaps[cmap_name]

    # this error message is a variant of _api.check_in_list but gives
    # additional hints as to how to access multivariate colormaps

    msg = f"{cmap!r} is not a valid value for cmap"
    msg += "; supported values for scalar colormaps are "
    msg += f"{', '.join(map(repr, sorted(mpl.colormaps)))}\n"
    msg += "See matplotlib.bivar_colormaps() and"
    msg += " matplotlib.multivar_colormaps() for"
    msg += " bivariate and multivariate colormaps."

    raise ValueError(msg)


def _iterable_variates_in_data(data):
    """
    Provides an interable over the variates contained in the data.

    The returned list has length 1 if 'data' contains scalar data.
    If 'data' has a dtype type with multiple fields, the returned list
    has a length equal to the number of field in the data type.

    Often used with VectorMappable._norm, which similarly has a list
    of norms equal to the number of variates

    Parameters
    ----------
    data : np.ndarray


    Returns
    -------
        list of np.ndarray

    """
    if data.dtype.fields is None:
        return [data]
    else:
        return [data[descriptor[0]] for descriptor in data.dtype.descr]


def _ensure_multivariate_norm(n_variates, norm):
    """
    Ensure that the nor
    m has the correct number of elements.
    If n_variates > 1: A single argument for norm will be repeated n
    times in the output.

    Parameters
    ----------
    n_variates : int
        -  number of variates in the data
    norm : `.Normalize` (or subclass thereof) or str or None or iterable

        - If iterable, the length must be equal to n_variates

    Returns
    -------
        if n_variates == 1:
            norm returned unchanged
        if n_variates > 1:
            an iterable of length n_variates
    """
    if isinstance(norm, str) or not np.iterable(norm):
        norm = [norm for i in range(n_variates)]
    else:
        if len(norm) != n_variates:
            raise ValueError(
                f'Unable to map the input for norm ({norm}) to {n_variates} '
                f'variables.')
    return norm


def _ensure_multivariate_data(n_variates, data):
    """
    Ensure that the data has dtype with n_variates.

    Input data of shape (n_variates, n, m) is converted to an array of shape
    (n, m) with data type np.dtype(f'{data.dtype}, ' * n_variates)

    Complex data is returned as a view with dtype np.dtype('float64, float64')
    or np.dtype('float32, float32')

    If n_variates is 1 and data is not of type np.ndarray (i.e. PIL.Image),
    the data is returned unchanged.

    If data is None, the function returns None

    Parameters
    ----------
    n_variates : int
        -  number of variates in the data
    data : np.ndarray, PIL.Image or None

    Returns
    -------
        np.ndarray, PIL.Image or None

    """

    if isinstance(data, np.ndarray):
        if len(data.dtype.descr) == n_variates:
            return data
        elif data.dtype in [np.complex64, np.complex128]:
            if data.dtype == np.complex128:
                dt = np.dtype('float64, float64')
            else:
                dt = np.dtype('float32, float32')
            reconstructed = np.ma.frombuffer(data.data, dtype=dt).reshape(data.shape)
            if np.ma.is_masked(data):
                for descriptor in dt.descr:
                    reconstructed[descriptor[0]][data.mask] = np.ma.masked
            return reconstructed

    if n_variates > 1 and len(data) == n_variates:
        # convert data from shape (n_variates, n, m)
        # to (n,m) with a new dtype
        data = [np.ma.array(part, copy=False) for part in data]
        dt = np.dtype(', '.join([f'{part.dtype}' for part in data]))
        fields = [descriptor[0] for descriptor in dt.descr]
        reconstructed = np.ma.empty(data[0].shape, dtype=dt)
        for i, f in enumerate(fields):
            if data[i].shape != reconstructed.shape:
                raise ValueError("For multivariate data all variates must have same "
                                 f"shape, not {data[0].shape} and {data[i].shape}")
            reconstructed[f] = data[i]
            if np.ma.is_masked(data[i]):
                reconstructed[f][data[i].mask] = np.ma.masked
        return reconstructed

    if data is None:
        return data

    if n_variates == 1:
        # PIL.Image gets passed here
        return data
    elif n_variates == 2:
        raise ValueError("Invalid data entry for mutlivariate data. The data"
                         " must contain complex numbers, or have a first dimension 2,"
                         " or be of a dtype with 2 fields")
    else:
        raise ValueError("Invalid data entry for mutlivariate data. The shape"
                         f" of the data must have a first dimension {n_variates}"
                         f" or be of a dtype with {n_variates} fields")


def _ensure_multivariate_clim(n_variates, vmin=None, vmax=None):
    """
    Ensure that vmin and vmax have the correct number of elements.
    If n_variates > 1: A single argument for vmin/vmax will be repeated n
    times in the output.

    Parameters
    ----------
    n_variates : int
        -  number of variates in the data
    vmin and vmax : float or iterable
        -  if iterable, the length must be n_variates

    Returns
    -------
            vmin, vmax as iterables of length n_variates
    """
    if not np.iterable(vmin):
        vmin = [vmin for i in range(n_variates)]
    else:
        if len(vmin) != n_variates:
            raise ValueError(
                f'Unable to map the input for vmin ({vmin}) to {n_variates} '
                f'variables.')

    if not np.iterable(vmax):
        vmax = [vmax for i in range(n_variates)]
    else:
        if len(vmax) != n_variates:
            raise ValueError(
                f'Unable to map the input for vmax ({vmax}) to {n_variates} '
                f'variables.')
        vmax = vmax

    return vmin, vmax


def _ensure_multivariate_params(n_variates, data, norm, vmin, vmax):
    """
    Ensure that the data, norm, vmin and vmax have the correct number of elements.
    If n_variates == 1, the norm, vmin and vmax are returned as lists of length 1.
    If n_variates > 1, the length of each input is checked for consistency and
    single arguments are repeated as necessary to form lists of length n_variates.
    Scalar data is returned unchanged, but multivariate data is restructured to a dtype
    with n_variate fields.
    See the component functions for details.
    """
    norm = _ensure_multivariate_norm(n_variates, norm)
    vmin, vmax = _ensure_multivariate_clim(n_variates, vmin, vmax)
    data = _ensure_multivariate_data(n_variates, data)
    return data, norm, vmin, vmax
