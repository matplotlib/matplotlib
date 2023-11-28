"""
Builtin colormaps, colormap handling utilities, and the `ScalarMappable` mixin.

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
from matplotlib._cm_multivar import cmap_families as multivar_cmaps
from matplotlib._cm_bivar import cmaps as bivar_cmaps


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
        if isinstance(cmap, colors.Colormap) \
                or isinstance(cmap, colors.MultivarColormap)\
                or isinstance(cmap, colors.BivarColormap):
            return cmap
        if isinstance(cmap, str):
            _api.check_in_list(sorted(self._cmaps), cmap=cmap)
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

_multivar_colormaps = ColormapRegistry(multivar_cmaps)
globals().update(_multivar_colormaps)

_bivar_colormaps = ColormapRegistry(bivar_cmaps)
globals().update(_bivar_colormaps)


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


def _merge_signals(fn):
    """
    If multiple ScalarMappables part of a VectorMappable emit 'changed' signals
    this decorator works to merge them into one so that only one  'changed'
    signal is emitted by the VectorMappable
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        # suspend signals
        self._pause_signals = True
        self.intercepted_changed = False
        # run fn
        out = fn(self, *args, **kwargs)
        # resume signals
        self._pause_signals = False
        # send 'changed' signal if any were intercepted
        if self.intercepted_changed:
            self.changed()
        return out

    return wrapper


class VectorMappable:
    """
    A mixin class to map multiple scalar data to RGBA.

    The VectorMappable applies data normalization before returning RGBA colors
    from the given MultivarColormap or BivarColormap.
    """
    def __init__(self, norm=None, cmap=None):
        """
        For multivariate data, a MultivariateColormap must be provided and
        norm must be a list of valid objects with length matching the colormap

        Parameters
        ----------
        norm : `.Normalize` (or subclass thereof) or str or None or list thereof
            The normalizing object which scales data, typically into the
            interval ``[0, 1]``.
            If a `str`, a `.Normalize` subclass is dynamically generated based
            on the scale with the corresponding name.
            If *None*, *norm* defaults to a *colors.Normalize* object which
            initializes its scaling based on the first data processed.
        cmap : str or `~matplotlib.colors.Colormap`,
            or `~matplotlib.colors.BivarColormap`
            or `~matplotlib.colors.MultivarColormap`
            The colormap used to map normalized data values to RGBA colors.
        """

        self._pause_signals = False

        cmap = ensure_cmap(cmap)
        norm = ensure_multivariate_norm(cmap.n_variates, norm)

        self.callbacks = cbook.CallbackRegistry(signals=["changed"])
        if isinstance(cmap, colors.MultivarColormap) or \
           isinstance(cmap, colors.BivarColormap):
            if isinstance(cmap, colors.MultivarColormap):
                self.scalars = [ScalarMappable(n, c) for n, c in zip(norm, cmap)]
            else:
                self.scalars = [ScalarMappable(n) for n in norm]
            for i, sca in enumerate(self.scalars):
                sca.callbacks.connect('changed', self._on_changed)
            self._cmap = cmap
            self._multivar_A = None
            self._colorbar = None
        else:
            # i.e type(cmap) is Str
            # or issubclass(type(cmap), colors.Colorbar)
            # or cmap is None
            self.scalars = [ScalarMappable(norm, cmap)]
            # it would be tempting to just use one layer of callbacks here
            # i.e. self.callbacks = self.scalars[0].callbacks
            # but this will require more reformating as it fails in cleanup
            # if a colorbar is removed
            self.scalars[0].callbacks.connect('changed', self._on_changed)

    @property
    def _A(self):
        if len(self.scalars) == 1:
            return self.scalars[0]._A
        else:
            # _A on the scalars should be views of the same array
            # or None
            # If they views of the same array –> ok, return it
            # else if they are all None, None
            # else, raise error
            if any([s is None for s in self.scalars]):
                if all([s is None for s in self.scalars]):
                    return None
                else:
                    raise AttributeError(
                        "Attempting to get _A on a VectorMappable but _A on"
                        " the ScalarMappables are not views of the same base"
                        " array (at least one is None)."
                        " Most likely sm.set_array(A) or sm._A = A has been"
                        " called, which is not supported when the"
                        " ScalarMappable belongs to a VectorMappable."
                    )
            if self._multivar_A is None:
                if all([s._A is None for s in self.scalars]):
                    return None
                raise AttributeError(
                    "Attempting to get _A on a VectorMappable but _A on"
                    " has only been indedependently set on"
                    " ScalarMappables, which is unsupported."
                    " Please use VectorMappable.set_array(A) instead."
                )

            if all([np.shares_memory(s._A, self._multivar_A) for s in self.scalars]):
                return self._multivar_A
            raise AttributeError(
                "Attempting to get _A on a VectorMappable but _A on"
                " the ScalarMappables are not views of _multivar_A"
                " on the VectorMappable."
                " Most likely sm.set_array(A) or sm._A = A has been"
                " called, which is not supported when the"
                " ScalarMappable belongs to a VectorMappable."
            )

    @_A.setter
    def _A(self, A):
        ensure_multivariate_data(len(self.scalars), A)
        if len(self.scalars) == 1:
            self.scalars[0]._A = A
        else:
            self._multivar_A = A
            for s, a in zip(self.scalars, A):
                s._A = a

    @property
    def cmap(self):
        return self.get_cmap()

    @cmap.setter
    def cmap(self, cmap):
        self.set_cmap(cmap)

    @property
    def colorbar(self):
        if len(self.scalars) == 1:
            return self.scalars[0].colorbar
        else:
            return self._colorbar

    @colorbar.setter
    def colorbar(self, colorbar):
        if len(self.scalars) == 1:
            self.scalars[0].colorbar = colorbar
        else:
            self._colorbar = colorbar

    def get_cmap(self):
        if len(self.scalars) == 1:
            return self.scalars[0].get_cmap()
        else:
            return self._cmap

    @_merge_signals
    def set_cmap(self, cmap):
        """
        Set the colormaps for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        cmap = ensure_cmap(cmap)

        if not cmap.n_variates == len(self.scalars):
            raise ValueError("The number of variates cannot be changed"
                             " after creation."
                             " Instead use the cmap keyword during creation"
                             " of the object."
                             )
        if len(self.scalars) == 1:
            self.scalars[0].set_cmap(cmap)
        else:
            if isinstance(cmap, colors.MultivarColormap):
                for s, cm in zip(self.scalars, cmap):
                    s.set_cmap(cm)
            self._cmap = cmap

    @_merge_signals
    def _scale_norm(self, norm, vmin, vmax):
        """
        Helper for initial scaling.

        Used by public functions that create a VectorMappable and support
        parameters *vmin*, *vmax* and *norm*. This makes sure that a *norm*
        will take precedence over *vmin*, *vmax*.

        Note that this method does not check input for consistency,
        nor does it set the norm.
        """
        if len(self.scalars) == 1:
            self.scalars[0]._scale_norm(norm, vmin, vmax)
        else:
            for s, n, vm, vx in zip(self.scalars, norm, vmin, vmax):
                s._scale_norm(n, vm, vx)

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """
        Return a normalized RGBA array corresponding to *arr*.

        See ScalarMappable.to_rgba for behaviour with scalar or RGBA data


        For multivariate data, each variate is converted independently before
        combination in sRGB space according to the rules of the colormap

        If alpha is set (float or np.ndarray) it replaces the alpha channel
        in the output image, except where the array is masked or np.nan

        If norm is False, no normalization of the input data is performed
        and it is assumed to be in the range (0-1).

        If bytes = True, conversion to bytes is done after image conversion.
        This is less efficient than converting the colormap before making the image
        (as is done in ScalarMappable.to_rgba) but prevents branching in the code.
        """
        if len(self.scalars) == 1:
            rgba = self.scalars[0].to_rgba(x, alpha=alpha, bytes=bytes, norm=norm)
        else:  # multivariate
            ensure_multivariate_data(len(self.scalars), x)
            if isinstance(self._cmap, colors.BivarColormap):
                if norm:
                    normed_0 = self.scalars[0].norm(x[0])
                    normed_1 = self.scalars[1].norm(x[1])
                else:
                    normed_0 = x[0].copy()
                    normed_1 = x[1].copy()
                # in-place clip to shape of colormap: square, circle, etc.
                rgba = self.cmap((normed_0, normed_1))
            else:  # i.e. isinstance(self._cmaps, colors.MultivarColormap)
                # This loops through the data and does norm+cmap for each variate.
                # The alternative is to first to do norm for all, then cmap for all.
                # Looping norm+cmap for each variate has better memory utilization,
                # but requires a separate implementation, rather than calling cmap().
                s = self.scalars[0]
                rgba = np.array(s.to_rgba(x[0], bytes=False, norm=norm, alpha=1))
                for s, xx in zip(self.scalars[1:], x[1:]):
                    sub_rgba = np.array(s.to_rgba(xx, bytes=False, norm=norm, alpha=1))
                    rgba[..., :3] += sub_rgba[..., :3]  # add colors
                    rgba[..., 3] *= sub_rgba[..., 3]  # multiply alpha
                # MultivarColormap require alpha = 0 for bad values
                # giving the following condition to get the bad_mask
                mask_bad = rgba[..., 3] == 0
                rgba[mask_bad] = self.cmap.get_bad()

                if self._cmap.combination_mode == 'Sub':
                    rgba[..., :3] -= len(self.scalars) - 1

            rgba = np.clip(rgba, 0, 1)

            # If manually specified alpha
            # rgba[..., -1] is at this point determined by the multivariate
            # colormaps. (typically 'bad' values set alpha = 0).
            #
            if alpha is not None:
                alpha = np.clip(alpha, 0, 1)
                if alpha.shape not in [(), x[0].shape]:
                    raise ValueError(
                        f"alpha is array-like but its shape {alpha.shape} does "
                        f"not match that of the input {x[0].shape}")
                rgba[..., -1] *= alpha

            if bytes:
                rgba = (rgba * 255).astype('uint8')

        return rgba

    @_merge_signals
    def set_array(self, A):
        """
        Set the value array from array-like *A*.

        Parameters
        ----------
        A : array-like or None
            The values that are mapped to colors.

            The base class `.ScalarMappable` does not make any assumptions on
            the dimensionality and shape of the value array *A*.
        """
        if len(self.scalars) == 1:
            return self.scalars[0].set_array(A)

        A = cbook.safe_masked_invalid(A, copy=True)
        if not np.can_cast(A.dtype, float, "same_kind"):
            raise TypeError(f"Image data of dtype {A.dtype} cannot be "
                            "converted to float")
        self._A = A

    def get_array(self):
        """
        Return the array of values, that are mapped to colors.
        """
        return self._A

    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
        if len(self.scalars) == 1:
            return self.scalars[0].get_clim()
        else:
            vmin = []
            vmax = []
            for s in self.scalars:
                vn, vx = s.get_clim()
                vmin.append(vn)
                vmax.append(vx)
            return vmin, vmax

    @_merge_signals
    def set_clim(self, vmin=None, vmax=None):
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             For scalar data the limits may also be passed as a tuple
             (*vmin*, *vmax*) as a single positional argument.

             For vector data *vmin* and *vmax* must be passed as lists
             of floats

             .. ACCEPTS: (vmin: float, vmax: float)
        """
        if len(self.scalars) == 1:
            self.scalars[0].set_clim(vmin=vmin, vmax=vmax)
        else:
            vmin, vmax = ensure_multivariate_clim(len(self.scalars), vmin, vmax)
            for s, vm, vx in zip(self.scalars, vmin, vmax):
                s.set_clim(vmin=vm, vmax=vx)

    def get_alpha(self):
        """
        Returns
        -------
        float
            Always returns 1.
        """
        # This method is intended to be overridden by Artist sub-classes
        return 1

    @property
    def norm(self):
        if len(self.scalars) == 1:
            return self.scalars[0].norm
        else:
            return ([s.norm for s in self.scalars])

    @norm.setter
    @_merge_signals
    def norm(self, norm):
        self.set_norm(norm)

    def set_norm(self, norm):
        """
        Set the normalization instance.

        Parameters
        ----------
        norm : `.Normalize` or str or None or list thereof

        Notes
        -----
        If there are any colorbars using the mappable for this norm, setting
        the norm of the mappable will reset the norm, locator, and formatters
        on the colorbar to default.
        """
        norm = ensure_multivariate_norm(len(self.scalars), norm)
        if len(self.scalars) == 1:
            return self.scalars[0].set_norm(norm)
        else:
            for s, n in zip(self.scalars, norm):
                s.set_norm(n)

    @_merge_signals
    def autoscale(self):
        """
        Autoscale the scalar limits on the norms using the current arrays
        """
        for s in self.scalars:
            s.autoscale()

    @_merge_signals
    def autoscale_None(self):
        """
        Autoscale the scalar limits on the norms using the
        current arrays, changing only limits that are None
        """
        for s in self.scalars:
            s.autoscale_None()

    def changed(self):
        """
        Call this whenever a ScalarMappable is changed to notify all the
        callbackSM listeners listening to the VectorMappable to the 'changed'
        signal.
        """
        if not self._pause_signals:
            self.callbacks.process('changed', self)
        else:
            self.intercepted_changed = True  # for the _merge_signals decorator

    def _on_changed(self, obj=None):
        """
        Called on the signal 'changed' from the ScalarMappables

        Propagate the signal to listeners on the VectorMappable
        """
        self.changed()

class ScalarMappable:
    """
    A mixin class to map scalar data to RGBA.

    The ScalarMappable applies data normalization before returning RGBA colors
    from the given colormap.
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
        self._norm = None  # So that the setter knows we're initializing.
        self.set_norm(norm)  # The Normalize instance of this ScalarMappable.
        self.cmap = None  # So that the setter knows we're initializing.
        self.set_cmap(cmap)  # The Colormap instance of this ScalarMappable.
        #: The last colorbar associated with this ScalarMappable. May be None.
        self.colorbar = None
        self.callbacks = cbook.CallbackRegistry(signals=["changed"])

    def _scale_norm(self, norm, vmin, vmax):
        """
        Helper for initial scaling.

        Used by public functions that create a ScalarMappable and support
        parameters *vmin*, *vmax* and *norm*. This makes sure that a *norm*
        will take precedence over *vmin*, *vmax*.

        Note that this method does not set the norm.
        """
        if vmin is not None or vmax is not None:
            self.set_clim(vmin, vmax)
            if isinstance(norm, colors.Normalize):
                raise ValueError(
                    "Passing a Normalize instance simultaneously with "
                    "vmin/vmax is not supported.  Please pass vmin/vmax "
                    "directly to the norm when creating it.")

        # always resolve the autoscaling so we have concrete limits
        # rather than deferring to draw time.
        self.autoscale_None()

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """
        Return a normalized RGBA array corresponding to *x*.

        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding `~numpy.ndarray` of RGBA values will be returned,
        based on the norm and colormap set for this ScalarMappable.

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
            x = self.norm(x)
        rgba = self.cmap(x, alpha=alpha, bytes=bytes)
        return rgba

    def set_array(self, A):
        """
        Set the value array from array-like *A*.

        Parameters
        ----------
        A : array-like or None
            The values that are mapped to colors.

            The base class `.ScalarMappable` does not make any assumptions on
            the dimensionality and shape of the value array *A*.
        """
        if A is None:
            self._A = None
            return

        A = cbook.safe_masked_invalid(A, copy=True)
        if not np.can_cast(A.dtype, float, "same_kind"):
            raise TypeError(f"Image data of dtype {A.dtype} cannot be "
                            "converted to float")

        self._A = A
        if not self.norm.scaled():
            self.norm.autoscale_None(A)

    def get_array(self):
        """
        Return the array of values, that are mapped to colors.

        The base class `.ScalarMappable` does not make any assumptions on
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
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        if vmax is None:
            try:
                vmin, vmax = vmin
            except (TypeError, ValueError):
                pass
        if vmin is not None:
            self.norm.vmin = colors._sanitize_extrema(vmin)
        if vmax is not None:
            self.norm.vmax = colors._sanitize_extrema(vmax)

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

        self.cmap = ensure_cmap(cmap, accept_multivariate=False)
        if not in_init:
            self.changed()  # Things are not set up properly yet.

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, norm):
        _api.check_isinstance((colors.Normalize, str, None), norm=norm)
        if norm is None:
            norm = colors.Normalize()
        elif isinstance(norm, str):
            try:
                scale_cls = scale._scale_mapping[norm]
            except KeyError:
                raise ValueError(
                    "Invalid norm str name; the following values are "
                    f"supported: {', '.join(scale._scale_mapping)}"
                ) from None
            norm = _auto_norm_from_scale(scale_cls)()

        if norm is self.norm:
            # We aren't updating anything
            return

        in_init = self.norm is None
        # Remove the current callback and connect to the new one
        if not in_init:
            self.norm.callbacks.disconnect(self._id_norm)
        self._norm = norm
        self._id_norm = self.norm.callbacks.connect('changed',
                                                    self.changed)
        if not in_init:
            self.changed()

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
        self.norm = norm

    def autoscale(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.norm.autoscale(self._A)

    def autoscale_None(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.norm.autoscale_None(self._A)

    def changed(self):
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        """
        self.callbacks.process('changed', self)
        self.stale = True


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


@_api.deprecated
def _ensure_cmap(cmap):
    """
    Ensure that we have a `.Colormap` object.

    For internal use to preserve type stability of errors.

    This function is depreciated with the introduction of cm.ensure_cmap().
    cm.ensure_cmap(cmap, multivariate = False) should be used instead.

    Parameters
    ----------
    cmap : None, str, Colormap

        - if a `Colormap`, return it
        - if a string, look it up in mpl.colormaps
        - if None, look up the default color map in mpl.colormaps

    Returns
    -------
    Colormap
    """

    return ensure_cmap(cmap, accept_multivariate=False)


def ensure_cmap(cmap, accept_multivariate=True):
    """
    For internal use to preserve type stability of errors, and
    for external use to ensure that we have a `~matplotlib.colors.Colormap`,
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

    if isinstance(cmap, colors.Colormap) or \
            isinstance(cmap, colors.MultivarColormap) or \
            isinstance(cmap, colors.BivarColormap):
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


def ensure_multivariate_norm(n_variates, norm):
    """
    Ensure that the norm has the correct number of elements.
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
    if n_variates > 1:
        if isinstance(norm, str) or not np.iterable(norm):
            norm = [norm for i in range(n_variates)]
        else:
            if len(norm) != n_variates:
                raise ValueError(
                    f'Unable to map the input for norm ({norm}) to {n_variates} '
                    f'variables.')
    return norm


def ensure_multivariate_data(n_variates, data):
    """
    Ensure that the data has the correct number of elements.
    Raises a ValueError if the data does not match the number of variates

    Parameters
    ----------
    n_variates : int
        -  number of variates in the data
    data : np.ndarray

    """
    if n_variates > 1:
        if len(data) != n_variates:
            raise ValueError(
                f'For the selected colormap the data must have a first dimension '
                f'{n_variates}, not {len(data)}')


def ensure_multivariate_clim(n_variates, vmin=None, vmax=None):
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
        if n_variates == 1:
            vmin, vmax returned unchanged
        if n_variates > 1:
            vmin, vmax as iterables of length n_variates
    """
    if n_variates > 1:
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


def ensure_multivariate_params(n_variates, data, norm, vmin, vmax):
    """
    Ensure that the data, norm, vmin and vmax have the correct number of elements.
    If n_variates == 1, the norm, vmin and vmax are returned unchanged.
    If n_variates > 1, the length of each input is checked for consistency and
    single arguments are repeated as necessary.
    See the component functions for details.
    """
    norm = ensure_multivariate_norm(n_variates, norm)
    vmin, vmax = ensure_multivariate_clim(n_variates, vmin, vmax)
    ensure_multivariate_data(n_variates, data)

    return norm, vmin, vmax
