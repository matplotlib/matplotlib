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

import matplotlib as mpl
from matplotlib import _api, colors
# TODO make this warn on access
from matplotlib.colorizer import _ScalarMappable as ScalarMappable  # noqa
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed
from matplotlib._cm_multivar import (
    cmap_families as multivar_cmaps, cmap_init as _multivar_cmap_init)
from matplotlib._cm_bivar import cmaps as bivar_cmaps, cmap_init as _bivar_cmap_init


_LUTSIZE = mpl.rcParams['image.lut']


def _gen_cmap_registry():
    """
    Generate a dict mapping standard colormap names to standard colormaps, as
    well as the reversed colormaps.
    """
    # Register colormap aliases for gray and grey.
    aliases = {
        # alias -> original name
        'grey': 'gray',
        'gist_grey': 'gist_gray',
        'gist_yerg': 'gist_yarg',
        'Grays': 'Greys',
    }
    cmap_d = {**cmaps_listed, **datad, **aliases}
    # Reversed cmaps.
    for name in list(cmap_d.keys()):
        cmap_d[f'{name}_r'] = name

    def cmap_init(name, cmap_spec):
        if name in datad:
            spec = cmap_spec
            return (  # Precache the cmaps at a fixed lutsize..
                colors.LinearSegmentedColormap(name, spec, _LUTSIZE)
                if 'red' in spec else
                colors.ListedColormap(spec['listed'], name)
                if 'listed' in spec else
                colors.LinearSegmentedColormap.from_list(name, spec, _LUTSIZE))
        if name in aliases:
            cmap = cmap_init(cmap_spec, cmap_d[cmap_spec]).copy()
            cmap.name = name
            return cmap
        if name.endswith('_r'):
            # Generate reversed cmaps.
            return cmap_init(cmap_spec, cmap_d[cmap_spec]).reversed()
        return colors.ListedColormap(cmap_spec, name=name)

    return cmap_d, cmap_init


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
    def __init__(self, cmaps, init_func):
        self._cmaps = {name: None for name in cmaps}
        self._init_func = init_func
        self._builtin_cmaps = cmaps

    def __getitem__(self, item):
        cmap = _api.check_getitem(self._cmaps, colormap=item, _error_cls=KeyError)
        if cmap is None:
            cmap = self._cmaps[item] = self._init_func(item, self._builtin_cmaps[item])
        return cmap.copy()

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
_colormaps = ColormapRegistry(*_gen_cmap_registry())

_multivar_colormaps = ColormapRegistry(multivar_cmaps, _multivar_cmap_init)

_bivar_colormaps = ColormapRegistry(bivar_cmaps, _bivar_cmap_init)


def _ensure_cmap(cmap):
    """
    Ensure that we have a `.Colormap` object.

    For internal use to preserve type stability of errors.

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
    if isinstance(cmap, colors.Colormap):
        return cmap
    cmap_name = mpl._val_or_rc(cmap, "image.cmap")
    # use check_in_list to ensure type stability of the exception raised by
    # the internal usage of this (ValueError vs KeyError)
    if cmap_name not in _colormaps:
        _api.check_in_list(sorted(_colormaps), cmap=cmap_name)
    return mpl.colormaps[cmap_name]


@functools.cache
def __getattr__(name):
    if name in _colormaps._builtin_cmaps:
        return _colormaps[name]
    raise AttributeError(f"module 'matplotlib.cm' has no attribute {name!r}")
