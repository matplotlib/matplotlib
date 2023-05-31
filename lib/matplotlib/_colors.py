from collections.abc import Sequence, Mapping
import base64
import numpy as np
import io
import re
import matplotlib as mpl

from numbers import Real
from matplotlib import cbook, _cm, _api

from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS


class _ColorMapping(dict):
    def __init__(self, mapping):
        super().__init__(mapping)
        self.cache = {}

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.cache.clear()

    def __delitem__(self, key):
        super().__delitem__(key)
        self.cache.clear()


_colors_full_map = {}
# Set by reverse priority order.
_colors_full_map.update(XKCD_COLORS)
_colors_full_map.update({k.replace('grey', 'gray'): v
                         for k, v in XKCD_COLORS.items()
                         if 'grey' in k})
_colors_full_map.update(CSS4_COLORS)
_colors_full_map.update(TABLEAU_COLORS)
_colors_full_map.update({k.replace('gray', 'grey'): v
                         for k, v in TABLEAU_COLORS.items()
                         if 'gray' in k})
_colors_full_map.update(BASE_COLORS)
_colors_full_map = _ColorMapping(_colors_full_map)


_REPR_PNG_SIZE = (512, 64)


def get_named_colors_mapping():
    """Return the global mapping of names to named colors."""
    return _colors_full_map


class Colormap:
    """
    Baseclass for all scalar to RGBA mappings.

    Typically, Colormap instances are used to convert data values (floats)
    from the interval ``[0, 1]`` to the RGBA color that the respective
    Colormap represents. For scaling of data into the ``[0, 1]`` interval see
    `matplotlib.colors.Normalize`. Subclasses of `matplotlib.cm.ScalarMappable`
    make heavy use of this ``data -> normalize -> map-to-color`` processing
    chain.
    """

    def __init__(self, name, N=256):
        """
        Parameters
        ----------
        name : str
            The name of the colormap.
        N : int
            The number of RGB quantization levels.
        """
        self.name = name
        self.N = int(N)  # ensure that N is always int
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0)  # If bad, don't paint anything.
        self._rgba_under = None
        self._rgba_over = None
        self._i_under = self.N
        self._i_over = self.N + 1
        self._i_bad = self.N + 2
        self._isinit = False
        #: When this colormap exists on a scalar mappable and colorbar_extend
        #: is not False, colorbar creation will pick up ``colorbar_extend`` as
        #: the default value for the ``extend`` keyword in the
        #: `matplotlib.colorbar.Colorbar` constructor.
        self.colorbar_extend = False

    def __call__(self, X, alpha=None, bytes=False):
        r"""
        Parameters
        ----------
        X : float or int, `~numpy.ndarray` or scalar
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X, or None.
        bytes : bool
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\s in the
            interval ``[0, 255]``.

        Returns
        -------
        Tuple of RGBA values if X is scalar, otherwise an array of
        RGBA values with a shape of ``X.shape + (4, )``.
        """
        if not self._isinit:
            self._init()

        xa = np.array(X, copy=True)
        if not xa.dtype.isnative:
            xa = xa.byteswap().newbyteorder()  # Native byteorder is faster.
        if xa.dtype.kind == "f":
            xa *= self.N
            # xa == 1 (== N after multiplication) is not out of range.
            xa[xa == self.N] = self.N - 1
        # Pre-compute the masks before casting to int (which can truncate
        # negative values to zero or wrap large floats to negative ints).
        mask_under = xa < 0
        mask_over = xa >= self.N
        # If input was masked, get the bad mask from it; else mask out nans.
        mask_bad = X.mask if np.ma.is_masked(X) else np.isnan(xa)
        with np.errstate(invalid="ignore"):
            # We need this cast for unsigned ints as well as floats
            xa = xa.astype(int)
        xa[mask_under] = self._i_under
        xa[mask_over] = self._i_over
        xa[mask_bad] = self._i_bad

        lut = self._lut
        if bytes:
            lut = (lut * 255).astype(np.uint8)

        rgba = lut.take(xa, axis=0, mode='clip')

        if alpha is not None:
            alpha = np.clip(alpha, 0, 1)
            if bytes:
                alpha *= 255  # Will be cast to uint8 upon assignment.
            if alpha.shape not in [(), xa.shape]:
                raise ValueError(
                    f"alpha is array-like but its shape {alpha.shape} does "
                    f"not match that of X {xa.shape}")
            rgba[..., -1] = alpha
            # If the "bad" color is all zeros, then ignore alpha input.
            if (lut[-1] == 0).all():
                rgba[mask_bad] = (0, 0, 0, 0)

        if not np.iterable(X):
            rgba = tuple(rgba)
        return rgba

    def __copy__(self):
        cls = self.__class__
        cmapobject = cls.__new__(cls)
        cmapobject.__dict__.update(self.__dict__)
        if self._isinit:
            cmapobject._lut = np.copy(self._lut)
        return cmapobject

    def __eq__(self, other):
        if (not isinstance(other, Colormap) or
                self.colorbar_extend != other.colorbar_extend):
            return False
        # To compare lookup tables the Colormaps have to be initialized
        if not self._isinit:
            self._init()
        if not other._isinit:
            other._init()
        return np.array_equal(self._lut, other._lut)

    def get_bad(self):
        """Get the color for masked values."""
        if not self._isinit:
            self._init()
        return np.array(self._lut[self._i_bad])

    def set_bad(self, color='k', alpha=None):
        """Set the color for masked values."""
        self._rgba_bad = to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def get_under(self):
        """Get the color for low out-of-range values."""
        if not self._isinit:
            self._init()
        return np.array(self._lut[self._i_under])

    def set_under(self, color='k', alpha=None):
        """Set the color for low out-of-range values."""
        self._rgba_under = to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def get_over(self):
        """Get the color for high out-of-range values."""
        if not self._isinit:
            self._init()
        return np.array(self._lut[self._i_over])

    def set_over(self, color='k', alpha=None):
        """Set the color for high out-of-range values."""
        self._rgba_over = to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def set_extremes(self, *, bad=None, under=None, over=None):
        """
        Set the colors for masked (*bad*) values and, when ``norm.clip =
        False``, low (*under*) and high (*over*) out-of-range values.
        """
        if bad is not None:
            self.set_bad(bad)
        if under is not None:
            self.set_under(under)
        if over is not None:
            self.set_over(over)

    def with_extremes(self, *, bad=None, under=None, over=None):
        """
        Return a copy of the colormap, for which the colors for masked (*bad*)
        values and, when ``norm.clip = False``, low (*under*) and high (*over*)
        out-of-range values, have been set accordingly.
        """
        new_cm = self.copy()
        new_cm.set_extremes(bad=bad, under=under, over=over)
        return new_cm

    def _set_extremes(self):
        if self._rgba_under:
            self._lut[self._i_under] = self._rgba_under
        else:
            self._lut[self._i_under] = self._lut[0]
        if self._rgba_over:
            self._lut[self._i_over] = self._rgba_over
        else:
            self._lut[self._i_over] = self._lut[self.N - 1]
        self._lut[self._i_bad] = self._rgba_bad

    def _init(self):
        """Generate the lookup table, ``self._lut``."""
        raise NotImplementedError("Abstract class only")

    def is_gray(self):
        """Return whether the colormap is grayscale."""
        if not self._isinit:
            self._init()
        return (np.all(self._lut[:, 0] == self._lut[:, 1]) and
                np.all(self._lut[:, 0] == self._lut[:, 2]))

    def resampled(self, lutsize):
        """Return a new colormap with *lutsize* entries."""
        if hasattr(self, '_resample'):
            _api.warn_external(
                "The ability to resample a color map is now public API "
                f"However the class {type(self)} still only implements "
                "the previous private _resample method.  Please update "
                "your class."
            )
            return self._resample(lutsize)

        raise NotImplementedError()

    def reversed(self, name=None):
        """
        Return a reversed instance of the Colormap.

        .. note:: This function is not implemented for the base class.

        Parameters
        ----------
        name : str, optional
            The name for the reversed colormap. If None, the
            name is set to ``self.name + "_r"``.

        See Also
        --------
        LinearSegmentedColormap.reversed
        ListedColormap.reversed
        """
        raise NotImplementedError()

    def _repr_png_(self):
        """Generate a PNG representation of the Colormap."""
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        X = np.tile(np.linspace(0, 1, _REPR_PNG_SIZE[0]),
                    (_REPR_PNG_SIZE[1], 1))
        pixels = self(X, bytes=True)
        png_bytes = io.BytesIO()
        title = self.name + ' colormap'
        author = f'Matplotlib v{mpl.__version__}, https://matplotlib.org'
        pnginfo = PngInfo()
        pnginfo.add_text('Title', title)
        pnginfo.add_text('Description', title)
        pnginfo.add_text('Author', author)
        pnginfo.add_text('Software', author)
        Image.fromarray(pixels).save(png_bytes, format='png', pnginfo=pnginfo)
        return png_bytes.getvalue()

    def _repr_html_(self):
        """Generate an HTML representation of the Colormap."""
        png_bytes = self._repr_png_()
        png_base64 = base64.b64encode(png_bytes).decode('ascii')
        def color_block(color):
            hex_color = to_hex(color, keep_alpha=True)
            return (f'<div title="{hex_color}" '
                    'style="display: inline-block; '
                    'width: 1em; height: 1em; '
                    'margin: 0; '
                    'vertical-align: middle; '
                    'border: 1px solid #555; '
                    f'background-color: {hex_color};"></div>')

        return ('<div style="vertical-align: middle;">'
                f'<strong>{self.name}</strong> '
                '</div>'
                '<div class="cmap"><img '
                f'alt="{self.name} colormap" '
                f'title="{self.name}" '
                'style="border: 1px solid #555;" '
                f'src="data:image/png;base64,{png_base64}"></div>'
                '<div style="vertical-align: middle; '
                f'max-width: {_REPR_PNG_SIZE[0]+2}px; '
                'display: flex; justify-content: space-between;">'
                '<div style="float: left;">'
                f'{color_block(self.get_under())} under'
                '</div>'
                '<div style="margin: 0 auto; display: inline-block;">'
                f'bad {color_block(self.get_bad())}'
                '</div>'
                '<div style="float: right;">'
                f'over {color_block(self.get_over())}'
                '</div>')

    def copy(self):
        """Return a copy of the colormap."""
        return self.__copy__()


def to_hex(c, keep_alpha=False):
    """
    Convert *c* to a hex color.

    Parameters
    ----------
    c : :ref:`color <colors_def>` or `numpy.ma.masked`

    keep_alpha : bool, default: False
      If False, use the ``#rrggbb`` format, otherwise use ``#rrggbbaa``.

    Returns
    -------
    str
      ``#rrggbb`` or ``#rrggbbaa`` hex color string
    """
    c = to_rgba(c)
    if not keep_alpha:
        c = c[:3]
    return "#" + "".join(format(round(val * 255), "02x") for val in c)


def _is_nth_color(c):
    """Return whether *c* can be interpreted as an item in the color cycle."""
    return isinstance(c, str) and re.match(r"\AC[0-9]+\Z", c)


def is_color_like(c):
    """Return whether *c* can be interpreted as an RGB(A) color."""
    # Special-case nth color syntax because it cannot be parsed during setup.
    if _is_nth_color(c):
        return True
    try:
        to_rgba(c)
    except ValueError:
        return False
    else:
        return True


def to_rgba(c, alpha=None):
    """
    Convert *c* to an RGBA color.

    Parameters
    ----------
    c : Matplotlib color or ``np.ma.masked``

    alpha : float, optional
        If *alpha* is given, force the alpha value of the returned RGBA tuple
        to *alpha*.

        If None, the alpha value from *c* is used. If *c* does not have an
        alpha channel, then alpha defaults to 1.

        *alpha* is ignored for the color value ``"none"`` (case-insensitive),
        which always maps to ``(0, 0, 0, 0)``.

    Returns
    -------
    tuple
        Tuple of floats ``(r, g, b, a)``, where each channel (red, green, blue,
        alpha) can assume values between 0 and 1.
    """
    # Special-case nth color syntax because it should not be cached.
    if _is_nth_color(c):
        prop_cycler = mpl.rcParams['axes.prop_cycle']
        colors = prop_cycler.by_key().get('color', ['k'])
        c = colors[int(c[1:]) % len(colors)]
    try:
        rgba = _colors_full_map.cache[c, alpha]
    except (KeyError, TypeError):  # Not in cache, or unhashable.
        rgba = None
    if rgba is None:  # Suppress exception chaining of cache lookup failure.
        rgba = _to_rgba_no_colorcycle(c, alpha)
        try:
            _colors_full_map.cache[c, alpha] = rgba
        except TypeError:
            pass
    return rgba


def to_rgb(c):
    """Convert *c* to an RGB color, silently dropping the alpha channel."""
    return to_rgba(c)[:3]


def to_rgba_array(c, alpha=None):
    """
    Convert *c* to a (n, 4) array of RGBA colors.

    Parameters
    ----------
    c : Matplotlib color or array of colors
        If *c* is a masked array, an `~numpy.ndarray` is returned with a
        (0, 0, 0, 0) row for each masked value or row in *c*.

    alpha : float or sequence of floats, optional
        If *alpha* is given, force the alpha value of the returned RGBA tuple
        to *alpha*.

        If None, the alpha value from *c* is used. If *c* does not have an
        alpha channel, then alpha defaults to 1.

        *alpha* is ignored for the color value ``"none"`` (case-insensitive),
        which always maps to ``(0, 0, 0, 0)``.

        If *alpha* is a sequence and *c* is a single color, *c* will be
        repeated to match the length of *alpha*.

    Returns
    -------
    array
        (n, 4) array of RGBA colors,  where each channel (red, green, blue,
        alpha) can assume values between 0 and 1.
    """
    if isinstance(c, tuple) and len(c) == 2:
        if alpha is None:
            c, alpha = c
        else:
            c = c[0]
    # Special-case inputs that are already arrays, for performance.  (If the
    # array has the wrong kind or shape, raise the error during one-at-a-time
    # conversion.)
    if np.iterable(alpha):
        alpha = np.asarray(alpha).ravel()
    if (isinstance(c, np.ndarray) and c.dtype.kind in "if"
            and c.ndim == 2 and c.shape[1] in [3, 4]):
        mask = c.mask.any(axis=1) if np.ma.is_masked(c) else None
        c = np.ma.getdata(c)
        if np.iterable(alpha):
            if c.shape[0] == 1 and alpha.shape[0] > 1:
                c = np.tile(c, (alpha.shape[0], 1))
            elif c.shape[0] != alpha.shape[0]:
                raise ValueError("The number of colors must match the number"
                                 " of alpha values if there are more than one"
                                 " of each.")
        if c.shape[1] == 3:
            result = np.column_stack([c, np.zeros(len(c))])
            result[:, -1] = alpha if alpha is not None else 1.
        elif c.shape[1] == 4:
            result = c.copy()
            if alpha is not None:
                result[:, -1] = alpha
        if mask is not None:
            result[mask] = 0
        if np.any((result < 0) | (result > 1)):
            raise ValueError("RGBA values should be within 0-1 range")
        return result
    # Handle single values.
    # Note that this occurs *after* handling inputs that are already arrays, as
    # `to_rgba(c, alpha)` (below) is expensive for such inputs, due to the need
    # to format the array in the ValueError message(!).
    if cbook._str_lower_equal(c, "none"):
        return np.zeros((0, 4), float)
    try:
        if np.iterable(alpha):
            return np.array([to_rgba(c, a) for a in alpha], float)
        else:
            return np.array([to_rgba(c, alpha)], float)
    except TypeError:
        pass
    except ValueError as e:
        if e.args == ("'alpha' must be between 0 and 1, inclusive", ):
            # ValueError is from _to_rgba_no_colorcycle().
            raise e
    if isinstance(c, str):
        raise ValueError(f"{c!r} is not a valid color value.")

    if len(c) == 0:
        return np.zeros((0, 4), float)

    # Quick path if the whole sequence can be directly converted to a numpy
    # array in one shot.
    if isinstance(c, Sequence):
        lens = {len(cc) if isinstance(cc, (list, tuple)) else -1 for cc in c}
        if lens == {3}:
            rgba = np.column_stack([c, np.ones(len(c))])
        elif lens == {4}:
            rgba = np.array(c)
        else:
            rgba = np.array([to_rgba(cc) for cc in c])
    else:
        rgba = np.array([to_rgba(cc) for cc in c])

    if alpha is not None:
        rgba[:, 3] = alpha
    return rgba


def _to_rgba_no_colorcycle(c, alpha=None):
    """
    Convert *c* to an RGBA color, with no support for color-cycle syntax.

    If *alpha* is given, force the alpha value of the returned RGBA tuple
    to *alpha*. Otherwise, the alpha value from *c* is used, if it has alpha
    information, or defaults to 1.

    *alpha* is ignored for the color value ``"none"`` (case-insensitive),
    which always maps to ``(0, 0, 0, 0)``.
    """
    if isinstance(c, tuple) and len(c) == 2:
        if alpha is None:
            c, alpha = c
        else:
            c = c[0]
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError("'alpha' must be between 0 and 1, inclusive")
    orig_c = c
    if c is np.ma.masked:
        return (0., 0., 0., 0.)
    if isinstance(c, str):
        if c.lower() == "none":
            return (0., 0., 0., 0.)
        # Named color.
        try:
            # This may turn c into a non-string, so we check again below.
            c = _colors_full_map[c]
        except KeyError:
            if len(orig_c) != 1:
                try:
                    c = _colors_full_map[c.lower()]
                except KeyError:
                    pass
    if isinstance(c, str):
        # hex color in #rrggbb format.
        match = re.match(r"\A#[a-fA-F0-9]{6}\Z", c)
        if match:
            return (tuple(int(n, 16) / 255
                          for n in [c[1:3], c[3:5], c[5:7]])
                    + (alpha if alpha is not None else 1.,))
        # hex color in #rgb format, shorthand for #rrggbb.
        match = re.match(r"\A#[a-fA-F0-9]{3}\Z", c)
        if match:
            return (tuple(int(n, 16) / 255
                          for n in [c[1]*2, c[2]*2, c[3]*2])
                    + (alpha if alpha is not None else 1.,))
        # hex color with alpha in #rrggbbaa format.
        match = re.match(r"\A#[a-fA-F0-9]{8}\Z", c)
        if match:
            color = [int(n, 16) / 255
                     for n in [c[1:3], c[3:5], c[5:7], c[7:9]]]
            if alpha is not None:
                color[-1] = alpha
            return tuple(color)
        # hex color with alpha in #rgba format, shorthand for #rrggbbaa.
        match = re.match(r"\A#[a-fA-F0-9]{4}\Z", c)
        if match:
            color = [int(n, 16) / 255
                     for n in [c[1]*2, c[2]*2, c[3]*2, c[4]*2]]
            if alpha is not None:
                color[-1] = alpha
            return tuple(color)
        # string gray.
        try:
            c = float(c)
        except ValueError:
            pass
        else:
            if not (0 <= c <= 1):
                raise ValueError(
                    f"Invalid string grayscale value {orig_c!r}. "
                    f"Value must be within 0-1 range")
            return c, c, c, alpha if alpha is not None else 1.
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # turn 2-D array into 1-D array
    if isinstance(c, np.ndarray):
        if c.ndim == 2 and c.shape[0] == 1:
            c = c.reshape(-1)
    # tuple color.
    if not np.iterable(c):
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    if len(c) not in [3, 4]:
        raise ValueError("RGBA sequence should have length 3 or 4")
    if not all(isinstance(x, Real) for x in c):
        # Checks that don't work: `map(float, ...)`, `np.array(..., float)` and
        # `np.array(...).astype(float)` would all convert "0.5" to 0.5.
        raise ValueError(f"Invalid RGBA argument: {orig_c!r}")
    # Return a tuple to prevent the cached value from being modified.
    c = tuple(map(float, c))
    if len(c) == 3 and alpha is None:
        alpha = 1
    if alpha is not None:
        c = c[:3] + (alpha,)
    if any(elem < 0 or elem > 1 for elem in c):
        raise ValueError("RGBA values should be within 0-1 range")
    return c


class ColorConverter:
    """
    A class only kept for backwards compatibility.

    Its functionality is entirely provided by module-level functions.
    """
    colors = _colors_full_map
    cache = _colors_full_map.cache
    to_rgb = staticmethod(to_rgb)
    to_rgba = staticmethod(to_rgba)
    to_rgba_array = staticmethod(to_rgba_array)


class ColorSequenceRegistry(Mapping):
    r"""
    Container for sequences of colors that are known to Matplotlib by name.

    The universal registry instance is `matplotlib.color_sequences`. There
    should be no need for users to instantiate `.ColorSequenceRegistry`
    themselves.

    Read access uses a dict-like interface mapping names to lists of colors::

        import matplotlib as mpl
        cmap = mpl.color_sequences['tab10']

    The returned lists are copies, so that their modification does not change
    the global definition of the color sequence.

    Additional color sequences can be added via
    `.ColorSequenceRegistry.register`::

        mpl.color_sequences.register('rgb', ['r', 'g', 'b'])
    """

    _BUILTIN_COLOR_SEQUENCES = {
        'tab10': _cm._tab10_data,
        'tab20': _cm._tab20_data,
        'tab20b': _cm._tab20b_data,
        'tab20c': _cm._tab20c_data,
        'Pastel1': _cm._Pastel1_data,
        'Pastel2': _cm._Pastel2_data,
        'Paired': _cm._Paired_data,
        'Accent': _cm._Accent_data,
        'Dark2': _cm._Dark2_data,
        'Set1': _cm._Set1_data,
        'Set2': _cm._Set1_data,
        'Set3': _cm._Set1_data,
    }

    def __init__(self):
        self._color_sequences = {**self._BUILTIN_COLOR_SEQUENCES}

    def __getitem__(self, item):
        try:
            return list(self._color_sequences[item])
        except KeyError:
            raise KeyError(f"{item!r} is not a known color sequence name")

    def __iter__(self):
        return iter(self._color_sequences)

    def __len__(self):
        return len(self._color_sequences)

    def __str__(self):
        return ('ColorSequenceRegistry; available colormaps:\n' +
                ', '.join(f"'{name}'" for name in self))

    def register(self, name, color_list):
        """
        Register a new color sequence.

        The color sequence registry stores a copy of the given *color_list*, so
        that future changes to the original list do not affect the registered
        color sequence. Think of this as the registry taking a snapshot
        of *color_list* at registration.

        Parameters
        ----------
        name : str
            The name for the color sequence.

        color_list : list of colors
            An iterable returning valid Matplotlib colors when iterating over.
            Note however that the returned color sequence will always be a
            list regardless of the input type.

        """
        if name in self._BUILTIN_COLOR_SEQUENCES:
            raise ValueError(f"{name!r} is a reserved name for a builtin "
                             "color sequence")

        color_list = list(color_list)  # force copy and coerce type to list
        for color in color_list:
            try:
                to_rgba(color)
            except ValueError:
                raise ValueError(
                    f"{color!r} is not a valid color specification")

        self._color_sequences[name] = color_list

    def unregister(self, name):
        """
        Remove a sequence from the registry.

        You cannot remove built-in color sequences.

        If the name is not registered, returns with no error.
        """
        if name in self._BUILTIN_COLOR_SEQUENCES:
            raise ValueError(
                f"Cannot unregister builtin color sequence {name!r}")
        self._color_sequences.pop(name, None)


_color_sequences = ColorSequenceRegistry()
