"""
The rcsetup module contains the default values and the validation code for
customization using matplotlib's rc settings.

Each rc setting is assigned a default value and a function used to validate
any attempted changes to that setting. The default values and validation
functions are defined in the rcsetup module, and are used to construct the
rcParams global object which stores the settings and is referenced throughout
matplotlib.

These default values should be consistent with the default matplotlibrc file
that actually reflects the values given here. Any additions or deletions to the
parameter set listed here should also be visited to the
:file:`matplotlibrc.template` in matplotlib's root source directory.
"""

import ast
from functools import partial, reduce
import logging
from numbers import Number
import operator
import os
import re

import numpy as np

from matplotlib import animation, cbook
from matplotlib.cbook import ls_mapper
from matplotlib.fontconfig_pattern import parse_fontconfig_pattern
from matplotlib.colors import is_color_like

# Don't let the original cycler collide with our validating cycler
from cycler import Cycler, cycler as ccycler


_log = logging.getLogger(__name__)
# The capitalized forms are needed for ipython at present; this may
# change for later versions.
interactive_bk = ['GTK3Agg', 'GTK3Cairo',
                  'MacOSX',
                  'nbAgg',
                  'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo',
                  'TkAgg', 'TkCairo',
                  'WebAgg',
                  'WX', 'WXAgg', 'WXCairo']
non_interactive_bk = ['agg', 'cairo',
                      'pdf', 'pgf', 'ps', 'svg', 'template']
all_backends = interactive_bk + non_interactive_bk


class ValidateInStrings:
    def __init__(self, key, valid, ignorecase=False, *,
                 _deprecated_since=None):
        """*valid* is a list of legal strings."""
        self.key = key
        self.ignorecase = ignorecase
        self._deprecated_since = _deprecated_since

        def func(s):
            if ignorecase:
                return s.lower()
            else:
                return s
        self.valid = {func(k): k for k in valid}

    def __call__(self, s):
        if self._deprecated_since:
            name, = (k for k, v in globals().items() if v is self)
            cbook.warn_deprecated(
                self._deprecated_since, name=name, obj_type="function")
        if self.ignorecase:
            s = s.lower()
        if s in self.valid:
            return self.valid[s]
        msg = (f"{s!r} is not a valid value for {self.key}; supported values "
               f"are {[*self.valid.values()]}")
        if (isinstance(s, str)
                and (s.startswith('"') and s.endswith('"')
                     or s.startswith("'") and s.endswith("'"))
                and s[1:-1] in self.valid):
            msg += "; remove quotes surrounding your string"
        raise ValueError(msg)


def _listify_validator(scalar_validator, allow_stringlist=False, *, doc=None):
    def f(s):
        if isinstance(s, str):
            try:
                return [scalar_validator(v.strip()) for v in s.split(',')
                        if v.strip()]
            except Exception:
                if allow_stringlist:
                    # Sometimes, a list of colors might be a single string
                    # of single-letter colornames. So give that a shot.
                    return [scalar_validator(v.strip())
                            for v in s if v.strip()]
                else:
                    raise
        # Allow any ordered sequence type -- generators, np.ndarray, pd.Series
        # -- but not sets, whose iteration order is non-deterministic.
        elif np.iterable(s) and not isinstance(s, (set, frozenset)):
            # The condition on this list comprehension will preserve the
            # behavior of filtering out any empty strings (behavior was
            # from the original validate_stringlist()), while allowing
            # any non-string/text scalar values such as numbers and arrays.
            return [scalar_validator(v) for v in s
                    if not isinstance(v, str) or v]
        else:
            raise ValueError("{!r} must be of type: str or non-dictionary "
                             "iterable".format(s))
    try:
        f.__name__ = "{}list".format(scalar_validator.__name__)
    except AttributeError:  # class instance.
        f.__name__ = "{}List".format(type(scalar_validator).__name__)
    f.__qualname__ = f.__qualname__.rsplit(".", 1)[0] + "." + f.__name__
    f.__doc__ = doc if doc is not None else scalar_validator.__doc__
    return f


def validate_any(s):
    return s
validate_anylist = _listify_validator(validate_any)


def validate_date(s):
    try:
        np.datetime64(s)
        return s
    except ValueError:
        raise RuntimeError('"%s" should be a string that can be parsed by ',
                           'numpy.datetime64.' % s)


@cbook.deprecated("3.2", alternative="os.path.exists")
def validate_path_exists(s):
    """If s is a path, return s, else False"""
    if s is None:
        return None
    if os.path.exists(s):
        return s
    else:
        raise RuntimeError('"%s" should be a path but it does not exist' % s)


def validate_bool(b):
    """Convert b to ``bool`` or raise."""
    if isinstance(b, str):
        b = b.lower()
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError('Could not convert "%s" to bool' % b)


@cbook.deprecated("3.3")
def validate_bool_maybe_none(b):
    """Convert b to ``bool`` or raise, passing through *None*."""
    if isinstance(b, str):
        b = b.lower()
    if b is None or b == 'none':
        return None
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError('Could not convert "%s" to bool' % b)


def _validate_tex_preamble(s):
    message = (
        f"Support for setting the 'text.latex.preamble' and 'pgf.preamble' "
        f"rcParams to {s!r} is deprecated since %(since)s and will be "
        f"removed %(removal)s; please set them to plain (possibly empty) "
        f"strings instead.")
    if s is None or s == 'None':
        cbook.warn_deprecated("3.3", message=message)
        return ""
    try:
        if isinstance(s, str):
            return s
        elif np.iterable(s):
            cbook.warn_deprecated("3.3", message=message)
            return '\n'.join(s)
        else:
            raise TypeError
    except TypeError as e:
        raise ValueError('Could not convert "%s" to string' % s) from e


def validate_axisbelow(s):
    try:
        return validate_bool(s)
    except ValueError:
        if isinstance(s, str):
            if s == 'line':
                return 'line'
            if s.lower().startswith('line'):
                cbook.warn_deprecated(
                    "3.3", message=f"Support for setting axes.axisbelow to "
                    f"{s!r} to mean 'line' is deprecated since %(since)s and "
                    f"will be removed %(removal)s; set it to 'line' instead.")
                return 'line'
    raise ValueError('%s cannot be interpreted as'
                     ' True, False, or "line"' % s)


def validate_dpi(s):
    """Confirm s is string 'figure' or convert s to float or raise."""
    if s == 'figure':
        return s
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f'{s!r} is not string "figure" and '
                         f'could not convert {s!r} to float') from e


def _make_type_validator(cls, *, allow_none=False):
    """
    Return a validator that converts inputs to *cls* or raises (and possibly
    allows ``None`` as well).
    """

    def validator(s):
        if (allow_none and
                (s is None or isinstance(s, str) and s.lower() == "none")):
            return None
        try:
            return cls(s)
        except ValueError as e:
            raise ValueError(f'Could not convert {s!r} to {cls.__name__}') from e

    validator.__name__ = f"validate_{cls.__name__}"
    if allow_none:
        validator.__name__ += "_or_None"
    validator.__qualname__ = (
        validator.__qualname__.rsplit(".", 1)[0] + "." + validator.__name__)
    return validator


validate_string = _make_type_validator(str)
validate_string_or_None = _make_type_validator(str, allow_none=True)
validate_stringlist = _listify_validator(
    validate_string, doc='return a list or strings')
validate_int = _make_type_validator(int)
validate_int_or_None = _make_type_validator(int, allow_none=True)
validate_float = _make_type_validator(float)
validate_float_or_None = _make_type_validator(float, allow_none=True)
validate_floatlist = _listify_validator(
    validate_float, doc='return a list of floats')


def validate_fonttype(s):
    """
    Confirm that this is a Postscript or PDF font type that we know how to
    convert to.
    """
    fonttypes = {'type3':    3,
                 'truetype': 42}
    try:
        fonttype = validate_int(s)
    except ValueError:
        try:
            return fonttypes[s.lower()]
        except KeyError as e:
            raise ValueError(
                'Supported Postscript/PDF font types are %s' % list(fonttypes)) from e
    else:
        if fonttype not in fonttypes.values():
            raise ValueError(
                'Supported Postscript/PDF font types are %s' %
                list(fonttypes.values()))
        return fonttype


_validate_standard_backends = ValidateInStrings(
    'backend', all_backends, ignorecase=True)
_auto_backend_sentinel = object()


def validate_backend(s):
    backend = (
        s if s is _auto_backend_sentinel or s.startswith("module://")
        else _validate_standard_backends(s))
    return backend


validate_toolbar = ValidateInStrings(
    'toolbar', ['None', 'toolbar2', 'toolmanager'], ignorecase=True,
    _deprecated_since="3.3")


def _make_nseq_validator(cls, n=None, allow_none=False):

    def validator(s):
        """Convert *n* objects using ``cls``, or raise."""
        if isinstance(s, str):
            s = [x.strip() for x in s.split(',')]
            if n is not None and len(s) != n:
                raise ValueError(
                    f'Expected exactly {n} comma-separated values, '
                    f'but got {len(s)} comma-separated values: {s}')
        else:
            if n is not None and len(s) != n:
                raise ValueError(
                    f'Expected exactly {n} values, '
                    f'but got {len(s)} values: {s}')
        try:
            return [cls(val) if not allow_none or val is not None else val
                    for val in s]
        except ValueError as e:
            raise ValueError(
                f'Could not convert all entries to {cls.__name__}s') from e

    return validator


validate_nseq_float = partial(_make_nseq_validator, float)
validate_nseq_int = partial(_make_nseq_validator, int)


def validate_color_or_inherit(s):
    """Return a valid color arg."""
    if s == 'inherit':
        return s
    return validate_color(s)


def validate_color_or_auto(s):
    if s == 'auto':
        return s
    return validate_color(s)


def validate_color_for_prop_cycle(s):
    # N-th color cycle syntax can't go into the color cycle.
    if isinstance(s, str) and re.match("^C[0-9]$", s):
        raise ValueError(f"Cannot put cycle reference ({s!r}) in prop_cycler")
    return validate_color(s)


def validate_color(s):
    """Return a valid color arg."""
    if isinstance(s, str):
        if s.lower() == 'none':
            return 'none'
        if len(s) == 6 or len(s) == 8:
            stmp = '#' + s
            if is_color_like(stmp):
                return stmp

    if is_color_like(s):
        return s

    # If it is still valid, it must be a tuple (as a string from matplotlibrc).
    try:
        color = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        pass
    else:
        if is_color_like(color):
            return color

    raise ValueError(f'{s!r} does not look like a color arg')


validate_colorlist = _listify_validator(
    validate_color, allow_stringlist=True, doc='return a list of colorspecs')
validate_orientation = ValidateInStrings(
    'orientation', ['landscape', 'portrait'], _deprecated_since="3.3")


def validate_aspect(s):
    if s in ('auto', 'equal'):
        return s
    try:
        return float(s)
    except ValueError as e:
        raise ValueError('not a valid aspect specification') from e


def validate_fontsize_None(s):
    if s is None or s == 'None':
        return None
    else:
        return validate_fontsize(s)


def validate_fontsize(s):
    fontsizes = ['xx-small', 'x-small', 'small', 'medium', 'large',
                 'x-large', 'xx-large', 'smaller', 'larger']
    if isinstance(s, str):
        s = s.lower()
    if s in fontsizes:
        return s
    try:
        return float(s)
    except ValueError as e:
        raise ValueError("%s is not a valid font size. Valid font sizes "
                         "are %s." % (s, ", ".join(fontsizes))) from e


validate_fontsizelist = _listify_validator(validate_fontsize)


def validate_fontweight(s):
    weights = [
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman',
        'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black']
    # Note: Historically, weights have been case-sensitive in Matplotlib
    if s in weights:
        return s
    try:
        return int(s)
    except (ValueError, TypeError) as e:
        raise ValueError(f'{s} is not a valid font weight.') from e


def validate_font_properties(s):
    parse_fontconfig_pattern(s)
    return s


def _validate_mathtext_fallback_to_cm(b):
    """
    Temporary validate for fallback_to_cm, while deprecated

    """
    if isinstance(b, str):
        b = b.lower()
    if b is None or b == 'none':
        return None
    else:
        cbook.warn_deprecated(
            "3.3", message="Support for setting the 'mathtext.fallback_to_cm' rcParam "
            "is deprecated since %(since)s and will be removed "
            "%(removal)s; use 'mathtext.fallback : 'cm' instead.")
        return validate_bool_maybe_none(b)


def _validate_mathtext_fallback(s):
    _fallback_fonts = ['cm', 'stix', 'stixsans']
    if isinstance(s, str):
        s = s.lower()
    if s is None or s == 'none':
        return None
    elif s.lower() in _fallback_fonts:
        return s
    else:
        raise ValueError(f"{s} is not a valid fallback font name. Valid fallback "
                         f"font names are {','.join(_fallback_fonts)}. Passing "
                         f"'None' will turn fallback off.")


validate_fontset = ValidateInStrings(
    'fontset',
    ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom'],
    _deprecated_since="3.3")
validate_mathtext_default = ValidateInStrings(
    'default', "rm cal it tt sf bf default bb frak scr regular".split(),
    _deprecated_since="3.3")
_validate_alignment = ValidateInStrings(
    'alignment',
    ['center', 'top', 'bottom', 'baseline', 'center_baseline'],
    _deprecated_since="3.3")


def validate_whiskers(s):
    if s == 'range':
        cbook.warn_deprecated(
            "3.2", message="Support for setting the boxplot.whiskers rcParam "
            "to 'range' is deprecated since %(since)s and will be removed "
            "%(removal)s; set it to 0, 100 instead.")
        return 'range'
    else:
        try:
            v = validate_nseq_float(2)(s)
            return v
        except (TypeError, ValueError):
            try:
                v = float(s)
                return v
            except ValueError as e:
                raise ValueError("Not a valid whisker value ['range', float, "
                                 "(float, float)]") from e


@cbook.deprecated("3.2")
def update_savefig_format(value):
    # The old savefig.extension could also have a value of "auto", but
    # the new savefig.format does not.  We need to fix this here.
    value = validate_string(value)
    if value == 'auto':
        cbook.warn_deprecated(
            "3.2", message="Support for setting the 'savefig.format' rcParam "
            "to 'auto' is deprecated since %(since)s and will be removed "
            "%(removal)s; set it to 'png' instead.")
        value = 'png'
    return value


# Replace by validate_string once deprecation period passes.
def _update_savefig_format(value):
    # The old savefig.extension could also have a value of "auto", but
    # the new savefig.format does not.  We need to fix this here.
    value = validate_string(value)
    if value == 'auto':
        cbook.warn_deprecated(
            "3.2", message="Support for setting the 'savefig.format' rcParam "
            "to 'auto' is deprecated since %(since)s and will be removed "
            "%(removal)s; set it to 'png' instead.")
        value = 'png'
    return value


validate_ps_papersize = ValidateInStrings(
    'ps_papersize',
    ['auto', 'letter', 'legal', 'ledger',
     'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10',
     'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10',
     ], ignorecase=True, _deprecated_since="3.3")


def validate_ps_distiller(s):
    if isinstance(s, str):
        s = s.lower()
    if s in ('none', None, 'false', False):
        return None
    else:
        return ValidateInStrings('ps.usedistiller', ['ghostscript', 'xpdf'])(s)


# A validator dedicated to the named line styles, based on the items in
# ls_mapper, and a list of possible strings read from Line2D.set_linestyle
_validate_named_linestyle = ValidateInStrings(
    'linestyle',
    [*ls_mapper.keys(), *ls_mapper.values(), 'None', 'none', ' ', ''],
    ignorecase=True)


def _validate_linestyle(ls):
    """
    A validator for all possible line styles, the named ones *and*
    the on-off ink sequences.
    """
    if isinstance(ls, str):
        try:  # Look first for a valid named line style, like '--' or 'solid'.
            return _validate_named_linestyle(ls)
        except ValueError:
            pass
        try:
            ls = ast.literal_eval(ls)  # Parsing matplotlibrc.
        except (SyntaxError, ValueError):
            pass  # Will error with the ValueError at the end.

    def _is_iterable_not_string_like(x):
        # Explicitly exclude bytes/bytearrays so that they are not
        # nonsensically interpreted as sequences of numbers (codepoints).
        return np.iterable(x) and not isinstance(x, (str, bytes, bytearray))

    # (offset, (on, off, on, off, ...))
    if (_is_iterable_not_string_like(ls)
            and len(ls) == 2
            and isinstance(ls[0], (type(None), Number))
            and _is_iterable_not_string_like(ls[1])
            and len(ls[1]) % 2 == 0
            and all(isinstance(elem, Number) for elem in ls[1])):
        if ls[0] is None:
            cbook.warn_deprecated(
                "3.3", message="Passing the dash offset as None is deprecated "
                "since %(since)s and support for it will be removed "
                "%(removal)s; pass it as zero instead.")
            ls = (0, ls[1])
        return ls
    # For backcompat: (on, off, on, off, ...); the offset is implicitly None.
    if (_is_iterable_not_string_like(ls)
            and len(ls) % 2 == 0
            and all(isinstance(elem, Number) for elem in ls)):
        return (0, ls)
    raise ValueError(f"linestyle {ls!r} is not a valid on-off ink sequence.")


def _deprecate_case_insensitive_join_cap(s):
    s_low = s.lower()
    if s != s_low:
        if s_low in ['miter', 'round', 'bevel']:
            cbook.warn_deprecated(
                "3.3", message="Case-insensitive capstyles are deprecated "
                "since %(since)s and support for them will be removed "
                "%(removal)s; please pass them in lowercase.")
        elif s_low in ['butt', 'round', 'projecting']:
            cbook.warn_deprecated(
                "3.3", message="Case-insensitive joinstyles are deprecated "
                "since %(since)s and support for them will be removed "
                "%(removal)s; please pass them in lowercase.")
        # Else, error out at the check_in_list stage.
    return s_low


def validate_joinstyle(s):
    s = _deprecate_case_insensitive_join_cap(s)
    cbook._check_in_list(['miter', 'round', 'bevel'], joinstyle=s)
    return s


def validate_capstyle(s):
    s = _deprecate_case_insensitive_join_cap(s)
    cbook._check_in_list(['butt', 'round', 'projecting'], capstyle=s)
    return s


validate_fillstyle = ValidateInStrings(
    'markers.fillstyle', ['full', 'left', 'right', 'bottom', 'top', 'none'])


validate_joinstylelist = _listify_validator(validate_joinstyle)
validate_capstylelist = _listify_validator(validate_capstyle)
validate_fillstylelist = _listify_validator(validate_fillstyle)


def validate_markevery(s):
    """
    Validate the markevery property of a Line2D object.

    Parameters
    ----------
    s : None, int, float, slice, length-2 tuple of ints,
        length-2 tuple of floats, list of ints

    Returns
    -------
    None, int, float, slice, length-2 tuple of ints,
        length-2 tuple of floats, list of ints

    """
    # Validate s against type slice float int and None
    if isinstance(s, (slice, float, int, type(None))):
        return s
    # Validate s against type tuple
    if isinstance(s, tuple):
        if (len(s) == 2
                and (all(isinstance(e, int) for e in s)
                     or all(isinstance(e, float) for e in s))):
            return s
        else:
            raise TypeError(
                "'markevery' tuple must be pair of ints or of floats")
    # Validate s against type list
    if isinstance(s, list):
        if all(isinstance(e, int) for e in s):
            return s
        else:
            raise TypeError(
                "'markevery' list must have all elements of type int")
    raise TypeError("'markevery' is of an invalid type")


validate_markeverylist = _listify_validator(validate_markevery)

validate_legend_loc = ValidateInStrings(
    'legend_loc',
    ['best',
     'upper right',
     'upper left',
     'lower left',
     'lower right',
     'right',
     'center left',
     'center right',
     'lower center',
     'upper center',
     'center'], ignorecase=True, _deprecated_since="3.3")

validate_svg_fonttype = ValidateInStrings(
    'svg.fonttype', ['none', 'path'], _deprecated_since="3.3")


@cbook.deprecated("3.3")
def validate_hinting(s):
    return _validate_hinting(s)


# Replace by plain list in _prop_validators after deprecation period.
def _validate_hinting(s):
    if s in (True, False):
        cbook.warn_deprecated(
            "3.2", message="Support for setting the text.hinting rcParam to "
            "True or False is deprecated since %(since)s and will be removed "
            "%(removal)s; set it to its synonyms 'auto' or 'none' instead.")
        return s
    if s.lower() in ('auto', 'native', 'either', 'none'):
        return s.lower()
    raise ValueError("hinting should be 'auto', 'native', 'either' or 'none'")


validate_pgf_texsystem = ValidateInStrings(
    'pgf.texsystem', ['xelatex', 'lualatex', 'pdflatex'],
    _deprecated_since="3.3")


@cbook.deprecated("3.3")
def validate_movie_writer(s):
    # writers.list() would only list actually available writers, but
    # FFMpeg.isAvailable is slow and not worth paying for at every import.
    if s in animation.writers._registered:
        return s
    else:
        raise ValueError(f"Supported animation writers are "
                         f"{sorted(animation.writers._registered)}")


validate_movie_frame_fmt = ValidateInStrings(
    'animation.frame_format', ['png', 'jpeg', 'tiff', 'raw', 'rgba'],
    _deprecated_since="3.3")
validate_axis_locator = ValidateInStrings(
    'major', ['minor', 'both', 'major'], _deprecated_since="3.3")
validate_movie_html_fmt = ValidateInStrings(
    'animation.html', ['html5', 'jshtml', 'none'], _deprecated_since="3.3")


def validate_bbox(s):
    if isinstance(s, str):
        s = s.lower()
        if s == 'tight':
            return s
        if s == 'standard':
            return None
        raise ValueError("bbox should be 'tight' or 'standard'")
    elif s is not None:
        # Backwards compatibility. None is equivalent to 'standard'.
        raise ValueError("bbox should be 'tight' or 'standard'")
    return s


def validate_sketch(s):
    if isinstance(s, str):
        s = s.lower()
    if s == 'none' or s is None:
        return None
    try:
        return tuple(validate_nseq_float(3)(s))
    except ValueError:
        raise ValueError("Expected a (scale, length, randomness) triplet")


def _validate_greaterequal0_lessthan1(s):
    s = validate_float(s)
    if 0 <= s < 1:
        return s
    else:
        raise RuntimeError(f'Value must be >=0 and <1; got {s}')


def _validate_greaterequal0_lessequal1(s):
    s = validate_float(s)
    if 0 <= s <= 1:
        return s
    else:
        raise RuntimeError(f'Value must be >=0 and <=1; got {s}')


_range_validators = {  # Slightly nicer (internal) API.
    "0 <= x < 1": _validate_greaterequal0_lessthan1,
    "0 <= x <= 1": _validate_greaterequal0_lessequal1,
}


validate_grid_axis = ValidateInStrings(
    'axes.grid.axis', ['x', 'y', 'both'], _deprecated_since="3.3")


def validate_hatch(s):
    r"""
    Validate a hatch pattern.
    A hatch pattern string can have any sequence of the following
    characters: ``\ / | - + * . x o O``.
    """
    if not isinstance(s, str):
        raise ValueError("Hatch pattern must be a string")
    cbook._check_isinstance(str, hatch_pattern=s)
    unknown = set(s) - {'\\', '/', '|', '-', '+', '*', '.', 'x', 'o', 'O'}
    if unknown:
        raise ValueError("Unknown hatch symbol(s): %s" % list(unknown))
    return s


validate_hatchlist = _listify_validator(validate_hatch)
validate_dashlist = _listify_validator(validate_nseq_float(allow_none=True))


_prop_validators = {
        'color': _listify_validator(validate_color_for_prop_cycle,
                                    allow_stringlist=True),
        'linewidth': validate_floatlist,
        'linestyle': _listify_validator(_validate_linestyle),
        'facecolor': validate_colorlist,
        'edgecolor': validate_colorlist,
        'joinstyle': validate_joinstylelist,
        'capstyle': validate_capstylelist,
        'fillstyle': validate_fillstylelist,
        'markerfacecolor': validate_colorlist,
        'markersize': validate_floatlist,
        'markeredgewidth': validate_floatlist,
        'markeredgecolor': validate_colorlist,
        'markevery': validate_markeverylist,
        'alpha': validate_floatlist,
        'marker': validate_stringlist,
        'hatch': validate_hatchlist,
        'dashes': validate_dashlist,
    }
_prop_aliases = {
        'c': 'color',
        'lw': 'linewidth',
        'ls': 'linestyle',
        'fc': 'facecolor',
        'ec': 'edgecolor',
        'mfc': 'markerfacecolor',
        'mec': 'markeredgecolor',
        'mew': 'markeredgewidth',
        'ms': 'markersize',
    }


def cycler(*args, **kwargs):
    """
    Create a `~cycler.Cycler` object much like :func:`cycler.cycler`,
    but includes input validation.

    Call signatures::

      cycler(cycler)
      cycler(label=values[, label2=values2[, ...]])
      cycler(label, values)

    Form 1 copies a given `~cycler.Cycler` object.

    Form 2 creates a `~cycler.Cycler` which cycles over one or more
    properties simultaneously. If multiple properties are given, their
    value lists must have the same length.

    Form 3 creates a `~cycler.Cycler` for a single property. This form
    exists for compatibility with the original cycler. Its use is
    discouraged in favor of the kwarg form, i.e. ``cycler(label=values)``.

    Parameters
    ----------
    cycler : Cycler
        Copy constructor for Cycler.

    label : str
        The property key. Must be a valid `.Artist` property.
        For example, 'color' or 'linestyle'. Aliases are allowed,
        such as 'c' for 'color' and 'lw' for 'linewidth'.

    values : iterable
        Finite-length iterable of the property values. These values
        are validated and will raise a ValueError if invalid.

    Returns
    -------
    Cycler
        A new :class:`~cycler.Cycler` for the given properties.

    Examples
    --------
    Creating a cycler for a single property:

    >>> c = cycler(color=['red', 'green', 'blue'])

    Creating a cycler for simultaneously cycling over multiple properties
    (e.g. red circle, green plus, blue cross):

    >>> c = cycler(color=['red', 'green', 'blue'],
    ...            marker=['o', '+', 'x'])

    """
    if args and kwargs:
        raise TypeError("cycler() can only accept positional OR keyword "
                        "arguments -- not both.")
    elif not args and not kwargs:
        raise TypeError("cycler() must have positional OR keyword arguments")

    if len(args) == 1:
        if not isinstance(args[0], Cycler):
            raise TypeError("If only one positional argument given, it must "
                            " be a Cycler instance.")
        return validate_cycler(args[0])
    elif len(args) == 2:
        pairs = [(args[0], args[1])]
    elif len(args) > 2:
        raise TypeError("No more than 2 positional arguments allowed")
    else:
        pairs = kwargs.items()

    validated = []
    for prop, vals in pairs:
        norm_prop = _prop_aliases.get(prop, prop)
        validator = _prop_validators.get(norm_prop, None)
        if validator is None:
            raise TypeError("Unknown artist property: %s" % prop)
        vals = validator(vals)
        # We will normalize the property names as well to reduce
        # the amount of alias handling code elsewhere.
        validated.append((norm_prop, vals))

    return reduce(operator.add, (ccycler(k, v) for k, v in validated))


def validate_cycler(s):
    """Return a Cycler object from a string repr or the object itself."""
    if isinstance(s, str):
        try:
            # TODO: We might want to rethink this...
            # While I think I have it quite locked down,
            # it is execution of arbitrary code without
            # sanitation.
            # Combine this with the possibility that rcparams
            # might come from the internet (future plans), this
            # could be downright dangerous.
            # I locked it down by only having the 'cycler()' function
            # available.
            # UPDATE: Partly plugging a security hole.
            # I really should have read this:
            # http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
            # We should replace this eval with a combo of PyParsing and
            # ast.literal_eval()
            if '.__' in s.replace(' ', ''):
                raise ValueError("'%s' seems to have dunder methods. Raising"
                                 " an exception for your safety")
            s = eval(s, {'cycler': cycler, '__builtins__': {}})
        except BaseException as e:
            raise ValueError("'%s' is not a valid cycler construction: %s" %
                             (s, e)) from e
    # Should make sure what comes from the above eval()
    # is a Cycler object.
    if isinstance(s, Cycler):
        cycler_inst = s
    else:
        raise ValueError("object was not a string or Cycler instance: %s" % s)

    unknowns = cycler_inst.keys - (set(_prop_validators) | set(_prop_aliases))
    if unknowns:
        raise ValueError("Unknown artist properties: %s" % unknowns)

    # Not a full validation, but it'll at least normalize property names
    # A fuller validation would require v0.10 of cycler.
    checker = set()
    for prop in cycler_inst.keys:
        norm_prop = _prop_aliases.get(prop, prop)
        if norm_prop != prop and norm_prop in cycler_inst.keys:
            raise ValueError("Cannot specify both '{0}' and alias '{1}'"
                             " in the same prop_cycle".format(norm_prop, prop))
        if norm_prop in checker:
            raise ValueError("Another property was already aliased to '{0}'."
                             " Collision normalizing '{1}'.".format(norm_prop,
                                                                    prop))
        checker.update([norm_prop])

    # This is just an extra-careful check, just in case there is some
    # edge-case I haven't thought of.
    assert len(checker) == len(cycler_inst.keys)

    # Now, it should be safe to mutate this cycler
    for prop in cycler_inst.keys:
        norm_prop = _prop_aliases.get(prop, prop)
        cycler_inst.change_key(prop, norm_prop)

    for key, vals in cycler_inst.by_key().items():
        _prop_validators[key](vals)

    return cycler_inst


def validate_hist_bins(s):
    valid_strs = ["auto", "sturges", "fd", "doane", "scott", "rice", "sqrt"]
    if isinstance(s, str) and s in valid_strs:
        return s
    try:
        return int(s)
    except (TypeError, ValueError):
        pass
    try:
        return validate_floatlist(s)
    except ValueError:
        pass
    raise ValueError("'hist.bins' must be one of {}, an int or"
                     " a sequence of floats".format(valid_strs))


@cbook.deprecated("3.2")
def validate_animation_writer_path(p):
    # Make sure it's a string and then figure out if the animations
    # are already loaded and reset the writers (which will validate
    # the path on next call)
    cbook._check_isinstance(str, path=p)
    from sys import modules
    # set dirty, so that the next call to the registry will re-evaluate
    # the state.
    # only set dirty if already loaded. If not loaded, the load will
    # trigger the checks.
    if "matplotlib.animation" in modules:
        modules["matplotlib.animation"].writers.set_dirty()
    return p


@cbook.deprecated("3.3")
def validate_webagg_address(s):
    if s is not None:
        import socket
        try:
            socket.inet_aton(s)
        except socket.error as e:
            raise ValueError("'webagg.address' is not a valid IP address") from e
        return s
    raise ValueError("'webagg.address' is not a valid IP address")


validate_axes_titlelocation = ValidateInStrings(
    'axes.titlelocation', ['left', 'center', 'right'], _deprecated_since="3.3")


class _ignorecase(list):
    """A marker class indicating that a list-of-str is case-insensitive."""


def _convert_validator_spec(key, conv):
    if isinstance(conv, list):
        ignorecase = isinstance(conv, _ignorecase)
        return ValidateInStrings(key, conv, ignorecase=ignorecase)
    else:
        return conv


# A map of key -> [value, converter].
# Converters given as lists or _ignorecase are converted to ValidateInStrings
# immediately below.
defaultParams = {
    'backend':           [_auto_backend_sentinel, validate_backend],
    'backend_fallback':  [True, validate_bool],
    'webagg.port':       [8988, validate_int],
    'webagg.address':    ['127.0.0.1', validate_string],
    'webagg.open_in_browser': [True, validate_bool],
    'webagg.port_retries': [50, validate_int],
    'toolbar':           ['toolbar2', _ignorecase(['none', 'toolbar2', 'toolmanager'])],
    'datapath':          [None, validate_any],  # see _get_data_path_cached
    'interactive':       [False, validate_bool],
    'timezone':          ['UTC', validate_string],

    # line props
    'lines.linewidth':       [1.5, validate_float],  # line width in points
    'lines.linestyle':       ['-', _validate_linestyle],  # solid line
    'lines.color':           ['C0', validate_color],  # first color in color cycle
    'lines.marker':          ['None', validate_string],  # marker name
    'lines.markerfacecolor': ['auto', validate_color_or_auto],  # default color
    'lines.markeredgecolor': ['auto', validate_color_or_auto],  # default color
    'lines.markeredgewidth': [1.0, validate_float],
    'lines.markersize':      [6, validate_float],    # markersize, in points
    'lines.antialiased':     [True, validate_bool],  # antialiased (no jaggies)
    'lines.dash_joinstyle':  ['round', validate_joinstyle],
    'lines.solid_joinstyle': ['round', validate_joinstyle],
    'lines.dash_capstyle':   ['butt', validate_capstyle],
    'lines.solid_capstyle':  ['projecting', validate_capstyle],
    'lines.dashed_pattern':  [[3.7, 1.6], validate_nseq_float(allow_none=True)],
    'lines.dashdot_pattern': [[6.4, 1.6, 1, 1.6],
                              validate_nseq_float(allow_none=True)],
    'lines.dotted_pattern':  [[1, 1.65], validate_nseq_float(allow_none=True)],
    'lines.scale_dashes':  [True, validate_bool],

    # marker props
    'markers.fillstyle': ['full', validate_fillstyle],

    ## pcolor(mesh) props:
    'pcolor.shading': ['flat', validate_string],  # auto,flat,nearest,gouraud

    ## patch props
    'patch.linewidth':   [1.0, validate_float],     # line width in points
    'patch.edgecolor':   ['black', validate_color],
    'patch.force_edgecolor': [False, validate_bool],
    'patch.facecolor':   ['C0', validate_color],    # first color in cycle
    'patch.antialiased': [True, validate_bool],     # antialiased (no jaggies)

    ## hatch props
    'hatch.color': ['black', validate_color],
    'hatch.linewidth': [1.0, validate_float],

    ## Histogram properties
    'hist.bins': [10, validate_hist_bins],

    ## Boxplot properties
    'boxplot.notch': [False, validate_bool],
    'boxplot.vertical': [True, validate_bool],
    'boxplot.whiskers': [1.5, validate_whiskers],
    'boxplot.bootstrap': [None, validate_int_or_None],
    'boxplot.patchartist': [False, validate_bool],
    'boxplot.showmeans': [False, validate_bool],
    'boxplot.showcaps': [True, validate_bool],
    'boxplot.showbox': [True, validate_bool],
    'boxplot.showfliers': [True, validate_bool],
    'boxplot.meanline': [False, validate_bool],

    'boxplot.flierprops.color': ['black', validate_color],
    'boxplot.flierprops.marker': ['o', validate_string],
    'boxplot.flierprops.markerfacecolor': ['none', validate_color_or_auto],
    'boxplot.flierprops.markeredgecolor': ['black', validate_color],
    'boxplot.flierprops.markeredgewidth': [1.0, validate_float],
    'boxplot.flierprops.markersize': [6, validate_float],
    'boxplot.flierprops.linestyle': ['none', _validate_linestyle],
    'boxplot.flierprops.linewidth': [1.0, validate_float],

    'boxplot.boxprops.color': ['black', validate_color],
    'boxplot.boxprops.linewidth': [1.0, validate_float],
    'boxplot.boxprops.linestyle': ['-', _validate_linestyle],

    'boxplot.whiskerprops.color': ['black', validate_color],
    'boxplot.whiskerprops.linewidth': [1.0, validate_float],
    'boxplot.whiskerprops.linestyle': ['-', _validate_linestyle],

    'boxplot.capprops.color': ['black', validate_color],
    'boxplot.capprops.linewidth': [1.0, validate_float],
    'boxplot.capprops.linestyle': ['-', _validate_linestyle],

    'boxplot.medianprops.color': ['C1', validate_color],
    'boxplot.medianprops.linewidth': [1.0, validate_float],
    'boxplot.medianprops.linestyle': ['-', _validate_linestyle],

    'boxplot.meanprops.color': ['C2', validate_color],
    'boxplot.meanprops.marker': ['^', validate_string],
    'boxplot.meanprops.markerfacecolor': ['C2', validate_color],
    'boxplot.meanprops.markeredgecolor': ['C2', validate_color],
    'boxplot.meanprops.markersize': [6, validate_float],
    'boxplot.meanprops.linestyle': ['--', _validate_linestyle],
    'boxplot.meanprops.linewidth': [1.0, validate_float],

    ## font props
    'font.family':     [['sans-serif'], validate_stringlist],  # used by text object
    'font.style':      ['normal', validate_string],
    'font.variant':    ['normal', validate_string],
    'font.stretch':    ['normal', validate_string],
    'font.weight':     ['normal', validate_fontweight],
    'font.size':       [10, validate_float],      # Base font size in points
    'font.serif':      [['DejaVu Serif', 'Bitstream Vera Serif',
                         'Computer Modern Roman',
                         'New Century Schoolbook', 'Century Schoolbook L',
                         'Utopia', 'ITC Bookman', 'Bookman',
                         'Nimbus Roman No9 L', 'Times New Roman',
                         'Times', 'Palatino', 'Charter', 'serif'],
                        validate_stringlist],
    'font.sans-serif': [['DejaVu Sans', 'Bitstream Vera Sans',
                         'Computer Modern Sans Serif',
                         'Lucida Grande', 'Verdana', 'Geneva', 'Lucid',
                         'Arial', 'Helvetica', 'Avant Garde', 'sans-serif'],
                        validate_stringlist],
    'font.cursive':    [['Apple Chancery', 'Textile', 'Zapf Chancery',
                         'Sand', 'Script MT', 'Felipa', 'cursive'],
                        validate_stringlist],
    'font.fantasy':    [['Comic Neue', 'Comic Sans MS', 'Chicago', 'Charcoal',
                         'Impact', 'Western', 'Humor Sans', 'xkcd', 'fantasy'],
                        validate_stringlist],
    'font.monospace':  [['DejaVu Sans Mono', 'Bitstream Vera Sans Mono',
                         'Computer Modern Typewriter',
                         'Andale Mono', 'Nimbus Mono L', 'Courier New',
                         'Courier', 'Fixed', 'Terminal', 'monospace'],
                        validate_stringlist],

    # text props
    'text.color':          ['black', validate_color],
    'text.usetex':         [False, validate_bool],
    'text.latex.preamble': ['', _validate_tex_preamble],
    'text.latex.preview':  [False, validate_bool],
    'text.hinting':        ['auto', _validate_hinting],
    'text.hinting_factor': [8, validate_int],
    'text.kerning_factor': [0, validate_int],
    'text.antialiased':    [True, validate_bool],

    'mathtext.cal':            ['cursive', validate_font_properties],
    'mathtext.rm':             ['sans', validate_font_properties],
    'mathtext.tt':             ['monospace', validate_font_properties],
    'mathtext.it':             ['sans:italic', validate_font_properties],
    'mathtext.bf':             ['sans:bold', validate_font_properties],
    'mathtext.sf':             ['sans', validate_font_properties],
    'mathtext.fontset':        [
        'dejavusans',
        ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']],
    'mathtext.default':        [
        'it',
        ['rm', 'cal', 'it', 'tt', 'sf', 'bf', 'default', 'bb', 'frak', 'scr', 'regular']],
    'mathtext.fallback_to_cm': [None, _validate_mathtext_fallback_to_cm],
    'mathtext.fallback':       ['cm', _validate_mathtext_fallback],

    'image.aspect':        ['equal', validate_aspect],  # equal, auto, a number
    'image.interpolation': ['antialiased', validate_string],
    'image.cmap':          ['viridis', validate_string],  # gray, jet, etc.
    'image.lut':           [256, validate_int],  # lookup table
    'image.origin':        ['upper', ['upper', 'lower']],
    'image.resample':      [True, validate_bool],
    # Specify whether vector graphics backends will combine all images on a
    # set of axes into a single composite image
    'image.composite_image': [True, validate_bool],

    # contour props
    'contour.negative_linestyle': ['dashed', _validate_linestyle],
    'contour.corner_mask':        [True, validate_bool],

    # errorbar props
    'errorbar.capsize':      [0, validate_float],

    # axis props
    'xaxis.labellocation':    ['center', ['left', 'center', 'right']],  # alignment of x axis title
    'yaxis.labellocation':    ['center', ['bottom', 'center', 'top']],  # alignment of y axis title

    # axes props
    'axes.axisbelow':        ['line', validate_axisbelow],
    'axes.facecolor':        ['white', validate_color],  # background color
    'axes.edgecolor':        ['black', validate_color],  # edge color
    'axes.linewidth':        [0.8, validate_float],  # edge linewidth

    'axes.spines.left':      [True, validate_bool],  # Set visibility of axes
    'axes.spines.right':     [True, validate_bool],  # 'spines', the lines
    'axes.spines.bottom':    [True, validate_bool],  # around the chart
    'axes.spines.top':       [True, validate_bool],  # denoting data boundary

    'axes.titlesize':        ['large', validate_fontsize],  # fontsize of the
                                                            # axes title
    'axes.titlelocation':    ['center', ['left', 'center', 'right']],  # alignment of axes title
    'axes.titleweight':      ['normal', validate_fontweight],  # font weight of axes title
    'axes.titlecolor':       ['auto', validate_color_or_auto],  # font color of axes title
    'axes.titley':           [None, validate_float_or_None],  # title location, axes units, None means auto
    'axes.titlepad':         [6.0, validate_float],  # pad from axes top decoration to title in points
    'axes.grid':             [False, validate_bool],   # display grid or not
    'axes.grid.which':       ['major', ['minor', 'both', 'major']],  # set whether the grid is drawn on
                                                                     # 'major' 'minor' or 'both' ticks
    'axes.grid.axis':        ['both', ['x', 'y', 'both']],  # grid type:
                                                            # 'x', 'y', or 'both'
    'axes.labelsize':        ['medium', validate_fontsize],  # fontsize of the
                                                             # x any y labels
    'axes.labelpad':         [4.0, validate_float],  # space between label and axis
    'axes.labelweight':      ['normal', validate_fontweight],  # fontsize of the x any y labels
    'axes.labelcolor':       ['black', validate_color],    # color of axis label
    'axes.formatter.limits': [[-5, 6], validate_nseq_int(2)],
                               # use scientific notation if log10
                               # of the axis range is smaller than the
                               # first or larger than the second
    'axes.formatter.use_locale': [False, validate_bool],
                               # Use the current locale to format ticks
    'axes.formatter.use_mathtext': [False, validate_bool],
    'axes.formatter.min_exponent': [0, validate_int],  # minimum exponent to format in scientific notation
    'axes.formatter.useoffset': [True, validate_bool],
    'axes.formatter.offset_threshold': [4, validate_int],
    'axes.unicode_minus': [True, validate_bool],
    # This entry can be either a cycler object or a
    # string repr of a cycler-object, which gets eval()'ed
    # to create the object.
    'axes.prop_cycle': [
        ccycler('color',
                ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                 '#bcbd22', '#17becf']),
        validate_cycler],
    # If 'data', axes limits are set close to the data.
    # If 'round_numbers' axes limits are set to the nearest round numbers.
    'axes.autolimit_mode': ['data', ['data', 'round_numbers']],
    'axes.xmargin': [0.05, _range_validators["0 <= x <= 1"]],
    'axes.ymargin': [0.05, _range_validators["0 <= x <= 1"]],

    'polaraxes.grid': [True, validate_bool],  # display polar grid or not
    'axes3d.grid': [True, validate_bool],  # display 3d grid

    # scatter props
    'scatter.marker': ['o', validate_string],
    'scatter.edgecolors': ['face', validate_string],

    'date.epoch': ['1970-01-01T00:00', validate_date],
    # TODO validate that these are valid datetime format strings
    'date.autoformatter.year': ['%Y', validate_string],
    'date.autoformatter.month': ['%Y-%m', validate_string],
    'date.autoformatter.day': ['%Y-%m-%d', validate_string],
    'date.autoformatter.hour': ['%m-%d %H', validate_string],
    'date.autoformatter.minute': ['%d %H:%M', validate_string],
    'date.autoformatter.second': ['%H:%M:%S', validate_string],
    'date.autoformatter.microsecond': ['%M:%S.%f', validate_string],

    #legend properties
    'legend.fancybox': [True, validate_bool],
    'legend.loc': ['best',
                   _ignorecase(['best',
                                'upper right', 'upper left',
                                'lower left', 'lower right', 'right',
                                'center left', 'center right',
                                'lower center', 'upper center',
                                'center'])],
    # the number of points in the legend line
    'legend.numpoints': [1, validate_int],
    # the number of points in the legend line for scatter
    'legend.scatterpoints': [1, validate_int],
    'legend.fontsize': ['medium', validate_fontsize],
    'legend.title_fontsize': [None, validate_fontsize_None],
     # the relative size of legend markers vs. original
    'legend.markerscale': [1.0, validate_float],
    'legend.shadow': [False, validate_bool],
     # whether or not to draw a frame around legend
    'legend.frameon': [True, validate_bool],
     # alpha value of the legend frame
    'legend.framealpha': [0.8, validate_float_or_None],

    ## the following dimensions are in fraction of the font size
    'legend.borderpad': [0.4, validate_float],  # units are fontsize
    # the vertical space between the legend entries
    'legend.labelspacing': [0.5, validate_float],
    # the length of the legend lines
    'legend.handlelength': [2., validate_float],
    # the length of the legend lines
    'legend.handleheight': [0.7, validate_float],
    # the space between the legend line and legend text
    'legend.handletextpad': [.8, validate_float],
    # the border between the axes and legend edge
    'legend.borderaxespad': [0.5, validate_float],
    # the border between the axes and legend edge
    'legend.columnspacing': [2., validate_float],
    'legend.facecolor': ['inherit', validate_color_or_inherit],
    'legend.edgecolor': ['0.8', validate_color_or_inherit],

    # tick properties
    'xtick.top':         [False, validate_bool],   # draw ticks on the top side
    'xtick.bottom':      [True, validate_bool],   # draw ticks on the bottom side
    'xtick.labeltop':    [False, validate_bool],  # draw label on the top
    'xtick.labelbottom': [True, validate_bool],  # draw label on the bottom
    'xtick.major.size':  [3.5, validate_float],    # major xtick size in points
    'xtick.minor.size':  [2, validate_float],    # minor xtick size in points
    'xtick.major.width': [0.8, validate_float],  # major xtick width in points
    'xtick.minor.width': [0.6, validate_float],  # minor xtick width in points
    'xtick.major.pad':   [3.5, validate_float],    # distance to label in points
    'xtick.minor.pad':   [3.4, validate_float],    # distance to label in points
    'xtick.color':       ['black', validate_color],  # color of the xtick labels
    'xtick.minor.visible':   [False, validate_bool],    # visibility of the x axis minor ticks
    'xtick.minor.top':   [True, validate_bool],  # draw x axis top minor ticks
    'xtick.minor.bottom':    [True, validate_bool],    # draw x axis bottom minor ticks
    'xtick.major.top':   [True, validate_bool],  # draw x axis top major ticks
    'xtick.major.bottom':    [True, validate_bool],    # draw x axis bottom major ticks

    # fontsize of the xtick labels
    'xtick.labelsize':   ['medium', validate_fontsize],
    'xtick.direction':   ['out', validate_string],            # direction of xticks
    'xtick.alignment':   ['center',
                          ['center', 'top', 'bottom', 'baseline', 'center_baseline']],

    'ytick.left':        [True, validate_bool],  # draw ticks on the left side
    'ytick.right':       [False, validate_bool],  # draw ticks on the right side
    'ytick.labelleft':   [True, validate_bool],  # draw tick labels on the left side
    'ytick.labelright':  [False, validate_bool],  # draw tick labels on the right side
    'ytick.major.size':  [3.5, validate_float],     # major ytick size in points
    'ytick.minor.size':  [2, validate_float],     # minor ytick size in points
    'ytick.major.width': [0.8, validate_float],   # major ytick width in points
    'ytick.minor.width': [0.6, validate_float],   # minor ytick width in points
    'ytick.major.pad':   [3.5, validate_float],     # distance to label in points
    'ytick.minor.pad':   [3.4, validate_float],     # distance to label in points
    'ytick.color':       ['black', validate_color],   # color of the ytick labels
    'ytick.minor.visible':   [False, validate_bool],    # visibility of the y axis minor ticks
    'ytick.minor.left':   [True, validate_bool],  # draw y axis left minor ticks
    'ytick.minor.right':    [True, validate_bool],    # draw y axis right minor ticks
    'ytick.major.left':   [True, validate_bool],  # draw y axis left major ticks
    'ytick.major.right':    [True, validate_bool],    # draw y axis right major ticks

    # fontsize of the ytick labels
    'ytick.labelsize':   ['medium', validate_fontsize],
    'ytick.direction':   ['out', validate_string],            # direction of yticks
    'ytick.alignment':   ['center_baseline',
                          ['center', 'top', 'bottom', 'baseline', 'center_baseline']],

    'grid.color':        ['#b0b0b0', validate_color],  # grid color
    'grid.linestyle':    ['-', _validate_linestyle],  # solid
    'grid.linewidth':    [0.8, validate_float],     # in points
    'grid.alpha':        [1.0, validate_float],

    ## figure props
    # figure title
    'figure.titlesize':   ['large', validate_fontsize],
    'figure.titleweight': ['normal', validate_fontweight],

    # figure size in inches: width by height
    'figure.figsize':    [[6.4, 4.8], validate_nseq_float(2)],
    'figure.dpi':        [100, validate_float],  # DPI
    'figure.facecolor':  ['white', validate_color],
    'figure.edgecolor':  ['white', validate_color],
    'figure.frameon':    [True, validate_bool],
    'figure.autolayout': [False, validate_bool],
    'figure.max_open_warning': [20, validate_int],
    'figure.raise_window': [True, validate_bool],

    'figure.subplot.left': [0.125, _range_validators["0 <= x <= 1"]],
    'figure.subplot.right': [0.9, _range_validators["0 <= x <= 1"]],
    'figure.subplot.bottom': [0.11, _range_validators["0 <= x <= 1"]],
    'figure.subplot.top': [0.88, _range_validators["0 <= x <= 1"]],
    'figure.subplot.wspace': [0.2, _range_validators["0 <= x < 1"]],
    'figure.subplot.hspace': [0.2, _range_validators["0 <= x < 1"]],

    # do constrained_layout.
    'figure.constrained_layout.use': [False, validate_bool],
    # wspace and hspace are fraction of adjacent subplots to use
    # for space.  Much smaller than above because we don't need
    # room for the text.
    'figure.constrained_layout.hspace':
        [0.02, _range_validators["0 <= x < 1"]],
    'figure.constrained_layout.wspace':
        [0.02, _range_validators["0 <= x < 1"]],
    # This is a buffer around the axes in inches.  This is 3pts.
    'figure.constrained_layout.h_pad': [0.04167, validate_float],
    'figure.constrained_layout.w_pad': [0.04167, validate_float],

    ## Saving figure's properties
    'savefig.dpi':         ['figure', validate_dpi],  # DPI
    'savefig.facecolor':   ['white', validate_color],
    'savefig.edgecolor':   ['white', validate_color],
    'savefig.orientation': ['portrait', ['landscape', 'portrait']],
    'savefig.jpeg_quality': [95, validate_int],
    # value checked by backend at runtime
    'savefig.format':     ['png', _update_savefig_format],
    # options are 'tight', or 'standard'. 'standard' validates to None.
    'savefig.bbox':       ['standard', validate_bbox],
    'savefig.pad_inches': [0.1, validate_float],
    # default directory in savefig dialog box
    'savefig.directory': ['~', validate_string],
    'savefig.transparent': [False, validate_bool],

    # Maintain shell focus for TkAgg
    'tk.window_focus':  [False, validate_bool],

    # Set the papersize/type
    'ps.papersize':     ['letter',
                         _ignorecase(['auto', 'letter', 'legal', 'ledger',
                                      *[f'{ab}{i}' for ab in 'ab' for i in range(11)]])],
    'ps.useafm':        [False, validate_bool],
    # use ghostscript or xpdf to distill ps output
    'ps.usedistiller':  [False, validate_ps_distiller],
    'ps.distiller.res': [6000, validate_int],     # dpi
    'ps.fonttype':      [3, validate_fonttype],  # 3 (Type3) or 42 (Truetype)
    # compression level from 0 to 9; 0 to disable
    'pdf.compression':  [6, validate_int],
    # ignore any color-setting commands from the frontend
    'pdf.inheritcolor': [False, validate_bool],
    # use only the 14 PDF core fonts embedded in every PDF viewing application
    'pdf.use14corefonts': [False, validate_bool],
    'pdf.fonttype':     [3, validate_fonttype],  # 3 (Type3) or 42 (Truetype)

    # choose latex application for creating pdf files (xelatex/lualatex)
    'pgf.texsystem': ['xelatex', ['xelatex', 'lualatex', 'pdflatex']],
    # use matplotlib rc settings for font configuration
    'pgf.rcfonts':   [True, validate_bool],
    # provide a custom preamble for the latex process
    'pgf.preamble':  ['', _validate_tex_preamble],

    # write raster image data directly into the svg file
    'svg.image_inline':     [True, validate_bool],
    # True to save all characters as paths in the SVG
    'svg.fonttype':         ['path', ['none', 'path']],
    'svg.hashsalt':         [None, validate_string_or_None],

    # set this when you want to generate hardcopy docstring
    'docstring.hardcopy': [False, validate_bool],

    'path.simplify': [True, validate_bool],
    'path.simplify_threshold': [1 / 9, _range_validators["0 <= x <= 1"]],
    'path.snap': [True, validate_bool],
    'path.sketch': [None, validate_sketch],
    'path.effects': [[], validate_any],
    'agg.path.chunksize': [0, validate_int],       # 0 to disable chunking;

    # key-mappings (multi-character mappings should be a list/tuple)
    'keymap.fullscreen':   [['f', 'ctrl+f'], validate_stringlist],
    'keymap.home':         [['h', 'r', 'home'], validate_stringlist],
    'keymap.back':         [['left', 'c', 'backspace', 'MouseButton.BACK'],
                            validate_stringlist],
    'keymap.forward':      [['right', 'v', 'MouseButton.FORWARD'],
                            validate_stringlist],
    'keymap.pan':          [['p'], validate_stringlist],
    'keymap.zoom':         [['o'], validate_stringlist],
    'keymap.save':         [['s', 'ctrl+s'], validate_stringlist],
    'keymap.quit':         [['ctrl+w', 'cmd+w', 'q'], validate_stringlist],
    'keymap.quit_all':     [[], validate_stringlist],  # proposed values: 'W', 'cmd+W', 'Q'
    'keymap.grid':         [['g'], validate_stringlist],
    'keymap.grid_minor':   [['G'], validate_stringlist],
    'keymap.yscale':       [['l'], validate_stringlist],
    'keymap.xscale':       [['k', 'L'], validate_stringlist],
    'keymap.all_axes':     [['a'], validate_stringlist],
    'keymap.help':         [['f1'], validate_stringlist],
    'keymap.copy':         [['ctrl+c', 'cmd+c'], validate_stringlist],

    # Animation settings
    'animation.html':         ['none', ['html5', 'jshtml', 'none']],
    # Limit, in MB, of size of base64 encoded animation in HTML
    # (i.e. IPython notebook)
    'animation.embed_limit':  [20, validate_float],
    'animation.writer':       ['ffmpeg', validate_string],
    'animation.codec':        ['h264', validate_string],
    'animation.bitrate':      [-1, validate_int],
    # Controls image format when frames are written to disk
    'animation.frame_format': ['png', ['png', 'jpeg', 'tiff', 'raw', 'rgba']],
    # Additional arguments for HTML writer
    'animation.html_args':    [[], validate_stringlist],
    # Path to ffmpeg binary. If just binary name, subprocess uses $PATH.
    'animation.ffmpeg_path':  ['ffmpeg', validate_string],
    # Additional arguments for ffmpeg movie writer (using pipes)
    'animation.ffmpeg_args':   [[], validate_stringlist],
    # Path to AVConv binary. If just binary name, subprocess uses $PATH.
    'animation.avconv_path':   ['avconv', validate_string],
    # Additional arguments for avconv movie writer (using pipes)
    'animation.avconv_args':   [[], validate_stringlist],
     # Path to convert binary. If just binary name, subprocess uses $PATH.
    'animation.convert_path':  ['convert', validate_string],
     # Additional arguments for convert movie writer (using pipes)
    'animation.convert_args':  [[], validate_stringlist],

    'mpl_toolkits.legacy_colorbar': [True, validate_bool],

    # Classic (pre 2.0) compatibility mode
    # This is used for things that are hard to make backward compatible
    # with a sane rcParam alone.  This does *not* turn on classic mode
    # altogether.  For that use `matplotlib.style.use('classic')`.
    '_internal.classic_mode': [False, validate_bool]
}
defaultParams = {k: [default, _convert_validator_spec(k, conv)]
                 for k, (default, conv) in defaultParams.items()}
