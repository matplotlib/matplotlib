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

from collections.abc import Iterable, Mapping
from functools import partial, reduce
import logging
import operator
import os
import re

import matplotlib as mpl
from matplotlib import cbook
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
    def __init__(self, key, valid, ignorecase=False):
        'valid is a list of legal strings'
        self.key = key
        self.ignorecase = ignorecase

        def func(s):
            if ignorecase:
                return s.lower()
            else:
                return s
        self.valid = {func(k): k for k in valid}

    def __call__(self, s):
        if self.ignorecase:
            s = s.lower()
        if s in self.valid:
            return self.valid[s]
        raise ValueError('Unrecognized %s string %r: valid strings are %s'
                         % (self.key, s, list(self.valid.values())))


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
        # We should allow any generic sequence type, including generators,
        # Numpy ndarrays, and pandas data structures.  However, unordered
        # sequences, such as sets, should be allowed but discouraged unless the
        # user desires pseudorandom behavior.
        elif isinstance(s, Iterable) and not isinstance(s, Mapping):
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
    f.__doc__ = doc if doc is not None else scalar_validator.__doc__
    return f


def validate_any(s):
    return s
validate_anylist = _listify_validator(validate_any)


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
    """Convert b to a boolean or raise"""
    if isinstance(b, str):
        b = b.lower()
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError('Could not convert "%s" to boolean' % b)


def validate_bool_maybe_none(b):
    """Convert b to a boolean or raise."""
    if isinstance(b, str):
        b = b.lower()
    if b is None or b == 'none':
        return None
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError('Could not convert "%s" to boolean' % b)


def _validate_tex_preamble(s):
    if s is None or s == 'None':
        return ""
    try:
        if isinstance(s, str):
            return s
        elif isinstance(s, Iterable):
            return '\n'.join(s)
        else:
            raise TypeError
    except TypeError:
        raise ValueError('Could not convert "%s" to string' % s)


def validate_axisbelow(s):
    try:
        return validate_bool(s)
    except ValueError:
        if isinstance(s, str):
            s = s.lower()
            if s.startswith('line'):
                return 'line'
    raise ValueError('%s cannot be interpreted as'
                     ' True, False, or "line"' % s)


def validate_dpi(s):
    """Confirm s is string 'figure' or convert s to float or raise."""
    if s == 'figure':
        return s
    try:
        return float(s)
    except ValueError:
        raise ValueError('"%s" is not string "figure" or'
            ' could not convert "%s" to float' % (s, s))


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
        except ValueError:
            raise ValueError(f'Could not convert {s!r} to {cls.__name__}')

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
        except KeyError:
            raise ValueError(
                'Supported Postscript/PDF font types are %s' % list(fonttypes))
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


@cbook.deprecated("3.1")
def validate_qt4(s):
    if s is None:
        return None
    return ValidateInStrings("backend.qt4", ['PyQt4', 'PySide', 'PyQt4v2'])(s)


@cbook.deprecated("3.1")
def validate_qt5(s):
    if s is None:
        return None
    return ValidateInStrings("backend.qt5", ['PyQt5', 'PySide2'])(s)


validate_toolbar = ValidateInStrings(
    'toolbar', ['None', 'toolbar2', 'toolmanager'], ignorecase=True)


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
        except ValueError:
            raise ValueError(
                f'Could not convert all entries to {cls.__name__}s')

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
    # Special-case the N-th color cycle syntax, this obviously can not
    # go in the color cycle.
    if isinstance(s, bytes):
        match = re.match(b'^C[0-9]$', s)
        if match is not None:
            raise ValueError('Can not put cycle reference ({cn!r}) in '
                             'prop_cycler'.format(cn=s))
    elif isinstance(s, str):
        match = re.match('^C[0-9]$', s)
        if match is not None:
            raise ValueError('Can not put cycle reference ({cn!r}) in '
                             'prop_cycler'.format(cn=s))
    return validate_color(s)


def validate_color(s):
    """Return a valid color arg."""
    try:
        if s.lower() == 'none':
            return 'none'
    except AttributeError:
        pass

    if isinstance(s, str):
        if len(s) == 6 or len(s) == 8:
            stmp = '#' + s
            if is_color_like(stmp):
                return stmp

    if is_color_like(s):
        return s

    # If it is still valid, it must be a tuple.
    colorarg = s
    msg = ''
    if s.find(',') >= 0:
        # get rid of grouping symbols
        stmp = ''.join([c for c in s if c.isdigit() or c == '.' or c == ','])
        vals = stmp.split(',')
        if len(vals) not in [3, 4]:
            msg = '\nColor tuples must be of length 3 or 4'
        else:
            try:
                colorarg = [float(val) for val in vals]
            except ValueError:
                msg = '\nCould not convert all entries to floats'

    if not msg and is_color_like(colorarg):
        return colorarg

    raise ValueError('%s does not look like a color arg%s' % (s, msg))


validate_colorlist = _listify_validator(
    validate_color, allow_stringlist=True, doc='return a list of colorspecs')
validate_orientation = ValidateInStrings(
    'orientation', ['landscape', 'portrait'])


def validate_aspect(s):
    if s in ('auto', 'equal'):
        return s
    try:
        return float(s)
    except ValueError:
        raise ValueError('not a valid aspect specification')


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
    except ValueError:
        raise ValueError("%s is not a valid font size. Valid font sizes "
                         "are %s." % (s, ", ".join(fontsizes)))


validate_fontsizelist = _listify_validator(validate_fontsize)


def validate_font_properties(s):
    parse_fontconfig_pattern(s)
    return s


validate_fontset = ValidateInStrings(
    'fontset',
    ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom'])


def validate_mathtext_default(s):
    if s == "circled":
        cbook.warn_deprecated(
            "3.1", message="Support for setting the mathtext.default rcParam "
            "to 'circled' is deprecated since %(since)s and will be removed "
            "%(removal)s.")
    return ValidateInStrings(
        'default',
        "rm cal it tt sf bf default bb frak circled scr regular".split())(s)


_validate_alignment = ValidateInStrings(
    'alignment',
    ['center', 'top', 'bottom', 'baseline',
     'center_baseline'])


_validate_verbose = ValidateInStrings(
    'verbose',
    ['silent', 'helpful', 'debug', 'debug-annoying'])


@cbook.deprecated("3.1")
def validate_verbose(s):
    return _validate_verbose(s)


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
            except ValueError:
                raise ValueError("Not a valid whisker value ['range', float, "
                                 "(float, float)]")


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


validate_ps_papersize = ValidateInStrings(
    'ps_papersize',
    ['auto', 'letter', 'legal', 'ledger',
     'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10',
     'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10',
     ], ignorecase=True)


def validate_ps_distiller(s):
    if isinstance(s, str):
        s = s.lower()
    if s in ('none', None, 'false', False):
        return None
    elif s in ('ghostscript', 'xpdf'):
        try:
            mpl._get_executable_info("gs")
        except mpl.ExecutableNotFoundError:
            _log.warning("Setting rcParams['ps.usedistiller'] requires "
                         "ghostscript.")
            return None
        if s == "xpdf":
            try:
                mpl._get_executable_info("pdftops")
            except mpl.ExecutableNotFoundError:
                _log.warning("Setting rcParams['ps.usedistiller'] to 'xpdf' "
                             "requires xpdf.")
                return None
        return s
    else:
        raise ValueError('matplotlibrc ps.usedistiller must either be none, '
                         'ghostscript or xpdf')


validate_joinstyle = ValidateInStrings('joinstyle',
                                       ['miter', 'round', 'bevel'],
                                       ignorecase=True)
validate_joinstylelist = _listify_validator(validate_joinstyle)

validate_capstyle = ValidateInStrings('capstyle',
                                      ['butt', 'round', 'projecting'],
                                      ignorecase=True)
validate_capstylelist = _listify_validator(validate_capstyle)

validate_fillstyle = ValidateInStrings('markers.fillstyle',
                                       ['full', 'left', 'right', 'bottom',
                                        'top', 'none'])
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
    s : None, int, float, slice, length-2 tuple of ints,
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
     'center'], ignorecase=True)

validate_svg_fonttype = ValidateInStrings('svg.fonttype', ['none', 'path'])


def validate_hinting(s):
    if s in (True, False):
        cbook.warn_deprecated(
            "3.2", message="Support for setting the text.hinting rcParam to "
            "True or False is deprecated since %(since)s and will be removed "
            "%(removal)s; set it to its synonyms 'auto' or 'none' instead.")
        return s
    if s.lower() in ('auto', 'native', 'either', 'none'):
        return s.lower()
    raise ValueError("hinting should be 'auto', 'native', 'either' or 'none'")


validate_pgf_texsystem = ValidateInStrings('pgf.texsystem',
                                           ['xelatex', 'lualatex', 'pdflatex'])

validate_movie_writer = ValidateInStrings('animation.writer',
    ['ffmpeg', 'ffmpeg_file',
     'avconv', 'avconv_file',
     'imagemagick', 'imagemagick_file',
     'html'])

validate_movie_frame_fmt = ValidateInStrings('animation.frame_format',
    ['png', 'jpeg', 'tiff', 'raw', 'rgba'])

validate_axis_locator = ValidateInStrings('major', ['minor', 'both', 'major'])

validate_movie_html_fmt = ValidateInStrings('animation.html',
    ['html5', 'jshtml', 'none'])


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
    if isinstance(s, str):
        result = tuple([float(v.strip()) for v in s.split(',')])
    elif isinstance(s, (list, tuple)):
        result = tuple([float(v) for v in s])
    if len(result) != 3:
        raise ValueError(
            "path.sketch must be a tuple (scale, length, randomness)")
    return result


class ValidateInterval:
    """
    Value must be in interval
    """
    def __init__(self, vmin, vmax, closedmin=True, closedmax=True):
        self.vmin = vmin
        self.vmax = vmax
        self.cmin = closedmin
        self.cmax = closedmax

    def __call__(self, s):
        try:
            s = float(s)
        except ValueError:
            raise RuntimeError('Value must be a float; found "%s"' % s)

        if self.cmin and s < self.vmin:
            raise RuntimeError('Value must be >= %f; found "%f"' %
                               (self.vmin, s))
        elif not self.cmin and s <= self.vmin:
            raise RuntimeError('Value must be > %f; found "%f"' %
                               (self.vmin, s))

        if self.cmax and s > self.vmax:
            raise RuntimeError('Value must be <= %f; found "%f"' %
                               (self.vmax, s))
        elif not self.cmax and s >= self.vmax:
            raise RuntimeError('Value must be < %f; found "%f"' %
                               (self.vmax, s))
        return s


validate_grid_axis = ValidateInStrings('axes.grid.axis', ['x', 'y', 'both'])


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
        'linestyle': validate_stringlist,
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
    Creates a `~cycler.Cycler` object much like :func:`cycler.cycler`,
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
    cycler : Cycler
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
        # TODO: We might want to rethink this...
        # While I think I have it quite locked down, it is execution of
        # arbitrary code without sanitation.
        # Combine this with the possibility that rcparams might come from the
        # internet (future plans), this could be downright dangerous.
        # I locked it down by only having the 'cycler()' function available.
        # UPDATE: Partly plugging a security hole.
        # I really should have read this:
        # http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
        # We should replace this eval with a combo of PyParsing and
        # ast.literal_eval()
        try:
            if '.__' in s.replace(' ', ''):
                raise ValueError("'%s' seems to have dunder methods. Raising"
                                 " an exception for your safety")
            s = eval(s, {'cycler': cycler, '__builtins__': {}})
        except BaseException as e:
            raise ValueError("'%s' is not a valid cycler construction: %s" %
                             (s, e))
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


def validate_webagg_address(s):
    if s is not None:
        import socket
        try:
            socket.inet_aton(s)
        except socket.error:
            raise ValueError("'webagg.address' is not a valid IP address")
        return s
    raise ValueError("'webagg.address' is not a valid IP address")


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
    # Look first for a valid named line style, like '--' or 'solid' Also
    # includes bytes(-arrays) here (they all fail _validate_named_linestyle);
    # otherwise, if *ls* is of even-length, it will be passed to the instance
    # of validate_nseq_float, which will return an absurd on-off ink
    # sequence...
    if isinstance(ls, (str, bytes, bytearray)):
        return _validate_named_linestyle(ls)

    # Look for an on-off ink sequence (in points) *of even length*.
    # Offset is set to None.
    try:
        if len(ls) % 2 != 0:
            raise ValueError("the linestyle sequence {!r} is not of even "
                             "length.".format(ls))

        return (None, validate_nseq_float()(ls))

    except (ValueError, TypeError):
        # TypeError can be raised inside the instance of validate_nseq_float,
        # by wrong types passed to float(), like NoneType.
        raise ValueError("linestyle {!r} is not a valid on-off ink "
                         "sequence.".format(ls))


validate_axes_titlelocation = ValidateInStrings(
    'axes.titlelocation', ['left', 'center', 'right'])


_validators = {  # key -> validator
    'backend':           validate_backend,
    'backend_fallback':  validate_bool,
    'webagg.port':       validate_int,
    'webagg.address':    validate_webagg_address,
    'webagg.open_in_browser': validate_bool,
    'webagg.port_retries': validate_int,
    'toolbar':           validate_toolbar,
    'datapath':          validate_any,  # see _get_data_path_cached
    'interactive':       validate_bool,
    'timezone':          validate_string,

    # the verbosity setting
    'verbose.level': _validate_verbose,
    'verbose.fileo': validate_string,

    # line props
    'lines.linewidth':       validate_float,  # line width in points
    'lines.linestyle':       _validate_linestyle,  # solid line
    'lines.color':           validate_color,  # first color in color cycle
    'lines.marker':          validate_string,  # marker name
    'lines.markerfacecolor': validate_color_or_auto,  # default color
    'lines.markeredgecolor': validate_color_or_auto,  # default color
    'lines.markeredgewidth': validate_float,
    'lines.markersize':      validate_float,  # markersize, in points
    'lines.antialiased':     validate_bool,  # antialiased (no jaggies)
    'lines.dash_joinstyle':  validate_joinstyle,
    'lines.solid_joinstyle': validate_joinstyle,
    'lines.dash_capstyle':   validate_capstyle,
    'lines.solid_capstyle':  validate_capstyle,
    'lines.dashed_pattern':  validate_nseq_float(allow_none=True),
    'lines.dashdot_pattern': validate_nseq_float(allow_none=True),
    'lines.dotted_pattern':  validate_nseq_float(allow_none=True),
    'lines.scale_dashes':    validate_bool,

    # marker props
    'markers.fillstyle': validate_fillstyle,

    ## patch props
    'patch.linewidth': validate_float,  # line width in points
    'patch.edgecolor': validate_color,
    'patch.force_edgecolor': validate_bool,
    'patch.facecolor': validate_color,  # first color in cycle
    'patch.antialiased': validate_bool,  # antialiased (no jaggies)

    ## hatch props
    'hatch.color': validate_color,
    'hatch.linewidth': validate_float,

    ## Histogram properties
    'hist.bins': validate_hist_bins,

    ## Boxplot properties
    'boxplot.notch': validate_bool,
    'boxplot.vertical': validate_bool,
    'boxplot.whiskers': validate_whiskers,
    'boxplot.bootstrap': validate_int_or_None,
    'boxplot.patchartist': validate_bool,
    'boxplot.showmeans': validate_bool,
    'boxplot.showcaps': validate_bool,
    'boxplot.showbox': validate_bool,
    'boxplot.showfliers': validate_bool,
    'boxplot.meanline': validate_bool,

    'boxplot.flierprops.color': validate_color,
    'boxplot.flierprops.marker': validate_string,
    'boxplot.flierprops.markerfacecolor': validate_color_or_auto,
    'boxplot.flierprops.markeredgecolor': validate_color,
    'boxplot.flierprops.markeredgewidth': validate_float,
    'boxplot.flierprops.markersize': validate_float,
    'boxplot.flierprops.linestyle': _validate_linestyle,
    'boxplot.flierprops.linewidth': validate_float,

    'boxplot.boxprops.color': validate_color,
    'boxplot.boxprops.linewidth': validate_float,
    'boxplot.boxprops.linestyle': _validate_linestyle,

    'boxplot.whiskerprops.color': validate_color,
    'boxplot.whiskerprops.linewidth': validate_float,
    'boxplot.whiskerprops.linestyle': _validate_linestyle,

    'boxplot.capprops.color': validate_color,
    'boxplot.capprops.linewidth': validate_float,
    'boxplot.capprops.linestyle': _validate_linestyle,

    'boxplot.medianprops.color': validate_color,
    'boxplot.medianprops.linewidth': validate_float,
    'boxplot.medianprops.linestyle': _validate_linestyle,

    'boxplot.meanprops.color': validate_color,
    'boxplot.meanprops.marker': validate_string,
    'boxplot.meanprops.markerfacecolor': validate_color,
    'boxplot.meanprops.markeredgecolor': validate_color,
    'boxplot.meanprops.markersize': validate_float,
    'boxplot.meanprops.linestyle': _validate_linestyle,
    'boxplot.meanprops.linewidth': validate_float,

    ## font props
    'font.family':     validate_stringlist,  # used by text object
    'font.style':      validate_string,
    'font.variant':    validate_string,
    'font.stretch':    validate_string,
    'font.weight':     validate_string,
    'font.size':       validate_float,  # Base font size in points
    'font.serif':      validate_stringlist,
    'font.sans-serif': validate_stringlist,
    'font.cursive':    validate_stringlist,
    'font.fantasy':    validate_stringlist,
    'font.monospace':  validate_stringlist,

    # text props
    'text.color':          validate_color,
    'text.usetex':         validate_bool,
    'text.latex.unicode':  validate_bool,
    'text.latex.preamble': _validate_tex_preamble,
    'text.latex.preview':  validate_bool,
    'text.hinting':        validate_hinting,
    'text.hinting_factor': validate_int,
    'text.antialiased':    validate_bool,

    'mathtext.cal':            validate_font_properties,
    'mathtext.rm':             validate_font_properties,
    'mathtext.tt':             validate_font_properties,
    'mathtext.it':             validate_font_properties,
    'mathtext.bf':             validate_font_properties,
    'mathtext.sf':             validate_font_properties,
    'mathtext.fontset':        validate_fontset,
    'mathtext.default':        validate_mathtext_default,
    'mathtext.fallback_to_cm': validate_bool,

    'image.aspect':        validate_aspect,  # equal, auto, a number
    'image.interpolation': validate_string,
    'image.cmap':          validate_string,  # gray, jet, etc.
    'image.lut':           validate_int,  # lookup table
    'image.origin':        ValidateInStrings('image.origin',
                                             ['upper', 'lower']),
    'image.resample':      validate_bool,
    # Specify whether vector graphics backends will combine all images on a
    # set of axes into a single composite image
    'image.composite_image': validate_bool,

    # contour props
    'contour.negative_linestyle': _validate_linestyle,
    'contour.corner_mask':        validate_bool,

    # errorbar props
    'errorbar.capsize':      validate_float,

    # axes props
    'axes.axisbelow':        validate_axisbelow,
    'axes.facecolor':        validate_color,  # background color
    'axes.edgecolor':        validate_color,  # edge color
    'axes.linewidth':        validate_float,  # edge linewidth

    'axes.spines.left':      validate_bool,  # Set visibility of axes
    'axes.spines.right':     validate_bool,  # 'spines', the lines
    'axes.spines.bottom':    validate_bool,  # around the chart
    'axes.spines.top':       validate_bool,  # denoting data boundary

    'axes.titlesize':     validate_fontsize,  # axes title fontsize
    'axes.titlelocation': validate_axes_titlelocation,  # axes title alignment
    'axes.titleweight':   validate_string,  # axes title font weight
    'axes.titlecolor':    validate_color_or_auto,  # axes title font color
    'axes.titlepad':      validate_float,  # axes-top-to-title pad, in points
    'axes.grid':             validate_bool,  # display grid or not
    'axes.grid.which':       validate_axis_locator,  # whether the grid are by
                                                     # default drawn on 'major'
                                                     # 'minor' or 'both' axis
                                                     # locator
    'axes.grid.axis':        validate_grid_axis,  # grid type: 'x', 'y', 'both'
    'axes.labelsize':        validate_fontsize,  # fontsize of the x/y labels
    'axes.labelpad':         validate_float,  # space between label and axis
    'axes.labelweight':      validate_string,  # fontsize of the x any y labels
    'axes.labelcolor':       validate_color,  # color of axis label
    'axes.formatter.limits': validate_nseq_int(2),
                               # use scientific notation if log10
                               # of the axis range is smaller than the
                               # first or larger than the second
    'axes.formatter.use_locale': validate_bool,
                               # Use the current locale to format ticks
    'axes.formatter.use_mathtext': validate_bool,
    'axes.formatter.min_exponent': validate_int,  # minimum exponent to format
                                                  # in scientific notation
    'axes.formatter.useoffset': validate_bool,
    'axes.formatter.offset_threshold': validate_int,
    'axes.unicode_minus': validate_bool,
    # This entry can be either a cycler object or a
    # string repr of a cycler-object, which gets eval()'ed
    # to create the object.
    'axes.prop_cycle': validate_cycler,
    # If 'data', axes limits are set close to the data.
    # If 'round_numbers' axes limits are set to the nearest round numbers.
    'axes.autolimit_mode': ValidateInStrings('autolimit_mode',
                                             ['data', 'round_numbers']),
    'axes.xmargin': ValidateInterval(0, 1,
                                     closedmin=True,
                                     closedmax=True),  # margin added to xaxis
    'axes.ymargin': ValidateInterval(0, 1,
                                     closedmin=True,
                                     closedmax=True),  # margin added to yaxis

    'polaraxes.grid': validate_bool,  # display polar grid or not
    'axes3d.grid': validate_bool,  # display 3d grid

    # scatter props
    'scatter.marker': validate_string,
    'scatter.edgecolors': validate_string,

    # TODO validate that these are valid datetime format strings
    'date.autoformatter.year': validate_string,
    'date.autoformatter.month': validate_string,
    'date.autoformatter.day': validate_string,
    'date.autoformatter.hour': validate_string,
    'date.autoformatter.minute': validate_string,
    'date.autoformatter.second': validate_string,
    'date.autoformatter.microsecond': validate_string,

    #legend properties
    'legend.fancybox': validate_bool,
    'legend.loc': validate_legend_loc,
    # the number of points in the legend line
    'legend.numpoints': validate_int,
    # the number of points in the legend line for scatter
    'legend.scatterpoints': validate_int,
    'legend.fontsize': validate_fontsize,
    'legend.title_fontsize': validate_fontsize_None,
     # the relative size of legend markers vs. original
    'legend.markerscale': validate_float,
    'legend.shadow': validate_bool,
     # whether or not to draw a frame around legend
    'legend.frameon': validate_bool,
     # alpha value of the legend frame
    'legend.framealpha': validate_float_or_None,

    ## the following dimensions are in fraction of the font size
    'legend.borderpad': validate_float,  # units are fontsize
    # the vertical space between the legend entries
    'legend.labelspacing': validate_float,
    # the length of the legend lines
    'legend.handlelength': validate_float,
    # the length of the legend lines
    'legend.handleheight': validate_float,
    # the space between the legend line and legend text
    'legend.handletextpad': validate_float,
    # the border between the axes and legend edge
    'legend.borderaxespad': validate_float,
    # the border between the axes and legend edge
    'legend.columnspacing': validate_float,
    'legend.facecolor': validate_color_or_inherit,
    'legend.edgecolor': validate_color_or_inherit,

    # tick properties
    'xtick.top':           validate_bool,   # draw ticks on the top
    'xtick.bottom':        validate_bool,   # draw ticks on the bottom
    'xtick.labeltop':      validate_bool,   # draw label on the top
    'xtick.labelbottom':   validate_bool,   # draw label on the bottom
    'xtick.major.size':    validate_float,  # major xtick size in points
    'xtick.minor.size':    validate_float,  # minor xtick size in points
    'xtick.major.width':   validate_float,  # major xtick width in points
    'xtick.minor.width':   validate_float,  # minor xtick width in points
    'xtick.major.pad':     validate_float,  # distance to label in points
    'xtick.minor.pad':     validate_float,  # distance to label in points
    'xtick.color':         validate_color,  # color of the xtick labels
    'xtick.minor.visible': validate_bool,   # visibility of the minor xticks
    'xtick.minor.top':     validate_bool,   # draw x axis top minor ticks
    'xtick.minor.bottom':  validate_bool,   # draw x axis bottom minor ticks
    'xtick.major.top':     validate_bool,   # draw x axis top major ticks
    'xtick.major.bottom':  validate_bool,   # draw x axis bottom major ticks
    'xtick.labelsize': validate_fontsize,   # fontsize of the xtick labels
    'xtick.direction': validate_string,     # direction of xticks
    'xtick.alignment': _validate_alignment,

    'ytick.left':          validate_bool,   # draw ticks on the left
    'ytick.right':         validate_bool,   # draw ticks on the right
    'ytick.labelleft':     validate_bool,   # draw tick labels on the left
    'ytick.labelright':    validate_bool,   # draw tick labels on the right
    'ytick.major.size':    validate_float,  # major ytick size in points
    'ytick.minor.size':    validate_float,  # minor ytick size in points
    'ytick.major.width':   validate_float,  # major ytick width in points
    'ytick.minor.width':   validate_float,  # minor ytick width in points
    'ytick.major.pad':     validate_float,  # distance to label in points
    'ytick.minor.pad':     validate_float,  # distance to label in points
    'ytick.color':         validate_color,  # color of the ytick labels
    'ytick.minor.visible': validate_bool,   # visibility of the minor yticks
    'ytick.minor.left':    validate_bool,   # draw y axis left minor ticks
    'ytick.minor.right':   validate_bool,   # draw y axis right minor ticks
    'ytick.major.left':    validate_bool,   # draw y axis left major ticks
    'ytick.major.right':   validate_bool,   # draw y axis right major ticks
    'ytick.labelsize': validate_fontsize,   # fontsize of the ytick labels
    'ytick.direction': validate_string,     # direction of yticks
    'ytick.alignment': _validate_alignment,

    'grid.color':     validate_color,  # grid color
    'grid.linestyle': _validate_linestyle,  # solid
    'grid.linewidth': validate_float,  # in points
    'grid.alpha':     validate_float,

    ## figure props
    # figure title
    'figure.titlesize':   validate_fontsize,
    'figure.titleweight': validate_string,

    # figure size in inches: width by height
    'figure.figsize':    validate_nseq_float(2),
    'figure.dpi':        validate_float,  # DPI
    'figure.facecolor':  validate_color,
    'figure.edgecolor':  validate_color,
    'figure.frameon':    validate_bool,
    'figure.autolayout': validate_bool,
    'figure.max_open_warning': validate_int,

    'figure.subplot.left': ValidateInterval(0, 1, closedmin=True,
                                            closedmax=True),
    'figure.subplot.right': ValidateInterval(0, 1, closedmin=True,
                                             closedmax=True),
    'figure.subplot.bottom': ValidateInterval(0, 1, closedmin=True,
                                              closedmax=True),
    'figure.subplot.top': ValidateInterval(0, 1, closedmin=True,
                                           closedmax=True),
    'figure.subplot.wspace': ValidateInterval(0, 1, closedmin=True,
                                              closedmax=False),
    'figure.subplot.hspace': ValidateInterval(0, 1, closedmin=True,
                                              closedmax=False),

    # do constrained_layout.
    'figure.constrained_layout.use': validate_bool,
    # wspace and hspace are fraction of adjacent subplots to use
    # for space.  Much smaller than above because we don't need
    # room for the text.
    'figure.constrained_layout.hspace': ValidateInterval(
        0, 1, closedmin=True, closedmax=False),
    'figure.constrained_layout.wspace': ValidateInterval(
        0, 1, closedmin=True, closedmax=False),
    # This is a buffer around the axes in inches.  This is 3pts.
    'figure.constrained_layout.h_pad': validate_float,
    'figure.constrained_layout.w_pad': validate_float,

    ## Saving figure's properties
    'savefig.dpi':         validate_dpi,  # DPI
    'savefig.facecolor':   validate_color,
    'savefig.edgecolor':   validate_color,
    'savefig.frameon':     validate_bool,
    'savefig.orientation': validate_orientation,
    'savefig.jpeg_quality': validate_int,
    # value checked by backend at runtime
    'savefig.format':     update_savefig_format,
    # options are 'tight', or 'standard'. 'standard' validates to None.
    'savefig.bbox':       validate_bbox,
    'savefig.pad_inches': validate_float,
    # default directory in savefig dialog box
    'savefig.directory': validate_string,
    'savefig.transparent': validate_bool,

    # Maintain shell focus for TkAgg
    'tk.window_focus':  validate_bool,

    # Set the papersize/type
    'ps.papersize':     validate_ps_papersize,
    'ps.useafm':        validate_bool,
    # use ghostscript or xpdf to distill ps output
    'ps.usedistiller':  validate_ps_distiller,
    'ps.distiller.res': validate_int,  # dpi
    'ps.fonttype':      validate_fonttype,  # 3 (Type3) or 42 (Truetype)
    # compression level from 0 to 9; 0 to disable
    'pdf.compression':  validate_int,
    # ignore any color-setting commands from the frontend
    'pdf.inheritcolor': validate_bool,
    # use only the 14 PDF core fonts embedded in every PDF viewing application
    'pdf.use14corefonts': validate_bool,
    'pdf.fonttype':     validate_fonttype,  # 3 (Type3) or 42 (Truetype)

    'pgf.debug':     validate_bool,  # output debug information
    # choose latex application for creating pdf files (xelatex/lualatex)
    'pgf.texsystem': validate_pgf_texsystem,
    # use matplotlib rc settings for font configuration
    'pgf.rcfonts':   validate_bool,
    # provide a custom preamble for the latex process
    'pgf.preamble':  _validate_tex_preamble,

    # write raster image data directly into the svg file
    'svg.image_inline':     validate_bool,
    # True to save all characters as paths in the SVG
    'svg.fonttype':         validate_svg_fonttype,
    'svg.hashsalt':         validate_string_or_None,

    # set this when you want to generate hardcopy docstring
    'docstring.hardcopy': validate_bool,

    'path.simplify': validate_bool,
    'path.simplify_threshold': ValidateInterval(0.0, 1.0),
    'path.snap': validate_bool,
    'path.sketch': validate_sketch,
    'path.effects': validate_anylist,
    'agg.path.chunksize': validate_int,  # 0 to disable chunking

    # key-mappings (multi-character mappings should be a list/tuple)
    'keymap.fullscreen': validate_stringlist,
    'keymap.home':       validate_stringlist,
    'keymap.back':       validate_stringlist,
    'keymap.forward':    validate_stringlist,
    'keymap.pan':        validate_stringlist,
    'keymap.zoom':       validate_stringlist,
    'keymap.save':       validate_stringlist,
    'keymap.quit':       validate_stringlist,
    'keymap.quit_all':   validate_stringlist,
    'keymap.grid':       validate_stringlist,
    'keymap.grid_minor': validate_stringlist,
    'keymap.yscale':     validate_stringlist,
    'keymap.xscale':     validate_stringlist,
    'keymap.all_axes':   validate_stringlist,
    'keymap.help':       validate_stringlist,
    'keymap.copy':       validate_stringlist,

    # Animation settings
    'animation.html':         validate_movie_html_fmt,
    # Limit, in MB, of size of base64 encoded animation in HTML
    # (i.e. IPython notebook)
    'animation.embed_limit':  validate_float,
    'animation.writer':       validate_movie_writer,
    'animation.codec':        validate_string,
    'animation.bitrate':      validate_int,
    # Controls image format when frames are written to disk
    'animation.frame_format': validate_movie_frame_fmt,
    # Additional arguments for HTML writer
    'animation.html_args':    validate_stringlist,
    # Path to ffmpeg binary. If just binary name, subprocess uses $PATH.
    'animation.ffmpeg_path':  validate_string,
    # Additional arguments for ffmpeg movie writer (using pipes)
    'animation.ffmpeg_args':  validate_stringlist,
    # Path to AVConv binary. If just binary name, subprocess uses $PATH.
    'animation.avconv_path':  validate_string,
    # Additional arguments for avconv movie writer (using pipes)
    'animation.avconv_args':  validate_stringlist,
     # Path to convert binary. If just binary name, subprocess uses $PATH.
    'animation.convert_path': validate_string,
     # Additional arguments for convert movie writer (using pipes)
    'animation.convert_args': validate_stringlist,

    # Classic (pre 2.0) compatibility mode
    # This is used for things that are hard to make backward compatible
    # with a sane rcParam alone.  This does *not* turn on classic mode
    # altogether.  For that use `matplotlib.style.use('classic')`.
    '_internal.classic_mode': validate_bool
}


_hardcoded_defaults = {  # Defaults not inferred from matplotlibrc.template...
    # ... because it can't be:
    'backend': _auto_backend_sentinel,
    # ... because they are private:
    '_internal.classic_mode': False,
    # ... because they are deprecated:
    'text.latex.unicode': True,
    'savefig.frameon': True,
    'verbose.level': 'silent',
    'verbose.fileo': 'sys.stdout',
}
