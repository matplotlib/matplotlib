"""
A module for finding, managing, and using fonts across platforms,
using fontconfig, and the Python wrappers in fcpy, underneath.

The API is based on the `W3C Cascading Style Sheet, Level 1 (CSS1)
font specification <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_.
Future versions may implement the Level 2 or 2.1 specifications.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
from matplotlib.externals.six.moves import cPickle as pickle


import os, sys, warnings
try:
    set
except NameError:
    from sets import Set as set
from collections import Iterable
import matplotlib
from matplotlib import afm
from matplotlib import rcParams, get_cachedir
from matplotlib.cbook import is_string_like
import matplotlib.cbook as cbook

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache


import freetypy as ft
import fcpy


font_scalings = {
    'xx-small' : 0.579,
    'x-small'  : 0.694,
    'small'    : 0.833,
    'medium'   : 1.0,
    'large'    : 1.200,
    'x-large'  : 1.440,
    'xx-large' : 1.728,
    'larger'   : 1.2,
    'smaller'  : 0.833,
    None       : 1.0
}

stretch_dict = {
    'ultra-condensed' : 100,
    'extra-condensed' : 200,
    'condensed'       : 300,
    'semi-condensed'  : 400,
    'normal'          : 500,
    'semi-expanded'   : 600,
    'expanded'        : 700,
    'extra-expanded'  : 800,
    'ultra-expanded'  : 900
}

stretch_css_to_fontconfig = {
    100: 50,
    200: 63,
    300: 75,
    400: 87,
    500: 100,
    600: 113,
    700: 125,
    800: 150,
    900: 200
}


stretch_fontconfig_to_css = dict(
    (v, k) for (k, v) in stretch_css_to_fontconfig.items())


weight_dict = {
    'ultralight' : 100,
    'extralight' : 100,
    'light'      : 200,
    'demilight'  : 300,
    'semilight'  : 300,
    'normal'     : 400,
    'regular'    : 400,
    'book'       : 400,
    'medium'     : 500,
    'roman'      : 500,
    'semibold'   : 600,
    'demibold'   : 600,
    'demi'       : 600,
    'bold'       : 700,
    'heavy'      : 800,
    'extra bold' : 800,
    'black'      : 900,
    'extra black': 900,
    'ultra black': 900
}


weight_css_to_fontconfig = {
    100: 40,
    200: 50,
    300: 55,
    400: 80,
    500: 100,
    600: 180,
    700: 200,
    800: 210,
    900: 215
}


weight_fontconfig_to_css = {
    0: 100,
    40: 100,
    50: 200,
    55: 300,
    75: 400,
    80: 400,
    100: 500,
    180: 600,
    200: 700,
    205: 700,
    210: 800,
    215: 900
}


font_family_aliases = set([
    'serif',
    'sans-serif',
    'sans serif',
    'cursive',
    'fantasy',
    'monospace',
    'sans'])


slant_dict = {
    'roman': fcpy.SLANT.ROMAN,
    'italic': fcpy.SLANT.ITALIC,
    'oblique': fcpy.SLANT.OBLIQUE
}


slant_rdict = dict(
    (val, key) for (key, val) in slant_dict.items()
)


def _convert_weight(weight):
    try:
        weight = int(weight)
    except ValueError:
        if weight not in weight_dict:
            raise ValueError("weight is invalid")
        weight = weight_dict[weight]
    else:
        weight = min(max((weight // 100), 1), 9) * 100
    return weight_css_to_fontconfig[weight]


class FontProperties(object):
    """
    A class for storing and manipulating font properties.

    The font properties are those described in the `W3C Cascading
    Style Sheet, Level 1
    <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
    specification.  The six properties are:

      - family: A list of font names in decreasing order of priority.
        The items may include a generic font family name, either
        'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'.
        In that case, the actual font to be used will be looked up
        from the associated rcParam in :file:`matplotlibrc`.

      - style: Either 'normal', 'italic' or 'oblique'.

      - variant: Either 'normal' or 'small-caps'.

      - stretch: A numeric value in the range 0-1000 or one of
        'ultra-condensed', 'extra-condensed', 'condensed',
        'semi-condensed', 'normal', 'semi-expanded', 'expanded',
        'extra-expanded' or 'ultra-expanded'

      - weight: A numeric value in the range 0-1000 or one of
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
        'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
        'extra bold', 'black'

      - size: Either an relative value of 'xx-small', 'x-small',
        'small', 'medium', 'large', 'x-large', 'xx-large' or an
        absolute font size, e.g., 12

    The default font property for TrueType fonts (as specified in the
    default :file:`matplotlibrc` file) is::

      sans-serif, normal, normal, normal, normal, scalable.

    Alternatively, a font may be specified using an absolute path to a
    .ttf file, by using the *fname* kwarg.

    The preferred usage of font sizes is to use the relative values,
    e.g.,  'large', instead of absolute font sizes, e.g., 12.  This
    approach allows all text sizes to be made larger or smaller based
    on the font manager's default font size.

    This class will also accept a `fontconfig
    <http://www.fontconfig.org/>`_ pattern, if it is the only argument
    provided.  See the documentation on `fontconfig patterns
    <http://www.fontconfig.org/fontconfig-user.html>`_.
    """
    def __init__(self,
                 family = None,
                 style  = None,
                 variant= None,
                 weight = None,
                 stretch= None,
                 size   = None,
                 fname  = None, # if this is set, it's a hardcoded filename to use
                 _init   = None  # used only by copy()
                 ):
        # This is used only by copy()
        if _init is not None:
            self._pattern = fcpy.Pattern(str(_init))
            return

        if is_string_like(family):
            # Treat family as a fontconfig pattern if it is the only
            # parameter provided.
            if (style is None and
                variant is None and
                weight is None and
                stretch is None and
                size is None and
                fname is None):
                self._pattern = fcpy.Pattern(family)
                return

        self._pattern = fcpy.Pattern()
        self.set_family(family)
        self.set_style(style)
        if variant is not None:
            self.set_variant(variant)
        self.set_weight(weight)
        self.set_stretch(stretch)
        self.set_file(fname)
        self.set_size(size)

    def __hash__(self):
        return hash(self._pattern)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return hash(self) != hash(other)

    def __str__(self):
        return str(self._pattern)

    def get_family(self):
        """
        Return a list of font names that comprise the font family.
        """
        return list(self._pattern.get('family'))

    def get_name(self):
        """
        Return the name of the font that best matches the font
        properties.
        """
        return get_font(findfont(self)).family_name

    def get_style(self):
        """
        Return the font style.  Values are: 'normal', 'italic' or
        'oblique'.
        """
        slant = next(self._pattern.get('slant'))
        slant = slant_rdict[slant]
        if slant == 'roman':
            slant = 'normal'
        return slant
    get_slant = get_style

    @cbook.deprecated(
        '2.0',
        message="variant support is ignored, since it isn't supported by fontconfig")
    def get_variant(self):
        """
        Return the font variant.  Values are: 'normal' or
        'small-caps'.
        """
        return self._variant

    def get_weight(self):
        """
        Set the font weight.  Options are: A numeric value in the
        range 0-1000 or one of 'light', 'normal', 'regular', 'book',
        'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold',
        'heavy', 'extra bold', 'black'
        """
        # matplotlib uses CSS weights externally, but fontconfig
        # weights internally, which are on a different scale
        return weight_fontconfig_to_css[
            next(self._pattern.get('weight'))]

    def get_stretch(self):
        """
        Return the font stretch or width.  Options are: 'ultra-condensed',
        'extra-condensed', 'condensed', 'semi-condensed', 'normal',
        'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'.
        """
        return stretch_fontconfig_to_css[
            next(self._pattern.get('width'))]

    def get_size(self):
        """
        Return the font size.
        """
        return next(self._pattern.get('size'))

    def get_size_in_points(self):
        return self.get_size()

    def get_file(self):
        """
        Return the filename of the associated font.
        """
        return next(self._pattern.get('file'))

    def get_fontconfig_pattern(self):
        """
        Get a fontconfig pattern suitable for looking up the font as
        specified with fontconfig's ``fc-match`` utility.

        See the documentation on `fontconfig patterns
        <http://www.fontconfig.org/fontconfig-user.html>`_.

        This support does not require fontconfig to be installed or
        support for it to be enabled.  We are merely borrowing its
        pattern syntax for use here.
        """
        return str(self._pattern)

    def set_family(self, family):
        """
        Change the font family.  May be either an alias (generic name
        is CSS parlance), such as: 'serif', 'sans-serif', 'cursive',
        'fantasy', or 'monospace', a real font name or a list of real
        font names.  Real font names are not supported when
        `text.usetex` is `True`.
        """
        if family is None:
            family = rcParams['font.family']
        if is_string_like(family):
            family = [six.text_type(family)]
        elif (not is_string_like(family) and isinstance(family, Iterable)):
            family = [six.text_type(f) for f in family]

        families = []
        for entry in family:
            if entry in font_family_aliases:
                if entry.startswith('sans'):
                    entry = 'sans-serif'
                families.extend(rcParams['font.' + entry])
            else:
                families.append(entry)

        self._pattern.set('family', families)
    set_name = set_family

    def set_style(self, style):
        """
        Set the font style.  Values are: 'normal', 'italic' or
        'oblique'.
        """
        if style is None:
            style = rcParams['font.style']
        if style not in ('normal', 'italic', 'oblique', None):
            raise ValueError("style must be normal, italic or oblique")
        if style == 'normal':
            style = 'roman'
        self._pattern.set('slant', slant_dict.get(style))
    set_slant = set_style

    @cbook.deprecated(
        '2.0',
        message="variant support is ignored, since it isn't supported by fontconfig")
    def set_variant(self, variant):
        """
        Set the font variant.  Values are: 'normal' or 'small-caps'.
        """
        if variant not in ('normal', 'small-caps', None):
            raise ValueError("variant must be normal or small-caps")
        self._variant = variant

    def set_weight(self, weight):
        """
        Set the font weight.  May be either a numeric value in the
        range 0-1000 or one of 'ultralight', 'light', 'normal',
        'regular', 'book', 'medium', 'roman', 'semibold', 'demibold',
        'demi', 'bold', 'heavy', 'extra bold', 'black'
        """
        if weight is None:
            weight = rcParams['font.weight']
        self._pattern.set('weight', _convert_weight(weight))

    def set_stretch(self, stretch):
        """
        Set the font stretch or width.  Options are: 'ultra-condensed',
        'extra-condensed', 'condensed', 'semi-condensed', 'normal',
        'semi-expanded', 'expanded', 'extra-expanded' or
        'ultra-expanded', or a numeric value in the range 0-1000.
        """
        if stretch is None:
            stretch = rcParams['font.stretch']
        try:
            stretch = int(stretch)
        except ValueError:
            if stretch not in stretch_dict:
                raise ValueError("stretch is invalid")
            stretch = stretch_dict[stretch]
        else:
            stretch = min(max((stretch // 100), 1), 9) * 100
        self._pattern.set('width', stretch_css_to_fontconfig[stretch])

    def set_size(self, size):
        """
        Set the font size.  Either an relative value of 'xx-small',
        'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
        or an absolute font size, e.g., 12.
        """
        if size is None:
            size = rcParams['font.size']
        try:
            size = float(size)
        except ValueError:
            if size is not None and size not in font_scalings:
                raise ValueError(
                    "Size is invalid. Valid font size are " + ", ".join(
                        str(i) for i in font_scalings.keys()))
            size = font_scalings[size] * rcParams['font.size']
        self._pattern.set('size', size)

    def set_file(self, file):
        """
        Set the filename of the fontfile to use.  In this case, all
        other properties will be ignored.
        """
        if file is not None:
            self._pattern.set('file', file)

    def set_fontconfig_pattern(self, pattern):
        """
        Set the properties by parsing a fontconfig *pattern*.

        See the documentation on `fontconfig patterns
        <http://www.fontconfig.org/fontconfig-user.html>`_.

        This support does not require fontconfig to be installed or
        support for it to be enabled.  We are merely borrowing its
        pattern syntax for use here.
        """
        self._pattern = fcpy.Pattern(pattern)

    def copy(self):
        """Return a deep copy of self"""
        return FontProperties(_init = self)


get_font = lru_cache(64)(ft.Face)


@lru_cache(64)
def _get_afm_pattern(filename):
    """
    Adds a fontconfig pattern for a given AFM file.
    """
    with open(filename) as fd:
        font = afm.AFM(fd)

    pattern = fcpy.Pattern()
    name = font.get_familyname()

    pattern.set('file', filename)
    pattern.set('family', name)
    pattern.set('fullname', font.get_fontname().lower())

    if font.get_angle() != 0 or name.lower().find('italic') >= 0:
        style = fcpy.SLANT.ITALIC
    elif name.lower().find('oblique') >= 0:
        style = fcpy.SLANT.OBLIQUE
    else:
        style = fcpy.SLANT.ROMAN
    pattern.set('slant', style)
    pattern.set('weight', _convert_weight(font.get_weight().lower()))
    pattern.set('scalable', True)

    return pattern


@lru_cache(8)
def _get_font_cache(directory, fontext=None):
    """
    Create a fcpy.Config instance for a particular directory and kind of font.
    """
    def add_directory(path):
        if fontext == 'afm':
            for filename in os.listdir(path):
                filename = os.path.join(path, filename)
                if filename.endswith('.afm') and os.path.isfile(filename):
                    fcpy_config.add_file(filename)
                    pattern = _get_afm_pattern(filename)
                    fcpy_config.add_pattern(pattern)
        else:
            fcpy_config.add_dir(path)


    if directory is None:
        if fontext == 'afm':
            fcpy_config = fcpy.Config()
        else:
            fcpy_config = fcpy.default_config()
        # Add the directories of fonts that ship with matplotlib
        for path in ['ttf', 'afm', 'pdfcorefonts']:
            path = os.path.join(rcParams['datapath'], 'fonts', path)
            add_directory(path)
    else:
        fcpy_config = fcpy.Config()
        add_directory(directory)

    fcpy_config.build_fonts()
    return fcpy_config


def findfont(prop, fontext=None, directory=None, fallback_to_default=True):
    """
    Search the fontconfig database for the font that most closely
    matches the :class:`FontProperties` *prop*.

    If `directory`, is specified, will only return fonts from the
    given directory (or subdirectory of that directory).

    If `fallback_to_default` is True, will fallback to the default
    font family (usually "DejaVu Sans" or "Helvetica") if
    the first lookup hard-fails.
    """
    if isinstance(prop, FontProperties):
        pattern = prop._pattern
    else:
        pattern = FontProperties(prop)._pattern

    fcpy_config = _get_font_cache(directory, fontext)

    pattern = pattern.copy()

    fcpy_config.substitute(pattern)
    pattern.substitute()

    match = fcpy_config.match(pattern)

    try:
        result = next(match.get('file'))
    except StopIteration:
        if fallback_to_default:
            if fontext == 'afm':
                return os.path.join(
                    rcParams['datapath'], 'fonts', 'afm', 'hhvr8a.afm')
            else:
                return os.path.join(
                    rcParams['datapath'], 'fonts', 'ttf', 'DejaVuSans.ttf')
        else:
            raise ValueError("Could not find font for '%s'" % pattern)

    return result
