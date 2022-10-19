"""
A module for parsing and generating `fontconfig patterns`_.

.. _fontconfig patterns:
   https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
"""

# This class logically belongs in `matplotlib.font_manager`, but placing it
# there would have created cyclical dependency problems, because it also needs
# to be available from `matplotlib.rcsetup` (for parsing matplotlibrc files).

from functools import lru_cache, partial
import re

import numpy as np
from pyparsing import (
    Optional, ParseException, Regex, StringEnd, Suppress, ZeroOrMore)

from matplotlib import _api


family_punc = r'\\\-:,'
_family_unescape = partial(re.compile(r'\\(?=[%s])' % family_punc).sub, '')
family_escape = re.compile(r'([%s])' % family_punc).sub

value_punc = r'\\=_:,'
_value_unescape = partial(re.compile(r'\\(?=[%s])' % value_punc).sub, '')
value_escape = re.compile(r'([%s])' % value_punc).sub

# Remove after module deprecation elapses (3.8); then remove underscores
# from _family_unescape and _value_unescape.
family_unescape = re.compile(r'\\([%s])' % family_punc).sub
value_unescape = re.compile(r'\\([%s])' % value_punc).sub


class FontconfigPatternParser:
    """
    A simple pyparsing-based parser for `fontconfig patterns`_.

    .. _fontconfig patterns:
       https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
    """

    _constants = {
        'thin':           ('weight', 'light'),
        'extralight':     ('weight', 'light'),
        'ultralight':     ('weight', 'light'),
        'light':          ('weight', 'light'),
        'book':           ('weight', 'book'),
        'regular':        ('weight', 'regular'),
        'normal':         ('weight', 'normal'),
        'medium':         ('weight', 'medium'),
        'demibold':       ('weight', 'demibold'),
        'semibold':       ('weight', 'semibold'),
        'bold':           ('weight', 'bold'),
        'extrabold':      ('weight', 'extra bold'),
        'black':          ('weight', 'black'),
        'heavy':          ('weight', 'heavy'),
        'roman':          ('slant', 'normal'),
        'italic':         ('slant', 'italic'),
        'oblique':        ('slant', 'oblique'),
        'ultracondensed': ('width', 'ultra-condensed'),
        'extracondensed': ('width', 'extra-condensed'),
        'condensed':      ('width', 'condensed'),
        'semicondensed':  ('width', 'semi-condensed'),
        'expanded':       ('width', 'expanded'),
        'extraexpanded':  ('width', 'extra-expanded'),
        'ultraexpanded':  ('width', 'ultra-expanded'),
    }

    def __init__(self):
        def comma_separated(elem):
            return elem + ZeroOrMore(Suppress(",") + elem)

        family = Regex(r"([^%s]|(\\[%s]))*" % (family_punc, family_punc))
        size = Regex(r"([0-9]+\.?[0-9]*|\.[0-9]+)")
        name = Regex(r"[a-z]+")
        value = Regex(r"([^%s]|(\\[%s]))*" % (value_punc, value_punc))
        prop = (
            (name + Suppress("=") + comma_separated(value))
            | name  # replace by oneOf(self._constants) in mpl 3.9.
        )
        pattern = (
            Optional(comma_separated(family)("families"))
            + Optional("-" + comma_separated(size)("sizes"))
            + ZeroOrMore(":" + prop("properties*"))
            + StringEnd()
        )
        self._parser = pattern
        self.ParseException = ParseException

    def parse(self, pattern):
        """
        Parse the given fontconfig *pattern* and return a dictionary
        of key/value pairs useful for initializing a
        `.font_manager.FontProperties` object.
        """
        try:
            parse = self._parser.parseString(pattern)
        except ParseException as err:
            # explain becomes a plain method on pyparsing 3 (err.explain(0)).
            raise ValueError("\n" + ParseException.explain(err, 0)) from None
        self._parser.resetCache()
        props = {}
        if "families" in parse:
            props["family"] = [*map(_family_unescape, parse["families"])]
        if "sizes" in parse:
            props["size"] = [*parse["sizes"]]
        for prop in parse.get("properties", []):
            if len(prop) == 1:
                if prop[0] not in self._constants:
                    _api.warn_deprecated(
                        "3.7", message=f"Support for unknown constants "
                        f"({prop[0]!r}) is deprecated since %(since)s and "
                        f"will be removed %(removal)s.")
                    continue
                prop = self._constants[prop[0]]
            k, *v = prop
            props.setdefault(k, []).extend(map(_value_unescape, v))
        return props


# `parse_fontconfig_pattern` is a bottleneck during the tests because it is
# repeatedly called when the rcParams are reset (to validate the default
# fonts).  In practice, the cache size doesn't grow beyond a few dozen entries
# during the test suite.
parse_fontconfig_pattern = lru_cache()(FontconfigPatternParser().parse)


def _escape_val(val, escape_func):
    """
    Given a string value or a list of string values, run each value through
    the input escape function to make the values into legal font config
    strings.  The result is returned as a string.
    """
    if not np.iterable(val) or isinstance(val, str):
        val = [val]

    return ','.join(escape_func(r'\\\1', str(x)) for x in val
                    if x is not None)


def generate_fontconfig_pattern(d):
    """
    Given a dictionary of key/value pairs, generates a fontconfig
    pattern string.
    """
    props = []

    # Family is added first w/o a keyword
    family = d.get_family()
    if family is not None and family != []:
        props.append(_escape_val(family, family_escape))

    # The other keys are added as key=value
    for key in ['style', 'variant', 'weight', 'stretch', 'file', 'size']:
        val = getattr(d, 'get_' + key)()
        # Don't use 'if not val' because 0 is a valid input.
        if val is not None and val != []:
            props.append(":%s=%s" % (key, _escape_val(val, value_escape)))

    return ''.join(props)
