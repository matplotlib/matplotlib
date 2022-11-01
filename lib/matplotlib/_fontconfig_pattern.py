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

from pyparsing import (
    Optional, ParseException, Regex, StringEnd, Suppress, ZeroOrMore)

from matplotlib import _api


family_punc = r'\\\-:,'
_family_unescape = partial(re.compile(r'\\(?=[%s])' % family_punc).sub, '')
_family_escape = partial(re.compile(r'(?=[%s])' % family_punc).sub, r'\\')
value_punc = r'\\=_:,'
_value_unescape = partial(re.compile(r'\\(?=[%s])' % value_punc).sub, '')
_value_escape = partial(re.compile(r'(?=[%s])' % value_punc).sub, r'\\')

# Remove after module deprecation elapses (3.8); then remove underscores
# from _{family,value}_{un,}escape.
family_unescape = re.compile(r'\\([%s])' % family_punc).sub
value_unescape = re.compile(r'\\([%s])' % value_punc).sub
family_escape = re.compile(r'([%s])' % family_punc).sub
value_escape = re.compile(r'([%s])' % value_punc).sub


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


def generate_fontconfig_pattern(d):
    """Convert a `.FontProperties` to a fontconfig pattern string."""
    kvs = [(k, getattr(d, f"get_{k}")())
           for k in ["style", "variant", "weight", "stretch", "file", "size"]]
    # Families is given first without a leading keyword.  Other entries (which
    # are necessarily scalar) are given as key=value, skipping Nones.
    return (",".join(_family_escape(f) for f in d.get_family())
            + "".join(f":{k}={_value_escape(str(v))}"
                      for k, v in kvs if v is not None))
