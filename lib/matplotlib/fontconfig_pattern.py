"""
A module for parsing a fontconfig pattern.

This class is defined here because it must be available in:
  - The old-style config framework (rcsetup.py)
  - The traits-based config framework (mpltraits.py)
  - The font manager (font_manager.py)

It probably logically belongs in font_manager.py, but
placing it in any of these places would have created cyclical
dependency problems, or an undesired dependency on traits even
when the traits-based config framework is not used.

See here for a rough specification of these patterns:
http://www.fontconfig.org/fontconfig-user.html

Author : Michael Droettboom <mdroe@stsci.edu>
License   : matplotlib license (PSF compatible)
"""
import re
from matplotlib.pyparsing import Literal, OneOrMore, ZeroOrMore, Optional, Regex, \
    StringEnd, ParseException, Suppress

family_punc = r'\\\-:,'
family_unescape = re.compile(r'\\([%s])' % family_punc).sub
family_escape = re.compile(r'([%s])' % family_punc).sub

value_punc = r'\\=_:,'
value_unescape = re.compile(r'\\([%s])' % value_punc).sub
value_escape = re.compile(r'([%s])' % value_punc).sub

class FontconfigPatternParser:
    """A simple pyparsing-based parser for fontconfig-style patterns.

    See here for a rough specification of these patterns:
    http://www.fontconfig.org/fontconfig-user.html
    """
    

    _constants = {
        'thin'           : ('weight', 'light'),
        'extralight'     : ('weight', 'light'),
        'ultralight'     : ('weight', 'light'),
        'light'          : ('weight', 'light'),
        'book'           : ('weight', 'book'),
        'regular'        : ('weight', 'regular'),
        'normal'         : ('weight', 'normal'),
        'medium'         : ('weight', 'medium'),
        'demibold'       : ('weight', 'demibold'),
        'semibold'       : ('weight', 'semibold'),
        'bold'           : ('weight', 'bold'),
        'extrabold'      : ('weight', 'extra bold'),
        'black'          : ('weight', 'black'),
        'heavy'          : ('weight', 'heavy'),
        'roman'          : ('slant', 'normal'),
        'italic'         : ('slant', 'italic'),
        'oblique'        : ('slant', 'oblique'),
        'ultracondensed' : ('width', 'ultra-condensed'),
        'extracondensed' : ('width', 'extra-condensed'),
        'condensed'      : ('width', 'condensed'),
        'semicondensed'  : ('width', 'semi-condensed'),
        'expanded'       : ('width', 'expanded'),
        'extraexpanded'  : ('width', 'extra-expanded'),
        'ultraexpanded'  : ('width', 'ultra-expanded')
        }
    
    def __init__(self):
        family      = Regex(r'([^%s]|(\\[%s]))*' %
                            (family_punc, family_punc)) \
                      .setParseAction(self._family)
        size        = Regex(r'[0-9.]+') \
                      .setParseAction(self._size)
        name        = Regex(r'[a-z]+') \
                      .setParseAction(self._name)
        value       = Regex(r'([^%s]|(\\[%s]))*' %
                            (value_punc, value_punc)) \
                      .setParseAction(self._value)

        families    =(family
                    + ZeroOrMore(
                        Literal(',')
                      + family)  
                    ).setParseAction(self._families)

        point_sizes =(size
                    + ZeroOrMore(
                        Literal(',')
                      + size)
                    ).setParseAction(self._point_sizes)

        property    =( (name
                      + Suppress(Literal('='))
                      + value
                      + ZeroOrMore(
                          Suppress(Literal(','))
                        + value)
                      )
                     |  name
                    ).setParseAction(self._property)

        pattern     =(Optional(
                        families)
                    + Optional(
                        Literal('-')
                      + point_sizes)
                    + ZeroOrMore(
                        Literal(':')
                      + property)
                    + StringEnd()
                    )

        self._parser = pattern
        self.ParseException = ParseException

    def parse(self, pattern):
        props = self._properties = {}
        try:
            self._parser.parseString(pattern)
        except self.ParseException, e:
            raise ValueError("Could not parse font string: '%s'\n%s" % (pattern, e))
            
        self._properties = None
        return props
        
    def _family(self, s, loc, tokens):
        return [family_unescape(r'\1', tokens[0])]

    def _size(self, s, loc, tokens):
        return [float(tokens[0])]

    def _name(self, s, loc, tokens):
        return [tokens[0]]

    def _value(self, s, loc, tokens):
        return [value_unescape(r'\1', tokens[0])]

    def _families(self, s, loc, tokens):
        self._properties['family'] = tokens
        return []

    def _point_sizes(self, s, loc, tokens):
        self._properties['size'] = tokens
        return []
        
    def _property(self, s, loc, tokens):
        if len(tokens) == 1:
            if tokens[0] in self._constants:
                key, val = self._constants[tokens[0]]
                self._properties.setdefault(key, []).append(val)
        else:
            key = tokens[0]
            val = tokens[1:]
            self._properties.setdefault(key, []).extend(val)
        return []

parse_fontconfig_pattern = FontconfigPatternParser().parse

def generate_fontconfig_pattern(d):
    """Given a dictionary of key/value pairs, generates a fontconfig pattern
    string."""
    props = []
    families = ''
    size = ''
    for key, val in d.items():
        if val is not None and val != []:
            val = [value_escape(r'\\\1', str(x)) for x in val if x is not None]
            if val != []:
                val = ','.join(val)
                props.append(":%s=%s" % (key, val))
    return ''.join(props)
