"""
font data tables for truetype and afm computer modern fonts
"""

from __future__ import annotations
from typing import overload

from .ft2font import CharacterCodeType


latex_to_bakoma: dict[str, tuple[str, CharacterCodeType]] = {
    '\\__sqrt__'                 : ('cmex10', 0x70),
    '\\bigcap'                   : ('cmex10', 0x5c),
    '\\bigcup'                   : ('cmex10', 0x5b),
    '\\bigodot'                  : ('cmex10', 0x4b),
    '\\bigoplus'                 : ('cmex10', 0x4d),
    '\\bigotimes'                : ('cmex10', 0x4f),
    '\\biguplus'                 : ('cmex10', 0x5d),
    '\\bigvee'                   : ('cmex10', 0x5f),
    '\\bigwedge'                 : ('cmex10', 0x5e),
    '\\coprod'                   : ('cmex10', 0x61),
    '\\int'                      : ('cmex10', 0x5a),
    '\\langle'                   : ('cmex10', 0xad),
    '\\leftangle'                : ('cmex10', 0xad),
    '\\leftbrace'                : ('cmex10', 0xa9),
    '\\oint'                     : ('cmex10', 0x49),
    '\\prod'                     : ('cmex10', 0x59),
    '\\rangle'                   : ('cmex10', 0xae),
    '\\rightangle'               : ('cmex10', 0xae),
    '\\rightbrace'               : ('cmex10', 0xaa),
    '\\sum'                      : ('cmex10', 0x58),
    '\\widehat'                  : ('cmex10', 0x62),
    '\\widetilde'                : ('cmex10', 0x65),
    '\\{'                        : ('cmex10', 0xa9),
    '\\}'                        : ('cmex10', 0xaa),
    '{'                          : ('cmex10', 0xa9),
    '}'                          : ('cmex10', 0xaa),

    '\\__angbracketleft__'       : ('cmsy10', 0x68),
    '\\__angbracketright__'      : ('cmsy10', 0x69),
    '\\__angbracketleftbig__'    : ('cmex10', 0xad),
    '\\__angbracketleftBig__'    : ('cmex10', 0x44),
    '\\__angbracketleftbigg__'   : ('cmex10', 0xbf),
    '\\__angbracketleftBigg__'   : ('cmex10', 0x2a),
    '\\__angbracketrightbig__'   : ('cmex10', 0xae),
    '\\__angbracketrightBig__'   : ('cmex10', 0x45),
    '\\__angbracketrightbigg__'  : ('cmex10', 0xc0),
    '\\__angbracketrightBigg__'  : ('cmex10', 0x2b),
    '\\__backslashbig__'         : ('cmex10', 0xb2),
    '\\__backslashBig__'         : ('cmex10', 0x2f),
    '\\__backslashbigg__'        : ('cmex10', 0xc2),
    '\\__backslashBigg__'        : ('cmex10', 0x2d),
    '\\__braceleftbig__'         : ('cmex10', 0xa9),
    '\\__braceleftBig__'         : ('cmex10', 0x6e),
    '\\__braceleftbigg__'        : ('cmex10', 0xbd),
    '\\__braceleftBigg__'        : ('cmex10', 0x28),
    '\\__bracerightbig__'        : ('cmex10', 0xaa),
    '\\__bracerightBig__'        : ('cmex10', 0x6f),
    '\\__bracerightbigg__'       : ('cmex10', 0xbe),
    '\\__bracerightBigg__'       : ('cmex10', 0x29),
    '\\__bracketleftbig__'       : ('cmex10', 0xa3),
    '\\__bracketleftBig__'       : ('cmex10', 0x68),
    '\\__bracketleftbigg__'      : ('cmex10', 0x2219),
    '\\__bracketleftBigg__'      : ('cmex10', 0x22),
    '\\__bracketrightbig__'      : ('cmex10', 0xa4),
    '\\__bracketrightBig__'      : ('cmex10', 0x69),
    '\\__bracketrightbigg__'     : ('cmex10', 0xb8),
    '\\__bracketrightBigg__'     : ('cmex10', 0x23),
    '\\__ceilingleftbig__'       : ('cmex10', 0xa7),
    '\\__ceilingleftBig__'       : ('cmex10', 0x6c),
    '\\__ceilingleftbigg__'      : ('cmex10', 0xbb),
    '\\__ceilingleftBigg__'      : ('cmex10', 0x26),
    '\\__ceilingrightbig__'      : ('cmex10', 0xa8),
    '\\__ceilingrightBig__'      : ('cmex10', 0x6d),
    '\\__ceilingrightbigg__'     : ('cmex10', 0xbc),
    '\\__ceilingrightBigg__'     : ('cmex10', 0x27),
    '\\__floorleftbig__'         : ('cmex10', 0xa5),
    '\\__floorleftBig__'         : ('cmex10', 0x6a),
    '\\__floorleftbigg__'        : ('cmex10', 0xb9),
    '\\__floorleftBigg__'        : ('cmex10', 0x24),
    '\\__floorrightbig__'        : ('cmex10', 0xa6),
    '\\__floorrightBig__'        : ('cmex10', 0x6b),
    '\\__floorrightbigg__'       : ('cmex10', 0xba),
    '\\__floorrightBigg__'       : ('cmex10', 0x25),
    '\\__hatwide__'              : ('cmex10', 0x62),
    '\\__hatwider__'             : ('cmex10', 0x63),
    '\\__hatwidest__'            : ('cmex10', 0x64),
    '\\__parenleftbig__'         : ('cmex10', 0xa1),
    '\\__parenleftBig__'         : ('cmex10', 0xb3),
    '\\__parenleftbigg__'        : ('cmex10', 0xb5),
    '\\__parenleftBigg__'        : ('cmex10', 0xc3),
    '\\__parenrightbig__'        : ('cmex10', 0xa2),
    '\\__parenrightBig__'        : ('cmex10', 0xb4),
    '\\__parenrightbigg__'       : ('cmex10', 0xb6),
    '\\__parenrightBigg__'       : ('cmex10', 0x21),
    '\\__radicalbig__'           : ('cmex10', 0x70),
    '\\__radicalBig__'           : ('cmex10', 0x71),
    '\\__radicalbigg__'          : ('cmex10', 0x72),
    '\\__radicalBigg__'          : ('cmex10', 0x73),
    '\\__slashbig__'             : ('cmex10', 0xb1),
    '\\__slashBig__'             : ('cmex10', 0x2e),
    '\\__slashbigg__'            : ('cmex10', 0xc1),
    '\\__slashBigg__'            : ('cmex10', 0x2c),
    '\\__tildewide__'            : ('cmex10', 0x65),
    '\\__tildewider__'           : ('cmex10', 0x66),
    '\\__tildewidest__'          : ('cmex10', 0x67),

    ','                          : ('cmmi10', 0x3b),
    '.'                          : ('cmmi10', 0x3a),
    '/'                          : ('cmmi10', 0x3d),
    '<'                          : ('cmmi10', 0x3c),
    '>'                          : ('cmmi10', 0x3e),
    '\\alpha'                    : ('cmmi10', 0xae),
    '\\beta'                     : ('cmmi10', 0xaf),
    '\\chi'                      : ('cmmi10', 0xc2),
    '\\combiningrightarrowabove' : ('cmmi10', 0x7e),
    '\\delta'                    : ('cmmi10', 0xb1),
    '\\ell'                      : ('cmmi10', 0x60),
    '\\epsilon'                  : ('cmmi10', 0xb2),
    '\\eta'                      : ('cmmi10', 0xb4),
    '\\flat'                     : ('cmmi10', 0x5b),
    '\\frown'                    : ('cmmi10', 0x5f),
    '\\gamma'                    : ('cmmi10', 0xb0),
    '\\imath'                    : ('cmmi10', 0x7b),
    '\\iota'                     : ('cmmi10', 0xb6),
    '\\jmath'                    : ('cmmi10', 0x7c),
    '\\kappa'                    : ('cmmi10', 0x2219),
    '\\lambda'                   : ('cmmi10', 0xb8),
    '\\leftharpoondown'          : ('cmmi10', 0x29),
    '\\leftharpoonup'            : ('cmmi10', 0x28),
    '\\mu'                       : ('cmmi10', 0xb9),
    '\\natural'                  : ('cmmi10', 0x5c),
    '\\nu'                       : ('cmmi10', 0xba),
    '\\omega'                    : ('cmmi10', 0x21),
    '\\phi'                      : ('cmmi10', 0xc1),
    '\\pi'                       : ('cmmi10', 0xbc),
    '\\psi'                      : ('cmmi10', 0xc3),
    '\\rho'                      : ('cmmi10', 0xbd),
    '\\rightharpoondown'         : ('cmmi10', 0x2b),
    '\\rightharpoonup'           : ('cmmi10', 0x2a),
    '\\sharp'                    : ('cmmi10', 0x5d),
    '\\sigma'                    : ('cmmi10', 0xbe),
    '\\smile'                    : ('cmmi10', 0x5e),
    '\\tau'                      : ('cmmi10', 0xbf),
    '\\theta'                    : ('cmmi10', 0xb5),
    '\\triangleleft'             : ('cmmi10', 0x2f),
    '\\triangleright'            : ('cmmi10', 0x2e),
    '\\upsilon'                  : ('cmmi10', 0xc0),
    '\\varepsilon'               : ('cmmi10', 0x22),
    '\\varphi'                   : ('cmmi10', 0x27),
    '\\varrho'                   : ('cmmi10', 0x25),
    '\\varsigma'                 : ('cmmi10', 0x26),
    '\\vartheta'                 : ('cmmi10', 0x23),
    '\\wp'                       : ('cmmi10', 0x7d),
    '\\xi'                       : ('cmmi10', 0xbb),
    '\\zeta'                     : ('cmmi10', 0xb3),

    '!'                          : ('cmr10', 0x21),
    '%'                          : ('cmr10', 0x25),
    '&'                          : ('cmr10', 0x26),
    '('                          : ('cmr10', 0x28),
    ')'                          : ('cmr10', 0x29),
    '+'                          : ('cmr10', 0x2b),
    ':'                          : ('cmr10', 0x3a),
    ';'                          : ('cmr10', 0x3b),
    '='                          : ('cmr10', 0x3d),
    '?'                          : ('cmr10', 0x3f),
    '@'                          : ('cmr10', 0x40),
    '['                          : ('cmr10', 0x5b),
    '\\#'                        : ('cmr10', 0x23),
    '\\$'                        : ('cmr10', 0x24),
    '\\%'                        : ('cmr10', 0x25),
    '\\Delta'                    : ('cmr10', 0xa2),
    '\\Gamma'                    : ('cmr10', 0xa1),
    '\\Lambda'                   : ('cmr10', 0xa4),
    '\\Omega'                    : ('cmr10', 0xad),
    '\\Phi'                      : ('cmr10', 0xa9),
    '\\Pi'                       : ('cmr10', 0xa6),
    '\\Psi'                      : ('cmr10', 0xaa),
    '\\Sigma'                    : ('cmr10', 0xa7),
    '\\Theta'                    : ('cmr10', 0xa3),
    '\\Upsilon'                  : ('cmr10', 0xa8),
    '\\Xi'                       : ('cmr10', 0xa5),
    '\\circumflexaccent'         : ('cmr10', 0x5e),
    '\\combiningacuteaccent'     : ('cmr10', 0xb6),
    '\\combiningbreve'           : ('cmr10', 0xb8),
    '\\combiningdiaeresis'       : ('cmr10', 0xc4),
    '\\combiningdotabove'        : ('cmr10', 0x5f),
    '\\combininggraveaccent'     : ('cmr10', 0xb5),
    '\\combiningoverline'        : ('cmr10', 0xb9),
    '\\combiningtilde'           : ('cmr10', 0x7e),
    '\\leftbracket'              : ('cmr10', 0x5b),
    '\\leftparen'                : ('cmr10', 0x28),
    '\\rightbracket'             : ('cmr10', 0x5d),
    '\\rightparen'               : ('cmr10', 0x29),
    '\\widebar'                  : ('cmr10', 0xb9),
    ']'                          : ('cmr10', 0x5d),

    '*'                          : ('cmsy10', 0xa4),
    '\N{MINUS SIGN}'             : ('cmsy10', 0xa1),
    '\\Downarrow'                : ('cmsy10', 0x2b),
    '\\Im'                       : ('cmsy10', 0x3d),
    '\\Leftarrow'                : ('cmsy10', 0x28),
    '\\Leftrightarrow'           : ('cmsy10', 0x2c),
    '\\P'                        : ('cmsy10', 0x7b),
    '\\Re'                       : ('cmsy10', 0x3c),
    '\\Rightarrow'               : ('cmsy10', 0x29),
    '\\S'                        : ('cmsy10', 0x78),
    '\\Uparrow'                  : ('cmsy10', 0x2a),
    '\\Updownarrow'              : ('cmsy10', 0x6d),
    '\\Vert'                     : ('cmsy10', 0x6b),
    '\\aleph'                    : ('cmsy10', 0x40),
    '\\approx'                   : ('cmsy10', 0xbc),
    '\\ast'                      : ('cmsy10', 0xa4),
    '\\asymp'                    : ('cmsy10', 0xb3),
    '\\backslash'                : ('cmsy10', 0x6e),
    '\\bigcirc'                  : ('cmsy10', 0xb0),
    '\\bigtriangledown'          : ('cmsy10', 0x35),
    '\\bigtriangleup'            : ('cmsy10', 0x34),
    '\\bot'                      : ('cmsy10', 0x3f),
    '\\bullet'                   : ('cmsy10', 0xb2),
    '\\cap'                      : ('cmsy10', 0x5c),
    '\\cdot'                     : ('cmsy10', 0xa2),
    '\\circ'                     : ('cmsy10', 0xb1),
    '\\clubsuit'                 : ('cmsy10', 0x7c),
    '\\cup'                      : ('cmsy10', 0x5b),
    '\\dag'                      : ('cmsy10', 0x79),
    '\\dashv'                    : ('cmsy10', 0x61),
    '\\ddag'                     : ('cmsy10', 0x7a),
    '\\diamond'                  : ('cmsy10', 0xa6),
    '\\diamondsuit'              : ('cmsy10', 0x7d),
    '\\div'                      : ('cmsy10', 0xa5),
    '\\downarrow'                : ('cmsy10', 0x23),
    '\\emptyset'                 : ('cmsy10', 0x3b),
    '\\equiv'                    : ('cmsy10', 0xb4),
    '\\exists'                   : ('cmsy10', 0x39),
    '\\forall'                   : ('cmsy10', 0x38),
    '\\geq'                      : ('cmsy10', 0xb8),
    '\\gg'                       : ('cmsy10', 0xc0),
    '\\heartsuit'                : ('cmsy10', 0x7e),
    '\\in'                       : ('cmsy10', 0x32),
    '\\infty'                    : ('cmsy10', 0x31),
    '\\lbrace'                   : ('cmsy10', 0x66),
    '\\lceil'                    : ('cmsy10', 0x64),
    '\\leftarrow'                : ('cmsy10', 0xc3),
    '\\leftrightarrow'           : ('cmsy10', 0x24),
    '\\leq'                      : ('cmsy10', 0x2219),
    '\\lfloor'                   : ('cmsy10', 0x62),
    '\\ll'                       : ('cmsy10', 0xbf),
    '\\mid'                      : ('cmsy10', 0x6a),
    '\\mp'                       : ('cmsy10', 0xa8),
    '\\nabla'                    : ('cmsy10', 0x72),
    '\\nearrow'                  : ('cmsy10', 0x25),
    '\\neg'                      : ('cmsy10', 0x3a),
    '\\ni'                       : ('cmsy10', 0x33),
    '\\nwarrow'                  : ('cmsy10', 0x2d),
    '\\odot'                     : ('cmsy10', 0xaf),
    '\\ominus'                   : ('cmsy10', 0xaa),
    '\\oplus'                    : ('cmsy10', 0xa9),
    '\\oslash'                   : ('cmsy10', 0xae),
    '\\otimes'                   : ('cmsy10', 0xad),
    '\\pm'                       : ('cmsy10', 0xa7),
    '\\prec'                     : ('cmsy10', 0xc1),
    '\\preceq'                   : ('cmsy10', 0xb9),
    '\\prime'                    : ('cmsy10', 0x30),
    '\\propto'                   : ('cmsy10', 0x2f),
    '\\rbrace'                   : ('cmsy10', 0x67),
    '\\rceil'                    : ('cmsy10', 0x65),
    '\\rfloor'                   : ('cmsy10', 0x63),
    '\\rightarrow'               : ('cmsy10', 0x21),
    '\\searrow'                  : ('cmsy10', 0x26),
    '\\sim'                      : ('cmsy10', 0xbb),
    '\\simeq'                    : ('cmsy10', 0x27),
    '\\slash'                    : ('cmsy10', 0x36),
    '\\spadesuit'                : ('cmsy10', 0xc4),
    '\\sqcap'                    : ('cmsy10', 0x75),
    '\\sqcup'                    : ('cmsy10', 0x74),
    '\\sqsubseteq'               : ('cmsy10', 0x76),
    '\\sqsupseteq'               : ('cmsy10', 0x77),
    '\\subset'                   : ('cmsy10', 0xbd),
    '\\subseteq'                 : ('cmsy10', 0xb5),
    '\\succ'                     : ('cmsy10', 0xc2),
    '\\succeq'                   : ('cmsy10', 0xba),
    '\\supset'                   : ('cmsy10', 0xbe),
    '\\supseteq'                 : ('cmsy10', 0xb6),
    '\\swarrow'                  : ('cmsy10', 0x2e),
    '\\times'                    : ('cmsy10', 0xa3),
    '\\to'                       : ('cmsy10', 0x21),
    '\\top'                      : ('cmsy10', 0x3e),
    '\\uparrow'                  : ('cmsy10', 0x22),
    '\\updownarrow'              : ('cmsy10', 0x6c),
    '\\uplus'                    : ('cmsy10', 0x5d),
    '\\vdash'                    : ('cmsy10', 0x60),
    '\\vee'                      : ('cmsy10', 0x5f),
    '\\vert'                     : ('cmsy10', 0x6a),
    '\\wedge'                    : ('cmsy10', 0x5e),
    '\\wr'                       : ('cmsy10', 0x6f),
    '\\|'                        : ('cmsy10', 0x6b),
    '|'                          : ('cmsy10', 0x6a),

    '\\_'                        : ('cmtt10', 0x5f)
}

# Automatically generated.

type12uni: dict[str, CharacterCodeType] = {
    'aring'          : 229,
    'quotedblright'  : 8221,
    'V'              : 86,
    'dollar'         : 36,
    'four'           : 52,
    'Yacute'         : 221,
    'P'              : 80,
    'underscore'     : 95,
    'p'              : 112,
    'Otilde'         : 213,
    'perthousand'    : 8240,
    'zero'           : 48,
    'dotlessi'       : 305,
    'Scaron'         : 352,
    'zcaron'         : 382,
    'egrave'         : 232,
    'section'        : 167,
    'Icircumflex'    : 206,
    'ntilde'         : 241,
    'ampersand'      : 38,
    'dotaccent'      : 729,
    'degree'         : 176,
    'K'              : 75,
    'acircumflex'    : 226,
    'Aring'          : 197,
    'k'              : 107,
    'smalltilde'     : 732,
    'Agrave'         : 192,
    'divide'         : 247,
    'ocircumflex'    : 244,
    'asciitilde'     : 126,
    'two'            : 50,
    'E'              : 69,
    'scaron'         : 353,
    'F'              : 70,
    'bracketleft'    : 91,
    'asciicircum'    : 94,
    'f'              : 102,
    'ordmasculine'   : 186,
    'mu'             : 181,
    'paragraph'      : 182,
    'nine'           : 57,
    'v'              : 118,
    'guilsinglleft'  : 8249,
    'backslash'      : 92,
    'six'            : 54,
    'A'              : 65,
    'icircumflex'    : 238,
    'a'              : 97,
    'ogonek'         : 731,
    'q'              : 113,
    'oacute'         : 243,
    'ograve'         : 242,
    'edieresis'      : 235,
    'comma'          : 44,
    'otilde'         : 245,
    'guillemotright' : 187,
    'ecircumflex'    : 234,
    'greater'        : 62,
    'uacute'         : 250,
    'L'              : 76,
    'bullet'         : 8226,
    'cedilla'        : 184,
    'ydieresis'      : 255,
    'l'              : 108,
    'logicalnot'     : 172,
    'exclamdown'     : 161,
    'endash'         : 8211,
    'agrave'         : 224,
    'Adieresis'      : 196,
    'germandbls'     : 223,
    'Odieresis'      : 214,
    'space'          : 32,
    'quoteright'     : 8217,
    'ucircumflex'    : 251,
    'G'              : 71,
    'quoteleft'      : 8216,
    'W'              : 87,
    'Q'              : 81,
    'g'              : 103,
    'w'              : 119,
    'question'       : 63,
    'one'            : 49,
    'ring'           : 730,
    'figuredash'     : 8210,
    'B'              : 66,
    'iacute'         : 237,
    'Ydieresis'      : 376,
    'R'              : 82,
    'b'              : 98,
    'r'              : 114,
    'Ccedilla'       : 199,
    'minus'          : 8722,
    'Lslash'         : 321,
    'Uacute'         : 218,
    'yacute'         : 253,
    'Ucircumflex'    : 219,
    'quotedbl'       : 34,
    'onehalf'        : 189,
    'Thorn'          : 222,
    'M'              : 77,
    'eight'          : 56,
    'multiply'       : 215,
    'grave'          : 96,
    'Ocircumflex'    : 212,
    'm'              : 109,
    'Ugrave'         : 217,
    'guilsinglright' : 8250,
    'Ntilde'         : 209,
    'questiondown'   : 191,
    'Atilde'         : 195,
    'ccedilla'       : 231,
    'Z'              : 90,
    'copyright'      : 169,
    'yen'            : 165,
    'Eacute'         : 201,
    'H'              : 72,
    'X'              : 88,
    'Idieresis'      : 207,
    'bar'            : 124,
    'h'              : 104,
    'x'              : 120,
    'udieresis'      : 252,
    'ordfeminine'    : 170,
    'braceleft'      : 123,
    'macron'         : 175,
    'atilde'         : 227,
    'Acircumflex'    : 194,
    'Oslash'         : 216,
    'C'              : 67,
    'quotedblleft'   : 8220,
    'S'              : 83,
    'exclam'         : 33,
    'Zcaron'         : 381,
    'equal'          : 61,
    's'              : 115,
    'eth'            : 240,
    'Egrave'         : 200,
    'hyphen'         : 45,
    'period'         : 46,
    'igrave'         : 236,
    'colon'          : 58,
    'Ecircumflex'    : 202,
    'trademark'      : 8482,
    'Aacute'         : 193,
    'cent'           : 162,
    'lslash'         : 322,
    'c'              : 99,
    'N'              : 78,
    'breve'          : 728,
    'Oacute'         : 211,
    'guillemotleft'  : 171,
    'n'              : 110,
    'idieresis'      : 239,
    'braceright'     : 125,
    'seven'          : 55,
    'brokenbar'      : 166,
    'ugrave'         : 249,
    'periodcentered' : 183,
    'sterling'       : 163,
    'I'              : 73,
    'Y'              : 89,
    'Eth'            : 208,
    'emdash'         : 8212,
    'i'              : 105,
    'daggerdbl'      : 8225,
    'y'              : 121,
    'plusminus'      : 177,
    'less'           : 60,
    'Udieresis'      : 220,
    'D'              : 68,
    'five'           : 53,
    'T'              : 84,
    'oslash'         : 248,
    'acute'          : 180,
    'd'              : 100,
    'OE'             : 338,
    'Igrave'         : 204,
    't'              : 116,
    'parenright'     : 41,
    'adieresis'      : 228,
    'quotesingle'    : 39,
    'twodotenleader' : 8229,
    'slash'          : 47,
    'ellipsis'       : 8230,
    'numbersign'     : 35,
    'odieresis'      : 246,
    'O'              : 79,
    'oe'             : 339,
    'o'              : 111,
    'Edieresis'      : 203,
    'plus'           : 43,
    'dagger'         : 8224,
    'three'          : 51,
    'hungarumlaut'   : 733,
    'parenleft'      : 40,
    'fraction'       : 8260,
    'registered'     : 174,
    'J'              : 74,
    'dieresis'       : 168,
    'Ograve'         : 210,
    'j'              : 106,
    'z'              : 122,
    'ae'             : 230,
    'semicolon'      : 59,
    'at'             : 64,
    'Iacute'         : 205,
    'percent'        : 37,
    'bracketright'   : 93,
    'AE'             : 198,
    'asterisk'       : 42,
    'aacute'         : 225,
    'U'              : 85,
    'eacute'         : 233,
    'e'              : 101,
    'thorn'          : 254,
    'u'              : 117,
}

uni2type1 = {v: k for k, v in type12uni.items()}

#  The script below is to sort and format the tex2uni dict

## For decimal values: int(hex(v), 16)
#  newtex = {k: hex(v) for k, v in tex2uni.items()}
#  sd = dict(sorted(newtex.items(), key=lambda item: item[0]))
#
## For formatting the sorted dictionary with proper spacing
## the value '24' comes from finding the longest string in
## the newtex keys with len(max(newtex, key=len))
#  for key in sd:
#      print("{0:24} : {1: <s},".format("'" + key + "'", sd[key]))

tex2uni: dict[str, CharacterCodeType] = {
    '#'                      : 0x23,
    '$'                      : 0x24,
    '%'                      : 0x25,
    'AA'                     : 0xc5,
    'AE'                     : 0xc6,
    'BbbC'                   : 0x2102,
    'BbbN'                   : 0x2115,
    'BbbP'                   : 0x2119,
    'BbbQ'                   : 0x211a,
    'BbbR'                   : 0x211d,
    'BbbZ'                   : 0x2124,
    'Bumpeq'                 : 0x224e,
    'Cap'                    : 0x22d2,
    'Colon'                  : 0x2237,
    'Cup'                    : 0x22d3,
    'DH'                     : 0xd0,
    'Delta'                  : 0x394,
    'Doteq'                  : 0x2251,
    'Downarrow'              : 0x21d3,
    'Equiv'                  : 0x2263,
    'Finv'                   : 0x2132,
    'Game'                   : 0x2141,
    'Gamma'                  : 0x393,
    'H'                      : 0x30b,
    'Im'                     : 0x2111,
    'Join'                   : 0x2a1d,
    'L'                      : 0x141,
    'Lambda'                 : 0x39b,
    'Ldsh'                   : 0x21b2,
    'Leftarrow'              : 0x21d0,
    'Leftrightarrow'         : 0x21d4,
    'Lleftarrow'             : 0x21da,
    'Longleftarrow'          : 0x27f8,
    'Longleftrightarrow'     : 0x27fa,
    'Longrightarrow'         : 0x27f9,
    'Lsh'                    : 0x21b0,
    'Nearrow'                : 0x21d7,
    'Nwarrow'                : 0x21d6,
    'O'                      : 0xd8,
    'OE'                     : 0x152,
    'Omega'                  : 0x3a9,
    'P'                      : 0xb6,
    'Phi'                    : 0x3a6,
    'Pi'                     : 0x3a0,
    'Psi'                    : 0x3a8,
    'QED'                    : 0x220e,
    'Rdsh'                   : 0x21b3,
    'Re'                     : 0x211c,
    'Rightarrow'             : 0x21d2,
    'Rrightarrow'            : 0x21db,
    'Rsh'                    : 0x21b1,
    'S'                      : 0xa7,
    'Searrow'                : 0x21d8,
    'Sigma'                  : 0x3a3,
    'Subset'                 : 0x22d0,
    'Supset'                 : 0x22d1,
    'Swarrow'                : 0x21d9,
    'Theta'                  : 0x398,
    'Thorn'                  : 0xde,
    'Uparrow'                : 0x21d1,
    'Updownarrow'            : 0x21d5,
    'Upsilon'                : 0x3a5,
    'Vdash'                  : 0x22a9,
    'Vert'                   : 0x2016,
    'Vvdash'                 : 0x22aa,
    'Xi'                     : 0x39e,
    '_'                      : 0x5f,
    '__sqrt__'               : 0x221a,
    'aa'                     : 0xe5,
    'ac'                     : 0x223e,
    'acute'                  : 0x301,
    'acwopencirclearrow'     : 0x21ba,
    'adots'                  : 0x22f0,
    'ae'                     : 0xe6,
    'aleph'                  : 0x2135,
    'alpha'                  : 0x3b1,
    'amalg'                  : 0x2a3f,
    'angle'                  : 0x2220,
    'approx'                 : 0x2248,
    'approxeq'               : 0x224a,
    'approxident'            : 0x224b,
    'arceq'                  : 0x2258,
    'ast'                    : 0x2217,
    'asterisk'               : 0x2a,
    'asymp'                  : 0x224d,
    'backcong'               : 0x224c,
    'backepsilon'            : 0x3f6,
    'backprime'              : 0x2035,
    'backsim'                : 0x223d,
    'backsimeq'              : 0x22cd,
    'backslash'              : 0x5c,
    'bagmember'              : 0x22ff,
    'bar'                    : 0x304,
    'barleftarrow'           : 0x21e4,
    'barvee'                 : 0x22bd,
    'barwedge'               : 0x22bc,
    'because'                : 0x2235,
    'beta'                   : 0x3b2,
    'beth'                   : 0x2136,
    'between'                : 0x226c,
    'bigcap'                 : 0x22c2,
    'bigcirc'                : 0x25cb,
    'bigcup'                 : 0x22c3,
    'bigodot'                : 0x2a00,
    'bigoplus'               : 0x2a01,
    'bigotimes'              : 0x2a02,
    'bigsqcup'               : 0x2a06,
    'bigstar'                : 0x2605,
    'bigtriangledown'        : 0x25bd,
    'bigtriangleup'          : 0x25b3,
    'biguplus'               : 0x2a04,
    'bigvee'                 : 0x22c1,
    'bigwedge'               : 0x22c0,
    'blacksquare'            : 0x25a0,
    'blacktriangle'          : 0x25b4,
    'blacktriangledown'      : 0x25be,
    'blacktriangleleft'      : 0x25c0,
    'blacktriangleright'     : 0x25b6,
    'bot'                    : 0x22a5,
    'bowtie'                 : 0x22c8,
    'boxbar'                 : 0x25eb,
    'boxdot'                 : 0x22a1,
    'boxminus'               : 0x229f,
    'boxplus'                : 0x229e,
    'boxtimes'               : 0x22a0,
    'breve'                  : 0x306,
    'bullet'                 : 0x2219,
    'bumpeq'                 : 0x224f,
    'c'                      : 0x327,
    'candra'                 : 0x310,
    'cap'                    : 0x2229,
    'carriagereturn'         : 0x21b5,
    'cdot'                   : 0x22c5,
    'cdotp'                  : 0xb7,
    'cdots'                  : 0x22ef,
    'cent'                   : 0xa2,
    'check'                  : 0x30c,
    'checkmark'              : 0x2713,
    'chi'                    : 0x3c7,
    'circ'                   : 0x2218,
    'circeq'                 : 0x2257,
    'circlearrowleft'        : 0x21ba,
    'circlearrowright'       : 0x21bb,
    'circledR'               : 0xae,
    'circledS'               : 0x24c8,
    'circledast'             : 0x229b,
    'circledcirc'            : 0x229a,
    'circleddash'            : 0x229d,
    'circumflexaccent'       : 0x302,
    'clubsuit'               : 0x2663,
    'clubsuitopen'           : 0x2667,
    'colon'                  : 0x3a,
    'coloneq'                : 0x2254,
    'combiningacuteaccent'   : 0x301,
    'combiningbreve'         : 0x306,
    'combiningdiaeresis'     : 0x308,
    'combiningdotabove'      : 0x307,
    'combiningfourdotsabove' : 0x20dc,
    'combininggraveaccent'   : 0x300,
    'combiningoverline'      : 0x304,
    'combiningrightarrowabove' : 0x20d7,
    'combiningthreedotsabove' : 0x20db,
    'combiningtilde'         : 0x303,
    'complement'             : 0x2201,
    'cong'                   : 0x2245,
    'coprod'                 : 0x2210,
    'copyright'              : 0xa9,
    'cup'                    : 0x222a,
    'cupdot'                 : 0x228d,
    'cupleftarrow'           : 0x228c,
    'curlyeqprec'            : 0x22de,
    'curlyeqsucc'            : 0x22df,
    'curlyvee'               : 0x22ce,
    'curlywedge'             : 0x22cf,
    'curvearrowleft'         : 0x21b6,
    'curvearrowright'        : 0x21b7,
    'cwopencirclearrow'      : 0x21bb,
    'd'                      : 0x323,
    'dag'                    : 0x2020,
    'dagger'                 : 0x2020,
    'daleth'                 : 0x2138,
    'danger'                 : 0x2621,
    'dashleftarrow'          : 0x290e,
    'dashrightarrow'         : 0x290f,
    'dashv'                  : 0x22a3,
    'ddag'                   : 0x2021,
    'ddagger'                : 0x2021,
    'ddddot'                 : 0x20dc,
    'dddot'                  : 0x20db,
    'ddot'                   : 0x308,
    'ddots'                  : 0x22f1,
    'degree'                 : 0xb0,
    'delta'                  : 0x3b4,
    'dh'                     : 0xf0,
    'diamond'                : 0x22c4,
    'diamondsuit'            : 0x2662,
    'digamma'                : 0x3dd,
    'disin'                  : 0x22f2,
    'div'                    : 0xf7,
    'divideontimes'          : 0x22c7,
    'dot'                    : 0x307,
    'doteq'                  : 0x2250,
    'doteqdot'               : 0x2251,
    'dotminus'               : 0x2238,
    'dotplus'                : 0x2214,
    'dots'                   : 0x2026,
    'dotsminusdots'          : 0x223a,
    'doublebarwedge'         : 0x2306,
    'downarrow'              : 0x2193,
    'downdownarrows'         : 0x21ca,
    'downharpoonleft'        : 0x21c3,
    'downharpoonright'       : 0x21c2,
    'downzigzagarrow'        : 0x21af,
    'ell'                    : 0x2113,
    'emdash'                 : 0x2014,
    'emptyset'               : 0x2205,
    'endash'                 : 0x2013,
    'epsilon'                : 0x3b5,
    'eqcirc'                 : 0x2256,
    'eqcolon'                : 0x2255,
    'eqdef'                  : 0x225d,
    'eqgtr'                  : 0x22dd,
    'eqless'                 : 0x22dc,
    'eqsim'                  : 0x2242,
    'eqslantgtr'             : 0x2a96,
    'eqslantless'            : 0x2a95,
    'equal'                  : 0x3d,
    'equalparallel'          : 0x22d5,
    'equiv'                  : 0x2261,
    'eta'                    : 0x3b7,
    'eth'                    : 0xf0,
    'exists'                 : 0x2203,
    'fallingdotseq'          : 0x2252,
    'flat'                   : 0x266d,
    'forall'                 : 0x2200,
    'frakC'                  : 0x212d,
    'frakZ'                  : 0x2128,
    'frown'                  : 0x2322,
    'gamma'                  : 0x3b3,
    'geq'                    : 0x2265,
    'geqq'                   : 0x2267,
    'geqslant'               : 0x2a7e,
    'gg'                     : 0x226b,
    'ggg'                    : 0x22d9,
    'gimel'                  : 0x2137,
    'gnapprox'               : 0x2a8a,
    'gneqq'                  : 0x2269,
    'gnsim'                  : 0x22e7,
    'grave'                  : 0x300,
    'greater'                : 0x3e,
    'gtrapprox'              : 0x2a86,
    'gtrdot'                 : 0x22d7,
    'gtreqless'              : 0x22db,
    'gtreqqless'             : 0x2a8c,
    'gtrless'                : 0x2277,
    'gtrsim'                 : 0x2273,
    'guillemotleft'          : 0xab,
    'guillemotright'         : 0xbb,
    'guilsinglleft'          : 0x2039,
    'guilsinglright'         : 0x203a,
    'hat'                    : 0x302,
    'hbar'                   : 0x127,
    'heartsuit'              : 0x2661,
    'hermitmatrix'           : 0x22b9,
    'hookleftarrow'          : 0x21a9,
    'hookrightarrow'         : 0x21aa,
    'hslash'                 : 0x210f,
    'i'                      : 0x131,
    'iiiint'                 : 0x2a0c,
    'iiint'                  : 0x222d,
    'iint'                   : 0x222c,
    'imageof'                : 0x22b7,
    'imath'                  : 0x131,
    'in'                     : 0x2208,
    'increment'              : 0x2206,
    'infty'                  : 0x221e,
    'int'                    : 0x222b,
    'intercal'               : 0x22ba,
    'invnot'                 : 0x2310,
    'iota'                   : 0x3b9,
    'isinE'                  : 0x22f9,
    'isindot'                : 0x22f5,
    'isinobar'               : 0x22f7,
    'isins'                  : 0x22f4,
    'isinvb'                 : 0x22f8,
    'jmath'                  : 0x237,
    'k'                      : 0x328,
    'kappa'                  : 0x3ba,
    'kernelcontraction'      : 0x223b,
    'l'                      : 0x142,
    'lambda'                 : 0x3bb,
    'lambdabar'              : 0x19b,
    'langle'                 : 0x27e8,
    'lasp'                   : 0x2bd,
    'lbrace'                 : 0x7b,
    'lbrack'                 : 0x5b,
    'lceil'                  : 0x2308,
    'ldots'                  : 0x2026,
    'leadsto'                : 0x21dd,
    'leftarrow'              : 0x2190,
    'leftarrowtail'          : 0x21a2,
    'leftbrace'              : 0x7b,
    'leftharpoonaccent'      : 0x20d0,
    'leftharpoondown'        : 0x21bd,
    'leftharpoonup'          : 0x21bc,
    'leftleftarrows'         : 0x21c7,
    'leftparen'              : 0x28,
    'leftrightarrow'         : 0x2194,
    'leftrightarrows'        : 0x21c6,
    'leftrightharpoons'      : 0x21cb,
    'leftrightsquigarrow'    : 0x21ad,
    'leftsquigarrow'         : 0x219c,
    'leftthreetimes'         : 0x22cb,
    'leq'                    : 0x2264,
    'leqq'                   : 0x2266,
    'leqslant'               : 0x2a7d,
    'less'                   : 0x3c,
    'lessapprox'             : 0x2a85,
    'lessdot'                : 0x22d6,
    'lesseqgtr'              : 0x22da,
    'lesseqqgtr'             : 0x2a8b,
    'lessgtr'                : 0x2276,
    'lesssim'                : 0x2272,
    'lfloor'                 : 0x230a,
    'lgroup'                 : 0x27ee,
    'lhd'                    : 0x25c1,
    'll'                     : 0x226a,
    'llcorner'               : 0x231e,
    'lll'                    : 0x22d8,
    'lnapprox'               : 0x2a89,
    'lneqq'                  : 0x2268,
    'lnsim'                  : 0x22e6,
    'longleftarrow'          : 0x27f5,
    'longleftrightarrow'     : 0x27f7,
    'longmapsto'             : 0x27fc,
    'longrightarrow'         : 0x27f6,
    'looparrowleft'          : 0x21ab,
    'looparrowright'         : 0x21ac,
    'lq'                     : 0x2018,
    'lrcorner'               : 0x231f,
    'ltimes'                 : 0x22c9,
    'macron'                 : 0xaf,
    'maltese'                : 0x2720,
    'mapsdown'               : 0x21a7,
    'mapsfrom'               : 0x21a4,
    'mapsto'                 : 0x21a6,
    'mapsup'                 : 0x21a5,
    'measeq'                 : 0x225e,
    'measuredangle'          : 0x2221,
    'measuredrightangle'     : 0x22be,
    'merge'                  : 0x2a55,
    'mho'                    : 0x2127,
    'mid'                    : 0x2223,
    'minus'                  : 0x2212,
    'minuscolon'             : 0x2239,
    'models'                 : 0x22a7,
    'mp'                     : 0x2213,
    'mu'                     : 0x3bc,
    'multimap'               : 0x22b8,
    'nLeftarrow'             : 0x21cd,
    'nLeftrightarrow'        : 0x21ce,
    'nRightarrow'            : 0x21cf,
    'nVDash'                 : 0x22af,
    'nVdash'                 : 0x22ae,
    'nabla'                  : 0x2207,
    'napprox'                : 0x2249,
    'natural'                : 0x266e,
    'ncong'                  : 0x2247,
    'ne'                     : 0x2260,
    'nearrow'                : 0x2197,
    'neg'                    : 0xac,
    'neq'                    : 0x2260,
    'nequiv'                 : 0x2262,
    'nexists'                : 0x2204,
    'ngeq'                   : 0x2271,
    'ngtr'                   : 0x226f,
    'ngtrless'               : 0x2279,
    'ngtrsim'                : 0x2275,
    'ni'                     : 0x220b,
    'niobar'                 : 0x22fe,
    'nis'                    : 0x22fc,
    'nisd'                   : 0x22fa,
    'nleftarrow'             : 0x219a,
    'nleftrightarrow'        : 0x21ae,
    'nleq'                   : 0x2270,
    'nless'                  : 0x226e,
    'nlessgtr'               : 0x2278,
    'nlesssim'               : 0x2274,
    'nmid'                   : 0x2224,
    'not'                    : 0x338,
    'notin'                  : 0x2209,
    'notsmallowns'           : 0x220c,
    'nparallel'              : 0x2226,
    'nprec'                  : 0x2280,
    'npreccurlyeq'           : 0x22e0,
    'nrightarrow'            : 0x219b,
    'nsim'                   : 0x2241,
    'nsimeq'                 : 0x2244,
    'nsqsubseteq'            : 0x22e2,
    'nsqsupseteq'            : 0x22e3,
    'nsubset'                : 0x2284,
    'nsubseteq'              : 0x2288,
    'nsucc'                  : 0x2281,
    'nsucccurlyeq'           : 0x22e1,
    'nsupset'                : 0x2285,
    'nsupseteq'              : 0x2289,
    'ntriangleleft'          : 0x22ea,
    'ntrianglelefteq'        : 0x22ec,
    'ntriangleright'         : 0x22eb,
    'ntrianglerighteq'       : 0x22ed,
    'nu'                     : 0x3bd,
    'nvDash'                 : 0x22ad,
    'nvdash'                 : 0x22ac,
    'nwarrow'                : 0x2196,
    'o'                      : 0xf8,
    'obar'                   : 0x233d,
    'ocirc'                  : 0x30a,
    'odot'                   : 0x2299,
    'oe'                     : 0x153,
    'oequal'                 : 0x229c,
    'oiiint'                 : 0x2230,
    'oiint'                  : 0x222f,
    'oint'                   : 0x222e,
    'omega'                  : 0x3c9,
    'ominus'                 : 0x2296,
    'oplus'                  : 0x2295,
    'origof'                 : 0x22b6,
    'oslash'                 : 0x2298,
    'otimes'                 : 0x2297,
    'overarc'                : 0x311,
    'overleftarrow'          : 0x20d6,
    'overleftrightarrow'     : 0x20e1,
    'parallel'               : 0x2225,
    'partial'                : 0x2202,
    'perp'                   : 0x27c2,
    'perthousand'            : 0x2030,
    'phi'                    : 0x3d5,
    'pi'                     : 0x3c0,
    'pitchfork'              : 0x22d4,
    'plus'                   : 0x2b,
    'pm'                     : 0xb1,
    'prec'                   : 0x227a,
    'precapprox'             : 0x2ab7,
    'preccurlyeq'            : 0x227c,
    'preceq'                 : 0x227c,
    'precnapprox'            : 0x2ab9,
    'precnsim'               : 0x22e8,
    'precsim'                : 0x227e,
    'prime'                  : 0x2032,
    'prod'                   : 0x220f,
    'propto'                 : 0x221d,
    'prurel'                 : 0x22b0,
    'psi'                    : 0x3c8,
    'quad'                   : 0x2003,
    'questeq'                : 0x225f,
    'rangle'                 : 0x27e9,
    'rasp'                   : 0x2bc,
    'ratio'                  : 0x2236,
    'rbrace'                 : 0x7d,
    'rbrack'                 : 0x5d,
    'rceil'                  : 0x2309,
    'rfloor'                 : 0x230b,
    'rgroup'                 : 0x27ef,
    'rhd'                    : 0x25b7,
    'rho'                    : 0x3c1,
    'rightModels'            : 0x22ab,
    'rightangle'             : 0x221f,
    'rightarrow'             : 0x2192,
    'rightarrowbar'          : 0x21e5,
    'rightarrowtail'         : 0x21a3,
    'rightassert'            : 0x22a6,
    'rightbrace'             : 0x7d,
    'rightharpoonaccent'     : 0x20d1,
    'rightharpoondown'       : 0x21c1,
    'rightharpoonup'         : 0x21c0,
    'rightleftarrows'        : 0x21c4,
    'rightleftharpoons'      : 0x21cc,
    'rightparen'             : 0x29,
    'rightrightarrows'       : 0x21c9,
    'rightsquigarrow'        : 0x219d,
    'rightthreetimes'        : 0x22cc,
    'rightzigzagarrow'       : 0x21dd,
    'ring'                   : 0x2da,
    'risingdotseq'           : 0x2253,
    'rq'                     : 0x2019,
    'rtimes'                 : 0x22ca,
    'scrB'                   : 0x212c,
    'scrE'                   : 0x2130,
    'scrF'                   : 0x2131,
    'scrH'                   : 0x210b,
    'scrI'                   : 0x2110,
    'scrL'                   : 0x2112,
    'scrM'                   : 0x2133,
    'scrR'                   : 0x211b,
    'scre'                   : 0x212f,
    'scrg'                   : 0x210a,
    'scro'                   : 0x2134,
    'scurel'                 : 0x22b1,
    'searrow'                : 0x2198,
    'setminus'               : 0x2216,
    'sharp'                  : 0x266f,
    'sigma'                  : 0x3c3,
    'sim'                    : 0x223c,
    'simeq'                  : 0x2243,
    'simneqq'                : 0x2246,
    'sinewave'               : 0x223f,
    'slash'                  : 0x2215,
    'smallin'                : 0x220a,
    'smallintclockwise'      : 0x2231,
    'smallointctrcclockwise' : 0x2233,
    'smallowns'              : 0x220d,
    'smallsetminus'          : 0x2216,
    'smallvarointclockwise'  : 0x2232,
    'smile'                  : 0x2323,
    'solbar'                 : 0x233f,
    'spadesuit'              : 0x2660,
    'spadesuitopen'          : 0x2664,
    'sphericalangle'         : 0x2222,
    'sqcap'                  : 0x2293,
    'sqcup'                  : 0x2294,
    'sqsubset'               : 0x228f,
    'sqsubseteq'             : 0x2291,
    'sqsubsetneq'            : 0x22e4,
    'sqsupset'               : 0x2290,
    'sqsupseteq'             : 0x2292,
    'sqsupsetneq'            : 0x22e5,
    'ss'                     : 0xdf,
    'star'                   : 0x22c6,
    'stareq'                 : 0x225b,
    'sterling'               : 0xa3,
    'subset'                 : 0x2282,
    'subseteq'               : 0x2286,
    'subseteqq'              : 0x2ac5,
    'subsetneq'              : 0x228a,
    'subsetneqq'             : 0x2acb,
    'succ'                   : 0x227b,
    'succapprox'             : 0x2ab8,
    'succcurlyeq'            : 0x227d,
    'succeq'                 : 0x227d,
    'succnapprox'            : 0x2aba,
    'succnsim'               : 0x22e9,
    'succsim'                : 0x227f,
    'sum'                    : 0x2211,
    'supset'                 : 0x2283,
    'supseteq'               : 0x2287,
    'supseteqq'              : 0x2ac6,
    'supsetneq'              : 0x228b,
    'supsetneqq'             : 0x2acc,
    'swarrow'                : 0x2199,
    't'                      : 0x361,
    'tau'                    : 0x3c4,
    'textasciiacute'         : 0xb4,
    'textasciicircum'        : 0x5e,
    'textasciigrave'         : 0x60,
    'textasciitilde'         : 0x7e,
    'textexclamdown'         : 0xa1,
    'textquestiondown'       : 0xbf,
    'textquotedblleft'       : 0x201c,
    'textquotedblright'      : 0x201d,
    'therefore'              : 0x2234,
    'theta'                  : 0x3b8,
    'thickspace'             : 0x2005,
    'thorn'                  : 0xfe,
    'tilde'                  : 0x303,
    'times'                  : 0xd7,
    'to'                     : 0x2192,
    'top'                    : 0x22a4,
    'triangle'               : 0x25b3,
    'triangledown'           : 0x25bf,
    'triangleeq'             : 0x225c,
    'triangleleft'           : 0x25c1,
    'trianglelefteq'         : 0x22b4,
    'triangleq'              : 0x225c,
    'triangleright'          : 0x25b7,
    'trianglerighteq'        : 0x22b5,
    'turnednot'              : 0x2319,
    'twoheaddownarrow'       : 0x21a1,
    'twoheadleftarrow'       : 0x219e,
    'twoheadrightarrow'      : 0x21a0,
    'twoheaduparrow'         : 0x219f,
    'ulcorner'               : 0x231c,
    'underbar'               : 0x331,
    'unlhd'                  : 0x22b4,
    'unrhd'                  : 0x22b5,
    'uparrow'                : 0x2191,
    'updownarrow'            : 0x2195,
    'updownarrowbar'         : 0x21a8,
    'updownarrows'           : 0x21c5,
    'upharpoonleft'          : 0x21bf,
    'upharpoonright'         : 0x21be,
    'uplus'                  : 0x228e,
    'upsilon'                : 0x3c5,
    'upuparrows'             : 0x21c8,
    'urcorner'               : 0x231d,
    'vDash'                  : 0x22a8,
    'varepsilon'             : 0x3b5,
    'varisinobar'            : 0x22f6,
    'varisins'               : 0x22f3,
    'varkappa'               : 0x3f0,
    'varlrtriangle'          : 0x22bf,
    'varniobar'              : 0x22fd,
    'varnis'                 : 0x22fb,
    'varnothing'             : 0x2205,
    'varphi'                 : 0x3c6,
    'varpi'                  : 0x3d6,
    'varpropto'              : 0x221d,
    'varrho'                 : 0x3f1,
    'varsigma'               : 0x3c2,
    'vartheta'               : 0x3d1,
    'vartriangle'            : 0x25b5,
    'vartriangleleft'        : 0x22b2,
    'vartriangleright'       : 0x22b3,
    'vdash'                  : 0x22a2,
    'vdots'                  : 0x22ee,
    'vec'                    : 0x20d7,
    'vee'                    : 0x2228,
    'veebar'                 : 0x22bb,
    'veeeq'                  : 0x225a,
    'vert'                   : 0x7c,
    'wedge'                  : 0x2227,
    'wedgeq'                 : 0x2259,
    'widebar'                : 0x305,
    'widehat'                : 0x302,
    'widetilde'              : 0x303,
    'wp'                     : 0x2118,
    'wr'                     : 0x2240,
    'xi'                     : 0x3be,
    'yen'                    : 0xa5,
    'zeta'                   : 0x3b6,
    '{'                      : 0x7b,
    '|'                      : 0x2016,
    '}'                      : 0x7d,
}
tex2uni['__angbracketleft__'] = tex2uni['langle']
tex2uni['__angbracketright__'] = tex2uni['rangle']

# Each element is a 4-tuple of the form:
#   src_start, src_end, dst_font, dst_start

type _EntryTypeIn = tuple[str, str, str, str | CharacterCodeType]
type _EntryTypeOut = tuple[CharacterCodeType, CharacterCodeType, str, CharacterCodeType]

_stix_virtual_fonts: dict[str, dict[str, list[_EntryTypeIn]] | list[_EntryTypeIn]] = {
    'bb': {
        "rm": [
            ("\N{DIGIT ZERO}",
             "\N{DIGIT NINE}",
             "rm",
             "\N{MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO}"),
            ("\N{LATIN CAPITAL LETTER A}",
             "\N{LATIN CAPITAL LETTER B}",
             "rm",
             "\N{MATHEMATICAL DOUBLE-STRUCK CAPITAL A}"),
            ("\N{LATIN CAPITAL LETTER C}",
             "\N{LATIN CAPITAL LETTER C}",
             "rm",
             "\N{DOUBLE-STRUCK CAPITAL C}"),
            ("\N{LATIN CAPITAL LETTER D}",
             "\N{LATIN CAPITAL LETTER G}",
             "rm",
             "\N{MATHEMATICAL DOUBLE-STRUCK CAPITAL D}"),
            ("\N{LATIN CAPITAL LETTER H}",
             "\N{LATIN CAPITAL LETTER H}",
             "rm",
             "\N{DOUBLE-STRUCK CAPITAL H}"),
            ("\N{LATIN CAPITAL LETTER I}",
             "\N{LATIN CAPITAL LETTER M}",
             "rm",
             "\N{MATHEMATICAL DOUBLE-STRUCK CAPITAL I}"),
            ("\N{LATIN CAPITAL LETTER N}",
             "\N{LATIN CAPITAL LETTER N}",
             "rm",
             "\N{DOUBLE-STRUCK CAPITAL N}"),
            ("\N{LATIN CAPITAL LETTER O}",
             "\N{LATIN CAPITAL LETTER O}",
             "rm",
             "\N{MATHEMATICAL DOUBLE-STRUCK CAPITAL O}"),
            ("\N{LATIN CAPITAL LETTER P}",
             "\N{LATIN CAPITAL LETTER Q}",
             "rm",
             "\N{DOUBLE-STRUCK CAPITAL P}"),
            ("\N{LATIN CAPITAL LETTER R}",
             "\N{LATIN CAPITAL LETTER R}",
             "rm",
             "\N{DOUBLE-STRUCK CAPITAL R}"),
            ("\N{LATIN CAPITAL LETTER S}",
             "\N{LATIN CAPITAL LETTER Y}",
             "rm",
             "\N{MATHEMATICAL DOUBLE-STRUCK CAPITAL S}"),
            ("\N{LATIN CAPITAL LETTER Z}",
             "\N{LATIN CAPITAL LETTER Z}",
             "rm",
             "\N{DOUBLE-STRUCK CAPITAL Z}"),
            ("\N{LATIN SMALL LETTER A}",
             "\N{LATIN SMALL LETTER Z}",
             "rm",
             "\N{MATHEMATICAL DOUBLE-STRUCK SMALL A}"),
            ("\N{GREEK CAPITAL LETTER GAMMA}",
             "\N{GREEK CAPITAL LETTER GAMMA}",
             "rm",
             "\N{DOUBLE-STRUCK CAPITAL GAMMA}"),
            ("\N{GREEK CAPITAL LETTER PI}",
             "\N{GREEK CAPITAL LETTER PI}",
             "rm",
             "\N{DOUBLE-STRUCK CAPITAL PI}"),
            ("\N{GREEK CAPITAL LETTER SIGMA}",
             "\N{GREEK CAPITAL LETTER SIGMA}",
             "rm",
             "\N{DOUBLE-STRUCK N-ARY SUMMATION}"),
            ("\N{GREEK SMALL LETTER GAMMA}",
             "\N{GREEK SMALL LETTER GAMMA}",
             "rm",
             "\N{DOUBLE-STRUCK SMALL GAMMA}"),
            ("\N{GREEK SMALL LETTER PI}",
             "\N{GREEK SMALL LETTER PI}",
             "rm",
             "\N{DOUBLE-STRUCK SMALL PI}"),
        ],
        "it": [
            ("\N{DIGIT ZERO}",
             "\N{DIGIT NINE}",
             "rm",
             "\N{MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO}"),
            ("\N{LATIN CAPITAL LETTER A}",
             "\N{LATIN CAPITAL LETTER B}",
             "it",
             0xe154),
            ("\N{LATIN CAPITAL LETTER C}",
             "\N{LATIN CAPITAL LETTER C}",
             "it",
             "\N{DOUBLE-STRUCK CAPITAL C}"),
            ("\N{LATIN CAPITAL LETTER D}",
             "\N{LATIN CAPITAL LETTER D}",
             "it",
             "\N{DOUBLE-STRUCK ITALIC CAPITAL D}"),
            ("\N{LATIN CAPITAL LETTER E}",
             "\N{LATIN CAPITAL LETTER G}",
             "it",
             0xe156),
            ("\N{LATIN CAPITAL LETTER H}",
             "\N{LATIN CAPITAL LETTER H}",
             "it",
             "\N{DOUBLE-STRUCK CAPITAL H}"),
            ("\N{LATIN CAPITAL LETTER I}",
             "\N{LATIN CAPITAL LETTER M}",
             "it",
             0xe159),
            ("\N{LATIN CAPITAL LETTER N}",
             "\N{LATIN CAPITAL LETTER N}",
             "it",
             "\N{DOUBLE-STRUCK CAPITAL N}"),
            ("\N{LATIN CAPITAL LETTER O}",
             "\N{LATIN CAPITAL LETTER O}",
             "it",
             0xe15e),
            ("\N{LATIN CAPITAL LETTER P}",
             "\N{LATIN CAPITAL LETTER Q}",
             "it",
             "\N{DOUBLE-STRUCK CAPITAL P}"),
            ("\N{LATIN CAPITAL LETTER R}",
             "\N{LATIN CAPITAL LETTER R}",
             "it",
             "\N{DOUBLE-STRUCK CAPITAL R}"),
            ("\N{LATIN CAPITAL LETTER S}",
             "\N{LATIN CAPITAL LETTER Y}",
             "it",
             0xe15f),
            ("\N{LATIN CAPITAL LETTER Z}",
             "\N{LATIN CAPITAL LETTER Z}",
             "it",
             "\N{DOUBLE-STRUCK CAPITAL Z}"),
            ("\N{LATIN SMALL LETTER A}",
             "\N{LATIN SMALL LETTER C}",
             "it",
             0xe166),
            ("\N{LATIN SMALL LETTER D}",
             "\N{LATIN SMALL LETTER E}",
             "it",
             "\N{DOUBLE-STRUCK ITALIC SMALL D}"),
            ("\N{LATIN SMALL LETTER F}",
             "\N{LATIN SMALL LETTER H}",
             "it",
             0xe169),
            ("\N{LATIN SMALL LETTER I}",
             "\N{LATIN SMALL LETTER J}",
             "it",
             "\N{DOUBLE-STRUCK ITALIC SMALL I}"),
            ("\N{LATIN SMALL LETTER K}",
             "\N{LATIN SMALL LETTER Z}",
             "it",
             0xe16c),
            ("\N{GREEK CAPITAL LETTER GAMMA}",
             "\N{GREEK CAPITAL LETTER GAMMA}",
             "it",
             "\N{DOUBLE-STRUCK CAPITAL GAMMA}"),  # \Gamma (not in beta STIX fonts)
            ("\N{GREEK CAPITAL LETTER PI}",
             "\N{GREEK CAPITAL LETTER PI}",
             "it",
             "\N{DOUBLE-STRUCK CAPITAL PI}"),
            ("\N{GREEK CAPITAL LETTER SIGMA}",
             "\N{GREEK CAPITAL LETTER SIGMA}",
             "rm",  # not in STIX italic
             "\N{DOUBLE-STRUCK N-ARY SUMMATION}"),  # \Sigma (not in beta STIX fonts)
            ("\N{GREEK SMALL LETTER GAMMA}",
             "\N{GREEK SMALL LETTER GAMMA}",
             "it",
             "\N{DOUBLE-STRUCK SMALL GAMMA}"),  # \gamma (not in beta STIX fonts)
            ("\N{GREEK SMALL LETTER PI}",
             "\N{GREEK SMALL LETTER PI}",
             "it",
             "\N{DOUBLE-STRUCK SMALL PI}"),
        ],
        "bf": [
            ("\N{DIGIT ZERO}",
             "\N{DIGIT NINE}",
             "rm",
             "\N{MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO}"),
            ("\N{LATIN CAPITAL LETTER A}",
             "\N{LATIN CAPITAL LETTER B}",
             "bf",
             0xe38a),
            ("\N{LATIN CAPITAL LETTER C}",
             "\N{LATIN CAPITAL LETTER C}",
             "bf",
             "\N{DOUBLE-STRUCK CAPITAL C}"),
            ("\N{LATIN CAPITAL LETTER D}",
             "\N{LATIN CAPITAL LETTER D}",
             "bf",
             "\N{DOUBLE-STRUCK ITALIC CAPITAL D}"),
            ("\N{LATIN CAPITAL LETTER E}",
             "\N{LATIN CAPITAL LETTER G}",
             "bf",
             0xe38d),
            ("\N{LATIN CAPITAL LETTER H}",
             "\N{LATIN CAPITAL LETTER H}",
             "bf",
             "\N{DOUBLE-STRUCK CAPITAL H}"),
            ("\N{LATIN CAPITAL LETTER I}",
             "\N{LATIN CAPITAL LETTER M}",
             "bf",
             0xe390),
            ("\N{LATIN CAPITAL LETTER N}",
             "\N{LATIN CAPITAL LETTER N}",
             "bf",
             "\N{DOUBLE-STRUCK CAPITAL N}"),
            ("\N{LATIN CAPITAL LETTER O}",
             "\N{LATIN CAPITAL LETTER O}",
             "bf",
             0xe395),
            ("\N{LATIN CAPITAL LETTER P}",
             "\N{LATIN CAPITAL LETTER Q}",
             "bf",
             "\N{DOUBLE-STRUCK CAPITAL P}"),
            ("\N{LATIN CAPITAL LETTER R}",
             "\N{LATIN CAPITAL LETTER R}",
             "bf",
             "\N{DOUBLE-STRUCK CAPITAL R}"),
            ("\N{LATIN CAPITAL LETTER S}",
             "\N{LATIN CAPITAL LETTER Y}",
             "bf",
             0xe396),
            ("\N{LATIN CAPITAL LETTER Z}",
             "\N{LATIN CAPITAL LETTER Z}",
             "bf",
             "\N{DOUBLE-STRUCK CAPITAL Z}"),
            ("\N{LATIN SMALL LETTER A}",
             "\N{LATIN SMALL LETTER C}",
             "bf",
             0xe39d),
            ("\N{LATIN SMALL LETTER D}",
             "\N{LATIN SMALL LETTER E}",
             "bf",
             "\N{DOUBLE-STRUCK ITALIC SMALL D}"),
            ("\N{LATIN SMALL LETTER F}",
             "\N{LATIN SMALL LETTER H}",
             "bf",
             0xe3a2),
            ("\N{LATIN SMALL LETTER I}",
             "\N{LATIN SMALL LETTER J}",
             "bf",
             "\N{DOUBLE-STRUCK ITALIC SMALL I}"),
            ("\N{LATIN SMALL LETTER K}",
             "\N{LATIN SMALL LETTER Z}",
             "bf",
             0xe3a7),
            ("\N{GREEK CAPITAL LETTER GAMMA}",
             "\N{GREEK CAPITAL LETTER GAMMA}",
             "bf",
             "\N{DOUBLE-STRUCK CAPITAL GAMMA}"),
            ("\N{GREEK CAPITAL LETTER PI}",
             "\N{GREEK CAPITAL LETTER PI}",
             "bf",
             "\N{DOUBLE-STRUCK CAPITAL PI}"),
            ("\N{GREEK CAPITAL LETTER SIGMA}",
             "\N{GREEK CAPITAL LETTER SIGMA}",
             "bf",
             "\N{DOUBLE-STRUCK N-ARY SUMMATION}"),
            ("\N{GREEK SMALL LETTER GAMMA}",
             "\N{GREEK SMALL LETTER GAMMA}",
             "bf",
             "\N{DOUBLE-STRUCK SMALL GAMMA}"),
            ("\N{GREEK SMALL LETTER PI}",
             "\N{GREEK SMALL LETTER PI}",
             "bf",
             "\N{DOUBLE-STRUCK SMALL PI}"),
        ],
    },
    'cal': [
        ("\N{LATIN CAPITAL LETTER A}",
         "\N{LATIN CAPITAL LETTER Z}",
         "it",
         0xe22d),
    ],
    'frak': {
        "rm": [
            ("\N{LATIN CAPITAL LETTER A}",
             "\N{LATIN CAPITAL LETTER B}",
             "rm",
             "\N{MATHEMATICAL FRAKTUR CAPITAL A}"),
            ("\N{LATIN CAPITAL LETTER C}",
             "\N{LATIN CAPITAL LETTER C}",
             "rm",
             "\N{BLACK-LETTER CAPITAL C}"),
            ("\N{LATIN CAPITAL LETTER D}",
             "\N{LATIN CAPITAL LETTER G}",
             "rm",
             "\N{MATHEMATICAL FRAKTUR CAPITAL D}"),
            ("\N{LATIN CAPITAL LETTER H}",
             "\N{LATIN CAPITAL LETTER H}",
             "rm",
             "\N{BLACK-LETTER CAPITAL H}"),
            ("\N{LATIN CAPITAL LETTER I}",
             "\N{LATIN CAPITAL LETTER I}",
             "rm",
             "\N{BLACK-LETTER CAPITAL I}"),
            ("\N{LATIN CAPITAL LETTER J}",
             "\N{LATIN CAPITAL LETTER Q}",
             "rm",
             "\N{MATHEMATICAL FRAKTUR CAPITAL J}"),
            ("\N{LATIN CAPITAL LETTER R}",
             "\N{LATIN CAPITAL LETTER R}",
             "rm",
             "\N{BLACK-LETTER CAPITAL R}"),
            ("\N{LATIN CAPITAL LETTER S}",
             "\N{LATIN CAPITAL LETTER Y}",
             "rm",
             "\N{MATHEMATICAL FRAKTUR CAPITAL S}"),
            ("\N{LATIN CAPITAL LETTER Z}",
             "\N{LATIN CAPITAL LETTER Z}",
             "rm",
             "\N{BLACK-LETTER CAPITAL Z}"),
            ("\N{LATIN SMALL LETTER A}",
             "\N{LATIN SMALL LETTER Z}",
             "rm",
             "\N{MATHEMATICAL FRAKTUR SMALL A}"),
            ],
        "bf": [
            ("\N{LATIN CAPITAL LETTER A}",
             "\N{LATIN CAPITAL LETTER Z}",
             "bf",
             "\N{MATHEMATICAL BOLD FRAKTUR CAPITAL A}"),
            ("\N{LATIN SMALL LETTER A}",
             "\N{LATIN SMALL LETTER Z}",
             "bf",
             "\N{MATHEMATICAL BOLD FRAKTUR SMALL A}"),
        ],
    },
    'scr': [
        ("\N{LATIN CAPITAL LETTER A}",
         "\N{LATIN CAPITAL LETTER A}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL A}"),
        ("\N{LATIN CAPITAL LETTER B}",
         "\N{LATIN CAPITAL LETTER B}",
         "it",
         "\N{SCRIPT CAPITAL B}"),
        ("\N{LATIN CAPITAL LETTER C}",
         "\N{LATIN CAPITAL LETTER D}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL C}"),
        ("\N{LATIN CAPITAL LETTER E}",
         "\N{LATIN CAPITAL LETTER F}",
         "it",
         "\N{SCRIPT CAPITAL E}"),
        ("\N{LATIN CAPITAL LETTER G}",
         "\N{LATIN CAPITAL LETTER G}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL G}"),
        ("\N{LATIN CAPITAL LETTER H}",
         "\N{LATIN CAPITAL LETTER H}",
         "it",
         "\N{SCRIPT CAPITAL H}"),
        ("\N{LATIN CAPITAL LETTER I}",
         "\N{LATIN CAPITAL LETTER I}",
         "it",
         "\N{SCRIPT CAPITAL I}"),
        ("\N{LATIN CAPITAL LETTER J}",
         "\N{LATIN CAPITAL LETTER K}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL J}"),
        ("\N{LATIN CAPITAL LETTER L}",
         "\N{LATIN CAPITAL LETTER L}",
         "it",
         "\N{SCRIPT CAPITAL L}"),
        ("\N{LATIN CAPITAL LETTER M}",
         "\N{LATIN CAPITAL LETTER M}",
         "it",
         "\N{SCRIPT CAPITAL M}"),
        ("\N{LATIN CAPITAL LETTER N}",
         "\N{LATIN CAPITAL LETTER Q}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL N}"),
        ("\N{LATIN CAPITAL LETTER R}",
         "\N{LATIN CAPITAL LETTER R}",
         "it",
         "\N{SCRIPT CAPITAL R}"),
        ("\N{LATIN CAPITAL LETTER S}",
         "\N{LATIN CAPITAL LETTER Z}",
         "it",
         "\N{MATHEMATICAL SCRIPT CAPITAL S}"),
        ("\N{LATIN SMALL LETTER A}",
         "\N{LATIN SMALL LETTER D}",
         "it",
         "\N{MATHEMATICAL SCRIPT SMALL A}"),
        ("\N{LATIN SMALL LETTER E}",
         "\N{LATIN SMALL LETTER E}",
         "it",
         "\N{SCRIPT SMALL E}"),
        ("\N{LATIN SMALL LETTER F}",
         "\N{LATIN SMALL LETTER F}",
         "it",
         "\N{MATHEMATICAL SCRIPT SMALL F}"),
        ("\N{LATIN SMALL LETTER G}",
         "\N{LATIN SMALL LETTER G}",
         "it",
         "\N{SCRIPT SMALL G}"),
        ("\N{LATIN SMALL LETTER H}",
         "\N{LATIN SMALL LETTER N}",
         "it",
         "\N{MATHEMATICAL SCRIPT SMALL H}"),
        ("\N{LATIN SMALL LETTER O}",
         "\N{LATIN SMALL LETTER O}",
         "it",
         "\N{SCRIPT SMALL O}"),
        ("\N{LATIN SMALL LETTER P}",
         "\N{LATIN SMALL LETTER Z}",
         "it",
         "\N{MATHEMATICAL SCRIPT SMALL P}"),
    ],
    'sf': {
        "rm": [
            ("\N{DIGIT ZERO}",
             "\N{DIGIT NINE}",
             "rm",
             "\N{MATHEMATICAL SANS-SERIF DIGIT ZERO}"),
            ("\N{LATIN CAPITAL LETTER A}",
             "\N{LATIN CAPITAL LETTER Z}",
             "rm",
             "\N{MATHEMATICAL SANS-SERIF CAPITAL A}"),
            ("\N{LATIN SMALL LETTER A}",
             "\N{LATIN SMALL LETTER Z}",
             "rm",
             "\N{MATHEMATICAL SANS-SERIF SMALL A}"),
            ("\N{GREEK CAPITAL LETTER ALPHA}",
             "\N{GREEK CAPITAL LETTER OMEGA}",
             "rm",
             0xe17d),
            ("\N{GREEK SMALL LETTER ALPHA}",
             "\N{GREEK SMALL LETTER OMEGA}",
             "rm",
             0xe196),
            ("\N{GREEK THETA SYMBOL}",
             "\N{GREEK THETA SYMBOL}",
             "rm",
             0xe1b0),
            ("\N{GREEK PHI SYMBOL}",
             "\N{GREEK PHI SYMBOL}",
             "rm",
             0xe1b1),
            ("\N{GREEK PI SYMBOL}",
             "\N{GREEK PI SYMBOL}",
             "rm",
             0xe1b3),
            ("\N{GREEK RHO SYMBOL}",
             "\N{GREEK RHO SYMBOL}",
             "rm",
             0xe1b2),
            ("\N{GREEK LUNATE EPSILON SYMBOL}",
             "\N{GREEK LUNATE EPSILON SYMBOL}",
             "rm",
             0xe1af),
            ("\N{PARTIAL DIFFERENTIAL}",
             "\N{PARTIAL DIFFERENTIAL}",
             "rm",
             0xe17c),
        ],
        "it": [
            # These numerals are actually upright.  We don't actually
            # want italic numerals ever.
            ("\N{DIGIT ZERO}",
             "\N{DIGIT NINE}",
             "rm",
             "\N{MATHEMATICAL SANS-SERIF DIGIT ZERO}"),
            ("\N{LATIN CAPITAL LETTER A}",
             "\N{LATIN CAPITAL LETTER Z}",
             "it",
             "\N{MATHEMATICAL SANS-SERIF ITALIC CAPITAL A}"),
            ("\N{LATIN SMALL LETTER A}",
             "\N{LATIN SMALL LETTER Z}",
             "it",
             "\N{MATHEMATICAL SANS-SERIF ITALIC SMALL A}"),
            ("\N{GREEK CAPITAL LETTER ALPHA}",
             "\N{GREEK CAPITAL LETTER OMEGA}",
             "rm",
             0xe17d),
            ("\N{GREEK SMALL LETTER ALPHA}",
             "\N{GREEK SMALL LETTER OMEGA}",
             "it",
             0xe1d8),
            ("\N{GREEK THETA SYMBOL}",
             "\N{GREEK THETA SYMBOL}",
             "it",
             0xe1f2),
            ("\N{GREEK PHI SYMBOL}",
             "\N{GREEK PHI SYMBOL}",
             "it",
             0xe1f3),
            ("\N{GREEK PI SYMBOL}",
             "\N{GREEK PI SYMBOL}",
             "it",
             0xe1f5),
            ("\N{GREEK RHO SYMBOL}",
             "\N{GREEK RHO SYMBOL}",
             "it",
             0xe1f4),
            ("\N{GREEK LUNATE EPSILON SYMBOL}",
             "\N{GREEK LUNATE EPSILON SYMBOL}",
             "it",
             0xe1f1),
        ],
        "bf": [
            ("\N{DIGIT ZERO}",
             "\N{DIGIT NINE}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD DIGIT ZERO}"),
            ("\N{LATIN CAPITAL LETTER A}",
             "\N{LATIN CAPITAL LETTER Z}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD CAPITAL A}"),
            ("\N{LATIN SMALL LETTER A}",
             "\N{LATIN SMALL LETTER Z}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD SMALL A}"),
            ("\N{GREEK CAPITAL LETTER ALPHA}",
             "\N{GREEK CAPITAL LETTER OMEGA}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD CAPITAL ALPHA}"),
            ("\N{GREEK SMALL LETTER ALPHA}",
             "\N{GREEK SMALL LETTER OMEGA}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD SMALL ALPHA}"),
            ("\N{GREEK THETA SYMBOL}",
             "\N{GREEK THETA SYMBOL}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD THETA SYMBOL}"),
            ("\N{GREEK PHI SYMBOL}",
             "\N{GREEK PHI SYMBOL}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD PHI SYMBOL}"),
            ("\N{GREEK PI SYMBOL}",
             "\N{GREEK PI SYMBOL}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD PI SYMBOL}"),
            ("\N{GREEK KAPPA SYMBOL}",
             "\N{GREEK KAPPA SYMBOL}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD KAPPA SYMBOL}"),
            ("\N{GREEK RHO SYMBOL}",
             "\N{GREEK RHO SYMBOL}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD RHO SYMBOL}"),
            ("\N{GREEK LUNATE EPSILON SYMBOL}",
             "\N{GREEK LUNATE EPSILON SYMBOL}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD EPSILON SYMBOL}"),
            ("\N{PARTIAL DIFFERENTIAL}",
             "\N{PARTIAL DIFFERENTIAL}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD PARTIAL DIFFERENTIAL}"),
            ("\N{NABLA}",
             "\N{NABLA}",
             "bf",
             "\N{MATHEMATICAL SANS-SERIF BOLD NABLA}"),
        ],
        "bfit": [
            ("\N{LATIN CAPITAL LETTER A}",
             "\N{LATIN CAPITAL LETTER Z}",
             "bfit",
             "\N{MATHEMATICAL BOLD ITALIC CAPITAL A}"),
            ("\N{LATIN SMALL LETTER A}",
             "\N{LATIN SMALL LETTER Z}",
             "bfit",
             "\N{MATHEMATICAL BOLD ITALIC SMALL A}"),
            ("\N{GREEK CAPITAL LETTER GAMMA}",
             "\N{GREEK CAPITAL LETTER OMEGA}",
             "bfit",
             "\N{MATHEMATICAL BOLD ITALIC CAPITAL GAMMA}"),
            ("\N{GREEK SMALL LETTER ALPHA}",
             "\N{GREEK SMALL LETTER OMEGA}",
             "bfit",
             "\N{MATHEMATICAL BOLD ITALIC SMALL ALPHA}"),
        ],
    },
    'tt': [
        ("\N{DIGIT ZERO}",
         "\N{DIGIT NINE}",
         "rm",
         "\N{MATHEMATICAL MONOSPACE DIGIT ZERO}"),
        ("\N{LATIN CAPITAL LETTER A}",
         "\N{LATIN CAPITAL LETTER Z}",
         "rm",
         "\N{MATHEMATICAL MONOSPACE CAPITAL A}"),
        ("\N{LATIN SMALL LETTER A}",
         "\N{LATIN SMALL LETTER Z}",
         "rm",
         "\N{MATHEMATICAL MONOSPACE SMALL A}")
    ],
}

_stix_virtual_fonts['bb']['normal'] = _stix_virtual_fonts['bb']['it']  # type:ignore[call-overload]
_stix_virtual_fonts['sf']['normal'] = _stix_virtual_fonts['sf']['it']  # type:ignore[call-overload]


@overload
def _normalize_stix_fontcodes(d: _EntryTypeIn) -> _EntryTypeOut: ...


@overload
def _normalize_stix_fontcodes(d: list[_EntryTypeIn]) -> list[_EntryTypeOut]: ...


@overload
def _normalize_stix_fontcodes(d: dict[str, list[_EntryTypeIn] |
                                      dict[str, list[_EntryTypeIn]]]
                              ) -> dict[str, list[_EntryTypeOut] |
                                        dict[str, list[_EntryTypeOut]]]: ...


def _normalize_stix_fontcodes(d):
    if isinstance(d, tuple):
        return tuple(ord(x) if isinstance(x, str) and len(x) == 1 else x for x in d)
    elif isinstance(d, list):
        return [_normalize_stix_fontcodes(x) for x in d]
    elif isinstance(d, dict):
        return {k: _normalize_stix_fontcodes(v) for k, v in d.items()}


stix_virtual_fonts: dict[str, dict[str, list[_EntryTypeOut]] | list[_EntryTypeOut]]
stix_virtual_fonts = _normalize_stix_fontcodes(_stix_virtual_fonts)

# Free redundant list now that it has been normalized
del _stix_virtual_fonts

# Fix some incorrect glyphs.
stix_glyph_fixes: dict[CharacterCodeType, CharacterCodeType] = {
    # Cap and Cup glyphs are swapped.
    0x22d2: 0x22d3,
    0x22d3: 0x22d2,
}

unicode_math_lut: dict[str, dict[CharacterCodeType, CharacterCodeType]] = {
    'up': {
        # digits
        0x30: 0x30,  # 0 -> 0
        0x31: 0x31,  # 1 -> 1
        0x32: 0x32,  # 2 -> 2
        0x33: 0x33,  # 3 -> 3
        0x34: 0x34,  # 4 -> 4
        0x35: 0x35,  # 5 -> 5
        0x36: 0x36,  # 6 -> 6
        0x37: 0x37,  # 7 -> 7
        0x38: 0x38,  # 8 -> 8
        0x39: 0x39,  # 9 -> 9
        # latin upper case
        0x41: 0x41,  # A -> A
        0x42: 0x42,  # B -> B
        0x43: 0x43,  # C -> C
        0x44: 0x44,  # D -> D
        0x45: 0x45,  # E -> E
        0x46: 0x46,  # F -> F
        0x47: 0x47,  # G -> G
        0x48: 0x48,  # H -> H
        0x49: 0x49,  # I -> I
        0x4a: 0x4a,  # J -> J
        0x4b: 0x4b,  # K -> K
        0x4c: 0x4c,  # L -> L
        0x4d: 0x4d,  # M -> M
        0x4e: 0x4e,  # N -> N
        0x4f: 0x4f,  # O -> O
        0x50: 0x50,  # P -> P
        0x51: 0x51,  # Q -> Q
        0x52: 0x52,  # R -> R
        0x53: 0x53,  # S -> S
        0x54: 0x54,  # T -> T
        0x55: 0x55,  # U -> U
        0x56: 0x56,  # V -> V
        0x57: 0x57,  # W -> W
        0x58: 0x58,  # X -> X
        0x59: 0x59,  # Y -> Y
        0x5a: 0x5a,  # Z -> Z
        # latin lower case
        0x61: 0x61,  # a -> a
        0x62: 0x62,  # b -> b
        0x63: 0x63,  # c -> c
        0x64: 0x64,  # d -> d
        0x65: 0x65,  # e -> e
        0x66: 0x66,  # f -> f
        0x67: 0x67,  # g -> g
        0x68: 0x68,  # h -> h
        0x69: 0x69,  # i -> i
        0x6a: 0x6a,  # j -> j
        0x6b: 0x6b,  # k -> k
        0x6c: 0x6c,  # l -> l
        0x6d: 0x6d,  # m -> m
        0x6e: 0x6e,  # n -> n
        0x6f: 0x6f,  # o -> o
        0x70: 0x70,  # p -> p
        0x71: 0x71,  # q -> q
        0x72: 0x72,  # r -> r
        0x73: 0x73,  # s -> s
        0x74: 0x74,  # t -> t
        0x75: 0x75,  # u -> u
        0x76: 0x76,  # v -> v
        0x77: 0x77,  # w -> w
        0x78: 0x78,  # x -> x
        0x79: 0x79,  # y -> y
        0x7a: 0x7a,  # z -> z
        # greek upper case
        0x391: 0x391,  # Α -> Α
        0x392: 0x392,  # Β -> Β
        0x393: 0x393,  # Γ -> Γ
        0x394: 0x394,  # Δ -> Δ
        0x395: 0x395,  # Ε -> Ε
        0x396: 0x396,  # Ζ -> Ζ
        0x397: 0x397,  # Η -> Η
        0x398: 0x398,  # Θ -> Θ
        0x399: 0x399,  # Ι -> Ι
        0x39a: 0x39a,  # Κ -> Κ
        0x39b: 0x39b,  # Λ -> Λ
        0x39c: 0x39c,  # Μ -> Μ
        0x39d: 0x39d,  # Ν -> Ν
        0x39e: 0x39e,  # Ξ -> Ξ
        0x39f: 0x39f,  # Ο -> Ο
        0x3a0: 0x3a0,  # Π -> Π
        0x3a1: 0x3a1,  # Ρ -> Ρ
        0x3f4: 0x3f4,  # ϴ -> ϴ
        0x3a3: 0x3a3,  # Σ -> Σ
        0x3a4: 0x3a4,  # Τ -> Τ
        0x3a5: 0x3a5,  # Υ -> Υ
        0x3a6: 0x3a6,  # Φ -> Φ
        0x3a7: 0x3a7,  # Χ -> Χ
        0x3a8: 0x3a8,  # Ψ -> Ψ
        0x3a9: 0x3a9,  # Ω -> Ω
        # greek lower case
        0x3b1: 0x3b1,  # α -> α
        0x3b2: 0x3b2,  # β -> β
        0x3b3: 0x3b3,  # γ -> γ
        0x3b4: 0x3b4,  # δ -> δ
        0x3b5: 0x3b5,  # ε -> ε
        0x3b6: 0x3b6,  # ζ -> ζ
        0x3b7: 0x3b7,  # η -> η
        0x3b8: 0x3b8,  # θ -> θ
        0x3b9: 0x3b9,  # ι -> ι
        0x3ba: 0x3ba,  # κ -> κ
        0x3bb: 0x3bb,  # λ -> λ
        0x3bc: 0x3bc,  # μ -> μ
        0x3bd: 0x3bd,  # ν -> ν
        0x3be: 0x3be,  # ξ -> ξ
        0x3bf: 0x3bf,  # ο -> ο
        0x3c0: 0x3c0,  # π -> π
        0x3c1: 0x3c1,  # ρ -> ρ
        0x3c2: 0x3c2,  # ς -> ς
        0x3c3: 0x3c3,  # σ -> σ
        0x3c4: 0x3c4,  # τ -> τ
        0x3c5: 0x3c5,  # υ -> υ
        0x3c6: 0x3c6,  # φ -> φ
        0x3c7: 0x3c7,  # χ -> χ
        0x3c8: 0x3c8,  # ψ -> ψ
        0x3c9: 0x3c9,  # ω -> ω
        0x2202: 0x2202,  # ∂ -> ∂
        0x3f5: 0x3f5,  # ϵ -> ϵ
        0x3d1: 0x3d1,  # ϑ -> ϑ
        0x3f0: 0x3f0,  # ϰ -> ϰ
        0x3f1: 0x3d5,  # ϱ -> ϕ
        0x3cf: 0x3f1,  # Ϗ -> ϱ
        0x3d6: 0x3d6,  # ϖ -> ϖ
    },
    'bfup': {
        # digits
        0x30: 0x1d7ce,  # 0 -> 𝟎
        0x31: 0x1d7cf,  # 1 -> 𝟏
        0x32: 0x1d7d0,  # 2 -> 𝟐
        0x33: 0x1d7d1,  # 3 -> 𝟑
        0x34: 0x1d7d2,  # 4 -> 𝟒
        0x35: 0x1d7d3,  # 5 -> 𝟓
        0x36: 0x1d7d4,  # 6 -> 𝟔
        0x37: 0x1d7d5,  # 7 -> 𝟕
        0x38: 0x1d7d6,  # 8 -> 𝟖
        0x39: 0x1d7d7,  # 9 -> 𝟗
        # latin upper case
        0x41: 0x1d400,  # A -> 𝐀
        0x42: 0x1d401,  # B -> 𝐁
        0x43: 0x1d402,  # C -> 𝐂
        0x44: 0x1d403,  # D -> 𝐃
        0x45: 0x1d404,  # E -> 𝐄
        0x46: 0x1d405,  # F -> 𝐅
        0x47: 0x1d406,  # G -> 𝐆
        0x48: 0x1d407,  # H -> 𝐇
        0x49: 0x1d408,  # I -> 𝐈
        0x4a: 0x1d409,  # J -> 𝐉
        0x4b: 0x1d40a,  # K -> 𝐊
        0x4c: 0x1d40b,  # L -> 𝐋
        0x4d: 0x1d40c,  # M -> 𝐌
        0x4e: 0x1d40d,  # N -> 𝐍
        0x4f: 0x1d40e,  # O -> 𝐎
        0x50: 0x1d40f,  # P -> 𝐏
        0x51: 0x1d410,  # Q -> 𝐐
        0x52: 0x1d411,  # R -> 𝐑
        0x53: 0x1d412,  # S -> 𝐒
        0x54: 0x1d413,  # T -> 𝐓
        0x55: 0x1d414,  # U -> 𝐔
        0x56: 0x1d415,  # V -> 𝐕
        0x57: 0x1d416,  # W -> 𝐖
        0x58: 0x1d417,  # X -> 𝐗
        0x59: 0x1d418,  # Y -> 𝐘
        0x5a: 0x1d419,  # Z -> 𝐙
        # latin lower case
        0x61: 0x1d41a,  # a -> 𝐚
        0x62: 0x1d41b,  # b -> 𝐛
        0x63: 0x1d41c,  # c -> 𝐜
        0x64: 0x1d41d,  # d -> 𝐝
        0x65: 0x1d41e,  # e -> 𝐞
        0x66: 0x1d41f,  # f -> 𝐟
        0x67: 0x1d420,  # g -> 𝐠
        0x68: 0x1d421,  # h -> 𝐡
        0x69: 0x1d422,  # i -> 𝐢
        0x6a: 0x1d423,  # j -> 𝐣
        0x6b: 0x1d424,  # k -> 𝐤
        0x6c: 0x1d425,  # l -> 𝐥
        0x6d: 0x1d426,  # m -> 𝐦
        0x6e: 0x1d427,  # n -> 𝐧
        0x6f: 0x1d428,  # o -> 𝐨
        0x70: 0x1d429,  # p -> 𝐩
        0x71: 0x1d42a,  # q -> 𝐪
        0x72: 0x1d42b,  # r -> 𝐫
        0x73: 0x1d42c,  # s -> 𝐬
        0x74: 0x1d42d,  # t -> 𝐭
        0x75: 0x1d42e,  # u -> 𝐮
        0x76: 0x1d42f,  # v -> 𝐯
        0x77: 0x1d430,  # w -> 𝐰
        0x78: 0x1d431,  # x -> 𝐱
        0x79: 0x1d432,  # y -> 𝐲
        0x7a: 0x1d433,  # z -> 𝐳
        # greek upper case
        0x391: 0x1d6a8,  # Α -> 𝚨
        0x392: 0x1d6a9,  # Β -> 𝚩
        0x393: 0x1d6aa,  # Γ -> 𝚪
        0x394: 0x1d6ab,  # Δ -> 𝚫
        0x395: 0x1d6ac,  # Ε -> 𝚬
        0x396: 0x1d6ad,  # Ζ -> 𝚭
        0x397: 0x1d6ae,  # Η -> 𝚮
        0x398: 0x1d6af,  # Θ -> 𝚯
        0x399: 0x1d6b0,  # Ι -> 𝚰
        0x39a: 0x1d6b1,  # Κ -> 𝚱
        0x39b: 0x1d6b2,  # Λ -> 𝚲
        0x39c: 0x1d6b3,  # Μ -> 𝚳
        0x39d: 0x1d6b4,  # Ν -> 𝚴
        0x39e: 0x1d6b5,  # Ξ -> 𝚵
        0x39f: 0x1d6b6,  # Ο -> 𝚶
        0x3a0: 0x1d6b7,  # Π -> 𝚷
        0x3a1: 0x1d6b8,  # Ρ -> 𝚸
        0x3f4: 0x1d6b9,  # ϴ -> 𝚹
        0x3a3: 0x1d6ba,  # Σ -> 𝚺
        0x3a4: 0x1d6bb,  # Τ -> 𝚻
        0x3a5: 0x1d6bc,  # Υ -> 𝚼
        0x3a6: 0x1d6bd,  # Φ -> 𝚽
        0x3a7: 0x1d6be,  # Χ -> 𝚾
        0x3a8: 0x1d6bf,  # Ψ -> 𝚿
        0x3a9: 0x1d6c0,  # Ω -> 𝛀
        # greek lower case
        0x3b1: 0x1d6c2,  # α -> 𝛂
        0x3b2: 0x1d6c3,  # β -> 𝛃
        0x3b3: 0x1d6c4,  # γ -> 𝛄
        0x3b4: 0x1d6c5,  # δ -> 𝛅
        0x3b5: 0x1d6c6,  # ε -> 𝛆
        0x3b6: 0x1d6c7,  # ζ -> 𝛇
        0x3b7: 0x1d6c8,  # η -> 𝛈
        0x3b8: 0x1d6c9,  # θ -> 𝛉
        0x3b9: 0x1d6ca,  # ι -> 𝛊
        0x3ba: 0x1d6cb,  # κ -> 𝛋
        0x3bb: 0x1d6cc,  # λ -> 𝛌
        0x3bc: 0x1d6cd,  # μ -> 𝛍
        0x3bd: 0x1d6ce,  # ν -> 𝛎
        0x3be: 0x1d6cf,  # ξ -> 𝛏
        0x3bf: 0x1d6d0,  # ο -> 𝛐
        0x3c0: 0x1d6d1,  # π -> 𝛑
        0x3c1: 0x1d6d2,  # ρ -> 𝛒
        0x3c2: 0x1d6d3,  # ς -> 𝛓
        0x3c3: 0x1d6d4,  # σ -> 𝛔
        0x3c4: 0x1d6d5,  # τ -> 𝛕
        0x3c5: 0x1d6d6,  # υ -> 𝛖
        0x3c6: 0x1d6d7,  # φ -> 𝛗
        0x3c7: 0x1d6d8,  # χ -> 𝛘
        0x3c8: 0x1d6d9,  # ψ -> 𝛙
        0x3c9: 0x1d6da,  # ω -> 𝛚
        0x2202: 0x1d6db,  # ∂ -> 𝛛
        0x3f5: 0x1d6dc,  # ϵ -> 𝛜
        0x3d1: 0x1d6dd,  # ϑ -> 𝛝
        0x3f0: 0x1d6de,  # ϰ -> 𝛞
        0x3f1: 0x1d6df,  # ϱ -> 𝛟
        0x3cf: 0x1d6e0,  # Ϗ -> 𝛠
        0x3d6: 0x1d6e1,  # ϖ -> 𝛡
    },
    'it': {
        # digits
        # latin upper case
        0x41: 0x1d434,  # A -> 𝐴
        0x42: 0x1d435,  # B -> 𝐵
        0x43: 0x1d436,  # C -> 𝐶
        0x44: 0x1d437,  # D -> 𝐷
        0x45: 0x1d438,  # E -> 𝐸
        0x46: 0x1d439,  # F -> 𝐹
        0x47: 0x1d43a,  # G -> 𝐺
        0x48: 0x1d43b,  # H -> 𝐻
        0x49: 0x1d43c,  # I -> 𝐼
        0x4a: 0x1d43d,  # J -> 𝐽
        0x4b: 0x1d43e,  # K -> 𝐾
        0x4c: 0x1d43f,  # L -> 𝐿
        0x4d: 0x1d440,  # M -> 𝑀
        0x4e: 0x1d441,  # N -> 𝑁
        0x4f: 0x1d442,  # O -> 𝑂
        0x50: 0x1d443,  # P -> 𝑃
        0x51: 0x1d444,  # Q -> 𝑄
        0x52: 0x1d445,  # R -> 𝑅
        0x53: 0x1d446,  # S -> 𝑆
        0x54: 0x1d447,  # T -> 𝑇
        0x55: 0x1d448,  # U -> 𝑈
        0x56: 0x1d449,  # V -> 𝑉
        0x57: 0x1d44a,  # W -> 𝑊
        0x58: 0x1d44b,  # X -> 𝑋
        0x59: 0x1d44c,  # Y -> 𝑌
        0x5a: 0x1d44d,  # Z -> 𝑍
        # latin lower case
        0x61: 0x1d44e,  # a -> 𝑎
        0x62: 0x1d44f,  # b -> 𝑏
        0x63: 0x1d450,  # c -> 𝑐
        0x64: 0x1d451,  # d -> 𝑑
        0x65: 0x1d452,  # e -> 𝑒
        0x66: 0x1d453,  # f -> 𝑓
        0x67: 0x1d454,  # g -> 𝑔
        0x68: 0x210e,  # h -> ℎ
        0x69: 0x1d456,  # i -> 𝑖
        0x6a: 0x1d457,  # j -> 𝑗
        0x6b: 0x1d458,  # k -> 𝑘
        0x6c: 0x1d459,  # l -> 𝑙
        0x6d: 0x1d45a,  # m -> 𝑚
        0x6e: 0x1d45b,  # n -> 𝑛
        0x6f: 0x1d45c,  # o -> 𝑜
        0x70: 0x1d45d,  # p -> 𝑝
        0x71: 0x1d45e,  # q -> 𝑞
        0x72: 0x1d45f,  # r -> 𝑟
        0x73: 0x1d460,  # s -> 𝑠
        0x74: 0x1d461,  # t -> 𝑡
        0x75: 0x1d462,  # u -> 𝑢
        0x76: 0x1d463,  # v -> 𝑣
        0x77: 0x1d464,  # w -> 𝑤
        0x78: 0x1d465,  # x -> 𝑥
        0x79: 0x1d466,  # y -> 𝑦
        0x7a: 0x1d467,  # z -> 𝑧
        # greek upper case
        0x391: 0x1d6e2,  # Α -> 𝛢
        0x392: 0x1d6e3,  # Β -> 𝛣
        0x393: 0x1d6e4,  # Γ -> 𝛤
        0x394: 0x1d6e5,  # Δ -> 𝛥
        0x395: 0x1d6e6,  # Ε -> 𝛦
        0x396: 0x1d6e7,  # Ζ -> 𝛧
        0x397: 0x1d6e8,  # Η -> 𝛨
        0x398: 0x1d6e9,  # Θ -> 𝛩
        0x399: 0x1d6ea,  # Ι -> 𝛪
        0x39a: 0x1d6eb,  # Κ -> 𝛫
        0x39b: 0x1d6ec,  # Λ -> 𝛬
        0x39c: 0x1d6ed,  # Μ -> 𝛭
        0x39d: 0x1d6ee,  # Ν -> 𝛮
        0x39e: 0x1d6ef,  # Ξ -> 𝛯
        0x39f: 0x1d6f0,  # Ο -> 𝛰
        0x3a0: 0x1d6f1,  # Π -> 𝛱
        0x3a1: 0x1d6f2,  # Ρ -> 𝛲
        0x3f4: 0x1d6f3,  # ϴ -> 𝛳
        0x3a3: 0x1d6f4,  # Σ -> 𝛴
        0x3a4: 0x1d6f5,  # Τ -> 𝛵
        0x3a5: 0x1d6f6,  # Υ -> 𝛶
        0x3a6: 0x1d6f7,  # Φ -> 𝛷
        0x3a7: 0x1d6f8,  # Χ -> 𝛸
        0x3a8: 0x1d6f9,  # Ψ -> 𝛹
        0x3a9: 0x1d6fa,  # Ω -> 𝛺
        # greek lower case
        0x3b1: 0x1d6fc,  # α -> 𝛼
        0x3b2: 0x1d6fd,  # β -> 𝛽
        0x3b3: 0x1d6fe,  # γ -> 𝛾
        0x3b4: 0x1d6ff,  # δ -> 𝛿
        0x3b5: 0x1d700,  # ε -> 𝜀
        0x3b6: 0x1d701,  # ζ -> 𝜁
        0x3b7: 0x1d702,  # η -> 𝜂
        0x3b8: 0x1d703,  # θ -> 𝜃
        0x3b9: 0x1d704,  # ι -> 𝜄
        0x3ba: 0x1d705,  # κ -> 𝜅
        0x3bb: 0x1d706,  # λ -> 𝜆
        0x3bc: 0x1d707,  # μ -> 𝜇
        0x3bd: 0x1d708,  # ν -> 𝜈
        0x3be: 0x1d709,  # ξ -> 𝜉
        0x3bf: 0x1d70a,  # ο -> 𝜊
        0x3c0: 0x1d70b,  # π -> 𝜋
        0x3c1: 0x1d70c,  # ρ -> 𝜌
        0x3c2: 0x1d70d,  # ς -> 𝜍
        0x3c3: 0x1d70e,  # σ -> 𝜎
        0x3c4: 0x1d70f,  # τ -> 𝜏
        0x3c5: 0x1d710,  # υ -> 𝜐
        0x3c6: 0x1d711,  # φ -> 𝜑
        0x3c7: 0x1d712,  # χ -> 𝜒
        0x3c8: 0x1d713,  # ψ -> 𝜓
        0x3c9: 0x1d714,  # ω -> 𝜔
        0x2202: 0x1d715,  # ∂ -> 𝜕
        0x3f5: 0x1d716,  # ϵ -> 𝜖
        0x3d1: 0x1d717,  # ϑ -> 𝜗
        0x3f0: 0x1d718,  # ϰ -> 𝜘
        0x3f1: 0x1d719,  # ϱ -> 𝜙
        0x3cf: 0x1d71a,  # Ϗ -> 𝜚
        0x3d6: 0x1d71b,  # ϖ -> 𝜛
    },
    'bfit': {
        # digits
        # latin upper case
        0x41: 0x1d468,  # A -> 𝑨
        0x42: 0x1d469,  # B -> 𝑩
        0x43: 0x1d46a,  # C -> 𝑪
        0x44: 0x1d46b,  # D -> 𝑫
        0x45: 0x1d46c,  # E -> 𝑬
        0x46: 0x1d46d,  # F -> 𝑭
        0x47: 0x1d46e,  # G -> 𝑮
        0x48: 0x1d46f,  # H -> 𝑯
        0x49: 0x1d470,  # I -> 𝑰
        0x4a: 0x1d471,  # J -> 𝑱
        0x4b: 0x1d472,  # K -> 𝑲
        0x4c: 0x1d473,  # L -> 𝑳
        0x4d: 0x1d474,  # M -> 𝑴
        0x4e: 0x1d475,  # N -> 𝑵
        0x4f: 0x1d476,  # O -> 𝑶
        0x50: 0x1d477,  # P -> 𝑷
        0x51: 0x1d478,  # Q -> 𝑸
        0x52: 0x1d479,  # R -> 𝑹
        0x53: 0x1d47a,  # S -> 𝑺
        0x54: 0x1d47b,  # T -> 𝑻
        0x55: 0x1d47c,  # U -> 𝑼
        0x56: 0x1d47d,  # V -> 𝑽
        0x57: 0x1d47e,  # W -> 𝑾
        0x58: 0x1d47f,  # X -> 𝑿
        0x59: 0x1d480,  # Y -> 𝒀
        0x5a: 0x1d481,  # Z -> 𝒁
        # latin lower case
        0x61: 0x1d482,  # a -> 𝒂
        0x62: 0x1d483,  # b -> 𝒃
        0x63: 0x1d484,  # c -> 𝒄
        0x64: 0x1d485,  # d -> 𝒅
        0x65: 0x1d486,  # e -> 𝒆
        0x66: 0x1d487,  # f -> 𝒇
        0x67: 0x1d488,  # g -> 𝒈
        0x68: 0x1d489,  # h -> 𝒉
        0x69: 0x1d48a,  # i -> 𝒊
        0x6a: 0x1d48b,  # j -> 𝒋
        0x6b: 0x1d48c,  # k -> 𝒌
        0x6c: 0x1d48d,  # l -> 𝒍
        0x6d: 0x1d48e,  # m -> 𝒎
        0x6e: 0x1d48f,  # n -> 𝒏
        0x6f: 0x1d490,  # o -> 𝒐
        0x70: 0x1d491,  # p -> 𝒑
        0x71: 0x1d492,  # q -> 𝒒
        0x72: 0x1d493,  # r -> 𝒓
        0x73: 0x1d494,  # s -> 𝒔
        0x74: 0x1d495,  # t -> 𝒕
        0x75: 0x1d496,  # u -> 𝒖
        0x76: 0x1d497,  # v -> 𝒗
        0x77: 0x1d498,  # w -> 𝒘
        0x78: 0x1d499,  # x -> 𝒙
        0x79: 0x1d49a,  # y -> 𝒚
        0x7a: 0x1d49b,  # z -> 𝒛
        # greek upper case
        0x391: 0x1d71c,  # Α -> 𝜜
        0x392: 0x1d71d,  # Β -> 𝜝
        0x393: 0x1d71e,  # Γ -> 𝜞
        0x394: 0x1d71f,  # Δ -> 𝜟
        0x395: 0x1d720,  # Ε -> 𝜠
        0x396: 0x1d721,  # Ζ -> 𝜡
        0x397: 0x1d722,  # Η -> 𝜢
        0x398: 0x1d723,  # Θ -> 𝜣
        0x399: 0x1d724,  # Ι -> 𝜤
        0x39a: 0x1d725,  # Κ -> 𝜥
        0x39b: 0x1d726,  # Λ -> 𝜦
        0x39c: 0x1d727,  # Μ -> 𝜧
        0x39d: 0x1d728,  # Ν -> 𝜨
        0x39e: 0x1d729,  # Ξ -> 𝜩
        0x39f: 0x1d72a,  # Ο -> 𝜪
        0x3a0: 0x1d72b,  # Π -> 𝜫
        0x3a1: 0x1d72c,  # Ρ -> 𝜬
        0x3f4: 0x1d72d,  # ϴ -> 𝜭
        0x3a3: 0x1d72e,  # Σ -> 𝜮
        0x3a4: 0x1d72f,  # Τ -> 𝜯
        0x3a5: 0x1d730,  # Υ -> 𝜰
        0x3a6: 0x1d731,  # Φ -> 𝜱
        0x3a7: 0x1d732,  # Χ -> 𝜲
        0x3a8: 0x1d733,  # Ψ -> 𝜳
        0x3a9: 0x1d734,  # Ω -> 𝜴
        # greek lower case
        0x3b1: 0x1d736,  # α -> 𝜶
        0x3b2: 0x1d737,  # β -> 𝜷
        0x3b3: 0x1d738,  # γ -> 𝜸
        0x3b4: 0x1d739,  # δ -> 𝜹
        0x3b5: 0x1d73a,  # ε -> 𝜺
        0x3b6: 0x1d73b,  # ζ -> 𝜻
        0x3b7: 0x1d73c,  # η -> 𝜼
        0x3b8: 0x1d73d,  # θ -> 𝜽
        0x3b9: 0x1d73e,  # ι -> 𝜾
        0x3ba: 0x1d73f,  # κ -> 𝜿
        0x3bb: 0x1d740,  # λ -> 𝝀
        0x3bc: 0x1d741,  # μ -> 𝝁
        0x3bd: 0x1d742,  # ν -> 𝝂
        0x3be: 0x1d743,  # ξ -> 𝝃
        0x3bf: 0x1d744,  # ο -> 𝝄
        0x3c0: 0x1d745,  # π -> 𝝅
        0x3c1: 0x1d746,  # ρ -> 𝝆
        0x3c2: 0x1d747,  # ς -> 𝝇
        0x3c3: 0x1d748,  # σ -> 𝝈
        0x3c4: 0x1d749,  # τ -> 𝝉
        0x3c5: 0x1d74a,  # υ -> 𝝊
        0x3c6: 0x1d74b,  # φ -> 𝝋
        0x3c7: 0x1d74c,  # χ -> 𝝌
        0x3c8: 0x1d74d,  # ψ -> 𝝍
        0x3c9: 0x1d74e,  # ω -> 𝝎
        0x2202: 0x1d74f,  # ∂ -> 𝝏
        0x3f5: 0x1d750,  # ϵ -> 𝝐
        0x3d1: 0x1d751,  # ϑ -> 𝝑
        0x3f0: 0x1d752,  # ϰ -> 𝝒
        0x3f1: 0x1d753,  # ϱ -> 𝝓
        0x3cf: 0x1d754,  # Ϗ -> 𝝔
        0x3d6: 0x1d755,  # ϖ -> 𝝕
    },
    'sfup': {
        # digits
        0x30: 0x1d7e2,  # 0 -> 𝟢
        0x31: 0x1d7e3,  # 1 -> 𝟣
        0x32: 0x1d7e4,  # 2 -> 𝟤
        0x33: 0x1d7e5,  # 3 -> 𝟥
        0x34: 0x1d7e6,  # 4 -> 𝟦
        0x35: 0x1d7e7,  # 5 -> 𝟧
        0x36: 0x1d7e8,  # 6 -> 𝟨
        0x37: 0x1d7e9,  # 7 -> 𝟩
        0x38: 0x1d7ea,  # 8 -> 𝟪
        0x39: 0x1d7eb,  # 9 -> 𝟫
        # latin upper case
        0x41: 0x1d5a0,  # A -> 𝖠
        0x42: 0x1d5a1,  # B -> 𝖡
        0x43: 0x1d5a2,  # C -> 𝖢
        0x44: 0x1d5a3,  # D -> 𝖣
        0x45: 0x1d5a4,  # E -> 𝖤
        0x46: 0x1d5a5,  # F -> 𝖥
        0x47: 0x1d5a6,  # G -> 𝖦
        0x48: 0x1d5a7,  # H -> 𝖧
        0x49: 0x1d5a8,  # I -> 𝖨
        0x4a: 0x1d5a9,  # J -> 𝖩
        0x4b: 0x1d5aa,  # K -> 𝖪
        0x4c: 0x1d5ab,  # L -> 𝖫
        0x4d: 0x1d5ac,  # M -> 𝖬
        0x4e: 0x1d5ad,  # N -> 𝖭
        0x4f: 0x1d5ae,  # O -> 𝖮
        0x50: 0x1d5af,  # P -> 𝖯
        0x51: 0x1d5b0,  # Q -> 𝖰
        0x52: 0x1d5b1,  # R -> 𝖱
        0x53: 0x1d5b2,  # S -> 𝖲
        0x54: 0x1d5b3,  # T -> 𝖳
        0x55: 0x1d5b4,  # U -> 𝖴
        0x56: 0x1d5b5,  # V -> 𝖵
        0x57: 0x1d5b6,  # W -> 𝖶
        0x58: 0x1d5b7,  # X -> 𝖷
        0x59: 0x1d5b8,  # Y -> 𝖸
        0x5a: 0x1d5b9,  # Z -> 𝖹
        # latin lower case
        0x61: 0x1d5ba,  # a -> 𝖺
        0x62: 0x1d5bb,  # b -> 𝖻
        0x63: 0x1d5bc,  # c -> 𝖼
        0x64: 0x1d5bd,  # d -> 𝖽
        0x65: 0x1d5be,  # e -> 𝖾
        0x66: 0x1d5bf,  # f -> 𝖿
        0x67: 0x1d5c0,  # g -> 𝗀
        0x68: 0x1d5c1,  # h -> 𝗁
        0x69: 0x1d5c2,  # i -> 𝗂
        0x6a: 0x1d5c3,  # j -> 𝗃
        0x6b: 0x1d5c4,  # k -> 𝗄
        0x6c: 0x1d5c5,  # l -> 𝗅
        0x6d: 0x1d5c6,  # m -> 𝗆
        0x6e: 0x1d5c7,  # n -> 𝗇
        0x6f: 0x1d5c8,  # o -> 𝗈
        0x70: 0x1d5c9,  # p -> 𝗉
        0x71: 0x1d5ca,  # q -> 𝗊
        0x72: 0x1d5cb,  # r -> 𝗋
        0x73: 0x1d5cc,  # s -> 𝗌
        0x74: 0x1d5cd,  # t -> 𝗍
        0x75: 0x1d5ce,  # u -> 𝗎
        0x76: 0x1d5cf,  # v -> 𝗏
        0x77: 0x1d5d0,  # w -> 𝗐
        0x78: 0x1d5d1,  # x -> 𝗑
        0x79: 0x1d5d2,  # y -> 𝗒
        0x7a: 0x1d5d3,  # z -> 𝗓
        # greek upper case
        # greek lower case
    },
    'bfsfup': {
        # digits
        0x30: 0x1d7ec,  # 0 -> 𝟬
        0x31: 0x1d7ed,  # 1 -> 𝟭
        0x32: 0x1d7ee,  # 2 -> 𝟮
        0x33: 0x1d7ef,  # 3 -> 𝟯
        0x34: 0x1d7f0,  # 4 -> 𝟰
        0x35: 0x1d7f1,  # 5 -> 𝟱
        0x36: 0x1d7f2,  # 6 -> 𝟲
        0x37: 0x1d7f3,  # 7 -> 𝟳
        0x38: 0x1d7f4,  # 8 -> 𝟴
        0x39: 0x1d7f5,  # 9 -> 𝟵
        # latin upper case
        0x41: 0x1d5d4,  # A -> 𝗔
        0x42: 0x1d5d5,  # B -> 𝗕
        0x43: 0x1d5d6,  # C -> 𝗖
        0x44: 0x1d5d7,  # D -> 𝗗
        0x45: 0x1d5d8,  # E -> 𝗘
        0x46: 0x1d5d9,  # F -> 𝗙
        0x47: 0x1d5da,  # G -> 𝗚
        0x48: 0x1d5db,  # H -> 𝗛
        0x49: 0x1d5dc,  # I -> 𝗜
        0x4a: 0x1d5dd,  # J -> 𝗝
        0x4b: 0x1d5de,  # K -> 𝗞
        0x4c: 0x1d5df,  # L -> 𝗟
        0x4d: 0x1d5e0,  # M -> 𝗠
        0x4e: 0x1d5e1,  # N -> 𝗡
        0x4f: 0x1d5e2,  # O -> 𝗢
        0x50: 0x1d5e3,  # P -> 𝗣
        0x51: 0x1d5e4,  # Q -> 𝗤
        0x52: 0x1d5e5,  # R -> 𝗥
        0x53: 0x1d5e6,  # S -> 𝗦
        0x54: 0x1d5e7,  # T -> 𝗧
        0x55: 0x1d5e8,  # U -> 𝗨
        0x56: 0x1d5e9,  # V -> 𝗩
        0x57: 0x1d5ea,  # W -> 𝗪
        0x58: 0x1d5eb,  # X -> 𝗫
        0x59: 0x1d5ec,  # Y -> 𝗬
        0x5a: 0x1d5ed,  # Z -> 𝗭
        # latin lower case
        0x61: 0x1d5ee,  # a -> 𝗮
        0x62: 0x1d5ef,  # b -> 𝗯
        0x63: 0x1d5f0,  # c -> 𝗰
        0x64: 0x1d5f1,  # d -> 𝗱
        0x65: 0x1d5f2,  # e -> 𝗲
        0x66: 0x1d5f3,  # f -> 𝗳
        0x67: 0x1d5f4,  # g -> 𝗴
        0x68: 0x1d5f5,  # h -> 𝗵
        0x69: 0x1d5f6,  # i -> 𝗶
        0x6a: 0x1d5f7,  # j -> 𝗷
        0x6b: 0x1d5f8,  # k -> 𝗸
        0x6c: 0x1d5f9,  # l -> 𝗹
        0x6d: 0x1d5fa,  # m -> 𝗺
        0x6e: 0x1d5fb,  # n -> 𝗻
        0x6f: 0x1d5fc,  # o -> 𝗼
        0x70: 0x1d5fd,  # p -> 𝗽
        0x71: 0x1d5fe,  # q -> 𝗾
        0x72: 0x1d5ff,  # r -> 𝗿
        0x73: 0x1d600,  # s -> 𝘀
        0x74: 0x1d601,  # t -> 𝘁
        0x75: 0x1d602,  # u -> 𝘂
        0x76: 0x1d603,  # v -> 𝘃
        0x77: 0x1d604,  # w -> 𝘄
        0x78: 0x1d605,  # x -> 𝘅
        0x79: 0x1d606,  # y -> 𝘆
        0x7a: 0x1d607,  # z -> 𝘇
        # greek upper case
        0x391: 0x1d756,  # Α -> 𝝖
        0x392: 0x1d757,  # Β -> 𝝗
        0x393: 0x1d758,  # Γ -> 𝝘
        0x394: 0x1d759,  # Δ -> 𝝙
        0x395: 0x1d75a,  # Ε -> 𝝚
        0x396: 0x1d75b,  # Ζ -> 𝝛
        0x397: 0x1d75c,  # Η -> 𝝜
        0x398: 0x1d75d,  # Θ -> 𝝝
        0x399: 0x1d75e,  # Ι -> 𝝞
        0x39a: 0x1d75f,  # Κ -> 𝝟
        0x39b: 0x1d760,  # Λ -> 𝝠
        0x39c: 0x1d761,  # Μ -> 𝝡
        0x39d: 0x1d762,  # Ν -> 𝝢
        0x39e: 0x1d763,  # Ξ -> 𝝣
        0x39f: 0x1d764,  # Ο -> 𝝤
        0x3a0: 0x1d765,  # Π -> 𝝥
        0x3a1: 0x1d766,  # Ρ -> 𝝦
        0x3f4: 0x1d767,  # ϴ -> 𝝧
        0x3a3: 0x1d768,  # Σ -> 𝝨
        0x3a4: 0x1d769,  # Τ -> 𝝩
        0x3a5: 0x1d76a,  # Υ -> 𝝪
        0x3a6: 0x1d76b,  # Φ -> 𝝫
        0x3a7: 0x1d76c,  # Χ -> 𝝬
        0x3a8: 0x1d76d,  # Ψ -> 𝝭
        0x3a9: 0x1d76e,  # Ω -> 𝝮
        # greek lower case
        0x3b1: 0x1d770,  # α -> 𝝰
        0x3b2: 0x1d771,  # β -> 𝝱
        0x3b3: 0x1d772,  # γ -> 𝝲
        0x3b4: 0x1d773,  # δ -> 𝝳
        0x3b5: 0x1d774,  # ε -> 𝝴
        0x3b6: 0x1d775,  # ζ -> 𝝵
        0x3b7: 0x1d776,  # η -> 𝝶
        0x3b8: 0x1d777,  # θ -> 𝝷
        0x3b9: 0x1d778,  # ι -> 𝝸
        0x3ba: 0x1d779,  # κ -> 𝝹
        0x3bb: 0x1d77a,  # λ -> 𝝺
        0x3bc: 0x1d77b,  # μ -> 𝝻
        0x3bd: 0x1d77c,  # ν -> 𝝼
        0x3be: 0x1d77d,  # ξ -> 𝝽
        0x3bf: 0x1d77e,  # ο -> 𝝾
        0x3c0: 0x1d77f,  # π -> 𝝿
        0x3c1: 0x1d780,  # ρ -> 𝞀
        0x3c2: 0x1d781,  # ς -> 𝞁
        0x3c3: 0x1d782,  # σ -> 𝞂
        0x3c4: 0x1d783,  # τ -> 𝞃
        0x3c5: 0x1d784,  # υ -> 𝞄
        0x3c6: 0x1d785,  # φ -> 𝞅
        0x3c7: 0x1d786,  # χ -> 𝞆
        0x3c8: 0x1d787,  # ψ -> 𝞇
        0x3c9: 0x1d788,  # ω -> 𝞈
        0x2202: 0x1d789,  # ∂ -> 𝞉
        0x3f5: 0x1d78a,  # ϵ -> 𝞊
        0x3d1: 0x1d78b,  # ϑ -> 𝞋
        0x3f0: 0x1d78c,  # ϰ -> 𝞌
        0x3f1: 0x1d78d,  # ϱ -> 𝞍
        0x3cf: 0x1d78e,  # Ϗ -> 𝞎
        0x3d6: 0x1d78f,  # ϖ -> 𝞏
    },
    'sfit': {
        # digits
        # latin upper case
        0x41: 0x1d608,  # A -> 𝘈
        0x42: 0x1d609,  # B -> 𝘉
        0x43: 0x1d60a,  # C -> 𝘊
        0x44: 0x1d60b,  # D -> 𝘋
        0x45: 0x1d60c,  # E -> 𝘌
        0x46: 0x1d60d,  # F -> 𝘍
        0x47: 0x1d60e,  # G -> 𝘎
        0x48: 0x1d60f,  # H -> 𝘏
        0x49: 0x1d610,  # I -> 𝘐
        0x4a: 0x1d611,  # J -> 𝘑
        0x4b: 0x1d612,  # K -> 𝘒
        0x4c: 0x1d613,  # L -> 𝘓
        0x4d: 0x1d614,  # M -> 𝘔
        0x4e: 0x1d615,  # N -> 𝘕
        0x4f: 0x1d616,  # O -> 𝘖
        0x50: 0x1d617,  # P -> 𝘗
        0x51: 0x1d618,  # Q -> 𝘘
        0x52: 0x1d619,  # R -> 𝘙
        0x53: 0x1d61a,  # S -> 𝘚
        0x54: 0x1d61b,  # T -> 𝘛
        0x55: 0x1d61c,  # U -> 𝘜
        0x56: 0x1d61d,  # V -> 𝘝
        0x57: 0x1d61e,  # W -> 𝘞
        0x58: 0x1d61f,  # X -> 𝘟
        0x59: 0x1d620,  # Y -> 𝘠
        0x5a: 0x1d621,  # Z -> 𝘡
        # latin lower case
        0x61: 0x1d622,  # a -> 𝘢
        0x62: 0x1d623,  # b -> 𝘣
        0x63: 0x1d624,  # c -> 𝘤
        0x64: 0x1d625,  # d -> 𝘥
        0x65: 0x1d626,  # e -> 𝘦
        0x66: 0x1d627,  # f -> 𝘧
        0x67: 0x1d628,  # g -> 𝘨
        0x68: 0x1d629,  # h -> 𝘩
        0x69: 0x1d62a,  # i -> 𝘪
        0x6a: 0x1d62b,  # j -> 𝘫
        0x6b: 0x1d62c,  # k -> 𝘬
        0x6c: 0x1d62d,  # l -> 𝘭
        0x6d: 0x1d62e,  # m -> 𝘮
        0x6e: 0x1d62f,  # n -> 𝘯
        0x6f: 0x1d630,  # o -> 𝘰
        0x70: 0x1d631,  # p -> 𝘱
        0x71: 0x1d632,  # q -> 𝘲
        0x72: 0x1d633,  # r -> 𝘳
        0x73: 0x1d634,  # s -> 𝘴
        0x74: 0x1d635,  # t -> 𝘵
        0x75: 0x1d636,  # u -> 𝘶
        0x76: 0x1d637,  # v -> 𝘷
        0x77: 0x1d638,  # w -> 𝘸
        0x78: 0x1d639,  # x -> 𝘹
        0x79: 0x1d63a,  # y -> 𝘺
        0x7a: 0x1d63b,  # z -> 𝘻
        # greek upper case
        # greek lower case
    },
    'bfsfit': {
        # digits
        # latin upper case
        0x41: 0x1d63c,  # A -> 𝘼
        0x42: 0x1d63d,  # B -> 𝘽
        0x43: 0x1d63e,  # C -> 𝘾
        0x44: 0x1d63f,  # D -> 𝘿
        0x45: 0x1d640,  # E -> 𝙀
        0x46: 0x1d641,  # F -> 𝙁
        0x47: 0x1d642,  # G -> 𝙂
        0x48: 0x1d643,  # H -> 𝙃
        0x49: 0x1d644,  # I -> 𝙄
        0x4a: 0x1d645,  # J -> 𝙅
        0x4b: 0x1d646,  # K -> 𝙆
        0x4c: 0x1d647,  # L -> 𝙇
        0x4d: 0x1d648,  # M -> 𝙈
        0x4e: 0x1d649,  # N -> 𝙉
        0x4f: 0x1d64a,  # O -> 𝙊
        0x50: 0x1d64b,  # P -> 𝙋
        0x51: 0x1d64c,  # Q -> 𝙌
        0x52: 0x1d64d,  # R -> 𝙍
        0x53: 0x1d64e,  # S -> 𝙎
        0x54: 0x1d64f,  # T -> 𝙏
        0x55: 0x1d650,  # U -> 𝙐
        0x56: 0x1d651,  # V -> 𝙑
        0x57: 0x1d652,  # W -> 𝙒
        0x58: 0x1d653,  # X -> 𝙓
        0x59: 0x1d654,  # Y -> 𝙔
        0x5a: 0x1d655,  # Z -> 𝙕
        # latin lower case
        0x61: 0x1d656,  # a -> 𝙖
        0x62: 0x1d657,  # b -> 𝙗
        0x63: 0x1d658,  # c -> 𝙘
        0x64: 0x1d659,  # d -> 𝙙
        0x65: 0x1d65a,  # e -> 𝙚
        0x66: 0x1d65b,  # f -> 𝙛
        0x67: 0x1d65c,  # g -> 𝙜
        0x68: 0x1d65d,  # h -> 𝙝
        0x69: 0x1d65e,  # i -> 𝙞
        0x6a: 0x1d65f,  # j -> 𝙟
        0x6b: 0x1d660,  # k -> 𝙠
        0x6c: 0x1d661,  # l -> 𝙡
        0x6d: 0x1d662,  # m -> 𝙢
        0x6e: 0x1d663,  # n -> 𝙣
        0x6f: 0x1d664,  # o -> 𝙤
        0x70: 0x1d665,  # p -> 𝙥
        0x71: 0x1d666,  # q -> 𝙦
        0x72: 0x1d667,  # r -> 𝙧
        0x73: 0x1d668,  # s -> 𝙨
        0x74: 0x1d669,  # t -> 𝙩
        0x75: 0x1d66a,  # u -> 𝙪
        0x76: 0x1d66b,  # v -> 𝙫
        0x77: 0x1d66c,  # w -> 𝙬
        0x78: 0x1d66d,  # x -> 𝙭
        0x79: 0x1d66e,  # y -> 𝙮
        0x7a: 0x1d66f,  # z -> 𝙯
        # greek upper case
        0x391: 0x1d790,  # Α -> 𝞐
        0x392: 0x1d791,  # Β -> 𝞑
        0x393: 0x1d792,  # Γ -> 𝞒
        0x394: 0x1d793,  # Δ -> 𝞓
        0x395: 0x1d794,  # Ε -> 𝞔
        0x396: 0x1d795,  # Ζ -> 𝞕
        0x397: 0x1d796,  # Η -> 𝞖
        0x398: 0x1d797,  # Θ -> 𝞗
        0x399: 0x1d798,  # Ι -> 𝞘
        0x39a: 0x1d799,  # Κ -> 𝞙
        0x39b: 0x1d79a,  # Λ -> 𝞚
        0x39c: 0x1d79b,  # Μ -> 𝞛
        0x39d: 0x1d79c,  # Ν -> 𝞜
        0x39e: 0x1d79d,  # Ξ -> 𝞝
        0x39f: 0x1d79e,  # Ο -> 𝞞
        0x3a0: 0x1d79f,  # Π -> 𝞟
        0x3a1: 0x1d7a0,  # Ρ -> 𝞠
        0x3f4: 0x1d7a1,  # ϴ -> 𝞡
        0x3a3: 0x1d7a2,  # Σ -> 𝞢
        0x3a4: 0x1d7a3,  # Τ -> 𝞣
        0x3a5: 0x1d7a4,  # Υ -> 𝞤
        0x3a6: 0x1d7a5,  # Φ -> 𝞥
        0x3a7: 0x1d7a6,  # Χ -> 𝞦
        0x3a8: 0x1d7a7,  # Ψ -> 𝞧
        0x3a9: 0x1d7a8,  # Ω -> 𝞨
        # greek lower case
        0x3b1: 0x1d7aa,  # α -> 𝞪
        0x3b2: 0x1d7ab,  # β -> 𝞫
        0x3b3: 0x1d7ac,  # γ -> 𝞬
        0x3b4: 0x1d7ad,  # δ -> 𝞭
        0x3b5: 0x1d7ae,  # ε -> 𝞮
        0x3b6: 0x1d7af,  # ζ -> 𝞯
        0x3b7: 0x1d7b0,  # η -> 𝞰
        0x3b8: 0x1d7b1,  # θ -> 𝞱
        0x3b9: 0x1d7b2,  # ι -> 𝞲
        0x3ba: 0x1d7b3,  # κ -> 𝞳
        0x3bb: 0x1d7b4,  # λ -> 𝞴
        0x3bc: 0x1d7b5,  # μ -> 𝞵
        0x3bd: 0x1d7b6,  # ν -> 𝞶
        0x3be: 0x1d7b7,  # ξ -> 𝞷
        0x3bf: 0x1d7b8,  # ο -> 𝞸
        0x3c0: 0x1d7b9,  # π -> 𝞹
        0x3c1: 0x1d7ba,  # ρ -> 𝞺
        0x3c2: 0x1d7bb,  # ς -> 𝞻
        0x3c3: 0x1d7bc,  # σ -> 𝞼
        0x3c4: 0x1d7bd,  # τ -> 𝞽
        0x3c5: 0x1d7be,  # υ -> 𝞾
        0x3c6: 0x1d7bf,  # φ -> 𝞿
        0x3c7: 0x1d7c0,  # χ -> 𝟀
        0x3c8: 0x1d7c1,  # ψ -> 𝟁
        0x3c9: 0x1d7c2,  # ω -> 𝟂
        0x2202: 0x1d7c3,  # ∂ -> 𝟃
        0x3f5: 0x1d7c4,  # ϵ -> 𝟄
        0x3d1: 0x1d7c5,  # ϑ -> 𝟅
        0x3f0: 0x1d7c6,  # ϰ -> 𝟆
        0x3f1: 0x1d7c7,  # ϱ -> 𝟇
        0x3cf: 0x1d7c8,  # Ϗ -> 𝟈
        0x3d6: 0x1d7c9,  # ϖ -> 𝟉
    },
    'scr': {
        # digits
        # latin upper case
        0x41: 0x1d49c,  # A -> 𝒜
        0x42: 0x212c,  # B -> ℬ
        0x43: 0x1d49e,  # C -> 𝒞
        0x44: 0x1d49f,  # D -> 𝒟
        0x45: 0x2130,  # E -> ℰ
        0x46: 0x2131,  # F -> ℱ
        0x47: 0x1d4a2,  # G -> 𝒢
        0x48: 0x210b,  # H -> ℋ
        0x49: 0x2110,  # I -> ℐ
        0x4a: 0x1d4a5,  # J -> 𝒥
        0x4b: 0x1d4a6,  # K -> 𝒦
        0x4c: 0x2112,  # L -> ℒ
        0x4d: 0x2133,  # M -> ℳ
        0x4e: 0x1d4a9,  # N -> 𝒩
        0x4f: 0x1d4aa,  # O -> 𝒪
        0x50: 0x1d4ab,  # P -> 𝒫
        0x51: 0x1d4ac,  # Q -> 𝒬
        0x52: 0x211b,  # R -> ℛ
        0x53: 0x1d4ae,  # S -> 𝒮
        0x54: 0x1d4af,  # T -> 𝒯
        0x55: 0x1d4b0,  # U -> 𝒰
        0x56: 0x1d4b1,  # V -> 𝒱
        0x57: 0x1d4b2,  # W -> 𝒲
        0x58: 0x1d4b3,  # X -> 𝒳
        0x59: 0x1d4b4,  # Y -> 𝒴
        0x5a: 0x1d4b5,  # Z -> 𝒵
        # latin lower case
        0x61: 0x1d4b6,  # a -> 𝒶
        0x62: 0x1d4b7,  # b -> 𝒷
        0x63: 0x1d4b8,  # c -> 𝒸
        0x64: 0x1d4b9,  # d -> 𝒹
        0x65: 0x212f,  # e -> ℯ
        0x66: 0x1d4bb,  # f -> 𝒻
        0x67: 0x210a,  # g -> ℊ
        0x68: 0x1d4bd,  # h -> 𝒽
        0x69: 0x1d4be,  # i -> 𝒾
        0x6a: 0x1d4bf,  # j -> 𝒿
        0x6b: 0x1d4c0,  # k -> 𝓀
        0x6c: 0x1d4c1,  # l -> 𝓁
        0x6d: 0x1d4c2,  # m -> 𝓂
        0x6e: 0x1d4c3,  # n -> 𝓃
        0x6f: 0x2134,  # o -> ℴ
        0x70: 0x1d4c5,  # p -> 𝓅
        0x71: 0x1d4c6,  # q -> 𝓆
        0x72: 0x1d4c7,  # r -> 𝓇
        0x73: 0x1d4c8,  # s -> 𝓈
        0x74: 0x1d4c9,  # t -> 𝓉
        0x75: 0x1d4ca,  # u -> 𝓊
        0x76: 0x1d4cb,  # v -> 𝓋
        0x77: 0x1d4cc,  # w -> 𝓌
        0x78: 0x1d4cd,  # x -> 𝓍
        0x79: 0x1d4ce,  # y -> 𝓎
        0x7a: 0x1d4cf,  # z -> 𝓏
        # greek upper case
        # greek lower case
    },
    'bfscr': {
        # digits
        # latin upper case
        0x41: 0x1d4d0,  # A -> 𝓐
        0x42: 0x1d4d1,  # B -> 𝓑
        0x43: 0x1d4d2,  # C -> 𝓒
        0x44: 0x1d4d3,  # D -> 𝓓
        0x45: 0x1d4d4,  # E -> 𝓔
        0x46: 0x1d4d5,  # F -> 𝓕
        0x47: 0x1d4d6,  # G -> 𝓖
        0x48: 0x1d4d7,  # H -> 𝓗
        0x49: 0x1d4d8,  # I -> 𝓘
        0x4a: 0x1d4d9,  # J -> 𝓙
        0x4b: 0x1d4da,  # K -> 𝓚
        0x4c: 0x1d4db,  # L -> 𝓛
        0x4d: 0x1d4dc,  # M -> 𝓜
        0x4e: 0x1d4dd,  # N -> 𝓝
        0x4f: 0x1d4de,  # O -> 𝓞
        0x50: 0x1d4df,  # P -> 𝓟
        0x51: 0x1d4e0,  # Q -> 𝓠
        0x52: 0x1d4e1,  # R -> 𝓡
        0x53: 0x1d4e2,  # S -> 𝓢
        0x54: 0x1d4e3,  # T -> 𝓣
        0x55: 0x1d4e4,  # U -> 𝓤
        0x56: 0x1d4e5,  # V -> 𝓥
        0x57: 0x1d4e6,  # W -> 𝓦
        0x58: 0x1d4e7,  # X -> 𝓧
        0x59: 0x1d4e8,  # Y -> 𝓨
        0x5a: 0x1d4e9,  # Z -> 𝓩
        # latin lower case
        0x61: 0x1d4ea,  # a -> 𝓪
        0x62: 0x1d4eb,  # b -> 𝓫
        0x63: 0x1d4ec,  # c -> 𝓬
        0x64: 0x1d4ed,  # d -> 𝓭
        0x65: 0x1d4ee,  # e -> 𝓮
        0x66: 0x1d4ef,  # f -> 𝓯
        0x67: 0x1d4f0,  # g -> 𝓰
        0x68: 0x1d4f1,  # h -> 𝓱
        0x69: 0x1d4f2,  # i -> 𝓲
        0x6a: 0x1d4f3,  # j -> 𝓳
        0x6b: 0x1d4f4,  # k -> 𝓴
        0x6c: 0x1d4f5,  # l -> 𝓵
        0x6d: 0x1d4f6,  # m -> 𝓶
        0x6e: 0x1d4f7,  # n -> 𝓷
        0x6f: 0x1d4f8,  # o -> 𝓸
        0x70: 0x1d4f9,  # p -> 𝓹
        0x71: 0x1d4fa,  # q -> 𝓺
        0x72: 0x1d4fb,  # r -> 𝓻
        0x73: 0x1d4fc,  # s -> 𝓼
        0x74: 0x1d4fd,  # t -> 𝓽
        0x75: 0x1d4fe,  # u -> 𝓾
        0x76: 0x1d4ff,  # v -> 𝓿
        0x77: 0x1d500,  # w -> 𝔀
        0x78: 0x1d501,  # x -> 𝔁
        0x79: 0x1d502,  # y -> 𝔂
        0x7a: 0x1d503,  # z -> 𝔃
        # greek upper case
        # greek lower case
    },
    'frak': {
        # digits
        # latin upper case
        0x41: 0x1d504,  # A -> 𝔄
        0x42: 0x1d505,  # B -> 𝔅
        0x43: 0x212d,  # C -> ℭ
        0x44: 0x1d507,  # D -> 𝔇
        0x45: 0x1d508,  # E -> 𝔈
        0x46: 0x1d509,  # F -> 𝔉
        0x47: 0x1d50a,  # G -> 𝔊
        0x48: 0x210c,  # H -> ℌ
        0x49: 0x2111,  # I -> ℑ
        0x4a: 0x1d50d,  # J -> 𝔍
        0x4b: 0x1d50e,  # K -> 𝔎
        0x4c: 0x1d50f,  # L -> 𝔏
        0x4d: 0x1d510,  # M -> 𝔐
        0x4e: 0x1d511,  # N -> 𝔑
        0x4f: 0x1d512,  # O -> 𝔒
        0x50: 0x1d513,  # P -> 𝔓
        0x51: 0x1d514,  # Q -> 𝔔
        0x52: 0x211c,  # R -> ℜ
        0x53: 0x1d516,  # S -> 𝔖
        0x54: 0x1d517,  # T -> 𝔗
        0x55: 0x1d518,  # U -> 𝔘
        0x56: 0x1d519,  # V -> 𝔙
        0x57: 0x1d51a,  # W -> 𝔚
        0x58: 0x1d51b,  # X -> 𝔛
        0x59: 0x1d51c,  # Y -> 𝔜
        0x5a: 0x2128,  # Z -> ℨ
        # latin lower case
        0x61: 0x1d51e,  # a -> 𝔞
        0x62: 0x1d51f,  # b -> 𝔟
        0x63: 0x1d520,  # c -> 𝔠
        0x64: 0x1d521,  # d -> 𝔡
        0x65: 0x1d522,  # e -> 𝔢
        0x66: 0x1d523,  # f -> 𝔣
        0x67: 0x1d524,  # g -> 𝔤
        0x68: 0x1d525,  # h -> 𝔥
        0x69: 0x1d526,  # i -> 𝔦
        0x6a: 0x1d527,  # j -> 𝔧
        0x6b: 0x1d528,  # k -> 𝔨
        0x6c: 0x1d529,  # l -> 𝔩
        0x6d: 0x1d52a,  # m -> 𝔪
        0x6e: 0x1d52b,  # n -> 𝔫
        0x6f: 0x1d52c,  # o -> 𝔬
        0x70: 0x1d52d,  # p -> 𝔭
        0x71: 0x1d52e,  # q -> 𝔮
        0x72: 0x1d52f,  # r -> 𝔯
        0x73: 0x1d530,  # s -> 𝔰
        0x74: 0x1d531,  # t -> 𝔱
        0x75: 0x1d532,  # u -> 𝔲
        0x76: 0x1d533,  # v -> 𝔳
        0x77: 0x1d534,  # w -> 𝔴
        0x78: 0x1d535,  # x -> 𝔵
        0x79: 0x1d536,  # y -> 𝔶
        0x7a: 0x1d537,  # z -> 𝔷
        # greek upper case
        # greek lower case
    },
    'bffrak': {
        # digits
        # latin upper case
        0x41: 0x1d56c,  # A -> 𝕬
        0x42: 0x1d56d,  # B -> 𝕭
        0x43: 0x1d56e,  # C -> 𝕮
        0x44: 0x1d56f,  # D -> 𝕯
        0x45: 0x1d570,  # E -> 𝕰
        0x46: 0x1d571,  # F -> 𝕱
        0x47: 0x1d572,  # G -> 𝕲
        0x48: 0x1d573,  # H -> 𝕳
        0x49: 0x1d574,  # I -> 𝕴
        0x4a: 0x1d575,  # J -> 𝕵
        0x4b: 0x1d576,  # K -> 𝕶
        0x4c: 0x1d577,  # L -> 𝕷
        0x4d: 0x1d578,  # M -> 𝕸
        0x4e: 0x1d579,  # N -> 𝕹
        0x4f: 0x1d57a,  # O -> 𝕺
        0x50: 0x1d57b,  # P -> 𝕻
        0x51: 0x1d57c,  # Q -> 𝕼
        0x52: 0x1d57d,  # R -> 𝕽
        0x53: 0x1d57e,  # S -> 𝕾
        0x54: 0x1d57f,  # T -> 𝕿
        0x55: 0x1d580,  # U -> 𝖀
        0x56: 0x1d581,  # V -> 𝖁
        0x57: 0x1d582,  # W -> 𝖂
        0x58: 0x1d583,  # X -> 𝖃
        0x59: 0x1d584,  # Y -> 𝖄
        0x5a: 0x1d585,  # Z -> 𝖅
        # latin lower case
        0x61: 0x1d586,  # a -> 𝖆
        0x62: 0x1d587,  # b -> 𝖇
        0x63: 0x1d588,  # c -> 𝖈
        0x64: 0x1d589,  # d -> 𝖉
        0x65: 0x1d58a,  # e -> 𝖊
        0x66: 0x1d58b,  # f -> 𝖋
        0x67: 0x1d58c,  # g -> 𝖌
        0x68: 0x1d58d,  # h -> 𝖍
        0x69: 0x1d58e,  # i -> 𝖎
        0x6a: 0x1d58f,  # j -> 𝖏
        0x6b: 0x1d590,  # k -> 𝖐
        0x6c: 0x1d591,  # l -> 𝖑
        0x6d: 0x1d592,  # m -> 𝖒
        0x6e: 0x1d593,  # n -> 𝖓
        0x6f: 0x1d594,  # o -> 𝖔
        0x70: 0x1d595,  # p -> 𝖕
        0x71: 0x1d596,  # q -> 𝖖
        0x72: 0x1d597,  # r -> 𝖗
        0x73: 0x1d598,  # s -> 𝖘
        0x74: 0x1d599,  # t -> 𝖙
        0x75: 0x1d59a,  # u -> 𝖚
        0x76: 0x1d59b,  # v -> 𝖛
        0x77: 0x1d59c,  # w -> 𝖜
        0x78: 0x1d59d,  # x -> 𝖝
        0x79: 0x1d59e,  # y -> 𝖞
        0x7a: 0x1d59f,  # z -> 𝖟
        # greek upper case
        # greek lower case
    },
    'tt': {
        # digits
        0x30: 0x1d7f6,  # 0 -> 𝟶
        0x31: 0x1d7f7,  # 1 -> 𝟷
        0x32: 0x1d7f8,  # 2 -> 𝟸
        0x33: 0x1d7f9,  # 3 -> 𝟹
        0x34: 0x1d7fa,  # 4 -> 𝟺
        0x35: 0x1d7fb,  # 5 -> 𝟻
        0x36: 0x1d7fc,  # 6 -> 𝟼
        0x37: 0x1d7fd,  # 7 -> 𝟽
        0x38: 0x1d7fe,  # 8 -> 𝟾
        0x39: 0x1d7ff,  # 9 -> 𝟿
        # latin upper case
        0x41: 0x1d670,  # A -> 𝙰
        0x42: 0x1d671,  # B -> 𝙱
        0x43: 0x1d672,  # C -> 𝙲
        0x44: 0x1d673,  # D -> 𝙳
        0x45: 0x1d674,  # E -> 𝙴
        0x46: 0x1d675,  # F -> 𝙵
        0x47: 0x1d676,  # G -> 𝙶
        0x48: 0x1d677,  # H -> 𝙷
        0x49: 0x1d678,  # I -> 𝙸
        0x4a: 0x1d679,  # J -> 𝙹
        0x4b: 0x1d67a,  # K -> 𝙺
        0x4c: 0x1d67b,  # L -> 𝙻
        0x4d: 0x1d67c,  # M -> 𝙼
        0x4e: 0x1d67d,  # N -> 𝙽
        0x4f: 0x1d67e,  # O -> 𝙾
        0x50: 0x1d67f,  # P -> 𝙿
        0x51: 0x1d680,  # Q -> 𝚀
        0x52: 0x1d681,  # R -> 𝚁
        0x53: 0x1d682,  # S -> 𝚂
        0x54: 0x1d683,  # T -> 𝚃
        0x55: 0x1d684,  # U -> 𝚄
        0x56: 0x1d685,  # V -> 𝚅
        0x57: 0x1d686,  # W -> 𝚆
        0x58: 0x1d687,  # X -> 𝚇
        0x59: 0x1d688,  # Y -> 𝚈
        0x5a: 0x1d689,  # Z -> 𝚉
        # latin lower case
        0x61: 0x1d68a,  # a -> 𝚊
        0x62: 0x1d68b,  # b -> 𝚋
        0x63: 0x1d68c,  # c -> 𝚌
        0x64: 0x1d68d,  # d -> 𝚍
        0x65: 0x1d68e,  # e -> 𝚎
        0x66: 0x1d68f,  # f -> 𝚏
        0x67: 0x1d690,  # g -> 𝚐
        0x68: 0x1d691,  # h -> 𝚑
        0x69: 0x1d692,  # i -> 𝚒
        0x6a: 0x1d693,  # j -> 𝚓
        0x6b: 0x1d694,  # k -> 𝚔
        0x6c: 0x1d695,  # l -> 𝚕
        0x6d: 0x1d696,  # m -> 𝚖
        0x6e: 0x1d697,  # n -> 𝚗
        0x6f: 0x1d698,  # o -> 𝚘
        0x70: 0x1d699,  # p -> 𝚙
        0x71: 0x1d69a,  # q -> 𝚚
        0x72: 0x1d69b,  # r -> 𝚛
        0x73: 0x1d69c,  # s -> 𝚜
        0x74: 0x1d69d,  # t -> 𝚝
        0x75: 0x1d69e,  # u -> 𝚞
        0x76: 0x1d69f,  # v -> 𝚟
        0x77: 0x1d6a0,  # w -> 𝚠
        0x78: 0x1d6a1,  # x -> 𝚡
        0x79: 0x1d6a2,  # y -> 𝚢
        0x7a: 0x1d6a3,  # z -> 𝚣
        # greek upper case
        # greek lower case
    },
    'bb': {
        # digits
        0x30: 0x1d7d8,  # 0 -> 𝟘
        0x31: 0x1d7d9,  # 1 -> 𝟙
        0x32: 0x1d7da,  # 2 -> 𝟚
        0x33: 0x1d7db,  # 3 -> 𝟛
        0x34: 0x1d7dc,  # 4 -> 𝟜
        0x35: 0x1d7dd,  # 5 -> 𝟝
        0x36: 0x1d7de,  # 6 -> 𝟞
        0x37: 0x1d7df,  # 7 -> 𝟟
        0x38: 0x1d7e0,  # 8 -> 𝟠
        0x39: 0x1d7e1,  # 9 -> 𝟡
        # latin upper case
        0x41: 0x1d538,  # A -> 𝔸
        0x42: 0x1d539,  # B -> 𝔹
        0x43: 0x2102,  # C -> ℂ
        0x44: 0x1d53b,  # D -> 𝔻
        0x45: 0x1d53c,  # E -> 𝔼
        0x46: 0x1d53d,  # F -> 𝔽
        0x47: 0x1d53e,  # G -> 𝔾
        0x48: 0x210d,  # H -> ℍ
        0x49: 0x1d540,  # I -> 𝕀
        0x4a: 0x1d541,  # J -> 𝕁
        0x4b: 0x1d542,  # K -> 𝕂
        0x4c: 0x1d543,  # L -> 𝕃
        0x4d: 0x1d544,  # M -> 𝕄
        0x4e: 0x2115,  # N -> ℕ
        0x4f: 0x1d546,  # O -> 𝕆
        0x50: 0x2119,  # P -> ℙ
        0x51: 0x211a,  # Q -> ℚ
        0x52: 0x211d,  # R -> ℝ
        0x53: 0x1d54a,  # S -> 𝕊
        0x54: 0x1d54b,  # T -> 𝕋
        0x55: 0x1d54c,  # U -> 𝕌
        0x56: 0x1d54d,  # V -> 𝕍
        0x57: 0x1d54e,  # W -> 𝕎
        0x58: 0x1d54f,  # X -> 𝕏
        0x59: 0x1d550,  # Y -> 𝕐
        0x5a: 0x2124,  # Z -> ℤ
        # latin lower case
        0x61: 0x1d552,  # a -> 𝕒
        0x62: 0x1d553,  # b -> 𝕓
        0x63: 0x1d554,  # c -> 𝕔
        0x64: 0x1d555,  # d -> 𝕕
        0x65: 0x1d556,  # e -> 𝕖
        0x66: 0x1d557,  # f -> 𝕗
        0x67: 0x1d558,  # g -> 𝕘
        0x68: 0x1d559,  # h -> 𝕙
        0x69: 0x1d55a,  # i -> 𝕚
        0x6a: 0x1d55b,  # j -> 𝕛
        0x6b: 0x1d55c,  # k -> 𝕜
        0x6c: 0x1d55d,  # l -> 𝕝
        0x6d: 0x1d55e,  # m -> 𝕞
        0x6e: 0x1d55f,  # n -> 𝕟
        0x6f: 0x1d560,  # o -> 𝕠
        0x70: 0x1d561,  # p -> 𝕡
        0x71: 0x1d562,  # q -> 𝕢
        0x72: 0x1d563,  # r -> 𝕣
        0x73: 0x1d564,  # s -> 𝕤
        0x74: 0x1d565,  # t -> 𝕥
        0x75: 0x1d566,  # u -> 𝕦
        0x76: 0x1d567,  # v -> 𝕧
        0x77: 0x1d568,  # w -> 𝕨
        0x78: 0x1d569,  # x -> 𝕩
        0x79: 0x1d56a,  # y -> 𝕪
        0x7a: 0x1d56b,  # z -> 𝕫
        # greek upper case
        # greek lower case
    }
}

greek_uppercase_domain = {
    *range(0x0391, 0x0391+0x11),
    ord('ϴ'),
    *range(0x0391 + 0x12, 0x0391 + 0x19),
    ord('∇'),
}
greek_lowercase_domain = {*range(0x03B1, 0x03B1+0x19)} | {ord(c) for c in "∂ϵϑϰϕϱϖ"}
