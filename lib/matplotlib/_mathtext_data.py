"""
font data tables for truetype and afm computer modern fonts
"""

# this dict maps symbol names to fontnames, glyphindex.  To get the
# glyph index from the character code, you have to use get_charmap
"""
from matplotlib.ft2font import FT2Font
font = FT2Font('/usr/local/share/matplotlib/cmr10.ttf')
items = font.get_charmap().items()
items.sort()

for charcode, glyphind in items:
    print charcode, glyphind
"""

latex_to_bakoma = {
    r'\oint'                     : ('cmex10',  45),
    r'\bigodot'                  : ('cmex10',  50),
    r'\bigoplus'                 : ('cmex10',  55),
    r'\bigotimes'                : ('cmex10',  59),
    r'\sum'                      : ('cmex10',  51),
    r'\prod'                     : ('cmex10',  24),
    r'\int'                      : ('cmex10',  56),
    r'\bigcup'                   : ('cmex10',  28),
    r'\bigcap'                   : ('cmex10',  60),
    r'\biguplus'                 : ('cmex10',  32),
    r'\bigwedge'                 : ('cmex10',   4),
    r'\bigvee'                   : ('cmex10',  37),
    r'\coprod'                   : ('cmex10',  42),
    r'\__sqrt__'                 : ('cmex10',  48),
    r'\leftbrace'                : ('cmex10',  92),
    r'{'                         : ('cmex10',  92),
    r'\{'                        : ('cmex10',  92),
    r'\rightbrace'               : ('cmex10', 130),
    r'}'                         : ('cmex10', 130),
    r'\}'                        : ('cmex10', 130),
    r'\leftangle'                : ('cmex10',  97),
    r'\rightangle'               : ('cmex10',  64),
    r'\langle'                   : ('cmex10',  97),
    r'\rangle'                   : ('cmex10',  64),
    r'\widehat'                  : ('cmex10',  15),
    r'\widetilde'                : ('cmex10',  52),
    r'\widebar'                  : ('cmr10',  131),

    r'\omega'                    : ('cmmi10',  29),
    r'\varepsilon'               : ('cmmi10',  20),
    r'\vartheta'                 : ('cmmi10',  22),
    r'\varrho'                   : ('cmmi10',  61),
    r'\varsigma'                 : ('cmmi10',  41),
    r'\varphi'                   : ('cmmi10',   6),
    r'\leftharpoonup'            : ('cmmi10', 108),
    r'\leftharpoondown'          : ('cmmi10',  68),
    r'\rightharpoonup'           : ('cmmi10', 117),
    r'\rightharpoondown'         : ('cmmi10',  77),
    r'\triangleright'            : ('cmmi10', 130),
    r'\triangleleft'             : ('cmmi10',  89),
    r'.'                         : ('cmmi10',  51),
    r','                         : ('cmmi10',  44),
    r'<'                         : ('cmmi10',  99),
    r'/'                         : ('cmmi10',  98),
    r'>'                         : ('cmmi10', 107),
    r'\flat'                     : ('cmmi10', 131),
    r'\natural'                  : ('cmmi10',  90),
    r'\sharp'                    : ('cmmi10',  50),
    r'\smile'                    : ('cmmi10',  97),
    r'\frown'                    : ('cmmi10',  58),
    r'\ell'                      : ('cmmi10', 102),
    r'\imath'                    : ('cmmi10',   8),
    r'\jmath'                    : ('cmmi10',  65),
    r'\wp'                       : ('cmmi10',  14),
    r'\alpha'                    : ('cmmi10',  13),
    r'\beta'                     : ('cmmi10',  35),
    r'\gamma'                    : ('cmmi10',  24),
    r'\delta'                    : ('cmmi10',  38),
    r'\epsilon'                  : ('cmmi10',  54),
    r'\zeta'                     : ('cmmi10',  10),
    r'\eta'                      : ('cmmi10',   5),
    r'\theta'                    : ('cmmi10',  18),
    r'\iota'                     : ('cmmi10',  28),
    r'\lambda'                   : ('cmmi10',   9),
    r'\mu'                       : ('cmmi10',  32),
    r'\nu'                       : ('cmmi10',  34),
    r'\xi'                       : ('cmmi10',   7),
    r'\pi'                       : ('cmmi10',  36),
    r'\kappa'                    : ('cmmi10',  30),
    r'\rho'                      : ('cmmi10',  39),
    r'\sigma'                    : ('cmmi10',  21),
    r'\tau'                      : ('cmmi10',  43),
    r'\upsilon'                  : ('cmmi10',  25),
    r'\phi'                      : ('cmmi10',  42),
    r'\chi'                      : ('cmmi10',  17),
    r'\psi'                      : ('cmmi10',  31),
    r'|'                         : ('cmsy10',  47),
    r'\|'                        : ('cmsy10',  47),
    r'('                         : ('cmr10',  119),
    r'\leftparen'                : ('cmr10',  119),
    r'\rightparen'               : ('cmr10',   68),
    r')'                         : ('cmr10',   68),
    r'+'                         : ('cmr10',   76),
    r'0'                         : ('cmr10',   40),
    r'1'                         : ('cmr10',  100),
    r'2'                         : ('cmr10',   49),
    r'3'                         : ('cmr10',  110),
    r'4'                         : ('cmr10',   59),
    r'5'                         : ('cmr10',  120),
    r'6'                         : ('cmr10',   69),
    r'7'                         : ('cmr10',  127),
    r'8'                         : ('cmr10',   77),
    r'9'                         : ('cmr10',   22),
    r'                           :'                    : ('cmr10',   85),
    r';'                         : ('cmr10',   31),
    r'='                         : ('cmr10',   41),
    r'\leftbracket'              : ('cmr10',   62),
    r'['                         : ('cmr10',   62),
    r'\rightbracket'             : ('cmr10',   72),
    r']'                         : ('cmr10',   72),
    r'\%'                        : ('cmr10',   48),
    r'%'                         : ('cmr10',   48),
    r'\$'                        : ('cmr10',   99),
    r'@'                         : ('cmr10',  111),
    r'\#'                        : ('cmr10',   39),
    r'\_'                        : ('cmtt10', 79),
    r'\Gamma'                    : ('cmr10',  19),
    r'\Delta'                    : ('cmr10',   6),
    r'\Theta'                    : ('cmr10',   7),
    r'\Lambda'                   : ('cmr10',  14),
    r'\Xi'                       : ('cmr10',   3),
    r'\Pi'                       : ('cmr10',  17),
    r'\Sigma'                    : ('cmr10',  10),
    r'\Upsilon'                  : ('cmr10',  11),
    r'\Phi'                      : ('cmr10',   9),
    r'\Psi'                      : ('cmr10',  15),
    r'\Omega'                    : ('cmr10',  12),

    # these are mathml names, I think.  I'm just using them for the
    # tex methods noted
    r'\circumflexaccent'         : ('cmr10',   124), # for \hat
    r'\combiningbreve'           : ('cmr10',   81),  # for \breve
    r'\combiningoverline'        : ('cmr10',   131),  # for \bar
    r'\combininggraveaccent'     : ('cmr10', 114), # for \grave
    r'\combiningacuteaccent'     : ('cmr10', 63), # for \accute
    r'\combiningdiaeresis'       : ('cmr10', 91), # for \ddot
    r'\combiningtilde'           : ('cmr10', 75), # for \tilde
    r'\combiningrightarrowabove' : ('cmmi10', 110), # for \vec
    r'\combiningdotabove'        : ('cmr10', 26), # for \dot

    r'\leftarrow'                : ('cmsy10',  10),
    r'\uparrow'                  : ('cmsy10',  25),
    r'\downarrow'                : ('cmsy10',  28),
    r'\leftrightarrow'           : ('cmsy10',  24),
    r'\nearrow'                  : ('cmsy10',  99),
    r'\searrow'                  : ('cmsy10',  57),
    r'\simeq'                    : ('cmsy10', 108),
    r'\Leftarrow'                : ('cmsy10', 104),
    r'\Rightarrow'               : ('cmsy10', 112),
    r'\Uparrow'                  : ('cmsy10',  60),
    r'\Downarrow'                : ('cmsy10',  68),
    r'\Leftrightarrow'           : ('cmsy10',  51),
    r'\nwarrow'                  : ('cmsy10',  65),
    r'\swarrow'                  : ('cmsy10', 116),
    r'\propto'                   : ('cmsy10',  15),
    r'\infty'                    : ('cmsy10',  32),
    r'\in'                       : ('cmsy10',  59),
    r'\ni'                       : ('cmsy10', 122),
    r'\bigtriangleup'            : ('cmsy10',  80),
    r'\bigtriangledown'          : ('cmsy10', 132),
    r'\slash'                    : ('cmsy10',  87),
    r'\forall'                   : ('cmsy10',  21),
    r'\exists'                   : ('cmsy10',   5),
    r'\neg'                      : ('cmsy10',  20),
    r'\emptyset'                 : ('cmsy10',  33),
    r'\Re'                       : ('cmsy10',  95),
    r'\Im'                       : ('cmsy10',  52),
    r'\top'                      : ('cmsy10', 100),
    r'\bot'                      : ('cmsy10',  11),
    r'\aleph'                    : ('cmsy10',  26),
    r'\cup'                      : ('cmsy10',   6),
    r'\cap'                      : ('cmsy10',  19),
    r'\uplus'                    : ('cmsy10',  58),
    r'\wedge'                    : ('cmsy10',  43),
    r'\vee'                      : ('cmsy10',  96),
    r'\vdash'                    : ('cmsy10', 109),
    r'\dashv'                    : ('cmsy10',  66),
    r'\lfloor'                   : ('cmsy10', 117),
    r'\rfloor'                   : ('cmsy10',  74),
    r'\lceil'                    : ('cmsy10', 123),
    r'\rceil'                    : ('cmsy10',  81),
    r'\lbrace'                   : ('cmsy10',  92),
    r'\rbrace'                   : ('cmsy10', 105),
    r'\mid'                      : ('cmsy10',  47),
    r'\vert'                     : ('cmsy10',  47),
    r'\Vert'                     : ('cmsy10',  44),
    r'\updownarrow'              : ('cmsy10',  94),
    r'\Updownarrow'              : ('cmsy10',  53),
    r'\backslash'                : ('cmsy10', 126),
    r'\wr'                       : ('cmsy10', 101),
    r'\nabla'                    : ('cmsy10', 110),
    r'\sqcup'                    : ('cmsy10',  67),
    r'\sqcap'                    : ('cmsy10', 118),
    r'\sqsubseteq'               : ('cmsy10',  75),
    r'\sqsupseteq'               : ('cmsy10', 124),
    r'\S'                        : ('cmsy10', 129),
    r'\dag'                      : ('cmsy10',  71),
    r'\ddag'                     : ('cmsy10', 127),
    r'\P'                        : ('cmsy10', 130),
    r'\clubsuit'                 : ('cmsy10',  18),
    r'\diamondsuit'              : ('cmsy10',  34),
    r'\heartsuit'                : ('cmsy10',  22),
    r'-'                         : ('cmsy10',  17),
    r'\cdot'                     : ('cmsy10',  78),
    r'\times'                    : ('cmsy10',  13),
    r'*'                         : ('cmsy10',   9),
    r'\ast'                      : ('cmsy10',   9),
    r'\div'                      : ('cmsy10',  31),
    r'\diamond'                  : ('cmsy10',  48),
    r'\pm'                       : ('cmsy10',   8),
    r'\mp'                       : ('cmsy10',  98),
    r'\oplus'                    : ('cmsy10',  16),
    r'\ominus'                   : ('cmsy10',  56),
    r'\otimes'                   : ('cmsy10',  30),
    r'\oslash'                   : ('cmsy10', 107),
    r'\odot'                     : ('cmsy10',  64),
    r'\bigcirc'                  : ('cmsy10', 115),
    r'\circ'                     : ('cmsy10',  72),
    r'\bullet'                   : ('cmsy10',  84),
    r'\asymp'                    : ('cmsy10', 121),
    r'\equiv'                    : ('cmsy10',  35),
    r'\subseteq'                 : ('cmsy10', 103),
    r'\supseteq'                 : ('cmsy10',  42),
    r'\leq'                      : ('cmsy10',  14),
    r'\geq'                      : ('cmsy10',  29),
    r'\preceq'                   : ('cmsy10',  79),
    r'\succeq'                   : ('cmsy10', 131),
    r'\sim'                      : ('cmsy10',  27),
    r'\approx'                   : ('cmsy10',  23),
    r'\subset'                   : ('cmsy10',  50),
    r'\supset'                   : ('cmsy10',  86),
    r'\ll'                       : ('cmsy10',  85),
    r'\gg'                       : ('cmsy10',  40),
    r'\prec'                     : ('cmsy10',  93),
    r'\succ'                     : ('cmsy10',  49),
    r'\rightarrow'               : ('cmsy10',  12),
    r'\to'                       : ('cmsy10',  12),
    r'\spadesuit'                : ('cmsy10',   7),
}

latex_to_cmex = {
    r'\__sqrt__'   : 112,
    r'\bigcap'     : 92,
    r'\bigcup'     : 91,
    r'\bigodot'    : 75,
    r'\bigoplus'   : 77,
    r'\bigotimes'  : 79,
    r'\biguplus'   : 93,
    r'\bigvee'     : 95,
    r'\bigwedge'   : 94,
    r'\coprod'     : 97,
    r'\int'        : 90,
    r'\leftangle'  : 173,
    r'\leftbrace'  : 169,
    r'\oint'       : 73,
    r'\prod'       : 89,
    r'\rightangle' : 174,
    r'\rightbrace' : 170,
    r'\sum'        : 88,
    r'\widehat'    : 98,
    r'\widetilde'  : 101,
}

latex_to_standard = {
    r'\cong'                     : ('psyr', 64),
    r'\Delta'                    : ('psyr', 68),
    r'\Phi'                      : ('psyr', 70),
    r'\Gamma'                    : ('psyr', 89),
    r'\alpha'                    : ('psyr', 97),
    r'\beta'                     : ('psyr', 98),
    r'\chi'                      : ('psyr', 99),
    r'\delta'                    : ('psyr', 100),
    r'\varepsilon'               : ('psyr', 101),
    r'\phi'                      : ('psyr', 102),
    r'\gamma'                    : ('psyr', 103),
    r'\eta'                      : ('psyr', 104),
    r'\iota'                     : ('psyr', 105),
    r'\varpsi'                   : ('psyr', 106),
    r'\kappa'                    : ('psyr', 108),
    r'\nu'                       : ('psyr', 110),
    r'\pi'                       : ('psyr', 112),
    r'\theta'                    : ('psyr', 113),
    r'\rho'                      : ('psyr', 114),
    r'\sigma'                    : ('psyr', 115),
    r'\tau'                      : ('psyr', 116),
    r'\upsilon'                  : ('psyr', 117),
    r'\varpi'                    : ('psyr', 118),
    r'\omega'                    : ('psyr', 119),
    r'\xi'                       : ('psyr', 120),
    r'\psi'                      : ('psyr', 121),
    r'\zeta'                     : ('psyr', 122),
    r'\sim'                      : ('psyr', 126),
    r'\leq'                      : ('psyr', 163),
    r'\infty'                    : ('psyr', 165),
    r'\clubsuit'                 : ('psyr', 167),
    r'\diamondsuit'              : ('psyr', 168),
    r'\heartsuit'                : ('psyr', 169),
    r'\spadesuit'                : ('psyr', 170),
    r'\leftrightarrow'           : ('psyr', 171),
    r'\leftarrow'                : ('psyr', 172),
    r'\uparrow'                  : ('psyr', 173),
    r'\rightarrow'               : ('psyr', 174),
    r'\downarrow'                : ('psyr', 175),
    r'\pm'                       : ('psyr', 176),
    r'\geq'                      : ('psyr', 179),
    r'\times'                    : ('psyr', 180),
    r'\propto'                   : ('psyr', 181),
    r'\partial'                  : ('psyr', 182),
    r'\bullet'                   : ('psyr', 183),
    r'\div'                      : ('psyr', 184),
    r'\neq'                      : ('psyr', 185),
    r'\equiv'                    : ('psyr', 186),
    r'\approx'                   : ('psyr', 187),
    r'\ldots'                    : ('psyr', 188),
    r'\aleph'                    : ('psyr', 192),
    r'\Im'                       : ('psyr', 193),
    r'\Re'                       : ('psyr', 194),
    r'\wp'                       : ('psyr', 195),
    r'\otimes'                   : ('psyr', 196),
    r'\oplus'                    : ('psyr', 197),
    r'\oslash'                   : ('psyr', 198),
    r'\cap'                      : ('psyr', 199),
    r'\cup'                      : ('psyr', 200),
    r'\supset'                   : ('psyr', 201),
    r'\supseteq'                 : ('psyr', 202),
    r'\subset'                   : ('psyr', 204),
    r'\subseteq'                 : ('psyr', 205),
    r'\in'                       : ('psyr', 206),
    r'\notin'                    : ('psyr', 207),
    r'\angle'                    : ('psyr', 208),
    r'\nabla'                    : ('psyr', 209),
    r'\textregistered'           : ('psyr', 210),
    r'\copyright'                : ('psyr', 211),
    r'\texttrademark'            : ('psyr', 212),
    r'\Pi'                       : ('psyr', 213),
    r'\prod'                     : ('psyr', 213),
    r'\surd'                     : ('psyr', 214),
    r'\__sqrt__'                 : ('psyr', 214),
    r'\cdot'                     : ('psyr', 215),
    r'\urcorner'                 : ('psyr', 216),
    r'\vee'                      : ('psyr', 217),
    r'\wedge'                    : ('psyr', 218),
    r'\Leftrightarrow'           : ('psyr', 219),
    r'\Leftarrow'                : ('psyr', 220),
    r'\Uparrow'                  : ('psyr', 221),
    r'\Rightarrow'               : ('psyr', 222),
    r'\Downarrow'                : ('psyr', 223),
    r'\Diamond'                  : ('psyr', 224),
    r'\langle'                   : ('psyr', 225),
    r'\Sigma'                    : ('psyr', 229),
    r'\sum'                      : ('psyr', 229),
    r'\forall'                   : ('psyr',  34),
    r'\exists'                   : ('psyr',  36),
    r'\lceil'                    : ('psyr', 233),
    r'\lbrace'                   : ('psyr', 123),
    r'\Psi'                      : ('psyr',  89),
    r'\bot'                      : ('psyr', 0136),
    r'\Omega'                    : ('psyr', 0127),
    r'\leftbracket'              : ('psyr', 0133),
    r'\rightbracket'             : ('psyr', 0135),
    r'\leftbrace'                : ('psyr', 123),
    r'\leftparen'                : ('psyr', 050),
    r'\prime'                    : ('psyr', 0242),
    r'\sharp'                    : ('psyr', 043),
    r'\slash'                    : ('psyr', 057),
    r'\Lamda'                    : ('psyr', 0114),
    r'\neg'                      : ('psyr', 0330),
    r'\Upsilon'                  : ('psyr', 0241),
    r'\rightbrace'               : ('psyr', 0175),
    r'\rfloor'                   : ('psyr', 0373),
    r'\lambda'                   : ('psyr', 0154),
    r'\to'                       : ('psyr', 0256),
    r'\Xi'                       : ('psyr', 0130),
    r'\emptyset'                 : ('psyr', 0306),
    r'\lfloor'                   : ('psyr', 0353),
    r'\rightparen'               : ('psyr', 051),
    r'\rceil'                    : ('psyr', 0371),
    r'\ni'                       : ('psyr', 047),
    r'\epsilon'                  : ('psyr', 0145),
    r'\Theta'                    : ('psyr', 0121),
    r'\langle'                   : ('psyr', 0341),
    r'\leftangle'                : ('psyr', 0341),
    r'\rangle'                   : ('psyr', 0361),
    r'\rightangle'               : ('psyr', 0361),
    r'\rbrace'                   : ('psyr', 0175),
    r'\circ'                     : ('psyr', 0260),
    r'\diamond'                  : ('psyr', 0340),
    r'\mu'                       : ('psyr', 0155),
    r'\mid'                      : ('psyr', 0352),
    r'\imath'                    : ('pncri8a', 105),
    r'\%'                        : ('pncr8a',  37),
    r'\$'                        : ('pncr8a',  36),
    r'\{'                        : ('pncr8a', 123),
    r'\}'                        : ('pncr8a', 125),
    r'\backslash'                : ('pncr8a',  92),
    r'\ast'                      : ('pncr8a',  42),
    r'\#'                        : ('pncr8a',  35),

    r'\circumflexaccent'         : ('pncri8a',   124), # for \hat
    r'\combiningbreve'           : ('pncri8a',   81),  # for \breve
    r'\combininggraveaccent'     : ('pncri8a', 114), # for \grave
    r'\combiningacuteaccent'     : ('pncri8a', 63), # for \accute
    r'\combiningdiaeresis'       : ('pncri8a', 91), # for \ddot
    r'\combiningtilde'           : ('pncri8a', 75), # for \tilde
    r'\combiningrightarrowabove' : ('pncri8a', 110), # for \vec
    r'\combiningdotabove'        : ('pncri8a', 26), # for \dot
}

tex2uni = {
    'widehat'                  : 0x0302,
    'widetilde'                : 0x0303,
    'widebar'                  : 0x0305,
    'langle'                   : 0x27e8,
    'rangle'                   : 0x27e9,
    'perp'                     : 0x27c2,
    'neq'                      : 0x2260,
    'Join'                     : 0x2a1d,
    'leqslant'                 : 0x2a7d,
    'geqslant'                 : 0x2a7e,
    'lessapprox'               : 0x2a85,
    'gtrapprox'                : 0x2a86,
    'lesseqqgtr'               : 0x2a8b,
    'gtreqqless'               : 0x2a8c,
    'triangleeq'               : 0x225c,
    'eqslantless'              : 0x2a95,
    'eqslantgtr'               : 0x2a96,
    'backepsilon'              : 0x03f6,
    'precapprox'               : 0x2ab7,
    'succapprox'               : 0x2ab8,
    'fallingdotseq'            : 0x2252,
    'subseteqq'                : 0x2ac5,
    'supseteqq'                : 0x2ac6,
    'varpropto'                : 0x221d,
    'precnapprox'              : 0x2ab9,
    'succnapprox'              : 0x2aba,
    'subsetneqq'               : 0x2acb,
    'supsetneqq'               : 0x2acc,
    'lnapprox'                 : 0x2ab9,
    'gnapprox'                 : 0x2aba,
    'longleftarrow'            : 0x27f5,
    'longrightarrow'           : 0x27f6,
    'longleftrightarrow'       : 0x27f7,
    'Longleftarrow'            : 0x27f8,
    'Longrightarrow'           : 0x27f9,
    'Longleftrightarrow'       : 0x27fa,
    'longmapsto'               : 0x27fc,
    'leadsto'                  : 0x21dd,
    'dashleftarrow'            : 0x290e,
    'dashrightarrow'           : 0x290f,
    'circlearrowleft'          : 0x21ba,
    'circlearrowright'         : 0x21bb,
    'leftrightsquigarrow'      : 0x21ad,
    'leftsquigarrow'           : 0x219c,
    'rightsquigarrow'          : 0x219d,
    'Game'                     : 0x2141,
    'hbar'                     : 0x0127,
    'hslash'                   : 0x210f,
    'ldots'                    : 0x2026,
    'vdots'                    : 0x22ee,
    'doteqdot'                 : 0x2251,
    'doteq'                    : 8784,
    'partial'                  : 8706,
    'gg'                       : 8811,
    'asymp'                    : 8781,
    'blacktriangledown'        : 9662,
    'otimes'                   : 8855,
    'nearrow'                  : 8599,
    'varpi'                    : 982,
    'vee'                      : 8744,
    'vec'                      : 8407,
    'smile'                    : 8995,
    'succnsim'                 : 8937,
    'gimel'                    : 8503,
    'vert'                     : 124,
    '|'                        : 124,
    'varrho'                   : 1009,
    'P'                        : 182,
    'approxident'              : 8779,
    'Swarrow'                  : 8665,
    'textasciicircum'          : 94,
    'imageof'                  : 8887,
    'ntriangleleft'            : 8938,
    'nleq'                     : 8816,
    'div'                      : 247,
    'nparallel'                : 8742,
    'Leftarrow'                : 8656,
    'lll'                      : 8920,
    'oiint'                    : 8751,
    'ngeq'                     : 8817,
    'Theta'                    : 920,
    'origof'                   : 8886,
    'blacksquare'              : 9632,
    'solbar'                   : 9023,
    'neg'                      : 172,
    'sum'                      : 8721,
    'Vdash'                    : 8873,
    'coloneq'                  : 8788,
    'degree'                   : 176,
    'bowtie'                   : 8904,
    'blacktriangleright'       : 9654,
    'varsigma'                 : 962,
    'leq'                      : 8804,
    'ggg'                      : 8921,
    'lneqq'                    : 8808,
    'scurel'                   : 8881,
    'stareq'                   : 8795,
    'BbbN'                     : 8469,
    'nLeftarrow'               : 8653,
    'nLeftrightarrow'          : 8654,
    'k'                        : 808,
    'bot'                      : 8869,
    'BbbC'                     : 8450,
    'Lsh'                      : 8624,
    'leftleftarrows'           : 8647,
    'BbbZ'                     : 8484,
    'digamma'                  : 989,
    'BbbR'                     : 8477,
    'BbbP'                     : 8473,
    'BbbQ'                     : 8474,
    'vartriangleright'         : 8883,
    'succsim'                  : 8831,
    'wedge'                    : 8743,
    'lessgtr'                  : 8822,
    'veebar'                   : 8891,
    'mapsdown'                 : 8615,
    'Rsh'                      : 8625,
    'chi'                      : 967,
    'prec'                     : 8826,
    'nsubseteq'                : 8840,
    'therefore'                : 8756,
    'eqcirc'                   : 8790,
    'textexclamdown'           : 161,
    'nRightarrow'              : 8655,
    'flat'                     : 9837,
    'notin'                    : 8713,
    'llcorner'                 : 8990,
    'varepsilon'               : 949,
    'bigtriangleup'            : 9651,
    'aleph'                    : 8501,
    'dotminus'                 : 8760,
    'upsilon'                  : 965,
    'Lambda'                   : 923,
    'cap'                      : 8745,
    'barleftarrow'             : 8676,
    'mu'                       : 956,
    'boxplus'                  : 8862,
    'mp'                       : 8723,
    'circledast'               : 8859,
    'tau'                      : 964,
    'in'                       : 8712,
    'backslash'                : 92,
    'varnothing'               : 8709,
    'sharp'                    : 9839,
    'eqsim'                    : 8770,
    'gnsim'                    : 8935,
    'Searrow'                  : 8664,
    'updownarrows'             : 8645,
    'heartsuit'                : 9825,
    'trianglelefteq'           : 8884,
    'ddag'                     : 8225,
    'sqsubseteq'               : 8849,
    'mapsfrom'                 : 8612,
    'boxbar'                   : 9707,
    'sim'                      : 8764,
    'Nwarrow'                  : 8662,
    'nequiv'                   : 8802,
    'succ'                     : 8827,
    'vdash'                    : 8866,
    'Leftrightarrow'           : 8660,
    'parallel'                 : 8741,
    'invnot'                   : 8976,
    'natural'                  : 9838,
    'ss'                       : 223,
    'uparrow'                  : 8593,
    'nsim'                     : 8769,
    'hookrightarrow'           : 8618,
    'Equiv'                    : 8803,
    'approx'                   : 8776,
    'Vvdash'                   : 8874,
    'nsucc'                    : 8833,
    'leftrightharpoons'        : 8651,
    'Re'                       : 8476,
    'boxminus'                 : 8863,
    'equiv'                    : 8801,
    'Lleftarrow'               : 8666,
    'thinspace'                : 8201,
    'll'                       : 8810,
    'Cup'                      : 8915,
    'measeq'                   : 8798,
    'upharpoonleft'            : 8639,
    'lq'                       : 8216,
    'Upsilon'                  : 933,
    'subsetneq'                : 8842,
    'greater'                  : 62,
    'supsetneq'                : 8843,
    'Cap'                      : 8914,
    'L'                        : 321,
    'spadesuit'                : 9824,
    'lrcorner'                 : 8991,
    'not'                      : 824,
    'bar'                      : 772,
    'rightharpoonaccent'       : 8401,
    'boxdot'                   : 8865,
    'l'                        : 322,
    'leftharpoondown'          : 8637,
    'bigcup'                   : 8899,
    'iint'                     : 8748,
    'bigwedge'                 : 8896,
    'downharpoonleft'          : 8643,
    'textasciitilde'           : 126,
    'subset'                   : 8834,
    'leqq'                     : 8806,
    'mapsup'                   : 8613,
    'nvDash'                   : 8877,
    'looparrowleft'            : 8619,
    'nless'                    : 8814,
    'rightarrowbar'            : 8677,
    'Vert'                     : 8214,
    'downdownarrows'           : 8650,
    'uplus'                    : 8846,
    'simeq'                    : 8771,
    'napprox'                  : 8777,
    'ast'                      : 8727,
    'twoheaduparrow'           : 8607,
    'doublebarwedge'           : 8966,
    'Sigma'                    : 931,
    'leftharpoonaccent'        : 8400,
    'ntrianglelefteq'          : 8940,
    'nexists'                  : 8708,
    'times'                    : 215,
    'measuredangle'            : 8737,
    'bumpeq'                   : 8783,
    'carriagereturn'           : 8629,
    'adots'                    : 8944,
    'checkmark'                : 10003,
    'lambda'                   : 955,
    'xi'                       : 958,
    'rbrace'                   : 125,
    'rbrack'                   : 93,
    'Nearrow'                  : 8663,
    'maltese'                  : 10016,
    'clubsuit'                 : 9827,
    'top'                      : 8868,
    'overarc'                  : 785,
    'varphi'                   : 966,
    'Delta'                    : 916,
    'iota'                     : 953,
    'nleftarrow'               : 8602,
    'candra'                   : 784,
    'supset'                   : 8835,
    'triangleleft'             : 9665,
    'gtreqless'                : 8923,
    'ntrianglerighteq'         : 8941,
    'quad'                     : 8195,
    'Xi'                       : 926,
    'gtrdot'                   : 8919,
    'leftthreetimes'           : 8907,
    'minus'                    : 8722,
    'preccurlyeq'              : 8828,
    'nleftrightarrow'          : 8622,
    'lambdabar'                : 411,
    'blacktriangle'            : 9652,
    'kernelcontraction'        : 8763,
    'Phi'                      : 934,
    'angle'                    : 8736,
    'spadesuitopen'            : 9828,
    'eqless'                   : 8924,
    'mid'                      : 8739,
    'varkappa'                 : 1008,
    'Ldsh'                     : 8626,
    'updownarrow'              : 8597,
    'beta'                     : 946,
    'textquotedblleft'         : 8220,
    'rho'                      : 961,
    'alpha'                    : 945,
    'intercal'                 : 8890,
    'beth'                     : 8502,
    'grave'                    : 768,
    'acwopencirclearrow'       : 8634,
    'nmid'                     : 8740,
    'nsupset'                  : 8837,
    'sigma'                    : 963,
    'dot'                      : 775,
    'Rightarrow'               : 8658,
    'turnednot'                : 8985,
    'backsimeq'                : 8909,
    'leftarrowtail'            : 8610,
    'approxeq'                 : 8778,
    'curlyeqsucc'              : 8927,
    'rightarrowtail'           : 8611,
    'Psi'                      : 936,
    'copyright'                : 169,
    'yen'                      : 165,
    'vartriangleleft'          : 8882,
    'rasp'                     : 700,
    'triangleright'            : 9655,
    'precsim'                  : 8830,
    'infty'                    : 8734,
    'geq'                      : 8805,
    'updownarrowbar'           : 8616,
    'precnsim'                 : 8936,
    'H'                        : 779,
    'ulcorner'                 : 8988,
    'looparrowright'           : 8620,
    'ncong'                    : 8775,
    'downarrow'                : 8595,
    'circeq'                   : 8791,
    'subseteq'                 : 8838,
    'bigstar'                  : 9733,
    'prime'                    : 8242,
    'lceil'                    : 8968,
    'Rrightarrow'              : 8667,
    'oiiint'                   : 8752,
    'curlywedge'               : 8911,
    'vDash'                    : 8872,
    'lfloor'                   : 8970,
    'ddots'                    : 8945,
    'exists'                   : 8707,
    'underbar'                 : 817,
    'Pi'                       : 928,
    'leftrightarrows'          : 8646,
    'sphericalangle'           : 8738,
    'coprod'                   : 8720,
    'circledcirc'              : 8858,
    'gtrsim'                   : 8819,
    'gneqq'                    : 8809,
    'between'                  : 8812,
    'theta'                    : 952,
    'complement'               : 8705,
    'arceq'                    : 8792,
    'nVdash'                   : 8878,
    'S'                        : 167,
    'wr'                       : 8768,
    'wp'                       : 8472,
    'backcong'                 : 8780,
    'lasp'                     : 701,
    'c'                        : 807,
    'nabla'                    : 8711,
    'dotplus'                  : 8724,
    'eta'                      : 951,
    'forall'                   : 8704,
    'eth'                      : 240,
    'colon'                    : 58,
    'sqcup'                    : 8852,
    'rightrightarrows'         : 8649,
    'sqsupset'                 : 8848,
    'mapsto'                   : 8614,
    'bigtriangledown'          : 9661,
    'sqsupseteq'               : 8850,
    'propto'                   : 8733,
    'pi'                       : 960,
    'pm'                       : 177,
    'dots'                     : 0x2026,
    'nrightarrow'              : 8603,
    'textasciiacute'           : 180,
    'Doteq'                    : 8785,
    'breve'                    : 774,
    'sqcap'                    : 8851,
    'twoheadrightarrow'        : 8608,
    'kappa'                    : 954,
    'vartriangle'              : 9653,
    'diamondsuit'              : 9826,
    'pitchfork'                : 8916,
    'blacktriangleleft'        : 9664,
    'nprec'                    : 8832,
    'vdots'                    : 8942,
    'curvearrowright'          : 8631,
    'barwedge'                 : 8892,
    'multimap'                 : 8888,
    'textquestiondown'         : 191,
    'cong'                     : 8773,
    'rtimes'                   : 8906,
    'rightzigzagarrow'         : 8669,
    'rightarrow'               : 8594,
    'leftarrow'                : 8592,
    '__sqrt__'                 : 8730,
    'twoheaddownarrow'         : 8609,
    'oint'                     : 8750,
    'bigvee'                   : 8897,
    'eqdef'                    : 8797,
    'sterling'                 : 163,
    'phi'                      : 981,
    'Updownarrow'              : 8661,
    'backprime'                : 8245,
    'emdash'                   : 8212,
    'Gamma'                    : 915,
    'i'                        : 305,
    'rceil'                    : 8969,
    'leftharpoonup'            : 8636,
    'Im'                       : 8465,
    'curvearrowleft'           : 8630,
    'wedgeq'                   : 8793,
    'fallingdotseq'            : 8786,
    'curlyeqprec'              : 8926,
    'questeq'                  : 8799,
    'less'                     : 60,
    'upuparrows'               : 8648,
    'tilde'                    : 771,
    'textasciigrave'           : 96,
    'smallsetminus'            : 8726,
    'ell'                      : 8467,
    'cup'                      : 8746,
    'danger'                   : 9761,
    'nVDash'                   : 8879,
    'cdotp'                    : 183,
    'cdots'                    : 8943,
    'hat'                      : 770,
    'eqgtr'                    : 8925,
    'enspace'                  : 8194,
    'psi'                      : 968,
    'frown'                    : 8994,
    'acute'                    : 769,
    'downzigzagarrow'          : 8623,
    'ntriangleright'           : 8939,
    'cupdot'                   : 8845,
    'circleddash'              : 8861,
    'oslash'                   : 8856,
    'mho'                      : 8487,
    'd'                        : 803,
    'sqsubset'                 : 8847,
    'cdot'                     : 8901,
    'Omega'                    : 937,
    'OE'                       : 338,
    'veeeq'                    : 8794,
    'Finv'                     : 8498,
    't'                        : 865,
    'leftrightarrow'           : 8596,
    'swarrow'                  : 8601,
    'rightthreetimes'          : 8908,
    'rightleftharpoons'        : 8652,
    'lesssim'                  : 8818,
    'searrow'                  : 8600,
    'because'                  : 8757,
    'gtrless'                  : 8823,
    'star'                     : 8902,
    'nsubset'                  : 8836,
    'zeta'                     : 950,
    'dddot'                    : 8411,
    'bigcirc'                  : 9675,
    'Supset'                   : 8913,
    'circ'                     : 8728,
    'slash'                    : 8725,
    'ocirc'                    : 778,
    'prod'                     : 8719,
    'twoheadleftarrow'         : 8606,
    'daleth'                   : 8504,
    'upharpoonright'           : 8638,
    'odot'                     : 8857,
    'Uparrow'                  : 8657,
    'O'                        : 216,
    'hookleftarrow'            : 8617,
    'trianglerighteq'          : 8885,
    'nsime'                    : 8772,
    'oe'                       : 339,
    'nwarrow'                  : 8598,
    'o'                        : 248,
    'ddddot'                   : 8412,
    'downharpoonright'         : 8642,
    'succcurlyeq'              : 8829,
    'gamma'                    : 947,
    'scrR'                     : 8475,
    'dag'                      : 8224,
    'thickspace'               : 8197,
    'frakZ'                    : 8488,
    'lessdot'                  : 8918,
    'triangledown'             : 9663,
    'ltimes'                   : 8905,
    'scrB'                     : 8492,
    'endash'                   : 8211,
    'scrE'                     : 8496,
    'scrF'                     : 8497,
    'scrH'                     : 8459,
    'scrI'                     : 8464,
    'rightharpoondown'         : 8641,
    'scrL'                     : 8466,
    'scrM'                     : 8499,
    'frakC'                    : 8493,
    'nsupseteq'                : 8841,
    'circledR'                 : 174,
    'circledS'                 : 9416,
    'ngtr'                     : 8815,
    'bigcap'                   : 8898,
    'scre'                     : 8495,
    'Downarrow'                : 8659,
    'scrg'                     : 8458,
    'overleftrightarrow'       : 8417,
    'scro'                     : 8500,
    'lnsim'                    : 8934,
    'eqcolon'                  : 8789,
    'curlyvee'                 : 8910,
    'urcorner'                 : 8989,
    'lbrace'                   : 123,
    'Bumpeq'                   : 8782,
    'delta'                    : 948,
    'boxtimes'                 : 8864,
    'overleftarrow'            : 8406,
    'prurel'                   : 8880,
    'clubsuitopen'             : 9831,
    'cwopencirclearrow'        : 8635,
    'geqq'                     : 8807,
    'rightleftarrows'          : 8644,
    'ac'                       : 8766,
    'ae'                       : 230,
    'int'                      : 8747,
    'rfloor'                   : 8971,
    'risingdotseq'             : 8787,
    'nvdash'                   : 8876,
    'diamond'                  : 8900,
    'ddot'                     : 776,
    'backsim'                  : 8765,
    'oplus'                    : 8853,
    'triangleq'                : 8796,
    'check'                    : 780,
    'ni'                       : 8715,
    'iiint'                    : 8749,
    'ne'                       : 8800,
    'lesseqgtr'                : 8922,
    'obar'                     : 9021,
    'supseteq'                 : 8839,
    'nu'                       : 957,
    'AA'                       : 8491,
    'AE'                       : 198,
    'models'                   : 8871,
    'ominus'                   : 8854,
    'dashv'                    : 8867,
    'omega'                    : 969,
    'rq'                       : 8217,
    'Subset'                   : 8912,
    'rightharpoonup'           : 8640,
    'Rdsh'                     : 8627,
    'bullet'                   : 8729,
    'divideontimes'            : 8903,
    'lbrack'                   : 91,
    'textquotedblright'        : 8221,
    'Colon'                    : 8759,
    '%'                        : 37,
    '$'                        : 36,
    '{'                        : 123,
    '}'                        : 125,
    '_'                        : 95,
    '#'                        : 35,
    'imath'                    : 0x131,
    'circumflexaccent'         : 770,
    'combiningbreve'           : 774,
    'combiningoverline'        : 772,
    'combininggraveaccent'     : 768,
    'combiningacuteaccent'     : 769,
    'combiningdiaeresis'       : 776,
    'combiningtilde'           : 771,
    'combiningrightarrowabove' : 8407,
    'combiningdotabove'        : 775,
    'to'                       : 8594,
    'succeq'                   : 8829,
    'emptyset'                 : 8709,
    'leftparen'                : 40,
    'rightparen'               : 41,
    'bigoplus'                 : 10753,
    'leftangle'                : 10216,
    'rightangle'               : 10217,
    'leftbrace'                : 124,
    'rightbrace'               : 125,
    'jmath'                    : 567,
    'bigodot'                  : 10752,
    'preceq'                   : 8828,
    'biguplus'                 : 10756,
    'epsilon'                  : 949,
    'vartheta'                 : 977,
    'bigotimes'                : 10754,
    'guillemotleft'            : 171,
    'ring'                     : 730,
    'Thorn'                    : 222,
    'guilsinglright'           : 8250,
    'perthousand'              : 8240,
    'macron'                   : 175,
    'cent'                     : 162,
    'guillemotright'           : 187,
    'equal'                    : 61,
    'asterisk'                 : 42,
    'guilsinglleft'            : 8249,
    'plus'                     : 43,
    'thorn'                    : 254,
    'dagger'                   : 8224
}

# Each element is a 4-tuple of the form:
#   src_start, src_end, dst_font, dst_start
#
stix_virtual_fonts = {
    'bb':
        {
        'rm':
            [
            (0x0030, 0x0039, 'rm', 0x1d7d8), # 0-9
            (0x0041, 0x0042, 'rm', 0x1d538), # A-B
            (0x0043, 0x0043, 'rm', 0x2102),  # C
            (0x0044, 0x0047, 'rm', 0x1d53b), # D-G
            (0x0048, 0x0048, 'rm', 0x210d),  # H
            (0x0049, 0x004d, 'rm', 0x1d540), # I-M
            (0x004e, 0x004e, 'rm', 0x2115),  # N
            (0x004f, 0x004f, 'rm', 0x1d546), # O
            (0x0050, 0x0051, 'rm', 0x2119),  # P-Q
            (0x0052, 0x0052, 'rm', 0x211d),  # R
            (0x0053, 0x0059, 'rm', 0x1d54a), # S-Y
            (0x005a, 0x005a, 'rm', 0x2124),  # Z
            (0x0061, 0x007a, 'rm', 0x1d552), # a-z
            (0x0393, 0x0393, 'rm', 0x213e),  # \Gamma
            (0x03a0, 0x03a0, 'rm', 0x213f),  # \Pi
            (0x03a3, 0x03a3, 'rm', 0x2140),  # \Sigma
            (0x03b3, 0x03b3, 'rm', 0x213d),  # \gamma
            (0x03c0, 0x03c0, 'rm', 0x213c),  # \pi
            ],
        'it':
            [
            (0x0030, 0x0039, 'rm', 0x1d7d8), # 0-9
            (0x0041, 0x0042, 'it', 0xe154),  # A-B
            (0x0043, 0x0043, 'it', 0x2102),  # C
            (0x0044, 0x0044, 'it', 0x2145),  # D
            (0x0045, 0x0047, 'it', 0xe156),  # E-G
            (0x0048, 0x0048, 'it', 0x210d),  # H
            (0x0049, 0x004d, 'it', 0xe159),  # I-M
            (0x004e, 0x004e, 'it', 0x2115),  # N
            (0x004f, 0x004f, 'it', 0xe15e),  # O
            (0x0050, 0x0051, 'it', 0x2119),  # P-Q
            (0x0052, 0x0052, 'it', 0x211d),  # R
            (0x0053, 0x0059, 'it', 0xe15f),  # S-Y
            (0x005a, 0x005a, 'it', 0x2124),  # Z
            (0x0061, 0x0063, 'it', 0xe166),  # a-c
            (0x0064, 0x0065, 'it', 0x2146),  # d-e
            (0x0066, 0x0068, 'it', 0xe169),  # f-h
            (0x0069, 0x006a, 'it', 0x2148),  # i-j
            (0x006b, 0x007a, 'it', 0xe16c),  # k-z
            (0x0393, 0x0393, 'it', 0x213e),  # \Gamma (missing in beta STIX fonts)
            (0x03a0, 0x03a0, 'it', 0x213f),  # \Pi
            (0x03a3, 0x03a3, 'it', 0x2140),  # \Sigma (missing in beta STIX fonts)
            (0x03b3, 0x03b3, 'it', 0x213d),  # \gamma (missing in beta STIX fonts)
            (0x03c0, 0x03c0, 'it', 0x213c),  # \pi
            ],
        'bf':
            [
            (0x0030, 0x0039, 'rm', 0x1d7d8), # 0-9
            (0x0041, 0x0042, 'bf', 0xe38a),  # A-B
            (0x0043, 0x0043, 'bf', 0x2102),  # C
            (0x0044, 0x0044, 'bf', 0x2145),  # D
            (0x0045, 0x0047, 'bf', 0xe38d),  # E-G
            (0x0048, 0x0048, 'bf', 0x210d),  # H
            (0x0049, 0x004d, 'bf', 0xe390),  # I-M
            (0x004e, 0x004e, 'bf', 0x2115),  # N
            (0x004f, 0x004f, 'bf', 0xe395),  # O
            (0x0050, 0x0051, 'bf', 0x2119),  # P-Q
            (0x0052, 0x0052, 'bf', 0x211d),  # R
            (0x0053, 0x0059, 'bf', 0xe396),  # S-Y
            (0x005a, 0x005a, 'bf', 0x2124),  # Z
            (0x0061, 0x0063, 'bf', 0xe39d),  # a-c
            (0x0064, 0x0065, 'bf', 0x2146),  # d-e
            (0x0066, 0x0068, 'bf', 0xe3a2),  # f-h
            (0x0069, 0x006a, 'bf', 0x2148),  # i-j
            (0x006b, 0x007a, 'bf', 0xe3a7),  # k-z
            (0x0393, 0x0393, 'bf', 0x213e),  # \Gamma
            (0x03a0, 0x03a0, 'bf', 0x213f),  # \Pi
            (0x03a3, 0x03a3, 'bf', 0x2140),  # \Sigma
            (0x03b3, 0x03b3, 'bf', 0x213d),  # \gamma
            (0x03c0, 0x03c0, 'bf', 0x213c),  # \pi
            ],
        },
    'cal':
        [
        (0x0041, 0x005a, 'it', 0xe22d), # A-Z
        ],
    'circled':
        {
        'rm':
            [
            (0x0030, 0x0030, 'rm', 0x24ea), # 0
            (0x0031, 0x0039, 'rm', 0x2460), # 1-9
            (0x0041, 0x005a, 'rm', 0x24b6), # A-Z
            (0x0061, 0x007a, 'rm', 0x24d0)  # a-z
            ],
        'it':
            [
            (0x0030, 0x0030, 'rm', 0x24ea), # 0
            (0x0031, 0x0039, 'rm', 0x2460), # 1-9
            (0x0041, 0x005a, 'it', 0x24b6), # A-Z
            (0x0061, 0x007a, 'it', 0x24d0)  # a-z
            ],
        'bf':
            [
            (0x0030, 0x0030, 'bf', 0x24ea), # 0
            (0x0031, 0x0039, 'bf', 0x2460), # 1-9
            (0x0041, 0x005a, 'bf', 0x24b6), # A-Z
            (0x0061, 0x007a, 'bf', 0x24d0)  # a-z
            ],
        },
    'frak':
        {
        'rm':
            [
            (0x0041, 0x0042, 'rm', 0x1d504), # A-B
            (0x0043, 0x0043, 'rm', 0x212d),  # C
            (0x0044, 0x0047, 'rm', 0x1d507), # D-G
            (0x0048, 0x0048, 'rm', 0x210c),  # H
            (0x0049, 0x0049, 'rm', 0x2111),  # I
            (0x004a, 0x0051, 'rm', 0x1d50d), # J-Q
            (0x0052, 0x0052, 'rm', 0x211c),  # R
            (0x0053, 0x0059, 'rm', 0x1d516), # S-Y
            (0x005a, 0x005a, 'rm', 0x2128),  # Z
            (0x0061, 0x007a, 'rm', 0x1d51e), # a-z
            ],
        'it':
            [
            (0x0041, 0x0042, 'rm', 0x1d504), # A-B
            (0x0043, 0x0043, 'rm', 0x212d),  # C
            (0x0044, 0x0047, 'rm', 0x1d507), # D-G
            (0x0048, 0x0048, 'rm', 0x210c),  # H
            (0x0049, 0x0049, 'rm', 0x2111),  # I
            (0x004a, 0x0051, 'rm', 0x1d50d), # J-Q
            (0x0052, 0x0052, 'rm', 0x211c),  # R
            (0x0053, 0x0059, 'rm', 0x1d516), # S-Y
            (0x005a, 0x005a, 'rm', 0x2128),  # Z
            (0x0061, 0x007a, 'rm', 0x1d51e), # a-z
            ],
        'bf':
            [
            (0x0041, 0x005a, 'bf', 0x1d56c), # A-Z
            (0x0061, 0x007a, 'bf', 0x1d586), # a-z
            ],
        },
    'scr':
        [
        (0x0041, 0x0041, 'it', 0x1d49c), # A
        (0x0042, 0x0042, 'it', 0x212c),  # B
        (0x0043, 0x0044, 'it', 0x1d49e), # C-D
        (0x0045, 0x0046, 'it', 0x2130),  # E-F
        (0x0047, 0x0047, 'it', 0x1d4a2), # G
        (0x0048, 0x0048, 'it', 0x210b),  # H
        (0x0049, 0x0049, 'it', 0x2110),  # I
        (0x004a, 0x004b, 'it', 0x1d4a5), # J-K
        (0x004c, 0x004c, 'it', 0x2112),  # L
        (0x004d, 0x003d, 'it', 0x2133),  # M
        (0x004e, 0x0051, 'it', 0x1d4a9), # N-Q
        (0x0052, 0x0052, 'it', 0x211b),  # R
        (0x0053, 0x005a, 'it', 0x1d4ae), # S-Z
        (0x0061, 0x0064, 'it', 0x1d4b6), # a-d
        (0x0065, 0x0065, 'it', 0x212f),  # e
        (0x0066, 0x0066, 'it', 0x1d4bb), # f
        (0x0067, 0x0067, 'it', 0x210a),  # g
        (0x0068, 0x006e, 'it', 0x1d4bd), # h-n
        (0x006f, 0x006f, 'it', 0x2134),  # o
        (0x0070, 0x007a, 'it', 0x1d4c5), # p-z
        ],
    'sf':
        {
        'rm':
            [
            (0x0030, 0x0039, 'rm', 0x1d7e2), # 0-9
            (0x0041, 0x005a, 'rm', 0x1d5a0), # A-Z
            (0x0061, 0x007a, 'rm', 0x1d5ba), # a-z
            (0x0391, 0x03a9, 'rm', 0xe17d),  # \Alpha-\Omega
            (0x03b1, 0x03c9, 'rm', 0xe196),  # \alpha-\omega
            (0x03d1, 0x03d1, 'rm', 0xe1b0),  # theta variant
            (0x03d5, 0x03d5, 'rm', 0xe1b1),  # phi variant
            (0x03d6, 0x03d6, 'rm', 0xe1b3),  # pi variant
            (0x03f1, 0x03f1, 'rm', 0xe1b2),  # rho variant
            (0x03f5, 0x03f5, 'rm', 0xe1af),  # lunate epsilon
            (0x2202, 0x2202, 'rm', 0xe17c),  # partial differential
            ],
        'it':
            [
            # These numerals are actually upright.  We don't actually
            # want italic numerals ever.
            (0x0030, 0x0039, 'rm', 0x1d7e2), # 0-9
            (0x0041, 0x005a, 'it', 0x1d608), # A-Z
            (0x0061, 0x007a, 'it', 0x1d622), # a-z
            (0x0391, 0x03a9, 'rm', 0xe17d),  # \Alpha-\Omega
            (0x03b1, 0x03c9, 'it', 0xe1d8),  # \alpha-\omega
            (0x03d1, 0x03d1, 'it', 0xe1f2),  # theta variant
            (0x03d5, 0x03d5, 'it', 0xe1f3),  # phi variant
            (0x03d6, 0x03d6, 'it', 0xe1f5),  # pi variant
            (0x03f1, 0x03f1, 'it', 0xe1f4),  # rho variant
            (0x03f5, 0x03f5, 'it', 0xe1f1),  # lunate epsilon
            ],
        'bf':
            [
            (0x0030, 0x0039, 'bf', 0x1d7ec), # 0-9
            (0x0041, 0x005a, 'bf', 0x1d5d4), # A-Z
            (0x0061, 0x007a, 'bf', 0x1d5ee), # a-z
            (0x0391, 0x03a9, 'bf', 0x1d756), # \Alpha-\Omega
            (0x03b1, 0x03c9, 'bf', 0x1d770), # \alpha-\omega
            (0x03d1, 0x03d1, 'bf', 0x1d78b), # theta variant
            (0x03d5, 0x03d5, 'bf', 0x1d78d), # phi variant
            (0x03d6, 0x03d6, 'bf', 0x1d78f), # pi variant
            (0x03f0, 0x03f0, 'bf', 0x1d78c), # kappa variant
            (0x03f1, 0x03f1, 'bf', 0x1d78e), # rho variant
            (0x03f5, 0x03f5, 'bf', 0x1d78a), # lunate epsilon
            (0x2202, 0x2202, 'bf', 0x1d789), # partial differential
            (0x2207, 0x2207, 'bf', 0x1d76f), # \Nabla
            ],
        },
    'tt':
        [
        (0x0030, 0x0039, 'rm', 0x1d7f6), # 0-9
        (0x0041, 0x005a, 'rm', 0x1d670), # A-Z
        (0x0061, 0x007a, 'rm', 0x1d68a)  # a-z
        ],
    }
