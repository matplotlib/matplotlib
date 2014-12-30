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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import warnings
from matplotlib.fontconfig_pattern import parse_fontconfig_pattern
from matplotlib.colors import is_color_like

#interactive_bk = ['gtk', 'gtkagg', 'gtkcairo', 'qt4agg',
#                  'tkagg', 'wx', 'wxagg', 'cocoaagg', 'webagg']
# The capitalized forms are needed for ipython at present; this may
# change for later versions.

interactive_bk = ['GTK', 'GTKAgg', 'GTKCairo', 'MacOSX',
                  'Qt4Agg', 'Qt5Agg', 'TkAgg', 'WX', 'WXAgg', 'CocoaAgg',
                  'GTK3Cairo', 'GTK3Agg', 'WebAgg', 'nbAgg']


non_interactive_bk = ['agg', 'cairo', 'emf', 'gdk',
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
        self.valid = dict([(func(k), k) for k in valid])

    def __call__(self, s):
        if self.ignorecase:
            s = s.lower()
        if s in self.valid:
            return self.valid[s]
        raise ValueError('Unrecognized %s string "%s": valid strings are %s'
                         % (self.key, s, list(six.itervalues(self.valid))))


def validate_any(s):
    return s


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
    if isinstance(b, six.string_types):
        b = b.lower()
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError('Could not convert "%s" to boolean' % b)


def validate_bool_maybe_none(b):
    'Convert b to a boolean or raise'
    if isinstance(b, six.string_types):
        b = b.lower()
    if b is None or b == 'none':
        return None
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError('Could not convert "%s" to boolean' % b)


def validate_float(s):
    """convert s to float or raise"""
    try:
        return float(s)
    except ValueError:
        raise ValueError('Could not convert "%s" to float' % s)


def validate_float_or_None(s):
    """convert s to float or raise"""
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        raise ValueError('Could not convert "%s" to float' % s)


def validate_int(s):
    """convert s to int or raise"""
    try:
        return int(s)
    except ValueError:
        raise ValueError('Could not convert "%s" to int' % s)


def validate_fonttype(s):
    """
    confirm that this is a Postscript of PDF font type that we know how to
    convert to
    """
    fonttypes = {'type3':    3,
                 'truetype': 42}
    try:
        fonttype = validate_int(s)
    except ValueError:
        if s.lower() in six.iterkeys(fonttypes):
            return fonttypes[s.lower()]
        raise ValueError(
            'Supported Postscript/PDF font types are %s' %
            list(six.iterkeys(fonttypes)))
    else:
        if fonttype not in six.itervalues(fonttypes):
            raise ValueError(
                'Supported Postscript/PDF font types are %s' %
                list(six.itervalues(fonttypes)))
        return fonttype


_validate_standard_backends = ValidateInStrings('backend',
                                                all_backends,
                                                ignorecase=True)


def validate_backend(s):
    if s.startswith('module://'):
        return s
    else:
        return _validate_standard_backends(s)


validate_qt4 = ValidateInStrings('backend.qt4', ['PyQt4', 'PySide', 'PyQt4v2'])
validate_qt5 = ValidateInStrings('backend.qt5', ['PyQt5'])


def validate_toolbar(s):
    validator = ValidateInStrings(
                'toolbar',
                ['None', 'toolbar2'],
                ignorecase=True)
    return validator(s)


def validate_maskedarray(v):
    # 2008/12/12: start warning; later, remove all traces of maskedarray
    try:
        if v == 'obsolete':
            return v
    except ValueError:
        pass
    warnings.warn('rcParams key "maskedarray" is obsolete and has no effect;\n'
                  ' please delete it from your matplotlibrc file')


_seq_err_msg = ('You must supply exactly {n:d} values, you provided '
                   '{num:d} values: {s}')

_str_err_msg = ('You must supply exactly {n:d} comma-separated values, '
                'you provided '
                '{num:d} comma-separated values: {s}')


class validate_nseq_float(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, s):
        """return a seq of n floats or raise"""
        if isinstance(s, six.string_types):
            s = s.split(',')
            err_msg = _str_err_msg
        else:
            err_msg = _seq_err_msg

        if len(s) != self.n:
            raise ValueError(err_msg.format(n=self.n, num=len(s), s=s))

        try:
            return [float(val) for val in s]
        except ValueError:
            raise ValueError('Could not convert all entries to floats')


class validate_nseq_int(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, s):
        """return a seq of n ints or raise"""
        if isinstance(s, six.string_types):
            s = s.split(',')
            err_msg = _str_err_msg
        else:
            err_msg = _seq_err_msg

        if len(s) != self.n:
            raise ValueError(err_msg.format(n=self.n, num=len(s), s=s))

        try:
            return [int(val) for val in s]
        except ValueError:
            raise ValueError('Could not convert all entries to ints')


def validate_color(s):
    'return a valid color arg'
    try:
        if s.lower() == 'none':
            return 'None'
    except AttributeError:
        pass
    if is_color_like(s):
        return s
    stmp = '#' + s
    if is_color_like(stmp):
        return stmp
    # If it is still valid, it must be a tuple.
    colorarg = s
    msg = ''
    if s.find(',') >= 0:
        # get rid of grouping symbols
        stmp = ''.join([c for c in s if c.isdigit() or c == '.' or c == ','])
        vals = stmp.split(',')
        if len(vals) != 3:
            msg = '\nColor tuples must be length 3'
        else:
            try:
                colorarg = [float(val) for val in vals]
            except ValueError:
                msg = '\nCould not convert all entries to floats'

    if not msg and is_color_like(colorarg):
        return colorarg

    raise ValueError('%s does not look like a color arg%s' % (s, msg))


def validate_colorlist(s):
    'return a list of colorspecs'
    if isinstance(s, six.string_types):
        return [validate_color(c.strip()) for c in s.split(',')]
    else:
        assert type(s) in [list, tuple]
        return [validate_color(c) for c in s]


def validate_stringlist(s):
    'return a list'
    if isinstance(s, six.string_types):
        return [six.text_type(v.strip()) for v in s.split(',') if v.strip()]
    else:
        assert type(s) in [list, tuple]
        return [six.text_type(v) for v in s if v]


validate_orientation = ValidateInStrings(
    'orientation', ['landscape', 'portrait'])


def validate_aspect(s):
    if s in ('auto', 'equal'):
        return s
    try:
        return float(s)
    except ValueError:
        raise ValueError('not a valid aspect specification')


def validate_fontsize(s):
    if isinstance(s, six.string_types):
        s = s.lower()
    if s in ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large',
             'xx-large', 'smaller', 'larger']:
        return s
    try:
        return float(s)
    except ValueError:
        raise ValueError('not a valid font size')


def validate_font_properties(s):
    parse_fontconfig_pattern(s)
    return s


validate_fontset = ValidateInStrings(
    'fontset',
    ['cm', 'stix', 'stixsans', 'custom'])

validate_mathtext_default = ValidateInStrings(
    'default',
    "rm cal it tt sf bf default bb frak circled scr regular".split())

validate_verbose = ValidateInStrings(
    'verbose',
    ['silent', 'helpful', 'debug', 'debug-annoying'])


def deprecate_savefig_extension(value):
    warnings.warn("savefig.extension is deprecated.  Use savefig.format "
                  "instead. Will be removed in 1.4.x")
    return value


def update_savefig_format(value):
    # The old savefig.extension could also have a value of "auto", but
    # the new savefig.format does not.  We need to fix this here.
    value = six.text_type(value)
    if value == 'auto':
        value = 'png'
    return value


validate_ps_papersize = ValidateInStrings(
    'ps_papersize',
    ['auto', 'letter', 'legal', 'ledger',
    'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10',
    'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10',
    ], ignorecase=True)


def validate_ps_distiller(s):
    if isinstance(s, six.string_types):
        s = s.lower()
    if s in ('none', None):
        return None
    elif s in ('false', False):
        return False
    elif s in ('ghostscript', 'xpdf'):
        return s
    else:
        raise ValueError('matplotlibrc ps.usedistiller must either be none, '
                         'ghostscript or xpdf')

validate_joinstyle = ValidateInStrings('joinstyle',
                                       ['miter', 'round', 'bevel'],
                                       ignorecase=True)

validate_capstyle = ValidateInStrings('capstyle',
                                      ['butt', 'round', 'projecting'],
                                      ignorecase=True)

validate_negative_linestyle = ValidateInStrings('negative_linestyle',
                                                ['solid', 'dashed'],
                                                ignorecase=True)


def validate_negative_linestyle_legacy(s):
    try:
        res = validate_negative_linestyle(s)
        return res
    except ValueError:
        dashes = validate_nseq_float(2)(s)
        warnings.warn("Deprecated negative_linestyle specification; use "
                      "'solid' or 'dashed'")
        return (0, dashes)  # (offset, (solid, blank))


def validate_tkpythoninspect(s):
    # Introduced 2010/07/05
    warnings.warn("tk.pythoninspect is obsolete, and has no effect")
    return validate_bool(s)

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


def deprecate_svg_embed_char_paths(value):
    warnings.warn("svg.embed_char_paths is deprecated.  Use "
                  "svg.fonttype instead.")

validate_svg_fonttype = ValidateInStrings('svg.fonttype',
                                          ['none', 'path', 'svgfont'])


def validate_hinting(s):
    if s in (True, False):
        return s
    if s.lower() in ('auto', 'native', 'either', 'none'):
        return s.lower()
    raise ValueError("hinting should be 'auto', 'native', 'either' or 'none'")

validate_pgf_texsystem = ValidateInStrings('pgf.texsystem',
                                           ['xelatex', 'lualatex', 'pdflatex'])

validate_movie_writer = ValidateInStrings('animation.writer',
    ['ffmpeg', 'ffmpeg_file',
     'avconv', 'avconv_file',
     'mencoder', 'mencoder_file',
     'imagemagick', 'imagemagick_file'])

validate_movie_frame_fmt = ValidateInStrings('animation.frame_format',
    ['png', 'jpeg', 'tiff', 'raw', 'rgba'])

validate_axis_locator = ValidateInStrings('major', ['minor','both','major'])

def validate_bbox(s):
    if isinstance(s, six.string_types):
        s = s.lower()
        if s == 'tight':
            return s
        if s == 'standard':
            return None
        raise ValueError("bbox should be 'tight' or 'standard'")

def validate_sketch(s):
    if isinstance(s, six.string_types):
        s = s.lower()
    if s == 'none' or s is None:
        return None
    if isinstance(s, six.string_types):
        result = tuple([float(v.strip()) for v in s.split(',')])
    elif isinstance(s, (list, tuple)):
        result = tuple([float(v) for v in s])
    if len(result) != 3:
        raise ValueError("path.sketch must be a tuple (scale, length, randomness)")
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
        except:
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


# a map from key -> value, converter
defaultParams = {
    'backend':           ['Agg', validate_backend],  # agg is certainly
                                                      # present
    'backend_fallback':  [True, validate_bool],  # agg is certainly present
    'backend.qt4':       ['PyQt4', validate_qt4],
    'backend.qt5':       ['PyQt5', validate_qt5],
    'webagg.port':       [8988, validate_int],
    'webagg.open_in_browser': [True, validate_bool],
    'webagg.port_retries': [50, validate_int],
    'nbagg.transparent':       [True, validate_bool],
    'toolbar':           ['toolbar2', validate_toolbar],
    'datapath':          [None, validate_path_exists],  # handled by
                                                        # _get_data_path_cached
    'interactive':       [False, validate_bool],
    'timezone':          ['UTC', six.text_type],

    # the verbosity setting
    'verbose.level': ['silent', validate_verbose],
    'verbose.fileo': ['sys.stdout', six.text_type],

    # line props
    'lines.linewidth':       [1.0, validate_float],  # line width in points
    'lines.linestyle':       ['-', six.text_type],             # solid line
    'lines.color':           ['b', validate_color],  # blue
    'lines.marker':          ['None', six.text_type],     # black
    'lines.markeredgewidth': [0.5, validate_float],
    'lines.markersize':      [6, validate_float],    # markersize, in points
    'lines.antialiased':     [True, validate_bool],  # antialised (no jaggies)
    'lines.dash_joinstyle':  ['round', validate_joinstyle],
    'lines.solid_joinstyle': ['round', validate_joinstyle],
    'lines.dash_capstyle':   ['butt', validate_capstyle],
    'lines.solid_capstyle':  ['projecting', validate_capstyle],

    ## patch props
    'patch.linewidth':   [1.0, validate_float],  # line width in points
    'patch.edgecolor':   ['k', validate_color],  # black
    'patch.facecolor':   ['b', validate_color],  # blue
    'patch.antialiased': [True, validate_bool],  # antialised (no jaggies)


    ## font props
    'font.family':     [['sans-serif'], validate_stringlist],  # used by text object
    'font.style':      ['normal', six.text_type],
    'font.variant':    ['normal', six.text_type],
    'font.stretch':    ['normal', six.text_type],
    'font.weight':     ['normal', six.text_type],
    'font.size':       [12, validate_float],      # Base font size in points
    'font.serif':      [['Bitstream Vera Serif', 'DejaVu Serif',
                         'New Century Schoolbook', 'Century Schoolbook L',
                         'Utopia', 'ITC Bookman', 'Bookman',
                         'Nimbus Roman No9 L', 'Times New Roman',
                         'Times', 'Palatino', 'Charter', 'serif'],
                        validate_stringlist],
    'font.sans-serif': [['Bitstream Vera Sans', 'DejaVu Sans',
                         'Lucida Grande', 'Verdana', 'Geneva', 'Lucid',
                         'Arial', 'Helvetica', 'Avant Garde', 'sans-serif'],
                        validate_stringlist],
    'font.cursive':    [['Apple Chancery', 'Textile', 'Zapf Chancery',
                         'Sand', 'cursive'], validate_stringlist],
    'font.fantasy':    [['Comic Sans MS', 'Chicago', 'Charcoal', 'Impact'
                         'Western', 'fantasy'], validate_stringlist],
    'font.monospace':  [['Bitstream Vera Sans Mono', 'DejaVu Sans Mono',
                         'Andale Mono', 'Nimbus Mono L', 'Courier New',
                         'Courier', 'Fixed', 'Terminal', 'monospace'],
                        validate_stringlist],

    # text props
    'text.color':          ['k', validate_color],     # black
    'text.usetex':         [False, validate_bool],
    'text.latex.unicode':  [False, validate_bool],
    'text.latex.preamble': [[''], validate_stringlist],
    'text.latex.preview':  [False, validate_bool],
    'text.dvipnghack':     [None, validate_bool_maybe_none],
    'text.hinting':        [True, validate_hinting],
    'text.hinting_factor': [8, validate_int],
    'text.antialiased':    [True, validate_bool],

    'mathtext.cal':            ['cursive', validate_font_properties],
    'mathtext.rm':             ['serif', validate_font_properties],
    'mathtext.tt':             ['monospace', validate_font_properties],
    'mathtext.it':             ['serif:italic', validate_font_properties],
    'mathtext.bf':             ['serif:bold', validate_font_properties],
    'mathtext.sf':             ['sans\-serif', validate_font_properties],
    'mathtext.fontset':        ['cm', validate_fontset],
    'mathtext.default':        ['it', validate_mathtext_default],
    'mathtext.fallback_to_cm': [True, validate_bool],

    'image.aspect':        ['equal', validate_aspect],  # equal, auto, a number
    'image.interpolation': ['bilinear', six.text_type],
    'image.cmap':          ['jet', six.text_type],        # one of gray, jet, etc
    'image.lut':           [256, validate_int],  # lookup table
    'image.origin':        ['upper', six.text_type],  # lookup table
    'image.resample':      [False, validate_bool],

    'contour.negative_linestyle': ['dashed',
                                    validate_negative_linestyle_legacy],

    # axes props
    'axes.axisbelow':        [False, validate_bool],
    'axes.hold':             [True, validate_bool],
    'axes.facecolor':        ['w', validate_color],  # background color; white
    'axes.edgecolor':        ['k', validate_color],  # edge color; black
    'axes.linewidth':        [1.0, validate_float],  # edge linewidth
    'axes.titlesize':        ['large', validate_fontsize],  # fontsize of the
                                                            # axes title
    'axes.titleweight':      ['normal', six.text_type],  # font weight of axes title
    'axes.grid':             [False, validate_bool],   # display grid or not
    'axes.grid.which':       ['major', validate_axis_locator],  # set wether the gid are by
                                                                # default draw on 'major'
                                                                # 'minor' or 'both' kind of
                                                                # axis locator
    'axes.labelsize':        ['medium', validate_fontsize],  # fontsize of the
                                                             # x any y labels
    'axes.labelweight':      ['normal', six.text_type],  # fontsize of the x any y labels
    'axes.labelcolor':       ['k', validate_color],    # color of axis label
    'axes.formatter.limits': [[-7, 7], validate_nseq_int(2)],
                               # use scientific notation if log10
                               # of the axis range is smaller than the
                               # first or larger than the second
    'axes.formatter.use_locale': [False, validate_bool],
                               # Use the current locale to format ticks
    'axes.formatter.use_mathtext': [False, validate_bool],
    'axes.formatter.useoffset': [True, validate_bool],
    'axes.unicode_minus': [True, validate_bool],
    'axes.color_cycle': [['b', 'g', 'r', 'c', 'm', 'y', 'k'],
                         validate_colorlist],  # cycle of plot
                                               # line colors
    'axes.xmargin': [0, ValidateInterval(0, 1,
                                         closedmin=True,
                                         closedmax=True)],  # margin added to xaxis
    'axes.ymargin': [0, ValidateInterval(0, 1,
                                         closedmin=True,
                                         closedmax=True)],# margin added to yaxis

    'polaraxes.grid': [True, validate_bool],  # display polar grid or
                                                     # not
    'axes3d.grid': [True, validate_bool],  # display 3d grid

    #legend properties
    'legend.fancybox': [False, validate_bool],

    # at some point, legend.loc should be changed to 'best'
    'legend.loc': ['upper right', validate_legend_loc],

    # this option is internally ignored - it never served any useful purpose
    'legend.isaxes': [True, validate_bool],

    # the number of points in the legend line
    'legend.numpoints': [2, validate_int],
    # the number of points in the legend line for scatter
    'legend.scatterpoints': [3, validate_int],
    'legend.fontsize': ['large', validate_fontsize],
     # the relative size of legend markers vs. original
    'legend.markerscale': [1.0, validate_float],
    'legend.shadow': [False, validate_bool],
     # whether or not to draw a frame around legend
    'legend.frameon': [True, validate_bool],
     # alpha value of the legend frame
    'legend.framealpha': [None, validate_float_or_None],

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
    # the relative size of legend markers vs. original
    'legend.markerscale': [1.0, validate_float],
    'legend.shadow': [False, validate_bool],

    ## tick properties
    'xtick.major.size':  [4, validate_float],    # major xtick size in points
    'xtick.minor.size':  [2, validate_float],    # minor xtick size in points
    'xtick.major.width': [0.5, validate_float],  # major xtick width in points
    'xtick.minor.width': [0.5, validate_float],  # minor xtick width in points
    'xtick.major.pad':   [4, validate_float],    # distance to label in points
    'xtick.minor.pad':   [4, validate_float],    # distance to label in points
    'xtick.color':       ['k', validate_color],  # color of the xtick labels
    # fontsize of the xtick labels
    'xtick.labelsize':   ['medium', validate_fontsize],
    'xtick.direction':   ['in', six.text_type],            # direction of xticks

    'ytick.major.size':  [4, validate_float],     # major ytick size in points
    'ytick.minor.size':  [2, validate_float],     # minor ytick size in points
    'ytick.major.width': [0.5, validate_float],   # major ytick width in points
    'ytick.minor.width': [0.5, validate_float],   # minor ytick width in points
    'ytick.major.pad':   [4, validate_float],     # distance to label in points
    'ytick.minor.pad':   [4, validate_float],     # distance to label in points
    'ytick.color':       ['k', validate_color],   # color of the ytick labels
    # fontsize of the ytick labels
    'ytick.labelsize':   ['medium', validate_fontsize],
    'ytick.direction':   ['in', six.text_type],            # direction of yticks

    'grid.color':        ['k', validate_color],       # grid color
    'grid.linestyle':    [':', six.text_type],       # dotted
    'grid.linewidth':    [0.5, validate_float],     # in points
    'grid.alpha':        [1.0, validate_float],


    ## figure props
    # figure size in inches: width by height
    'figure.figsize':    [[8.0, 6.0], validate_nseq_float(2)],
    'figure.dpi':        [80, validate_float],   # DPI
    'figure.facecolor':  ['0.75', validate_color],  # facecolor; scalar gray
    'figure.edgecolor':  ['w', validate_color],  # edgecolor; white
    'figure.frameon':    [True, validate_bool],
    'figure.autolayout': [False, validate_bool],
    'figure.max_open_warning': [20, validate_int],

    'figure.subplot.left': [0.125, ValidateInterval(0, 1, closedmin=True,
                                                       closedmax=True)],
    'figure.subplot.right': [0.9, ValidateInterval(0, 1, closedmin=True,
                                                     closedmax=True)],
    'figure.subplot.bottom': [0.1, ValidateInterval(0, 1, closedmin=True,
                                                     closedmax=True)],
    'figure.subplot.top': [0.9, ValidateInterval(0, 1, closedmin=True,
                                                     closedmax=True)],
    'figure.subplot.wspace': [0.2, ValidateInterval(0, 1, closedmin=True,
                                                     closedmax=False)],
    'figure.subplot.hspace': [0.2, ValidateInterval(0, 1, closedmin=True,
                                                     closedmax=False)],

    ## Saving figure's properties
    'savefig.dpi':         [100, validate_float],   # DPI
    'savefig.facecolor':   ['w', validate_color],  # facecolor; white
    'savefig.edgecolor':   ['w', validate_color],  # edgecolor; white
    'savefig.frameon':     [True, validate_bool],
    'savefig.orientation': ['portrait', validate_orientation],  # edgecolor;
                                                                 #white
    'savefig.jpeg_quality': [95, validate_int],
    # what to add to extensionless filenames
    'savefig.extension':  ['png', deprecate_savefig_extension],
    # value checked by backend at runtime
    'savefig.format':     ['png', update_savefig_format],
    # options are 'tight', or 'standard'. 'standard' validates to None.
    'savefig.bbox':       [None, validate_bbox],
    'savefig.pad_inches': [0.1, validate_float],
    # default directory in savefig dialog box
    'savefig.directory': ['~', six.text_type],
    'savefig.transparent': [False, validate_bool],

    # Maintain shell focus for TkAgg
    'tk.window_focus':  [False, validate_bool],
    'tk.pythoninspect': [False, validate_tkpythoninspect],  # obsolete
    # Set the papersize/type
    'ps.papersize':     ['letter', validate_ps_papersize],
    'ps.useafm':        [False, validate_bool],  # Set PYTHONINSPECT
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

    'pgf.debug':     [False, validate_bool],  # output debug information
    # choose latex application for creating pdf files (xelatex/lualatex)
    'pgf.texsystem': ['xelatex', validate_pgf_texsystem],
    # use matplotlib rc settings for font configuration
    'pgf.rcfonts':   [True, validate_bool],
    # provide a custom preamble for the latex process
    'pgf.preamble':  [[''], validate_stringlist],

    # write raster image data directly into the svg file
    'svg.image_inline':     [True, validate_bool],
    # suppress scaling of raster data embedded in SVG
    'svg.image_noscale':    [False, validate_bool],
    # True to save all characters as paths in the SVG
    'svg.embed_char_paths': [True, deprecate_svg_embed_char_paths],
    'svg.fonttype':         ['path', validate_svg_fonttype],

    # set this when you want to generate hardcopy docstring
    'docstring.hardcopy': [False, validate_bool],
    # where plugin directory is locate
    'plugins.directory':  ['.matplotlib_plugins', six.text_type],

    'path.simplify': [True, validate_bool],
    'path.simplify_threshold': [1.0 / 9.0, ValidateInterval(0.0, 1.0)],
    'path.snap': [True, validate_bool],
    'path.sketch': [None, validate_sketch],
    'path.effects': [[], validate_any],
    'agg.path.chunksize': [0, validate_int],       # 0 to disable chunking;

    # key-mappings (multi-character mappings should be a list/tuple)
    'keymap.fullscreen':   [('f', 'ctrl+f'), validate_stringlist],
    'keymap.home':         [['h', 'r', 'home'], validate_stringlist],
    'keymap.back':         [['left', 'c', 'backspace'], validate_stringlist],
    'keymap.forward':      [['right', 'v'], validate_stringlist],
    'keymap.pan':          [['p'], validate_stringlist],
    'keymap.zoom':         [['o'], validate_stringlist],
    'keymap.save':         [['s', 'ctrl+s'], validate_stringlist],
    'keymap.quit':         [['ctrl+w', 'cmd+w'], validate_stringlist],
    'keymap.grid':         [['g'], validate_stringlist],
    'keymap.yscale':       [['l'], validate_stringlist],
    'keymap.xscale':       [['k', 'L'], validate_stringlist],
    'keymap.all_axes':     [['a'], validate_stringlist],

    # sample data
    'examples.directory': ['', six.text_type],

    # Animation settings
    'animation.writer':       ['ffmpeg', validate_movie_writer],
    'animation.codec':        ['mpeg4', six.text_type],
    'animation.bitrate':      [-1, validate_int],
    # Controls image format when frames are written to disk
    'animation.frame_format': ['png', validate_movie_frame_fmt],
    # Path to FFMPEG binary. If just binary name, subprocess uses $PATH.
    'animation.ffmpeg_path':  ['ffmpeg', six.text_type],

    # Additional arguments for ffmpeg movie writer (using pipes)
    'animation.ffmpeg_args':   [[], validate_stringlist],
    # Path to AVConv binary. If just binary name, subprocess uses $PATH.
    'animation.avconv_path':   ['avconv', six.text_type],
    # Additional arguments for avconv movie writer (using pipes)
    'animation.avconv_args':   [[], validate_stringlist],
    # Path to MENCODER binary. If just binary name, subprocess uses $PATH.
    'animation.mencoder_path': ['mencoder', six.text_type],
    # Additional arguments for mencoder movie writer (using pipes)
    'animation.mencoder_args': [[], validate_stringlist],
     # Path to convert binary. If just binary name, subprocess uses $PATH
    'animation.convert_path':  ['convert', six.text_type],
     # Additional arguments for mencoder movie writer (using pipes)

    'animation.convert_args':  [[], validate_stringlist]}


if __name__ == '__main__':
    rc = defaultParams
    rc['datapath'][0] = '/'
    for key in rc:
        if not rc[key][1](rc[key][0]) == rc[key][0]:
            print("%s: %s != %s" % (key, rc[key][1](rc[key][0]), rc[key][0]))
