"""Traits-based declaration for Matplotlib configuration.
"""

# stdlib imports
import os

# external imports
import enthought.traits.api as T

# internal imports
import mpltraits as mplT
import cutils
import checkdep
from tconfig import TConfig, TConfigManager, tconf2File
import pytz

# Code begins
DEBUG = False

##############################################################################
# Main Config class follows
##############################################################################
class MPLConfig(TConfig):
    """
    This is a sample matplotlib configuration file.  It should be placed
    in HOME/.matplotlib (unix/linux like systems) and
    C:\Documents and Settings\yourname\.matplotlib (win32 systems)

    By default, the installer will overwrite the existing file in the install
    path, so if you want to preserve yours, please move it to your HOME dir and
    set the environment variable if necessary.

    This file is best viewed in a editor which supports ini or conf mode syntax
    highlighting.

    Blank lines, or lines starting with a comment symbol, are ignored,
    as are trailing comments.  Other lines must have the format

      key = val   optional comment

    val should be valid python syntax, just as you would use when setting
    properties using rcParams. This should become more obvious by inspecting
    the default values listed herein.

    Colors: for the color values below, you can either use
     - a matplotlib color string, such as r | k | b
     - an rgb tuple, such as (1.0, 0.5, 0.0)
     - a hex string, such as #ff00ff or ff00ff
     - a scalar grayscale intensity such as 0.75
     - a legal html color name, eg red | blue | darkslategray

    Interactivity: see http://matplotlib.sourceforge.net/interactive.html.

    ### CONFIGURATION BEGINS HERE ###
    """

    interactive = T.Trait(False, mplT.BoolHandler())
    toolbar = T.Trait('toolbar2', 'toolbar2', None)
    timezone = T.Trait('UTC', pytz.all_timezones)
    datapath = T.Trait(cutils.get_data_path())
    numerix = T.Trait('numpy', 'numpy', 'numeric', 'numarray')
    maskedarray = T.false
    units = T.false

    class backend(TConfig):
        use = T.Trait('Agg', mplT.BackendHandler())

        class cairo(TConfig):
            format = T.Trait('png', 'png', 'ps', 'pdf', 'svg')

        class tk(TConfig):
            """
            window_focus : Maintain shell focus for TkAgg
            pythoninspect: tk sets PYTHONINSPECT
            """

            window_focus = T.false
            pythoninspect = T.false

        class ps(TConfig):
            papersize = T.Trait('letter', 'auto', 'letter', 'legal', 'ledger',
                                'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
                                'A8', 'A9', 'A10',
                                'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                                'B8', 'B9', 'B10')
            useafm = T.false
            fonttype = T.Trait(3, 42)

            class distiller(TConfig):
                use = T.Trait(None, None, 'ghostscript', 'xpdf', False)
                resolution = T.Float(6000)

        class pdf(TConfig):
            compression = T.Range(0, 9, 6)
            fonttype = T.Trait(3, 42)
            inheritcolor = T.false
            use14corefonts = T.false

        class svg(TConfig):
            image_inline = T.true
            image_noscale = T.false
            embed_chars = T.true

    class lines(TConfig):
        linewidth = T.Float(1.0)
        linestyle = T.Trait('-','--','-.', ':', 'steps', '', ' ', None)
        color = T.Trait('blue',mplT.ColorHandler())
        solid_joinstyle = T.Trait('miter', 'miter', 'round', 'bevel')
        solid_capstyle = T.Trait('butt', 'butt', 'round', 'projecting')
        dash_joinstyle = T.Trait('miter', 'miter', 'round', 'bevel')
        dash_capstyle = T.Trait('butt', 'butt', 'round', 'projecting')
        marker = T.Trait('None', 'None', 'o', '.', ',', '^', 'v', '<', '>', 's',
                         '+', 'x', 'D','d', '1', '2', '3', '4', 'h', 'H', 'p',
                         '|', '_')
        markeredgewidth = T.Float(0.5)
        markersize = T.Float(6)
        antialiased = T.true

    class patch(TConfig):
        linewidth = T.Float(1.0)
        facecolor = T.Trait('blue', mplT.ColorHandler())
        edgecolor = T.Trait('black', mplT.ColorHandler())
        antialiased = T.true

    class font(TConfig):
        family = T.Trait('sans-serif', 'sans-serif', 'serif', 'cursive',
                         'fantasy', 'monospace')
        style = T.Trait('normal', 'normal', 'italic', 'oblique')
        variant = T.Trait('normal', 'normal', 'small-caps')
        weight = T.Trait('normal', 'normal', 'bold', 'bolder', 'lighter',
                          100, 200, 300, 400, 500, 600, 700, 800, 900)
        stretch = T.Trait('normal', 'ultra-condensed', 'extra-condensed',
                          'condensed', 'semi-condensed', 'normal', 'semi-expanded',
                          'expanded', 'extra-expanded', 'ultra-expanded',
                          'wider', 'narrower')
        size = T.Float(12.0)
        serif = T.ListStr(["Bitstream Vera Serif", "New Century Schoolbook",
                 "Century Schoolbook L", "Utopia", "ITC Bookman", "Bookman",
                 "Nimbus Roman No9 L", "Times New Roman", "Times", "Palatino",
                 "Charter", "serif"])
        sans_serif = T.ListStr(["Bitstream Vera Sans", "Lucida Grande", "Verdana",
                      "Geneva", "Lucid", "Arial", "Helvetica", "Avant Garde",
                      "sans-serif"])
        cursive = T.ListStr(["Apple Chancery", "Textile", "Zapf Chancery", "Sand",
                   "cursive"])
        fantasy = T.ListStr(["Comic Sans MS", "Chicago", "Charcoal", "Impact", "Western",
                   "fantasy"])
        monospace = T.ListStr(["Bitstream Vera Sans Mono", "Andale Mono", "Nimbus Mono L",
                     "Courier New", "Courier", "Fixed", "Terminal", "monospace"])

    class text(TConfig):
        color = T.Trait('black',mplT.ColorHandler())
        usetex = T.false

        class latex(TConfig):
            unicode = T.false
            preamble = T.ListStr([])
            dvipnghack = T.false

    class mathtext(TConfig):
        cal = T.Trait('cursive'       , mplT.FontconfigPatternHandler())
        rm  = T.Trait('serif'         , mplT.FontconfigPatternHandler())
        tt  = T.Trait('monospace'     , mplT.FontconfigPatternHandler())
        it  = T.Trait('serif:oblique' , mplT.FontconfigPatternHandler())
        bf  = T.Trait('serif:bold'    , mplT.FontconfigPatternHandler())
        sf  = T.Trait('sans'          , mplT.FontconfigPatternHandler())
        fontset = T.Trait('cm', 'cm', 'stix', 'stixsans', 'custom')
        fallback_to_cm = T.true

    class axes(TConfig):
        hold = T.Trait(True, mplT.BoolHandler())
        facecolor = T.Trait('white', mplT.ColorHandler())
        edgecolor = T.Trait('black', mplT.ColorHandler())
        linewidth = T.Float(1.0)
        grid = T.Trait(False, mplT.BoolHandler())
        polargrid = T.Trait(True, mplT.BoolHandler())
        titlesize = T.Trait('large', 'xx-small', 'x-small', 'small', 'medium',
                            'large', 'x-large', 'xx-large', T.Float)
        labelsize = T.Trait('medium', 'xx-small', 'x-small', 'small', 'medium',
                            'large', 'x-large', 'xx-large', T.Float)
        labelcolor = T.Trait('black', mplT.ColorHandler())
        axisbelow = T.false

        class formatter(TConfig):
            limits = T.List(T.Float, [-7, 7], minlen=2, maxlen=2)

    class xticks(TConfig):
        color = T.Trait('black', mplT.ColorHandler())
        labelsize = T.Trait('small', 'xx-small', 'x-small', 'small', 'medium',
                            'large', 'x-large', 'xx-large', T.Float)
        direction = T.Trait('in', 'out')

        class major(TConfig):
            size = T.Float(4)
            pad = T.Float(4)

        class minor(TConfig):
            size = T.Float(2)
            pad = T.Float(4)

    class yticks(TConfig):
        color = T.Trait('black', mplT.ColorHandler())
        labelsize = T.Trait('small', 'xx-small', 'x-small', 'small', 'medium',
                            'large', 'x-large', 'xx-large', T.Float)
        direction = T.Trait('in', 'out')

        class major(TConfig):
            size = T.Float(4)
            pad = T.Float(4)

        class minor(TConfig):
            size = T.Float(2)
            pad = T.Float(4)

    class grid(TConfig):
        color = T.Trait('black', mplT.ColorHandler())
        linestyle = T.Trait(':','-','--','-.', ':', 'steps', '', ' ')
        linewidth = T.Float(0.5)

    class legend(TConfig):
        loc = T.Trait('upper right', 'best', 'upper right', 'upper left',
                      'lower left', 'lower right', 'right', 'center left',
                      'center right', 'lower center', 'upper center', 'center')
        isaxes = T.true
        numpoints = T.Int(3)
        fontsize = T.Trait('medium', 'xx-small', 'x-small', 'small', 'medium',
                           'large', 'x-large', 'xx-large', T.Float)
        pad = T.Float(0.2)
        markerscale = T.Float(1.0)
        labelsep = T.Float(0.01)
        handlelen = T.Float(0.05)
        handletextsep = T.Float(0.02)
        axespad = T.Float(0.02)
        shadow = T.false

    class figure(TConfig):
        figsize = T.List(T.Float, [8,6], maxlen=2, minlen=2)
        dpi = T.Float(80)
        facecolor = T.Trait('0.75', mplT.ColorHandler())
        edgecolor = T.Trait('white', mplT.ColorHandler())

        class subplot(TConfig):
            """The figure subplot parameters.  All dimensions are fraction
            of the figure width or height"""
            left = T.Float(0.125)
            right = T.Float(0.9)
            bottom = T.Float(0.1)
            top = T.Float(0.9)
            wspace = T.Float(0.2)
            hspace = T.Float(0.2)

    class image(TConfig):
        aspect = T.Trait('equal', 'equal', 'auto', T.Float)
        interpolation = T.Trait('bilinear', 'bilinear', 'nearest', 'bicubic',
                                'spline16', 'spline36', 'hanning', 'hamming',
                                'hermite', 'kaiser', 'quadric', 'catrom',
                                'gaussian', 'bessel', 'mitchell', 'sinc',
                                'lanczos', 'blackman')
        cmap = T.Trait('jet', *mplT.colormaps)
        lut = T.Int(256)
        origin = T.Trait('upper', 'upper', 'lower')

    class contour(TConfig):
        negative_linestyle = T.Trait('dashed', 'dashed', 'solid')

    class savefig(TConfig):
        dpi = T.Float(100)
        facecolor = T.Trait('white', mplT.ColorHandler())
        edgecolor = T.Trait('white', mplT.ColorHandler())
        orientation = T.Trait('portrait', 'portrait', 'landscape')

    class verbose(TConfig):
        level = T.Trait('silent', 'silent', 'helpful', 'debug', 'debug-annoying')
        fileo = T.Trait('sys.stdout', 'sys.stdout', T.File)


class RcParamsWrapper(dict):

    """A backwards-compatible interface to a traited config object
    """

    def __init__(self, tconfig):
        self.tconfig = tconfig

        self.tconfig_map = {
        'backend' : (self.tconfig.backend, 'use'),
        'numerix' : (self.tconfig, 'numerix'),
        'maskedarray' : (self.tconfig, 'maskedarray'),
        'toolbar' : (self.tconfig, 'toolbar'),
        'datapath' : (self.tconfig, 'datapath'),
        'units' : (self.tconfig, 'units'),
        'interactive' : (self.tconfig, 'interactive'),
        'timezone' : (self.tconfig, 'timezone'),

        # the verbosity setting
        'verbose.level' : (self.tconfig.verbose, 'level'),
        'verbose.fileo' : (self.tconfig.verbose, 'fileo'),

        # line props
        'lines.linewidth' : (self.tconfig.lines, 'linewidth'),
        'lines.linestyle' : (self.tconfig.lines, 'linestyle'),
        'lines.color' : (self.tconfig.lines, 'color'),
        'lines.marker' : (self.tconfig.lines, 'marker'),
        'lines.markeredgewidth' : (self.tconfig.lines, 'markeredgewidth'),
        'lines.markersize' : (self.tconfig.lines, 'markersize'),
        'lines.antialiased' : (self.tconfig.lines, 'antialiased'),
        'lines.dash_joinstyle' : (self.tconfig.lines, 'dash_joinstyle'),
        'lines.solid_joinstyle' : (self.tconfig.lines, 'solid_joinstyle'),
        'lines.dash_capstyle' : (self.tconfig.lines, 'dash_capstyle'),
        'lines.solid_capstyle' : (self.tconfig.lines, 'solid_capstyle'),

        # patch props
        'patch.linewidth' : (self.tconfig.patch, 'linewidth'),
        'patch.edgecolor' : (self.tconfig.patch, 'edgecolor'),
        'patch.facecolor' : (self.tconfig.patch, 'facecolor'),
        'patch.antialiased' : (self.tconfig.patch, 'antialiased'),


        # font props
        'font.family' : (self.tconfig.font, 'family'),
        'font.style' : (self.tconfig.font, 'style'),
        'font.variant' : (self.tconfig.font, 'variant'),
        'font.stretch' : (self.tconfig.lines, 'color'),
        'font.weight' : (self.tconfig.font, 'weight'),
        'font.size' : (self.tconfig.font, 'size'),
        'font.serif' : (self.tconfig.font, 'serif'),
        'font.sans-serif' : (self.tconfig.font, 'sans_serif'),
        'font.cursive' : (self.tconfig.font, 'cursive'),
        'font.fantasy' : (self.tconfig.font, 'fantasy'),
        'font.monospace' : (self.tconfig.font, 'monospace'),

        # text props
        'text.color' : (self.tconfig.text, 'color'),
        'text.usetex' : (self.tconfig.text, 'usetex'),
        'text.latex.unicode' : (self.tconfig.text.latex, 'unicode'),
        'text.latex.preamble' : (self.tconfig.text.latex, 'preamble'),
        'text.dvipnghack' : (self.tconfig.text.latex, 'dvipnghack'),

        'mathtext.cal'        : (self.tconfig.mathtext, 'cal'),
        'mathtext.rm'         : (self.tconfig.mathtext, 'rm'),
        'mathtext.tt'         : (self.tconfig.mathtext, 'tt'),
        'mathtext.it'         : (self.tconfig.mathtext, 'it'),
        'mathtext.bf'         : (self.tconfig.mathtext, 'bf'),
        'mathtext.sf'         : (self.tconfig.mathtext, 'sf'),
        'mathtext.fontset'    : (self.tconfig.mathtext, 'fontset'),
        'mathtext.fallback_to_cm' : (self.tconfig.mathtext, 'fallback_to_cm'),

        'image.aspect' : (self.tconfig.image, 'aspect'),
        'image.interpolation' : (self.tconfig.image, 'interpolation'),
        'image.cmap' : (self.tconfig.image, 'cmap'),
        'image.lut' : (self.tconfig.image, 'lut'),
        'image.origin' : (self.tconfig.image, 'origin'),

        'contour.negative_linestyle' : (self.tconfig.contour, 'negative_linestyle'),

        # axes props
        'axes.axisbelow' : (self.tconfig.axes, 'axisbelow'),
        'axes.hold' : (self.tconfig.axes, 'hold'),
        'axes.facecolor' : (self.tconfig.axes, 'facecolor'),
        'axes.edgecolor' : (self.tconfig.axes, 'edgecolor'),
        'axes.linewidth' : (self.tconfig.axes, 'linewidth'),
        'axes.titlesize' : (self.tconfig.axes, 'titlesize'),
        'axes.grid' : (self.tconfig.axes, 'grid'),
        'axes.labelsize' : (self.tconfig.axes, 'labelsize'),
        'axes.labelcolor' : (self.tconfig.axes, 'labelcolor'),
        'axes.formatter.limits' : (self.tconfig.axes.formatter, 'limits'),

        'polaraxes.grid' : (self.tconfig.axes, 'polargrid'),

        #legend properties
        'legend.loc' : (self.tconfig.legend, 'loc'),
        'legend.isaxes' : (self.tconfig.legend, 'isaxes'),
        'legend.numpoints' : (self.tconfig.legend, 'numpoints'),
        'legend.fontsize' : (self.tconfig.legend, 'fontsize'),
        'legend.pad' : (self.tconfig.legend, 'pad'),
        'legend.markerscale' : (self.tconfig.legend, 'markerscale'),
        'legend.labelsep' : (self.tconfig.legend, 'labelsep'),
        'legend.handlelen' : (self.tconfig.legend, 'handlelen'),
        'legend.handletextsep' : (self.tconfig.legend, 'handletextsep'),
        'legend.axespad' : (self.tconfig.legend, 'axespad'),
        'legend.shadow' : (self.tconfig.legend, 'shadow'),

        # tick properties
        'xtick.major.size' : (self.tconfig.xticks.major, 'size'),
        'xtick.minor.size' : (self.tconfig.xticks.minor, 'size'),
        'xtick.major.pad' : (self.tconfig.xticks.major, 'pad'),
        'xtick.minor.pad' : (self.tconfig.xticks.minor, 'pad'),
        'xtick.color' : (self.tconfig.xticks, 'color'),
        'xtick.labelsize' : (self.tconfig.xticks, 'labelsize'),
        'xtick.direction' : (self.tconfig.xticks, 'direction'),

        'ytick.major.size' : (self.tconfig.yticks.major, 'size'),
        'ytick.minor.size' : (self.tconfig.yticks.minor, 'size'),
        'ytick.major.pad' : (self.tconfig.yticks.major, 'pad'),
        'ytick.minor.pad' : (self.tconfig.yticks.minor, 'pad'),
        'ytick.color' : (self.tconfig.yticks, 'color'),
        'ytick.labelsize' : (self.tconfig.yticks, 'labelsize'),
        'ytick.direction' : (self.tconfig.yticks, 'direction'),

        'grid.color' : (self.tconfig.grid, 'color'),
        'grid.linestyle' : (self.tconfig.grid, 'linestyle'),
        'grid.linewidth' : (self.tconfig.grid, 'linewidth'),


        # figure props
        'figure.figsize' : (self.tconfig.figure, 'figsize'),
        'figure.dpi' : (self.tconfig.figure, 'dpi'),
        'figure.facecolor' : (self.tconfig.figure, 'facecolor'),
        'figure.edgecolor' : (self.tconfig.figure, 'edgecolor'),

        'figure.subplot.left' : (self.tconfig.figure.subplot, 'left'),
        'figure.subplot.right' : (self.tconfig.figure.subplot, 'right'),
        'figure.subplot.bottom' : (self.tconfig.figure.subplot, 'bottom'),
        'figure.subplot.top' : (self.tconfig.figure.subplot, 'top'),
        'figure.subplot.wspace' : (self.tconfig.figure.subplot, 'wspace'),
        'figure.subplot.hspace' : (self.tconfig.figure.subplot, 'hspace'),


        'savefig.dpi' : (self.tconfig.savefig, 'dpi'),
        'savefig.facecolor' : (self.tconfig.savefig, 'facecolor'),
        'savefig.edgecolor' : (self.tconfig.savefig, 'edgecolor'),
        'savefig.orientation' : (self.tconfig.savefig, 'orientation'),

        'cairo.format' : (self.tconfig.backend.cairo, 'format'),
        'tk.window_focus' : (self.tconfig.backend.tk, 'window_focus'),
        'tk.pythoninspect' : (self.tconfig.backend.tk, 'pythoninspect'),
        'ps.papersize' : (self.tconfig.backend.ps, 'papersize'),
        'ps.useafm' : (self.tconfig.backend.ps, 'useafm'),
        'ps.usedistiller' : (self.tconfig.backend.ps.distiller, 'use'),
        'ps.distiller.res' : (self.tconfig.backend.ps.distiller, 'resolution'),
        'ps.fonttype' : (self.tconfig.backend.ps, 'fonttype'),
        'pdf.compression' : (self.tconfig.backend.pdf, 'compression'),
        'pdf.inheritcolor' : (self.tconfig.backend.pdf, 'inheritcolor'),
        'pdf.use14corefonts' : (self.tconfig.backend.pdf, 'use14corefonts'),
        'pdf.fonttype' : (self.tconfig.backend.pdf, 'fonttype'),
        'svg.image_inline' : (self.tconfig.backend.svg, 'image_inline'),
        'svg.image_noscale' : (self.tconfig.backend.svg, 'image_noscale'),
        'svg.embed_char_paths' : (self.tconfig.backend.svg, 'embed_chars'),

        }

    def __setitem__(self, key, val):
        try:
            obj, attr = self.tconfig_map[key]
            setattr(obj, attr, val)
        except KeyError:
            raise KeyError('%s is not a valid rc parameter.\
See rcParams.keys() for a list of valid parameters.'%key)

    def __getitem__(self, key):
        obj, attr = self.tconfig_map[key]
        return getattr(obj, attr)

    def keys(self):
        return self.tconfig_map.keys()

    def has_key(self, val):
        return self.tconfig_map.has_key(val)

    def update(self, arg, **kwargs):
        try:
            for key in arg:
                self[key] = arg[key]
        except AttributeError:
            for key, val in arg:
                self[key] = val

        for key in kwargs:
            self[key] = kwargs[key]


old_config_file = cutils.get_config_file(tconfig=False)
old_config_path = os.path.split(old_config_file)[0]
config_file = os.path.join(old_config_path, 'matplotlib.conf')

if os.path.exists(old_config_file) and not os.path.exists(config_file):
    CONVERT = True
else:
    config_file = cutils.get_config_file(tconfig=True)
    CONVERT = False

if DEBUG: print 'loading', config_file

configManager = TConfigManager(MPLConfig,
                               config_file,
                               filePriority=True)
mplConfig = configManager.tconf
mplConfigDefault = MPLConfig()

# TODO: move into traits validation
mplConfig.backend.ps.distiller.use = \
    checkdep.ps_distiller(mplConfig.backend.ps.distiller.use)
mplConfig.text.usetex = checkdep.usetex(mplConfig.text.usetex)

def save_config():
    """Save mplConfig customizations to current matplotlib.conf
    """
    configManager.write()

rcParams = RcParamsWrapper(mplConfig)
rcParamsDefault = RcParamsWrapper(mplConfigDefault)

# convert old matplotlibrc to new matplotlib.conf
if CONVERT:
    from rcparams import rcParams as old_rcParams
    for key, val in old_rcParams.iteritems():
        rcParams[key] = val
    save_config()
    print '%s converted to %s'%(cutils.get_config_file(tconfig=False),
                                config_file)

def rcdefaults():
    """
    Restore the default rc params - the ones that were created at
    matplotlib load time
    """
    for key in rcParamsDefault.keys():
        rcParams[key] = rcParamsDefault[key]


##############################################################################
# Auto-generate the mpl-data/matplotlib.conf
##############################################################################
if __name__ == "__main__":
    mplConfig = MPLConfig()
    tconf2File(mplConfig, '../mpl-data/matplotlib.conf.template', force=True)
    print 'matplotlib.conf.template created in ../mpl-data'
