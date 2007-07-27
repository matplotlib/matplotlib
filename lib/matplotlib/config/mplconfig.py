"""Traits-based declaration for Matplotlib configuration.
"""

# stdlib imports
import os

# external imports
import enthought.traits.api as T

# internal imports
import mpltraits as mplT
import cutils
from tconfig import TConfig, TConfigManager
import pytz

# Code begins

##############################################################################
# Main Config class follows
##############################################################################
class MPLConfig(TConfig):
    """
    This is a sample matplotlib configuration file.  It should be placed
    in HOME/.matplotlib/matplotlibrc (unix/linux like systems) and
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
     - a matplotlib color string, such as r, k, or b
     - an rgb tuple, such as (1.0, 0.5, 0.0)
     - a hex string, such as #ff00ff or ff00ff
     - a scalar grayscale intensity such as 0.75
     - a legal html color name, eg red, blue, darkslategray

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
        """Valid backends are: 'GTKAgg', 'GTKCairo', 'QtAgg', 'Qt4Agg',
        'TkAgg', 'Agg', 'Cairo', 'PS', 'PDF', 'SVG'"""
        use = T.Trait('TkAgg', mplT.BackendHandler())
        
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
                use = T.Trait(None, None, 'ghostscript', 'xpdf')
                resolution = T.Float(6000)
        
        class pdf(TConfig):
            compression = T.Range(0, 9, 6)
            fonttype = T.Trait(3, 42)
            inheritcolor = T.false
            use14corefonts = T.false
        
        class svg(TConfig):
            image_inline = T.true
            image_noscale = T.false
            embed_chars = T.false
    
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

        class math(TConfig):
                mathtext2 = T.false
                rm = T.Trait('cmr10.ttf')
                it = T.Trait('cmmi10.ttf')
                tt = T.Trait('cmtt10.ttf')
                mit = T.Trait('cmmi10.ttf')
                cal = T.Trait('cmsy10.ttf')
                nonascii = T.Trait('cmex10.ttf')

    class axes(TConfig):
        hold = T.Trait(True, mplT.BoolHandler())
        facecolor = T.Trait('white', mplT.ColorHandler())
        edgecolor = T.Trait('black', mplT.ColorHandler())
        linewidth = T.Float(1.0)
        grid = T.Trait(True, mplT.BoolHandler())
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
        linestyle = T.Trait('-','--','-.', ':', 'steps', '', ' ')
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


config_file = cutils.get_config_file(tconfig=True)
old_config_file = cutils.get_config_file(tconfig=False)
print 
if os.path.exists(old_config_file) and not os.path.exists(config_file):
    CONVERT = True
else: CONVERT = False
configManager = TConfigManager(MPLConfig,
                               cutils.get_config_file(tconfig=True),
                               filePriority=True)
mplConfig = configManager.tconf


def save_config():
    """Save mplConfig customizations to current matplotlib.conf
    """
    configManager.write()


class RcParamsWrapper(dict):
    
    """A backwards-compatible interface to a traited config object
    """
    
    tconf = {
    'backend' : (mplConfig.backend, 'use'),
    'numerix' : (mplConfig, 'numerix'),
    'maskedarray' : (mplConfig, 'maskedarray'),
    'toolbar' : (mplConfig, 'toolbar'),
    'datapath' : (mplConfig, 'datapath'),
    'units' : (mplConfig, 'units'),
    'interactive' : (mplConfig, 'interactive'),
    'timezone' : (mplConfig, 'timezone'),

    # the verbosity setting
    'verbose.level' : (mplConfig.verbose, 'level'),
    'verbose.fileo' : (mplConfig.verbose, 'fileo'),

    # line props
    'lines.linewidth' : (mplConfig.lines, 'linewidth'),
    'lines.linestyle' : (mplConfig.lines, 'linestyle'),
    'lines.color' : (mplConfig.lines, 'color'),
    'lines.marker' : (mplConfig.lines, 'marker'),
    'lines.markeredgewidth' : (mplConfig.lines, 'markeredgewidth'),
    'lines.markersize' : (mplConfig.lines, 'markersize'),
    'lines.antialiased' : (mplConfig.lines, 'antialiased'),
    'lines.dash_joinstyle' : (mplConfig.lines, 'dash_joinstyle'),
    'lines.solid_joinstyle' : (mplConfig.lines, 'solid_joinstyle'),
    'lines.dash_capstyle' : (mplConfig.lines, 'dash_capstyle'),
    'lines.solid_capstyle' : (mplConfig.lines, 'solid_capstyle'),

    # patch props
    'patch.linewidth' : (mplConfig.patch, 'linewidth'),
    'patch.edgecolor' : (mplConfig.patch, 'edgecolor'),
    'patch.facecolor' : (mplConfig.patch, 'facecolor'),
    'patch.antialiased' : (mplConfig.patch, 'antialiased'),


    # font props
    'font.family' : (mplConfig.font, 'family'),
    'font.style' : (mplConfig.font, 'style'),
    'font.variant' : (mplConfig.font, 'variant'),
    'font.stretch' : (mplConfig.lines, 'color'),
    'font.weight' : (mplConfig.font, 'weight'),
    'font.size' : (mplConfig.font, 'size'),
    'font.serif' : (mplConfig.font, 'serif'),
    'font.sans-serif' : (mplConfig.font, 'sans_serif'),
    'font.cursive' : (mplConfig.font, 'cursive'),
    'font.fantasy' : (mplConfig.font, 'fantasy'),
    'font.monospace' : (mplConfig.font, 'monospace'),

    # text props
    'text.color' : (mplConfig.text, 'color'),
    'text.usetex' : (mplConfig.text, 'usetex'),
    'text.latex.unicode' : (mplConfig.text.latex, 'unicode'),
    'text.latex.preamble' : (mplConfig.text.latex, 'preamble'),
    'text.dvipnghack' : (mplConfig.text.latex, 'dvipnghack'),

    'image.aspect' : (mplConfig.image, 'aspect'),
    'image.interpolation' : (mplConfig.image, 'interpolation'),
    'image.cmap' : (mplConfig.image, 'cmap'),
    'image.lut' : (mplConfig.image, 'lut'),
    'image.origin' : (mplConfig.image, 'origin'),

    'contour.negative_linestyle' : (mplConfig.contour, 'negative_linestyle'),

    # axes props
    'axes.axisbelow' : (mplConfig.axes, 'axisbelow'),
    'axes.hold' : (mplConfig.axes, 'hold'),
    'axes.facecolor' : (mplConfig.axes, 'facecolor'),
    'axes.edgecolor' : (mplConfig.axes, 'edgecolor'),
    'axes.linewidth' : (mplConfig.axes, 'linewidth'),
    'axes.titlesize' : (mplConfig.axes, 'titlesize'),
    'axes.grid' : (mplConfig.axes, 'grid'),
    'axes.labelsize' : (mplConfig.axes, 'labelsize'),
    'axes.labelcolor' : (mplConfig.axes, 'labelcolor'),
    'axes.formatter.limits' : (mplConfig.axes.formatter, 'limits'),

    'polaraxes.grid' : (mplConfig.axes, 'polargrid'),

    #legend properties
    'legend.loc' : (mplConfig.legend, 'loc'),
    'legend.isaxes' : (mplConfig.legend, 'isaxes'),
    'legend.numpoints' : (mplConfig.legend, 'numpoints'),
    'legend.fontsize' : (mplConfig.legend, 'fontsize'),
    'legend.pad' : (mplConfig.legend, 'pad'),
    'legend.markerscale' : (mplConfig.legend, 'markerscale'),
    'legend.labelsep' : (mplConfig.legend, 'labelsep'),
    'legend.handlelen' : (mplConfig.legend, 'handlelen'),
    'legend.handletextsep' : (mplConfig.legend, 'handletextsep'),
    'legend.axespad' : (mplConfig.legend, 'axespad'),
    'legend.shadow' : (mplConfig.legend, 'shadow'),

    # tick properties
    'xtick.major.size' : (mplConfig.xticks.major, 'size'),
    'xtick.minor.size' : (mplConfig.xticks.minor, 'size'),
    'xtick.major.pad' : (mplConfig.xticks.major, 'pad'),
    'xtick.minor.pad' : (mplConfig.xticks.minor, 'pad'),
    'xtick.color' : (mplConfig.xticks, 'color'),
    'xtick.labelsize' : (mplConfig.xticks, 'labelsize'),
    'xtick.direction' : (mplConfig.xticks, 'direction'),

    'ytick.major.size' : (mplConfig.yticks.major, 'size'),
    'ytick.minor.size' : (mplConfig.yticks.minor, 'size'),
    'ytick.major.pad' : (mplConfig.yticks.major, 'pad'),
    'ytick.minor.pad' : (mplConfig.yticks.minor, 'pad'),
    'ytick.color' : (mplConfig.yticks, 'color'),
    'ytick.labelsize' : (mplConfig.yticks, 'labelsize'),
    'ytick.direction' : (mplConfig.yticks, 'direction'),

    'grid.color' : (mplConfig.grid, 'color'),
    'grid.linestyle' : (mplConfig.grid, 'linestyle'),
    'grid.linewidth' : (mplConfig.grid, 'linewidth'),


    # figure props
    'figure.figsize' : (mplConfig.figure, 'figsize'),
    'figure.dpi' : (mplConfig.figure, 'dpi'),
    'figure.facecolor' : (mplConfig.figure, 'facecolor'),
    'figure.edgecolor' : (mplConfig.figure, 'edgecolor'),

    'figure.subplot.left' : (mplConfig.figure.subplot, 'left'),
    'figure.subplot.right' : (mplConfig.figure.subplot, 'right'),
    'figure.subplot.bottom' : (mplConfig.figure.subplot, 'bottom'),
    'figure.subplot.top' : (mplConfig.figure.subplot, 'top'),
    'figure.subplot.wspace' : (mplConfig.figure.subplot, 'wspace'),
    'figure.subplot.hspace' : (mplConfig.figure.subplot, 'hspace'),


    'savefig.dpi' : (mplConfig.savefig, 'dpi'),
    'savefig.facecolor' : (mplConfig.savefig, 'facecolor'),
    'savefig.edgecolor' : (mplConfig.savefig, 'edgecolor'),
    'savefig.orientation' : (mplConfig.savefig, 'orientation'),

    'cairo.format' : (mplConfig.backend.cairo, 'format'),
    'tk.window_focus' : (mplConfig.backend.tk, 'window_focus'),
    'tk.pythoninspect' : (mplConfig.backend.tk, 'pythoninspect'),
    'ps.papersize' : (mplConfig.backend.ps, 'papersize'),
    'ps.useafm' : (mplConfig.backend.ps, 'useafm'),
    'ps.usedistiller' : (mplConfig.backend.ps.distiller, 'use'),
    'ps.distiller.res' : (mplConfig.backend.ps.distiller, 'resolution'),
    'ps.fonttype' : (mplConfig.backend.ps, 'fonttype'),
    'pdf.compression' : (mplConfig.backend.pdf, 'compression'),
    'pdf.inheritcolor' : (mplConfig.backend.pdf, 'inheritcolor'),
    'pdf.use14corefonts' : (mplConfig.backend.pdf, 'use14corefonts'),
    'pdf.fonttype' : (mplConfig.backend.pdf, 'fonttype'),
    'svg.image_inline' : (mplConfig.backend.svg, 'image_inline'),
    'svg.image_noscale' : (mplConfig.backend.svg, 'image_noscale'),
    'svg.embed_char_paths' : (mplConfig.backend.svg, 'embed_chars'),

    # mathtext settings
    'mathtext.mathtext2' : (mplConfig.text.math, 'mathtext2'),
    'mathtext.rm' : (mplConfig.text.math, 'rm'),
    'mathtext.it' : (mplConfig.text.math, 'it'),
    'mathtext.tt' : (mplConfig.text.math, 'tt'),
    'mathtext.mit' : (mplConfig.text.math, 'mit'),
    'mathtext.cal' : (mplConfig.text.math, 'cal'),
    'mathtext.nonascii' : (mplConfig.text.math, 'nonascii'),
    }

    def __setitem__(self, key, val):
        try:
            obj, attr = self.tconf[key]
            setattr(obj, attr, val)
        except KeyError:
            raise KeyError('%s is not a valid rc parameter.\
See rcParams.keys() for a list of valid parameters.'%key)

    def __getitem__(self, key):
        obj, attr = self.tconf[key]
        return getattr(obj, attr)

    def keys():
        return self.tconf.keys()


rcParams = RcParamsWrapper()

# convert old matplotlibrc to new matplotlib.conf
if CONVERT:
    from rcparams import rcParams as old_rcParams
    for key, val in old_rcParams.iteritems():
        rcParams[key] = val
    save_config()
    print '%s converted to %s'%(cutils.get_config_file(tconfig=False),
                                config_file)

##############################################################################
# Simple testing
##############################################################################
if __name__ == "__main__":
    mplConfig = MPLConfig()
    mplConfig.backend.pdf.compression = 1.1
    print mplConfig
