"""
This is a matlab(TM) style functional interface the matplotlib.

The following matlab(TM) compatible commands are provided by

   >>> from pylab import *

Plotting commands

  axes     - Create a new axes
  axhline  - draw a horizontal line across axes
  axvline  - draw a vertical line across axes
  axhspan  - draw a horizontal bar across axes
  axvspan  - draw a vertical bar across axes
  axis     - Set or return the current axis limits
  bar      - make a bar chart
  barh     - a horizontal bar chart
  boxplot  - make a box and whisker plot
  cla      - clear current axes
  clf      - clear a figure window
  close    - close a figure window
  colorbar - add a colorbar to the current figure
  cohere   - make a plot of coherence
  contour  - make a contour plot
  csd      - make a plot of cross spectral density
  draw     - Force a redraw of the current figure
  errorbar - make an errorbar graph
  figlegend - make legend on the figure rather than the axes
  figimage  - make a figure image
  figtext   - add text in figure coords
  figure   - create or change active figure
  fill     - make filled polygons
  gca      - return the current axes
  gcf      - return the current figure
  gci      - get the current image, or None
  get      - get a handle graphics property
  gray     - set the current colormap to gray
  jet      - set the current colormap to jet
  hist     - make a histogram
  hold     - set the axes hold state
  legend   - make an axes legend
  loglog   - a log log plot
  imread   - load image file into array
  imshow   - plot image data
  pcolor   - make a pseudocolor plot
  plot     - make a line plot
  psd      - make a plot of power spectral density
  rc       - control the default params
  savefig  - save the current figure
  scatter  - make a scatter plot
  set      - set a handle graphics property
  semilogx - log x axis
  semilogy - log y axis
  show     - show the figures
  specgram - a spectrogram plot
  stem     - make a stem plot
  subplot  - make a subplot (numrows, numcols, axesnum)
  table    - add a table to the plot
  text     - add some text at location x,y to the current axes
  title    - add a title to the current axes
  xlim     - set/get the xlimits
  ylim     - set/get the ylimits
  xticks   - set/get the xticks
  yticks   - set/get the yticks
  xlabel   - add an xlabel to the current axes
  ylabel   - add a ylabel to the current axes

Matrix commands

  cumprod   - the cumulative product along a dimension
  cumsum    - the cumulative sum along a dimension
  detrend   - remove the mean or besdt fit line from an array
  diag      - the k-th diagonal of matrix
  diff      - the n-th differnce of an array
  eig       - the eigenvalues and eigen vectors of v
  eye       - a matrix where the k-th diagonal is ones, else zero
  find      - return the indices where a condition is nonzero
  fliplr    - flip the rows of a matrix up/down
  flipud    - flip the columns of a matrix left/right
  linspace  - a linear spaced vector of N values from min to max inclusive
  ones      - an array of ones
  rand      - an array from the uniform distribution [0,1]
  randn     - an array from the normal distribution
  rot90     - rotate matrix k*90 degress counterclockwise
  squeeze   - squeeze an array removing any dimensions of length 1
  tri       - a triangular matrix
  tril      - a lower triangular matrix
  triu      - an upper triangular matrix
  vander    - the Vandermonde matrix of vector x
  svd       - singular value decomposition
  zeros     - a matrix of zeros

Probability

  levypdf   - The levy probability density function from the char. func.
  normpdf   - The Gaussian probability density function
  rand      - random numbers from the uniform distribution
  randn     - random numbers from the normal distribution

Statistics

  corrcoef  - correlation coefficient
  cov       - covariance matrix
  max       - the maximum along dimension m
  mean      - the mean along dimension m
  median    - the median along dimension m
  min       - the minimum along dimension m
  norm      - the norm of vector x
  prod      - the product along dimension m
  ptp       - the max-min along dimension m
  std       - the standard deviation along dimension m
  sum       - the sum along dimension m

Time series analysis

  bartlett  - M-point Bartlett window
  blackman  - M-point Blackman window
  cohere    - the coherence using average periodiogram
  csd       - the cross spectral density using average periodiogram
  fft       - the fast Fourier transform of vector x
  hamming   - M-point Hamming window
  hanning   - M-point Hanning window
  hist      - compute the histogram of x
  kaiser    - M length Kaiser window
  psd       - the power spectral density using average periodiogram
  sinc      - the sinc function of array x

Other

  angle     - the angle of a complex array
  polyfit   - fit x, y to an n-th order polynomial
  polyval   - evaluate an n-th order polynomial
  roots     - the roots of the polynomial coefficients in p
  trapz     - trapezoidal integration


Credits: The plotting commands were provided by
John D. Hunter <jdhunter@ace.bsd.uhicago.edu>

Most of the other commands are from the Numeric, MLab and FFT, with
the exception of those in mlab.py provided by matplotlib.
"""
from __future__ import generators


__version__  = '0.90.1'
__revision__ = '$Revision$'
__date__     = '$Date$'

import md5, os, re, shutil, sys, warnings
import distutils.sysconfig

# Needed for toolkit setuptools support
if 0:
    try:
        __import__('pkg_resources').declare_namespace(__name__)
    except ImportError:
        pass # must not have setuptools

if not hasattr(sys, 'argv'):  # for modpython
    sys.argv = ['modpython']

"""
Manage user customizations through a rc file.

The default file location is given in the following order

  - environment variable MATPLOTLIBRC

  - HOME/.matplotlib/matplotlibrc if HOME is defined

  - PATH/matplotlibrc where PATH is the return value of
    get_data_path()
"""

import sys, os, tempfile


major, minor1, minor2, s, tmp = sys.version_info
_python23 = major>=2 and minor1>=3

_havemath = _python23

try:
    import datetime
    import dateutil
    import pytz
except ImportError: _havedate = False
else: _havedate = True

#try:
#    import pkg_resources # pkg_resources is part of setuptools
#except ImportError: _have_pkg_resources = False
#else: _have_pkg_resources = True

if not _python23:
    def enumerate(seq):
         for i in range(len(seq)):
             yield i, seq[i]


def is_string_like(obj):
    if hasattr(obj, 'shape'): return 0 # this is a workaround
                                       # for a bug in numeric<23.1
    try: obj + ''
    except (TypeError, ValueError): return 0
    return 1


def _is_writable_dir(p):
    """
    p is a string pointing to a putative writable dir -- return True p
    is such a string, else False
    """
    try: p + ''  # test is string like
    except TypeError: return False
    try:
        t = tempfile.TemporaryFile(dir=p)
        t.write('1')
        t.close()
    except OSError: return False
    else: return True

class Verbose:
    """
    A class to handle reporting.  Set the fileo attribute to any file
    instance to handle the output.  Default is sys.stdout
    """
    levels = ('silent', 'helpful', 'debug', 'debug-annoying')
    vald = dict( [(level, i) for i,level in enumerate(levels)])

    # parse the verbosity from the command line; flags look like
    # --verbose-silent or --verbose-helpful
    _commandLineVerbose = None


    for arg in sys.argv[1:]:
        if not arg.startswith('--verbose-'): continue
        _commandLineVerbose = arg[10:]



    def __init__(self):
        self.set_level('silent')
        self.fileo = sys.stdout

    def set_level(self, level):
        'set the verbosity to one of the Verbose.levels strings'

        if self._commandLineVerbose is not None:
            level = self._commandLineVerbose
        if level not in self.levels:
            raise ValueError('Illegal verbose string "%s".  Legal values are %s'%(level, self.levels))
        self.level = level

    def set_fileo(self, fname):
        std = {
            'sys.stdout': sys.stdout,
            'sys.stderr': sys.stderr,
        }
        if fname in std:
            self.fileo = std[fname]
        else:
            try:
                fileo = file(fname, 'w')
            except IOError:
                raise ValueError('Verbose object could not open log file "%s" for writing.\nCheck your matplotlibrc verbose.fileo setting'%fname)
            else:
                self.fileo = fileo

    def report(self, s, level='helpful'):
        """
        print message s to self.fileo if self.level>=level.  Return
        value indicates whether a message was issued

        """
        if self.ge(level):
            print >>self.fileo, s
            return True
        return False



    def wrap(self, fmt, func, level='helpful', always=True):
        """
        return a callable function that wraps func and reports it
        output through the verbose handler if current verbosity level
        is higher than level

        if always is True, the report will occur on every function
        call; otherwise only on the first time the function is called
        """
        assert(callable, func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)

            if (always or not wrapper._spoke):
                spoke = self.report(fmt%ret, level)
                if not wrapper._spoke: wrapper._spoke = spoke
            return ret
        wrapper._spoke = False
        wrapper.__doc__ = func.__doc__
        return wrapper

    def ge(self, level):
        'return true if self.level is >= level'
        return self.vald[self.level]>=self.vald[level]


verbose=Verbose()

def _get_home():
    """Find user's home directory if possible.
    Otherwise raise error.

    :see:  http://mail.python.org/pipermail/python-list/2005-February/263921.html
    """
    path=''
    try:
        path=os.path.expanduser("~")
    except:
        pass
    if not os.path.isdir(path):
        for evar in ('HOME', 'USERPROFILE', 'TMP'):
            try:
                path = os.environ[evar]
                if os.path.isdir(path):
                    break
            except: pass
    if path:
        return path
    else:
        raise RuntimeError('please define environment variable $HOME')



get_home = verbose.wrap('$HOME=%s', _get_home, always=False)

def _get_configdir():
    """
    Return the string representing the configuration dir.  

    default is HOME/.matplotlib.  you can override this with the
    MPLCONFIGDIR environment variable
    """

    configdir = os.environ.get('MPLCONFIGDIR')
    if configdir is not None:
        if not _is_writable_dir(configdir):
            raise RuntimeError('Could not write to MPLCONFIGDIR="%s"'%configdir)
        return configdir
    
    h = get_home()
    p = os.path.join(get_home(), '.matplotlib')
    
    if os.path.exists(p):
        if not _is_writable_dir(p):
            raise RuntimeError("'%s' is not a writable dir; you must set %s/.matplotlib to be a writable dir.  You can also set environment variable MPLCONFIGDIR to any writable directory where you want matplotlib data stored "%h)
    else:
        if not _is_writable_dir(h):
            raise RuntimeError("Failed to create %s/.matplotlib; consider setting MPLCONFIGDIR to a writable directory for matplotlib configuration data"%h)

        os.mkdir(p)

    return p
get_configdir = verbose.wrap('CONFIGDIR=%s', _get_configdir, always=False)


def _get_data_path():
    'get the path to matplotlib data'

    if os.environ.has_key('MATPLOTLIBDATA'):
        path = os.environ['MATPLOTLIBDATA']
        if os.path.isdir(path): return path

    path = os.sep.join([os.path.dirname(__file__), 'mpl-data'])
    if os.path.isdir(path): return path

    # setuptools' namespace_packages may highjack this init file
    # so need to try something known to be in matplotlib, not basemap
    import matplotlib.afm
    path = os.sep.join([os.path.dirname(matplotlib.afm.__file__), 'mpl-data'])
    if os.path.isdir(path): return path

    # py2exe zips pure python, so still need special check
    if getattr(sys,'frozen',None):
        path = os.path.join(os.path.split(sys.path[0])[0], 'matplotlibdata')
        if os.path.isdir(path):  return path
        else:
            # Try again assuming sys.path[0] is a dir not a exe
            path = os.path.join(sys.path[0], 'matplotlibdata')
            if os.path.isdir(path): return path

    raise RuntimeError('Could not find the matplotlib data files')

get_data_path = verbose.wrap('matplotlib data path %s', _get_data_path, always=False)

def get_py2exe_datafiles():
    datapath = get_data_path()
    head, tail = os.path.split(datapath)
    d = {}
    for root, dirs, files in os.walk(datapath):
        # Need to explicitly remove cocoa_agg files or py2exe complains
        # NOTE I dont know why, but do as previous version
        if 'Matplotlib.nib' in files:
            files.remove('Matplotlib.nib')
        files = [os.path.join(root, filename) for filename in files]
        root = root.replace(tail, 'matplotlibdata')
        root = root[root.index('matplotlibdata'):]
        d[root] = files
    return d.items()

def checkdep_dvipng():
    try:
        stdin, stdout = os.popen4('dvipng -version')
        line = stdout.readlines()[1]
        v = line.split()[-1]
        float(v)
        return v
    except (IndexError, ValueError):
        return None

def checkdep_ghostscript():
    try:
        if sys.platform == 'win32':
            command = 'gswin32c -v'
        else:
            command = 'gs -v'
        stdin, stdout = os.popen4(command)
        line = stdout.readlines()[0]
        v = line.split()[2]
        vtest = '.'.join(v.split('.')[:2]) # deal with version numbers like '7.07.1'
        float(vtest)
        return vtest
    except (IndexError, ValueError):
        return None

def checkdep_tex():
    try:
        stdin, stdout = os.popen4('tex -version')
        line = stdout.readlines()[0]
        pattern = '3\.1\d+'
        match = re.search(pattern, line)
        v = match.group(0)
        float(v)
        return v
    except (IndexError, ValueError):
        return None

def checkdep_pdftops():
    try:
        stdin, stdout = os.popen4('pdftops -v')
        for line in stdout.readlines():
            if 'version' in line:
                v = line.split()[-1]
        float(v)
        return v
    except (IndexError, ValueError):
        return None

def compare_versions(a, b):
    "return True if a is greater than b"
    if a:
        a = [int(i) for i in a.split('.')]
        b = [int(i) for i in b.split('.')]
        if a[0]>b[0]: return True
        elif (a[0]==b[0]) and (a[1]>=b[1]): return True
        else: return False
    else: return False


##########################
# to go to rcdefaults.py
#

class ValidateInStrings:
    def __init__(self, key, valid, ignorecase=False):
        'valid is a list of legal strings'
	self.key = key
        self.ignorecase = ignorecase
        def func(s):
            if ignorecase: return s.lower()
            else: return s
        self.valid = dict([(func(k),k) for k in valid])

    def __call__(self, s):
        if self.ignorecase: s = s.lower()
        if s in self.valid: return self.valid[s]
        raise ValueError('Unrecognized %s string "%s": valid strings are %s'%(self.key, s, self.valid.values()))

def validate_path_exists(s):
    'If s is a path, return s, else False'
    if os.path.exists(s): return s
    else:
        raise RuntimeError('"%s" should be a path but it does not exist'%s)

def validate_bool(b):
    'Convert b to a boolean or raise'
    bl = b.lower()
    if bl in ('f', 'no', 'false', '0', 0): return False
    elif bl in ('t', 'yes', 'true', '1', 1): return True
    else:
        raise ValueError('Could not convert "%s" to boolean' % b)

def validate_float(s):
    'convert s to float or raise'
    try: return float(s)
    except ValueError:
        raise ValueError('Could not convert "%s" to float' % s)

def validate_int(s):
    'convert s to int or raise'
    try: return int(s)
    except ValueError:
        raise ValueError('Could not convert "%s" to int' % s)

validate_backend = ValidateInStrings('backend',[
    'Agg2', 'Agg', 'Aqt', 'Cairo', 'CocoaAgg', 'EMF', 'GD', 'GDK',
    'GTK', 'GTKAgg', 'GTKCairo', 'FltkAgg', 'Paint', 'Pdf', 'PS',
    'QtAgg', 'Qt4Agg', 'SVG', 'Template', 'TkAgg', 'WX', 'WXAgg',
    ], ignorecase=True)

validate_numerix = ValidateInStrings('numerix',[
    'Numeric','numarray','numpy',
    ], ignorecase=True)

validate_toolbar = ValidateInStrings('toolbar',[
    'None','classic','toolbar2',
    ], ignorecase=True)

class validate_nseq_float:
    def __init__(self, n):
        self.n = n
    def __call__(self, s):
        'return a seq of n floats or raise'
        ss = s.split(',')
        if len(ss) != self.n:
            raise ValueError('You must use exactly %d comma separated values'%self.n)
        try: return [float(val) for val in ss]
        except ValueError:
            raise ValueError('Could not convert all entries to floats')

class validate_nseq_int:
    def __init__(self, n):
        self.n = n
    def __call__(self, s):
        'return a seq of n ints or raise'
        ss = s.split(',')
        if len(ss) != self.n:
            raise ValueError('You must use exactly %d comma separated values'%self.n)
        try: return [int(val) for val in ss]
        except ValueError:
            raise ValueError('Could not convert all entries to ints')


def validate_color(s):
    'return a valid color arg'
    if s.lower() == 'none': return 'None'
    if len(s)==1 and s.isalpha(): return s
    if s.find(',')>=0: # looks like an rgb
        # get rid of grouping symbols
        s = ''.join([ c for c in s if c.isdigit() or c=='.' or c==','])
        vals = s.split(',')
        if len(vals)!=3:
            raise ValueError('Color tuples must be length 3')

        try: return [float(val) for val in vals]
        except ValueError:
            raise ValueError('Could not convert all entries "%s" to floats'%s)

    if s.replace('.', '').isdigit(): # looks like scalar (grayscale)
        return s

    if len(s)==6 and s.isalnum(): # looks like hex
        return '#' + s

    if s.isalpha():
        #assuming a color name, hold on
        return s

    raise ValueError('"s" does not look like color arg')

def validate_comma_sep_str(s):
    'return a list'
    ss = s.split(',')
    try:
        return [val.strip() for val in ss]
    except ValueError:
        raise ValueError('Could not convert all entries to strings')

validate_orientation = ValidateInStrings('orientation',[
    'landscape', 'portrait',
    ])

def validate_latex_preamble(s):
    'return a list'
    preamble_list = validate_comma_sep_str(s)
    if not preamble_list == ['']:
        verbose.report("""
*****************************************************************
You have the following UNSUPPORTED LaTeX preamble customizations:
%s
Please do not ask for support with these customizations active.
*****************************************************************
"""% '\n'.join(preamble_list), 'helpful')
    return preamble_list



def validate_aspect(s):
    if s in ('auto', 'equal'):
        return s
    try:
        return float(s)
    except ValueError:
        raise ValueError('not a valid aspect specification')

def validate_fontsize(s):
    if s.lower() in ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large',
             'xx-large', 'smaller', 'larger']:
        return s.lower()
    try:
        return float(s)
    except ValueError:
        raise ValueError('not a valid font size')

validate_verbose = ValidateInStrings('verbose',[
    'silent', 'helpful', 'debug', 'debug-annoying',
    ])

validate_ps_papersize = ValidateInStrings('ps_papersize',[
    'auto', 'letter', 'legal', 'ledger',
    'a0', 'a1', 'a2','a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10',
    'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10',
    ], ignorecase=True)

def validate_ps_distiller(s):
    s = s.lower()

    if s == 'none':
        return None
    elif s == 'false':
        return False
    elif s in ('ghostscript', 'xpdf'):
        flag = True
        gs_req = '7.07'
        gs_sugg = '7.07'
        gs_v = checkdep_ghostscript()
        if compare_versions(gs_v, gs_sugg): pass
        elif compare_versions(gs_v, gs_req):
            verbose.report( 'ghostscript-%s found. ghostscript-%s or later \
is recommended to use the ps.usedistiller option.' % (gs_v, gs_sugg))
        else:
            flag = False
            warnings.warn('matplotlibrc ps.usedistiller option can not be used \
unless ghostscript-%s or later is installed on your system'% gs_req)

        if s == 'xpdf':
            pdftops_req = '3.0'
            pdftops_v = checkdep_pdftops()
            if compare_versions(pdftops_v, pdftops_req): pass
            else:
                flag = False
                warnings.warn('matplotlibrc ps.usedistiller can not be set to \
xpdf unless xpdf-%s or later is installed on your system' % pdftops_req)

        if flag: return s
        else: return None
    else:
        raise ValueError('matplotlibrc ps.usedistiller must either be none, \
ghostscript or xpdf')

def validate_usetex(s):
    bl = validate_bool(s)
    if bl:
        tex_req = '3.1415'
        gs_req = '7.07'
        gs_sugg = '7.07'
        dvipng_req = '1.5'
        flag = True

        tex_v = checkdep_tex()
        if compare_versions(tex_v, tex_req): pass
        else:
            flag = False
            warnings.warn('matplotlibrc text.usetex option can not be used \
unless TeX-%s or later is installed on your system' % tex_req)

        dvipng_v = checkdep_dvipng()
        if compare_versions(dvipng_v, dvipng_req): pass
        else:
            flag = False
            warnings.warn( 'matplotlibrc text.usetex can not be used with *Agg \
backend unless dvipng-1.5 or later is installed on your system')

        gs_v = checkdep_ghostscript()
        if compare_versions(gs_v, gs_sugg): pass
        elif compare_versions(gs_v, gs_req):
            verbose.report( 'ghostscript-%s found. ghostscript-%s or later is \
recommended for use with the text.usetex option.' % (gs_v, gs_sugg))
        else:
            flag = False
            warnings.warn('matplotlibrc text.usetex can not be used \
unless ghostscript-%s or later is installed on your system'% gs_req)

        if flag: return True
        else: return False
    else:
        return bl

validate_joinstyle = ValidateInStrings('joinstyle',['miter', 'round', 'bevel'], ignorecase=True)

validate_capstyle = ValidateInStrings('capstyle',['butt', 'round', 'projecting'], ignorecase=True)

def validate_linecol_linestyle(s):
    try:
        dashes = validate_nseq_float(2)(s)
        warnings.warn("Deprecated negative_linestyle specification; use 'solid' or 'dashed'")
        return (0, dashes)  # (offset, (solid, blank))
    except ValueError:
        V = ValidateInStrings('linecol_linestyle',['solid', 'dashed'], ignorecase=True)
        return(V(s))

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
        try: s = float(s)
        except: raise RuntimeError('Value must be a float; found "%s"'%s)

        if self.cmin and s<self.vmin:
            raise RuntimeError('Value must be >= %f; found "%f"'%(self.vmin, s))
        elif not self.cmin and s<=self.vmin:
            raise RuntimeError('Value must be > %f; found "%f"'%(self.vmin, s))

        if self.cmax and s>self.vmax:
            raise RuntimeError('Value must be <= %f; found "%f"'%(self.vmax, s))
        elif not self.cmax and s>=self.vmax:
            raise RuntimeError('Value must be < %f; found "%f"'%(self.vmax, s))
        return s



# a map from key -> value, converter
defaultParams = {
    'backend'           : ['WXAgg', validate_backend],
    'numerix'           : ['numpy', validate_numerix],
    'maskedarray'       : [False, validate_bool],
    'toolbar'           : ['toolbar2', validate_toolbar],
    'datapath'          : [get_data_path(), validate_path_exists],
    'units'             : [False, validate_bool],
    'interactive'       : [False, validate_bool],
    'timezone'          : ['UTC', str],

    # the verbosity setting
    'verbose.level'           : ['silent', validate_verbose],
    'verbose.fileo'           : ['sys.stdout', str],

    # line props
    'lines.linewidth'   : [1.0, validate_float],     # line width in points
    'lines.linestyle'   : ['-', str],                # solid line
    'lines.color'       : ['b', validate_color],     # blue
    'lines.marker'       : ['None', str],     # black
    'lines.markeredgewidth'       : [0.5, validate_float],
    'lines.markersize'  : [6, validate_float],       # markersize, in points
    'lines.antialiased' : [True, validate_bool],     # antialised (no jaggies)
    'lines.dash_joinstyle' : ['miter', validate_joinstyle],
    'lines.solid_joinstyle' : ['miter', validate_joinstyle],
    'lines.dash_capstyle' : ['butt', validate_capstyle],
    'lines.solid_capstyle' : ['projecting', validate_capstyle],

    # patch props
    'patch.linewidth'   : [1.0, validate_float], # line width in points
    'patch.edgecolor'   : ['k', validate_color], # black
    'patch.facecolor'   : ['b', validate_color], # blue
    'patch.antialiased' : [True, validate_bool], # antialised (no jaggies)


    # font props
    'font.family'       : ['serif', str],            # used by text object
    'font.style'        : ['normal', str],           #
    'font.variant'      : ['normal', str],           #
    'font.stretch'      : ['normal', str],           #
    'font.weight'       : ['normal', str],           #
    'font.size'         : [12.0, validate_float], #
    'font.serif'        : ['Bitstream Vera Serif, New Century Schoolbook, Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Times New Roman, Times, Palatino, Charter, serif', validate_comma_sep_str],
    'font.sans-serif'   : ['Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif', validate_comma_sep_str],
    'font.cursive'      : ['Apple Chancery, Textile, Zapf Chancery, Sand, cursive', validate_comma_sep_str],
    'font.fantasy'      : ['Comic Sans MS, Chicago, Charcoal, Impact, Western, fantasy', validate_comma_sep_str],
    'font.monospace'    : ['Bitstream Vera Sans Mono, Andale Mono, Nimbus Mono L, Courier New, Courier, Fixed, Terminal, monospace', validate_comma_sep_str],

    # text props
    'text.color'        : ['k', validate_color],     # black
    'text.usetex'       : [False, validate_usetex],
    'text.latex.unicode': [False, validate_bool],
    'text.latex.preamble': ['', validate_latex_preamble],
    'text.dvipnghack'    : [False, validate_bool],
    'text.fontstyle'    : ['normal', str],
    'text.fontangle'    : ['normal', str],
    'text.fontvariant'  : ['normal', str],
    'text.fontweight'   : ['normal', str],
    'text.fontsize'     : ['medium', validate_fontsize],


    'image.aspect' : ['equal', validate_aspect],  # equal, auto, a number
    'image.interpolation'  : ['bilinear', str],
    'image.cmap'   : ['jet', str],        # one of gray, jet, etc
    'image.lut'    : [256, validate_int],  # lookup table
    'image.origin'    : ['upper', str],  # lookup table

    'contour.negative_linestyle' : ['dashed', validate_linecol_linestyle],

    # axes props
    'axes.axisbelow'    : [False, validate_bool],
    'axes.hold'         : [True, validate_bool],
    'axes.facecolor'    : ['w', validate_color],    # background color; white
    'axes.edgecolor'    : ['k', validate_color],    # edge color; black
    'axes.linewidth'    : [1.0, validate_float],    # edge linewidth
    'axes.titlesize'    : [14, validate_fontsize], # fontsize of the axes title
    'axes.grid'         : [False, validate_bool],   # display grid or not
    'axes.labelsize'    : [12, validate_fontsize], # fontsize of the x any y labels
    'axes.labelcolor'   : ['k', validate_color],    # color of axis label
    'axes.formatter.limits' : [(-7, 7), validate_nseq_int(2)],
                               # use scientific notation if log10
                               # of the axis range is smaller than the
                               # first or larger than the second


    'polaraxes.grid'         : [True, validate_bool],   # display polar grid or not

    #legend properties
    'legend.isaxes'    :       [True,validate_bool],
    'legend.numpoints' :       [ 2,validate_int],      # the number of points in the legend line
    'legend.fontsize'  :       [14,validate_fontsize],
    'legend.pad'       :       [ 0.2, validate_float],         # the fractional whitespace inside the legend border
    'legend.markerscale' :     [ 1.0, validate_float],    # the relative size of legend markers vs. original

    # the following dimensions are in axes coords
    'legend.labelsep'  :       [ 0.010, validate_float],    # the vertical space between the legend entries
    'legend.handlelen'     :   [ 0.05, validate_float],  # the length of the legend lines
    'legend.handletextsep' :   [ 0.02, validate_float], # the space between the legend line and legend text
    'legend.axespad'   :       [ 0.02, validate_float], # the border between the axes and legend edge
    'legend.shadow' :          [ False, validate_bool ],


    # tick properties
    'xtick.major.size'   : [4, validate_float],      # major xtick size in points
    'xtick.minor.size'   : [2, validate_float],      # minor xtick size in points
    'xtick.major.pad'    : [4, validate_float],      # distance to label in points
    'xtick.minor.pad'    : [4, validate_float],      # distance to label in points
    'xtick.color'        : ['k', validate_color],    # color of the xtick labels
    'xtick.labelsize'    : [12, validate_fontsize], # fontsize of the xtick labels
    'xtick.direction'    : ['in', str],            # direction of xticks

    'ytick.major.size'   : [4, validate_float],      # major ytick size in points
    'ytick.minor.size'   : [2, validate_float],      # minor ytick size in points
    'ytick.major.pad'    : [4, validate_float],      # distance to label in points
    'ytick.minor.pad'    : [4, validate_float],      # distance to label in points
    'ytick.color'        : ['k', validate_color],    # color of the ytick labels
    'ytick.labelsize'    : [12, validate_fontsize], # fontsize of the ytick labels
    'ytick.direction'    : ['in', str],            # direction of yticks

    'grid.color'       :   ['k', validate_color],       # grid color
    'grid.linestyle'   :   [':', str],       # dotted
    'grid.linewidth'   :   [0.5, validate_float],     # in points


    # figure props
    # figure size in inches: width by height
    'figure.figsize'    : [ (8,6), validate_nseq_float(2)],
    'figure.dpi'        : [ 80, validate_float],   # DPI
    'figure.facecolor'  : [ '0.75', validate_color], # facecolor; scalar gray
    'figure.edgecolor'  : [ 'w', validate_color],  # edgecolor; white

    'figure.subplot.left'   : [0.125, ValidateInterval(0, 1, closedmin=False, closedmax=False)],
    'figure.subplot.right'  : [0.9, ValidateInterval(0, 1, closedmin=False, closedmax=False)],
    'figure.subplot.bottom' : [0.1, ValidateInterval(0, 1, closedmin=False, closedmax=False)],
    'figure.subplot.top'    : [0.9, ValidateInterval(0, 1, closedmin=False, closedmax=False)],
    'figure.subplot.wspace' : [0.2, ValidateInterval(0, 1, closedmin=False, closedmax=True)],
    'figure.subplot.hspace' : [0.2, ValidateInterval(0, 1, closedmin=False, closedmax=True)],


    'savefig.dpi'       : [ 100, validate_float],   # DPI
    'savefig.facecolor' : [ 'w', validate_color],  # facecolor; white
    'savefig.edgecolor' : [ 'w', validate_color],  # edgecolor; white
    'savefig.orientation' : [ 'portait', validate_orientation],  # edgecolor; white

    'tk.window_focus'   : [ False, validate_bool],  # Maintain shell focus for TkAgg
    'tk.pythoninspect'  : [ False, validate_bool],  # Set PYTHONINSPECT
    'ps.papersize'      : [ 'letter', validate_ps_papersize], # Set the papersize/type
    'ps.useafm'         : [ False, validate_bool],  # Set PYTHONINSPECT
    'ps.usedistiller'   : [ False, validate_ps_distiller],  # use ghostscript or xpdf to distill ps output
    'ps.distiller.res'  : [6000, validate_int],     # dpi
    'pdf.compression'   : [6, validate_int],        # compression level from 0 to 9; 0 to disable
    'pdf.inheritcolor'  : [False, validate_bool],   # ignore any color-setting commands from the frontend
    'pdf.use14corefonts' : [False, validate_bool],  # use only the 14 PDF core fonts, embedded in every PDF viewing application
    'svg.image_inline'  : [True, validate_bool],    # write raster image data directly into the svg file
    'svg.image_noscale'  : [False, validate_bool],  # suppress scaling of raster data embedded in SVG
    'plugins.directory' : ['.matplotlib_plugins', str], # where plugin directory is locate

    # mathtext settings
    'mathtext.mathtext2'  :   [False, validate_bool], # Needed to enable Unicode
    # fonts used by mathtext. These ship with matplotlib
    'mathtext.rm'       :   ['cmr10.ttf', str], # Roman (normal)
    'mathtext.it'       :   ['cmmi10.ttf', str], # Italic
    'mathtext.tt'       :   ['cmtt10.ttf', str],  # Typewriter (monospaced)
    'mathtext.mit'       :   ['cmmi10.ttf', str], # Math italic
    'mathtext.cal'      :   ['cmsy10.ttf', str], # Caligraphic
    'mathtext.nonascii' :   ['cmex10.ttf', str], # All other nonascii fonts

    }


def matplotlib_fname():
    """
    Return the path to the rc file

    Search order:

     * current working dir
     * environ var MATPLOTLIBRC
     * HOME/.matplotlib/matplotlibrc
     * MATPLOTLIBDATA/matplotlibrc


    """

    oldname = os.path.join( os.getcwd(), '.matplotlibrc')
    if os.path.exists(oldname):
        print >> sys.stderr, """\
WARNING: Old rc filename ".matplotlibrc" found in working dir
  and and renamed to new default rc file name "matplotlibrc"
  (no leading"dot"). """
        shutil.move('.matplotlibrc', 'matplotlibrc')

    home = get_home()
    oldname = os.path.join( home, '.matplotlibrc')
    if os.path.exists(oldname):
        configdir = get_configdir()
        newname = os.path.join(configdir, 'matplotlibrc')
        print >> sys.stderr, """\
WARNING: Old rc filename "%s" found and renamed to
  new default rc file name "%s"."""%(oldname, newname)

        shutil.move(oldname, newname)


    fname = os.path.join( os.getcwd(), 'matplotlibrc')
    if os.path.exists(fname): return fname

    if os.environ.has_key('MATPLOTLIBRC'):
        path =  os.environ['MATPLOTLIBRC']
        if os.path.exists(path):
            fname = os.path.join(path, 'matplotlibrc')
            if os.path.exists(fname):
                return fname

    fname = os.path.join(get_configdir(), 'matplotlibrc')
    if os.path.exists(fname): return fname


    path =  get_data_path() # guaranteed to exist or raise
    fname = os.path.join(path, 'matplotlibrc')
    if not os.path.exists(fname):
        warnings.warn('Could not find matplotlibrc; using defaults')
    return fname


def validate_key(key, val, line, cnt, fname, fail_on_error):
    if key in _deprecated_map.keys():
        alt = _deprecated_map[key]
        warnings.warn('%s is deprecated in matplotlibrc - use %s instead.' % (key, alt))
        key = alt

    if not defaultParams.has_key(key):
        print >> sys.stderr, """\
Bad key "%s" on line %d in
%s.
You probably need to get an updated matplotlibrc file from
http://matplotlib.sf.net/matplotlibrc or from the matplotlib source
distribution""" % (key, cnt, fname)
        return None

    default, converter =  defaultParams[key]

    if fail_on_error:
        return converter(val)   # try to convert to proper type or raise
    else:
        try: cval = converter(val)   # try to convert to proper type or raise
        except Exception, msg:
            warnings.warn('Bad val "%s" on line #%d\n\t"%s"\n\tin file "%s"\n\t%s' % (
                val, cnt, line, fname, msg))
            return None
        else:
            return cval

_deprecated_map = {
    'text.fontstyle':   'font.style',
    'text.fontangle':   'font.style',
    'text.fontvariant': 'font.variant',
    'text.fontweight':  'font.weight',
    'text.fontsize':    'font.size',
    'tick.size' :       'tick.major.size',
    }

def rc_params(fail_on_error=False):
    'Return the default params updated from the values in the rc file'

    fname = matplotlib_fname()
    if not os.path.exists(fname):
        message = 'could not find rc file; returning defaults'
        ret =  dict([ (key, tup[0]) for key, tup in defaultParams.items()])
        warnings.warn(message)
        return ret

    cnt = 0
    rc_temp = {}
    for line in file(fname):
        cnt += 1
        strippedline = line.split('#',1)[0].strip()
        if not strippedline: continue
        tup = strippedline.split(':',1)
        if len(tup) !=2:
            warnings.warn('Illegal line #%d\n\t%s\n\tin file "%s"' % (cnt, line, fname))
            continue
        key, val = tup
        key = key.strip()
        val = val.strip()
        if key in rc_temp:
            warnings.warn('Duplicate key in file "%s", line #%d'%(fname,cnt))
        rc_temp[key] = (val, line, cnt)

    ret = dict([ (key,default) for key, (default, converter) in defaultParams.iteritems() ])

    for key in ('verbose.level', 'verbose.fileo'):
        if key in rc_temp:
            val, line, cnt = rc_temp.pop(key)
            cval = validate_key(key, val, line, cnt, fname, fail_on_error)
            if cval is not None:
                ret[key] = cval

    verbose.set_level(ret['verbose.level'])
    verbose.set_fileo(ret['verbose.fileo'])

    for key, (val, line, cnt) in rc_temp.iteritems():
        cval = validate_key(key, val, line, cnt, fname, fail_on_error)
        if cval is not None:
            ret[key] = cval

    verbose.report('loaded rc file %s'%fname)

    return ret


# this is the instance used by the matplotlib classes
rcParams = rc_params()

rcParamsDefault = dict(rcParams.items()) # a copy

def rc(group, **kwargs):
    """
    Set the current rc params.  Group is the grouping for the rc, eg
    for lines.linewidth the group is 'lines', for axes.facecolor, the
    group is 'axes', and so on.  Group may also be a list or tuple
    of group names, eg ('xtick','ytick').  kwargs is a list of
    attribute name/value pairs, eg

      rc('lines', linewidth=2, color='r')

    sets the current rc params and is equivalent to

      rcParams['lines.linewidth'] = 2
      rcParams['lines.color'] = 'r'

    The following aliases are available to save typing for interactive
    users
        'lw'  : 'linewidth'
        'ls'  : 'linestyle'
        'c'   : 'color'
        'fc'  : 'facecolor'
        'ec'  : 'edgecolor'
        'mew' : 'markeredgewidth'
        'aa'  : 'antialiased'

    Thus you could abbreviate the above rc command as

          rc('lines', lw=2, c='r')


    Note you can use python's kwargs dictionary facility to store
    dictionaries of default parameters.  Eg, you can customize the
    font rc as follows

      font = {'family' : 'monospace',
              'weight' : 'bold',
              'size'   : 'larger',
             }

      rc('font', **font)  # pass in the font dict as kwargs

    This enables you to easily switch between several configurations.
    Use rcdefaults to restore the default rc params after changes.
    """

    aliases = {
        'lw'  : 'linewidth',
        'ls'  : 'linestyle',
        'c'   : 'color',
        'fc'  : 'facecolor',
        'ec'  : 'edgecolor',
        'mew' : 'markeredgewidth',
        'aa'  : 'antialiased',
        }

    if is_string_like(group):
        group = (group,)
    for g in group:
        for k,v in kwargs.items():
            name = aliases.get(k) or k
            key = '%s.%s' % (g, name)
            if not rcParams.has_key(key):
                raise KeyError('Unrecognized key "%s" for group "%s" and name "%s"' %
                               (key, g, name))

            rcParams[key] = v

def rcdefaults():
    """
    Restore the default rc params - the ones that were created at
    matplotlib load time
    """
    rcParams.update(rcParamsDefault)

# Now allow command line to override

# Allow command line access to the backend with -d (matlab compatible
# flag)

for s in sys.argv[1:]:
    if s.startswith('-d'):  # look for a -d flag
        name = validate_backend(s[2:].strip(), fail_on_err = False)
        # we don't want to assume all -d flags are backends, eg -debug
        if name:
            rcParams['backend'] = name
            break

def use(arg):
    """
    Set the matplotlib backend to one of the known backends
    """
    rcParams['backend'] = validate_backend(arg)

def get_backend():
    return rcParams['backend']

def interactive(b):
    """
    Set interactive mode to boolean b.

    If b is True, then draw after every plotting command, eg, after xlabel
    """
    rcParams['interactive'] = b

def is_interactive():
    'Return true if plot mode is interactive'
    b = rcParams['interactive']
    return b

def tk_window_focus():
    """Return true if focus maintenance under TkAgg on win32 is on.
     This currently works only for python.exe and IPython.exe.
     Both IDLE and Pythonwin.exe fail badly when tk_window_focus is on."""
    if rcParams['backend'] != 'TkAgg':
        return False
    return rcParams['tk.window_focus']



verbose.report('matplotlib version %s'%__version__)
verbose.report('verbose.level %s'%verbose.level)
verbose.report('interactive is %s'%rcParams['interactive'])
verbose.report('units is %s'%rcParams['units'])
verbose.report('platform is %s'%sys.platform)
verbose.report('loaded modules: %s'%sys.modules.keys(), 'debug')

class ExampleInfo:
    pass

class ExampleManager:
    baseurl = 'http://matplotlib.sf.net'
    urls = ['%s/%s'%(baseurl, subdir) for subdir in
            ( 'examples', 'examples/widgets')]

    def get_examples(self):
        import urllib, re
        rgx = re.compile('.*<A HREF="([^.]+\.py)">.*')
        examples = []
        for url in urls:
            lines = urllib.urlopen(url).readlines()
            for line in lines:
                m = rgx.match(line)
                if m is not None:
                    examples.append('%s/%s'%(url, m.group(1)))
        return examples

    def get_info(self, s):
        """
        return an ExampleInfo instance from s, the string content of
        an example
        """
        pass
