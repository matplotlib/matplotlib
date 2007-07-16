"""
This is an object-orient plotting library.

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

from rcsetup import defaultParams, validate_backend, validate_toolbar
from rcsetup import validate_cairo_format

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

def checkdep_ps_distiller(s):
    if not s:
        return False

    flag = True
    gs_req = '7.07'
    gs_sugg = '7.07'
    gs_v = checkdep_ghostscript()
    if compare_versions(gs_v, gs_sugg): pass
    elif compare_versions(gs_v, gs_req):
        verbose.report(('ghostscript-%s found. ghostscript-%s or later '
                        'is recommended to use the ps.usedistiller option.') % (gs_v, gs_sugg))
    else:
        flag = False
        warnings.warn(('matplotlibrc ps.usedistiller option can not be used '
                       'unless ghostscript-%s or later is installed on your system') % gs_req)

    if s == 'xpdf':
        pdftops_req = '3.0'
        pdftops_v = checkdep_pdftops()
        if compare_versions(pdftops_v, pdftops_req): pass
        else:
            flag = False
            warnings.warn(('matplotlibrc ps.usedistiller can not be set to '
                           'xpdf unless xpdf-%s or later is installed on your system') % pdftops_req)

    if flag:
        return s
    else:
        return False

def checkdep_usetex(s):
    if not s:
        return False

    tex_req = '3.1415'
    gs_req = '7.07'
    gs_sugg = '7.07'
    dvipng_req = '1.5'
    flag = True

    tex_v = checkdep_tex()
    if compare_versions(tex_v, tex_req): pass
    else:
        flag = False
        warnings.warn(('matplotlibrc text.usetex option can not be used '
                       'unless TeX-%s or later is '
                       'installed on your system') % tex_req)

    dvipng_v = checkdep_dvipng()
    if compare_versions(dvipng_v, dvipng_req): pass
    else:
        flag = False
        warnings.warn( 'matplotlibrc text.usetex can not be used with *Agg '
                       'backend unless dvipng-1.5 or later is '
                       'installed on your system')

    gs_v = checkdep_ghostscript()
    if compare_versions(gs_v, gs_sugg): pass
    elif compare_versions(gs_v, gs_req):
        verbose.report(('ghostscript-%s found. ghostscript-%s or later is '
                        'recommended for use with the text.usetex '
                        'option.') % (gs_v, gs_sugg))
    else:
        flag = False
        warnings.warn(('matplotlibrc text.usetex can not be used '
                       'unless ghostscript-%s or later is '
                       'installed on your system') % gs_req)

    return flag


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
        if not os.path.isdir(path):
            raise RuntimeError('Path in environment MATPLOTLIBDATA not a directory')
        return path

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
        if os.path.isdir(path): return path
        else:
            # Try again assuming sys.path[0] is a dir not a exe
            path = os.path.join(sys.path[0], 'matplotlibdata')
            if os.path.isdir(path): return path

    raise RuntimeError('Could not find the matplotlib data files')

def _get_data_path_cached():
    if defaultParams['datapath'][0] is None:
        defaultParams['datapath'][0] = _get_data_path()
    return defaultParams['datapath'][0]

get_data_path = verbose.wrap('matplotlib data path %s', _get_data_path_cached, always=False)

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

    if ret['datapath'] is None:
        ret['datapath'] = get_data_path()

    verbose.report('loaded rc file %s'%fname)

    return ret


# this is the instance used by the matplotlib classes
rcParams = rc_params()

rcParamsDefault = dict(rcParams.items()) # a copy

rcParams['ps.usedistiller'] = checkdep_ps_distiller(rcParams['ps.usedistiller'])
rcParams['text.usetex'] = checkdep_usetex(rcParams['text.usetex'])

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


def use(arg):
    """
    Set the matplotlib backend to one of the known backends.

    The argument is case-insensitive.  For the Cairo backend,
    the argument can have an extension to indicate the type of
    output.  Example:

        use('cairo.pdf')

    will specify a default of pdf output generated by Cairo.

    Note: this function must be called *before* importing pylab
    for the first time; or, if you are not using pylab, it must
    be called before importing matplotlib.backends.
    """
    be_parts = arg.split('.')
    name = validate_backend(be_parts[0])
    rcParams['backend'] = name
    if name == 'Cairo' and len(be_parts) > 1:
        rcParams['cairo.format'] = validate_cairo_format(be_parts[1])

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

# Now allow command line to override

# Allow command line access to the backend with -d (matlab compatible
# flag)

for s in sys.argv[1:]:
    if s.startswith('-d') and len(s) > 2:  # look for a -d flag
        try:
            use(s[2:])
        except (KeyError, ValueError):
            pass
        # we don't want to assume all -d flags are backends, eg -debug


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



class Namespace:
    """
    A class which takes a list of modules and creates an object with
    the module naems at attrs
    """
    def __init__(self, namespace):
        for k,v in namespace.items():
            modname = getattr(v, '__name__', None)
            if modname is None: continue
            if modname.startswith('matplotlib.'):
                self.__dict__[modname.replace('matplotlib.', '')] = v


class Importer:
    def __init__(self, modstr):
        """
        import a bunch of matplotlib modules listed in modstr into a
        single namespace.  Eg,

        mpl = Importer('artist, cbook, lines, patches')
        print mpl.cbook.iterable(1)
        """
        for name in modstr.split(','):
            name = name.strip()
            wholename = '.'.join(['matplotlib', name])
            basemod = __import__(wholename)
            mod = getattr(basemod, name)
            setattr(self, name, mod)

