"""
This is an object-oriented plotting library.

A procedural interface is provided by the companion pyplot module,
which may be imported directly, e.g.::

    import matplotlib.pyplot as plt

or using ipython::

    ipython

at your terminal, followed by::

    In [1]: %matplotlib
    In [2]: import matplotlib.pyplot as plt

at the ipython shell prompt.

For the most part, direct use of the object-oriented library is
encouraged when programming; pyplot is primarily for working
interactively.  The
exceptions are the pyplot commands :func:`~matplotlib.pyplot.figure`,
:func:`~matplotlib.pyplot.subplot`,
:func:`~matplotlib.pyplot.subplots`, and
:func:`~pyplot.savefig`, which can greatly simplify scripting.

Modules include:

    :mod:`matplotlib.axes`
        defines the :class:`~matplotlib.axes.Axes` class.  Most pylab
        commands are wrappers for :class:`~matplotlib.axes.Axes`
        methods.  The axes module is the highest level of OO access to
        the library.

    :mod:`matplotlib.figure`
        defines the :class:`~matplotlib.figure.Figure` class.

    :mod:`matplotlib.artist`
        defines the :class:`~matplotlib.artist.Artist` base class for
        all classes that draw things.

    :mod:`matplotlib.lines`
        defines the :class:`~matplotlib.lines.Line2D` class for
        drawing lines and markers

    :mod:`matplotlib.patches`
        defines classes for drawing polygons

    :mod:`matplotlib.text`
        defines the :class:`~matplotlib.text.Text`,
        :class:`~matplotlib.text.TextWithDash`, and
        :class:`~matplotlib.text.Annotate` classes

    :mod:`matplotlib.image`
        defines the :class:`~matplotlib.image.AxesImage` and
        :class:`~matplotlib.image.FigureImage` classes

    :mod:`matplotlib.collections`
        classes for efficient drawing of groups of lines or polygons

    :mod:`matplotlib.colors`
        classes for interpreting color specifications and for making
        colormaps

    :mod:`matplotlib.cm`
        colormaps and the :class:`~matplotlib.image.ScalarMappable`
        mixin class for providing color mapping functionality to other
        classes

    :mod:`matplotlib.ticker`
        classes for calculating tick mark locations and for formatting
        tick labels

    :mod:`matplotlib.backends`
        a subpackage with modules for various gui libraries and output
        formats

The base matplotlib namespace includes:

    :data:`~matplotlib.rcParams`
        a global dictionary of default configuration settings.  It is
        initialized by code which may be overridded by a matplotlibrc
        file.

    :func:`~matplotlib.rc`
        a function for setting groups of rcParams values

    :func:`~matplotlib.use`
        a function for setting the matplotlib backend.  If used, this
        function must be called immediately after importing matplotlib
        for the first time.  In particular, it must be called
        **before** importing pylab (if pylab is imported).

matplotlib was initially written by John D. Hunter (1968-2012) and is now
developed and maintained by a host of others.

Occasionally the internal documentation (python docstrings) will refer
to MATLAB&reg;, a registered trademark of The MathWorks, Inc.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import sys
import distutils.version
from itertools import chain

__version__ = str('1.4.3')
__version__numpy__ = str('1.6')  # minimum required numpy version

try:
    import dateutil
except ImportError:
    raise ImportError("matplotlib requires dateutil")

def compare_versions(a, b):
    "return True if a is greater than or equal to b"
    if a:
        if six.PY3:
            if isinstance(a, bytes):
                a = a.decode('ascii')
            if isinstance(b, bytes):
                b = b.decode('ascii')
        a = distutils.version.LooseVersion(a)
        b = distutils.version.LooseVersion(b)
        return a >= b
    else:
        return False

if not compare_versions(six.__version__, '1.3'):
    raise ImportError(
        'six 1.3 or later is required; you have %s' % (
            six.__version__))

try:
    import pyparsing
except ImportError:
    raise ImportError("matplotlib requires pyparsing")
else:
    if not compare_versions(pyparsing.__version__, '1.5.6'):
        raise ImportError(
            "matplotlib requires pyparsing >= 1.5.6")

    # pyparsing 2.0.0 bug, but it may be patched in distributions
    try:
        f = pyparsing.Forward()
        f <<= pyparsing.Literal('a')
        bad_pyparsing = f is None
    except TypeError:
        bad_pyparsing = True

    # pyparsing 1.5.6 does not have <<= on the Forward class, but
    # pyparsing 2.0.0 and later will spew deprecation warnings if
    # using << instead.  Additionally, the <<= in pyparsing 1.5.7 is
    # broken, since it doesn't return self.  In order to support
    # pyparsing 1.5.6 and above with a common code base, this small
    # monkey patch is applied.
    if bad_pyparsing:
        def _forward_ilshift(self, other):
            self.__lshift__(other)
            return self
        pyparsing.Forward.__ilshift__ = _forward_ilshift

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

import os
import re
import tempfile
import warnings
import contextlib
import distutils.sysconfig

# cbook must import matplotlib only within function
# definitions, so it is safe to import from it here.
from matplotlib.cbook import is_string_like
from matplotlib.compat import subprocess

try:
    reload
except NameError:
    # Python 3
    from imp import reload


if not hasattr(sys, 'argv'):  # for modpython
    sys.argv = [str('modpython')]


from matplotlib.rcsetup import (defaultParams,
                                validate_backend)

major, minor1, minor2, s, tmp = sys.version_info
_python24 = (major == 2 and minor1 >= 4) or major >= 3

# the havedate check was a legacy from old matplotlib which preceeded
# datetime support
_havedate = True

#try:
#    import pkg_resources # pkg_resources is part of setuptools
#except ImportError: _have_pkg_resources = False
#else: _have_pkg_resources = True

if not _python24:
    raise ImportError('matplotlib requires Python 2.4 or later')


import numpy
if not compare_versions(numpy.__version__, __version__numpy__):
    raise ImportError(
        'numpy %s or later is required; you have %s' % (
            __version__numpy__, numpy.__version__))


def _is_writable_dir(p):
    """
    p is a string pointing to a putative writable dir -- return True p
    is such a string, else False
    """
    try:
        p + ''  # test is string like
    except TypeError:
        return False

    # Test whether the operating system thinks it's a writable directory.
    # Note that this check is necessary on Google App Engine, because the
    # subsequent check will succeed even though p may not be writable.
    if not os.access(p, os.W_OK) or not os.path.isdir(p):
        return False

    # Also test that it is actually possible to write to a file here.
    try:
        t = tempfile.TemporaryFile(dir=p)
        try:
            t.write(b'1')
        finally:
            t.close()
    except OSError:
        return False

    return True


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
        # cast to str because we are using unicode_literals,
        # and argv is always str
        if not arg.startswith(str('--verbose-')):
            continue
        level_str = arg[10:]
        # If it doesn't match one of ours, then don't even
        # bother noting it, we are just a 3rd-party library
        # to somebody else's script.
        if level_str in levels:
            _commandLineVerbose = level_str

    def __init__(self):
        self.set_level('silent')
        self.fileo = sys.stdout

    def set_level(self, level):
        'set the verbosity to one of the Verbose.levels strings'

        if self._commandLineVerbose is not None:
            level = self._commandLineVerbose
        if level not in self.levels:
            warnings.warn('matplotlib: unrecognized --verbose-* string "%s".'
                          ' Legal values are %s' % (level, self.levels))
        else:
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
                fileo = open(fname, 'w')
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
            print(s, file=self.fileo)
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
        assert six.callable(func)
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
        s = subprocess.Popen(['dvipng','-version'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = s.communicate()
        line = stdout.decode('ascii').split('\n')[1]
        v = line.split()[-1]
        return v
    except (IndexError, ValueError, OSError):
        return None

def checkdep_ghostscript():
    if sys.platform == 'win32':
        gs_execs = ['gswin32c', 'gswin64c', 'gs']
    else:
        gs_execs = ['gs']
    for gs_exec in gs_execs:
        try:
            s = subprocess.Popen(
                [gs_exec, '--version'], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            stdout, stderr = s.communicate()
            if s.returncode == 0:
                v = stdout[:-1].decode('ascii')
                return gs_exec, v
        except (IndexError, ValueError, OSError):
            pass
    return None, None

def checkdep_tex():
    try:
        s = subprocess.Popen(['tex','-version'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = s.communicate()
        line = stdout.decode('ascii').split('\n')[0]
        pattern = '3\.1\d+'
        match = re.search(pattern, line)
        v = match.group(0)
        return v
    except (IndexError, ValueError, AttributeError, OSError):
        return None

def checkdep_pdftops():
    try:
        s = subprocess.Popen(['pdftops','-v'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = s.communicate()
        lines = stderr.decode('ascii').split('\n')
        for line in lines:
            if 'version' in line:
                v = line.split()[-1]
        return v
    except (IndexError, ValueError, UnboundLocalError, OSError):
        return None

def checkdep_inkscape():
    try:
        s = subprocess.Popen(['inkscape','-V'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = s.communicate()
        lines = stdout.decode('ascii').split('\n')
        for line in lines:
            if 'Inkscape' in line:
                v = line.split()[1]
                break
        return v
    except (IndexError, ValueError, UnboundLocalError, OSError):
        return None

def checkdep_xmllint():
    try:
        s = subprocess.Popen(['xmllint','--version'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = s.communicate()
        lines = stderr.decode('ascii').split('\n')
        for line in lines:
            if 'version' in line:
                v = line.split()[-1]
                break
        return v
    except (IndexError, ValueError, UnboundLocalError, OSError):
        return None

def checkdep_ps_distiller(s):
    if not s:
        return False

    flag = True
    gs_req = '7.07'
    gs_sugg = '7.07'
    gs_exec, gs_v = checkdep_ghostscript()
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
        pdftops_req_alt = '0.9' # poppler version numbers, ugh
        pdftops_v = checkdep_pdftops()
        if compare_versions(pdftops_v, pdftops_req):
            pass
        elif compare_versions(pdftops_v, pdftops_req_alt) and not \
            compare_versions(pdftops_v, '1.0'):
            pass
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

    gs_exec, gs_v = checkdep_ghostscript()
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
    Otherwise, returns None.

    :see:  http://mail.python.org/pipermail/python-list/2005-February/325395.html
    """
    try:
        if six.PY2 and sys.platform == 'win32':
            path = os.path.expanduser(b"~").decode(sys.getfilesystemencoding())
        else:
            path = os.path.expanduser("~")
    except ImportError:
        # This happens on Google App Engine (pwd module is not present).
        pass
    else:
        if os.path.isdir(path):
            return path
    for evar in ('HOME', 'USERPROFILE', 'TMP'):
        path = os.environ.get(evar)
        if path is not None and os.path.isdir(path):
            return path
    return None


def _create_tmp_config_dir():
    """
    If the config directory can not be created, create a temporary
    directory.

    Returns None if a writable temporary directory could not be created.
    """
    import getpass
    import tempfile

    try:
        tempdir = tempfile.gettempdir()
    except NotImplementedError:
        # Some restricted platforms (such as Google App Engine) do not provide
        # gettempdir.
        return None
    tempdir = os.path.join(tempdir, 'matplotlib-%s' % getpass.getuser())
    os.environ['MPLCONFIGDIR'] = tempdir

    return tempdir


get_home = verbose.wrap('$HOME=%s', _get_home, always=False)

def _get_xdg_config_dir():
    """
    Returns the XDG configuration directory, according to the `XDG
    base directory spec
    <http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html>`_.
    """
    path = os.environ.get('XDG_CONFIG_HOME')
    if path is None:
        path = get_home()
        if path is not None:
            path = os.path.join(path, '.config')
    return path


def _get_xdg_cache_dir():
    """
    Returns the XDG cache directory, according to the `XDG
    base directory spec
    <http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html>`_.
    """
    path = os.environ.get('XDG_CACHE_HOME')
    if path is None:
        path = get_home()
        if path is not None:
            path = os.path.join(path, '.cache')
    return path


def _get_config_or_cache_dir(xdg_base):
    from matplotlib.cbook import mkdirs

    configdir = os.environ.get('MPLCONFIGDIR')
    if configdir is not None:
        configdir = os.path.abspath(configdir)
        if not os.path.exists(configdir):
            mkdirs(configdir)

        if not _is_writable_dir(configdir):
            return _create_tmp_config_dir()
        return configdir

    p = None
    h = get_home()
    if h is not None:
        p = os.path.join(h, '.matplotlib')
    if (sys.platform.startswith('linux') and xdg_base):
        p = os.path.join(xdg_base, 'matplotlib')

    if p is not None:
        if os.path.exists(p):
            if _is_writable_dir(p):
                return p
        else:
            try:
                mkdirs(p)
            except OSError:
                pass
            else:
                return p

    return _create_tmp_config_dir()


def _get_configdir():
    """
    Return the string representing the configuration directory.

    The directory is chosen as follows:

    1. If the MPLCONFIGDIR environment variable is supplied, choose that.

    2a. On Linux, if `$HOME/.matplotlib` exists, choose that, but warn that
        that is the old location.  Barring that, follow the XDG specification
        and look first in `$XDG_CONFIG_HOME`, if defined, or `$HOME/.config`.

    2b. On other platforms, choose `$HOME/.matplotlib`.

    3. If the chosen directory exists and is writable, use that as the
       configuration directory.
    4. If possible, create a temporary directory, and use it as the
       configuration directory.
    5. A writable directory could not be found or created; return None.
    """
    return _get_config_or_cache_dir(_get_xdg_config_dir())

get_configdir = verbose.wrap('CONFIGDIR=%s', _get_configdir, always=False)


def _get_cachedir():
    """
    Return the location of the cache directory.

    The procedure used to find the directory is the same as for
    _get_config_dir, except using `$XDG_CACHE_HOME`/`~/.cache` instead.
    """
    return _get_config_or_cache_dir(_get_xdg_cache_dir())

get_cachedir = verbose.wrap('CACHEDIR=%s', _get_cachedir, always=False)


def _get_data_path():
    'get the path to matplotlib data'

    if 'MATPLOTLIBDATA' in os.environ:
        path = os.environ['MATPLOTLIBDATA']
        if not os.path.isdir(path):
            raise RuntimeError('Path in environment MATPLOTLIBDATA not a directory')
        return path

    path = os.sep.join([os.path.dirname(__file__), 'mpl-data'])
    if os.path.isdir(path):
        return path

    # setuptools' namespace_packages may highjack this init file
    # so need to try something known to be in matplotlib, not basemap
    import matplotlib.afm
    path = os.sep.join([os.path.dirname(matplotlib.afm.__file__), 'mpl-data'])
    if os.path.isdir(path):
        return path

    # py2exe zips pure python, so still need special check
    if getattr(sys,'frozen',None):
        exe_path = os.path.dirname(sys.executable)
        path = os.path.join(exe_path, 'mpl-data')
        if os.path.isdir(path):
            return path

        # Try again assuming we need to step up one more directory
        path = os.path.join(os.path.split(exe_path)[0], 'mpl-data')
        if os.path.isdir(path):
            return path

        # Try again assuming sys.path[0] is a dir not a exe
        path = os.path.join(sys.path[0], 'mpl-data')
        if os.path.isdir(path):
            return path

    raise RuntimeError('Could not find the matplotlib data files')

def _get_data_path_cached():
    if defaultParams['datapath'][0] is None:
        defaultParams['datapath'][0] = _get_data_path()
    return defaultParams['datapath'][0]

get_data_path = verbose.wrap('matplotlib data path %s', _get_data_path_cached,
                             always=False)



def get_example_data(fname):
    """
    get_example_data is deprecated -- use matplotlib.cbook.get_sample_data instead
    """
    raise NotImplementedError('get_example_data is deprecated -- use matplotlib.cbook.get_sample_data instead')


def get_py2exe_datafiles():
    datapath = get_data_path()
    _, tail = os.path.split(datapath)
    d = {}
    for root, _, files in os.walk(datapath):
        # Need to explicitly remove cocoa_agg files or py2exe complains
        # NOTE I dont know why, but do as previous version
        if 'Matplotlib.nib' in files:
            files.remove('Matplotlib.nib')
        files = [os.path.join(root, filename) for filename in files]
        root = root.replace(tail, 'mpl-data')
        root = root[root.index('mpl-data'):]
        d[root] = files
    return list(d.items())


def matplotlib_fname():
    """
    Get the location of the config file.

    The file location is determined in the following order

    - `$PWD/matplotlibrc`

    - environment variable `MATPLOTLIBRC`

    - `$MPLCONFIGDIR/matplotlib`

    - On Linux,

          - `$HOME/.matplotlib/matplotlibrc`, if it exists

          - or `$XDG_CONFIG_HOME/matplotlib/matplotlibrc` (if
            $XDG_CONFIG_HOME is defined)

          - or `$HOME/.config/matplotlib/matplotlibrc` (if
            $XDG_CONFIG_HOME is not defined)

    - On other platforms,

         - `$HOME/.matplotlib/matplotlibrc` if `$HOME` is defined.

    - Lastly, it looks in `$MATPLOTLIBDATA/matplotlibrc` for a
      system-defined copy.
    """
    if six.PY2:
        cwd = os.getcwdu()
    else:
        cwd = os.getcwd()
    fname = os.path.join(cwd, 'matplotlibrc')
    if os.path.exists(fname):
        return fname

    if 'MATPLOTLIBRC' in os.environ:
        path =  os.environ['MATPLOTLIBRC']
        if os.path.exists(path):
            fname = os.path.join(path, 'matplotlibrc')
            if os.path.exists(fname):
                return fname

    configdir = _get_configdir()
    if configdir is not None:
        fname = os.path.join(configdir, 'matplotlibrc')
        if os.path.exists(fname):
            home = get_home()
            if (sys.platform.startswith('linux') and
                home is not None and
                os.path.exists(os.path.join(
                    home, '.matplotlib', 'matplotlibrc'))):
                warnings.warn(
                    "Found matplotlib configuration in ~/.matplotlib/. "
                    "To conform with the XDG base directory standard, "
                    "this configuration location has been deprecated "
                    "on Linux, and the new location is now %s/matplotlib/. "
                    "Please move your configuration there to ensure that "
                    "matplotlib will continue to find it in the future." %
                    _get_xdg_config_dir())
                return os.path.join(
                    home, '.matplotlib', 'matplotlibrc')
            return fname

    path = get_data_path()  # guaranteed to exist or raise
    fname = os.path.join(path, 'matplotlibrc')
    if not os.path.exists(fname):
        warnings.warn('Could not find matplotlibrc; using defaults')

    return fname


_deprecated_map = {
    'text.fontstyle':   ('font.style',lambda x: x),
    'text.fontangle':   ('font.style',lambda x: x),
    'text.fontvariant': ('font.variant',lambda x: x),
    'text.fontweight':  ('font.weight',lambda x: x),
    'text.fontsize':    ('font.size',lambda x: x),
    'tick.size' :       ('tick.major.size',lambda x: x),
    'svg.embed_char_paths' : ('svg.fonttype',lambda x: "path" if x else "none"),
    'savefig.extension' : ('savefig.format',lambda x: x),
    }

_deprecated_ignore_map = {
    }

_obsolete_set = set(['tk.pythoninspect', ])
_all_deprecated = set(chain(_deprecated_ignore_map,
                            _deprecated_map, _obsolete_set))

_rcparam_warn_str = ("Trying to set {key} to {value} via the {func} "
                     "method of RcParams which does not validate cleanly. "
                     "This warning will turn into an Exception in 1.5. "
                     "If you think {value} should validate correctly for "
                     "rcParams[{key}] "
                     "please create an issue on github."
                     )


class RcParams(dict):

    """
    A dictionary object including validation

    validating functions are defined and associated with rc parameters in
    :mod:`matplotlib.rcsetup`
    """

    validate = dict((key, converter) for key, (default, converter) in
                    six.iteritems(defaultParams)
                    if key not in _all_deprecated)
    msg_depr = "%s is deprecated and replaced with %s; please use the latter."
    msg_depr_ignore = "%s is deprecated and ignored. Use %s"

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        for k, v in six.iteritems(dict(*args, **kwargs)):
            try:
                self[k] = v
            except (ValueError, RuntimeError):
                # force the issue
                warnings.warn(_rcparam_warn_str.format(key=repr(k),
                                                       value=repr(v),
                                                       func='__init__'))
                dict.__setitem__(self, k, v)

    def __setitem__(self, key, val):
        try:
            if key in _deprecated_map:
                alt_key, alt_val = _deprecated_map[key]
                warnings.warn(self.msg_depr % (key, alt_key))
                key = alt_key
                val = alt_val(val)
            elif key in _deprecated_ignore_map:
                alt = _deprecated_ignore_map[key]
                warnings.warn(self.msg_depr_ignore % (key, alt))
                return
            try:
                cval = self.validate[key](val)
            except ValueError as ve:
                raise ValueError("Key %s: %s" % (key, str(ve)))
            dict.__setitem__(self, key, cval)
        except KeyError:
            raise KeyError('%s is not a valid rc parameter.\
See rcParams.keys() for a list of valid parameters.' % (key,))

    def __getitem__(self, key):
        if key in _deprecated_map:
            alt_key, alt_val = _deprecated_map[key]
            warnings.warn(self.msg_depr % (key, alt_key))
            key = alt_key
        elif key in _deprecated_ignore_map:
            alt = _deprecated_ignore_map[key]
            warnings.warn(self.msg_depr_ignore % (key, alt))
            key = alt
        return dict.__getitem__(self, key)

    # http://stackoverflow.com/questions/2390827/how-to-properly-subclass-dict-and-override-get-set
    # the default dict `update` does not use __setitem__
    # so rcParams.update(...) (such as in seaborn) side-steps
    # all of the validation over-ride update to force
    # through __setitem__
    def update(self, *args, **kwargs):
        for k, v in six.iteritems(dict(*args, **kwargs)):
            try:
                self[k] = v
            except (ValueError, RuntimeError):
                # force the issue
                warnings.warn(_rcparam_warn_str.format(key=repr(k),
                                                       value=repr(v),
                                                       func='update'))
                dict.__setitem__(self, k, v)

    def __repr__(self):
        import pprint
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        repr_split = pprint.pformat(dict(self), indent=1,
                                    width=80 - indent).split('\n')
        repr_indented = ('\n' + ' ' * indent).join(repr_split)
        return '{0}({1})'.format(class_name, repr_indented)

    def __str__(self):
        return '\n'.join('{0}: {1}'.format(k, v)
                         for k, v in sorted(self.items()))

    def keys(self):
        """
        Return sorted list of keys.
        """
        k = list(dict.keys(self))
        k.sort()
        return k

    def values(self):
        """
        Return values in order of sorted keys.
        """
        return [self[k] for k in self.keys()]

    def find_all(self, pattern):
        """
        Return the subset of this RcParams dictionary whose keys match,
        using :func:`re.search`, the given ``pattern``.

        .. note::

            Changes to the returned dictionary are *not* propagated to
            the parent RcParams dictionary.

        """
        import re
        pattern_re = re.compile(pattern)
        return RcParams((key, value)
                        for key, value in self.items()
                        if pattern_re.search(key))


def rc_params(fail_on_error=False):
    """Return a :class:`matplotlib.RcParams` instance from the
    default matplotlib rc file.
    """
    fname = matplotlib_fname()
    if not os.path.exists(fname):
        # this should never happen, default in mpl-data should always be found
        message = 'could not find rc file; returning defaults'
        ret = RcParams([(key, default) for key, (default, _) in
                        six.iteritems(defaultParams)
                        if key not in _all_deprecated])
        warnings.warn(message)
        return ret

    return rc_params_from_file(fname, fail_on_error)


URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')


def is_url(filename):
    """Return True if string is an http, ftp, or file URL path."""
    return URL_REGEX.match(filename) is not None


def _url_lines(f):
    # Compatibility for urlopen in python 3, which yields bytes.
    for line in f:
        yield line.decode('utf8')


@contextlib.contextmanager
def _open_file_or_url(fname):
    if is_url(fname):
        f = urlopen(fname)
        yield _url_lines(f)
        f.close()
    else:
        with open(fname) as f:
            yield f


_error_details_fmt = 'line #%d\n\t"%s"\n\tin file "%s"'


def _rc_params_in_file(fname, fail_on_error=False):
    """Return :class:`matplotlib.RcParams` from the contents of the given file.

    Unlike `rc_params_from_file`, the configuration class only contains the
    parameters specified in the file (i.e. default values are not filled in).
    """
    cnt = 0
    rc_temp = {}
    with _open_file_or_url(fname) as fd:
        for line in fd:
            cnt += 1
            strippedline = line.split('#', 1)[0].strip()
            if not strippedline: continue
            tup = strippedline.split(':', 1)
            if len(tup) != 2:
                error_details = _error_details_fmt % (cnt, line, fname)
                warnings.warn('Illegal %s' % error_details)
                continue
            key, val = tup
            key = key.strip()
            val = val.strip()
            if key in rc_temp:
                warnings.warn('Duplicate key in file "%s", line #%d' % \
                              (fname, cnt))
            rc_temp[key] = (val, line, cnt)

    config = RcParams()

    for key in ('verbose.level', 'verbose.fileo'):
        if key in rc_temp:
            val, line, cnt = rc_temp.pop(key)
            if fail_on_error:
                config[key] = val # try to convert to proper type or raise
            else:
                try:
                    config[key] = val # try to convert to proper type or skip
                except Exception as msg:
                    error_details = _error_details_fmt % (cnt, line, fname)
                    warnings.warn('Bad val "%s" on %s\n\t%s' %
                                  (val, error_details, msg))

    for key, (val, line, cnt) in six.iteritems(rc_temp):
        if key in defaultParams:
            if fail_on_error:
                config[key] = val # try to convert to proper type or raise
            else:
                try:
                    config[key] = val # try to convert to proper type or skip
                except Exception as msg:
                    error_details = _error_details_fmt % (cnt, line, fname)
                    warnings.warn('Bad val "%s" on %s\n\t%s' %
                                  (val, error_details, msg))
        elif key in _deprecated_ignore_map:
            warnings.warn('%s is deprecated. Update your matplotlibrc to use '
                          '%s instead.'% (key, _deprecated_ignore_map[key]))

        else:
            print("""
Bad key "%s" on line %d in
%s.
You probably need to get an updated matplotlibrc file from
http://matplotlib.sf.net/_static/matplotlibrc or from the matplotlib source
distribution""" % (key, cnt, fname), file=sys.stderr)

    return config


def rc_params_from_file(fname, fail_on_error=False, use_default_template=True):
    """Return :class:`matplotlib.RcParams` from the contents of the given file.

    Parameters
    ----------
    fname : str
        Name of file parsed for matplotlib settings.
    fail_on_error : bool
        If True, raise an error when the parser fails to convert a parameter.
    use_default_template : bool
        If True, initialize with default parameters before updating with those
        in the given file. If False, the configuration class only contains the
        parameters specified in the file. (Useful for updating dicts.)
    """
    config_from_file = _rc_params_in_file(fname, fail_on_error)

    if not use_default_template:
        return config_from_file

    iter_params = six.iteritems(defaultParams)
    config = RcParams([(key, default) for key, (default, _) in iter_params
                                      if key not in _all_deprecated])
    config.update(config_from_file)

    verbose.set_level(config['verbose.level'])
    verbose.set_fileo(config['verbose.fileo'])

    if config['datapath'] is None:
        config['datapath'] = get_data_path()

    if not config['text.latex.preamble'] == ['']:
        verbose.report("""
*****************************************************************
You have the following UNSUPPORTED LaTeX preamble customizations:
%s
Please do not ask for support with these customizations active.
*****************************************************************
"""% '\n'.join(config['text.latex.preamble']), 'helpful')

    verbose.report('loaded rc file %s'%fname)

    return config


# this is the instance used by the matplotlib classes
rcParams = rc_params()

if rcParams['examples.directory']:
    # paths that are intended to be relative to matplotlib_fname()
    # are allowed for the examples.directory parameter.
    # However, we will need to fully qualify the path because
    # Sphinx requires absolute paths.
    if not os.path.isabs(rcParams['examples.directory']):
        _basedir, _fname = os.path.split(matplotlib_fname())
        # Sometimes matplotlib_fname() can return relative paths,
        # Also, using realpath() guarentees that Sphinx will use
        # the same path that matplotlib sees (in case of weird symlinks).
        _basedir = os.path.realpath(_basedir)
        _fullpath = os.path.join(_basedir, rcParams['examples.directory'])
        rcParams['examples.directory'] = _fullpath

rcParamsOrig = rcParams.copy()

rcParamsDefault = RcParams([(key, default) for key, (default, converter) in
                            six.iteritems(defaultParams)
                            if key not in _all_deprecated])


rcParams['ps.usedistiller'] = checkdep_ps_distiller(
                      rcParams['ps.usedistiller'])
rcParams['text.usetex'] = checkdep_usetex(rcParams['text.usetex'])

if rcParams['axes.formatter.use_locale']:
    import locale
    locale.setlocale(locale.LC_ALL, '')


def rc(group, **kwargs):
    """
    Set the current rc params.  Group is the grouping for the rc, e.g.,
    for ``lines.linewidth`` the group is ``lines``, for
    ``axes.facecolor``, the group is ``axes``, and so on.  Group may
    also be a list or tuple of group names, e.g., (*xtick*, *ytick*).
    *kwargs* is a dictionary attribute name/value pairs, e.g.,::

      rc('lines', linewidth=2, color='r')

    sets the current rc params and is equivalent to::

      rcParams['lines.linewidth'] = 2
      rcParams['lines.color'] = 'r'

    The following aliases are available to save typing for interactive
    users:

    =====   =================
    Alias   Property
    =====   =================
    'lw'    'linewidth'
    'ls'    'linestyle'
    'c'     'color'
    'fc'    'facecolor'
    'ec'    'edgecolor'
    'mew'   'markeredgewidth'
    'aa'    'antialiased'
    =====   =================

    Thus you could abbreviate the above rc command as::

          rc('lines', lw=2, c='r')


    Note you can use python's kwargs dictionary facility to store
    dictionaries of default parameters.  e.g., you can customize the
    font rc as follows::

      font = {'family' : 'monospace',
              'weight' : 'bold',
              'size'   : 'larger'}

      rc('font', **font)  # pass in the font dict as kwargs

    This enables you to easily switch between several configurations.
    Use :func:`~matplotlib.pyplot.rcdefaults` to restore the default
    rc params after changes.
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
        for k, v in six.iteritems(kwargs):
            name = aliases.get(k) or k
            key = '%s.%s' % (g, name)
            try:
                rcParams[key] = v
            except KeyError:
                raise KeyError('Unrecognized key "%s" for group "%s" and name "%s"' %
                               (key, g, name))

def rcdefaults():
    """
    Restore the default rc params.  These are not the params loaded by
    the rc file, but mpl's internal params.  See rc_file_defaults for
    reloading the default params from the rc file
    """
    rcParams.clear()
    rcParams.update(rcParamsDefault)


def rc_file(fname):
    """
    Update rc params from file.
    """
    rcParams.update(rc_params_from_file(fname))


class rc_context(object):
    """
    Return a context manager for managing rc settings.

    This allows one to do::

        with mpl.rc_context(fname='screen.rc'):
            plt.plot(x, a)
            with mpl.rc_context(fname='print.rc'):
                plt.plot(x, b)
            plt.plot(x, c)

    The 'a' vs 'x' and 'c' vs 'x' plots would have settings from
    'screen.rc', while the 'b' vs 'x' plot would have settings from
    'print.rc'.

    A dictionary can also be passed to the context manager::

        with mpl.rc_context(rc={'text.usetex': True}, fname='screen.rc'):
            plt.plot(x, a)

    The 'rc' dictionary takes precedence over the settings loaded from
    'fname'.  Passing a dictionary only is also valid.
    """

    def __init__(self, rc=None, fname=None):
        self.rcdict = rc
        self.fname = fname
        self._rcparams = rcParams.copy()
        try:
            if self.fname:
                rc_file(self.fname)
            if self.rcdict:
                rcParams.update(self.rcdict)
        except:
            # if anything goes wrong, revert rc parameters and re-raise
            rcParams.clear()
            rcParams.update(self._rcparams)
            raise

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        rcParams.update(self._rcparams)


def rc_file_defaults():
    """
    Restore the default rc params from the original matplotlib rc that
    was loaded
    """
    rcParams.update(rcParamsOrig)

_use_error_msg = """ This call to matplotlib.use() has no effect
because the backend has already been chosen;
matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
or matplotlib.backends is imported for the first time.
"""

def use(arg, warn=True, force=False):
    """
    Set the matplotlib backend to one of the known backends.

    The argument is case-insensitive. *warn* specifies whether a
    warning should be issued if a backend has already been set up.
    *force* is an **experimental** flag that tells matplotlib to
    attempt to initialize a new backend by reloading the backend
    module.

    .. note::

        This function must be called *before* importing pyplot for
        the first time; or, if you are not using pyplot, it must be called
        before importing matplotlib.backends.  If warn is True, a warning
        is issued if you try and call this after pylab or pyplot have been
        loaded.  In certain black magic use cases, e.g.
        :func:`pyplot.switch_backend`, we are doing the reloading necessary to
        make the backend switch work (in some cases, e.g., pure image
        backends) so one can set warn=False to suppress the warnings.

    To find out which backend is currently set, see
    :func:`matplotlib.get_backend`.

    """
    # Lets determine the proper backend name first
    if arg.startswith('module://'):
        name = arg
    else:
        # Lowercase only non-module backend names (modules are case-sensitive)
        arg = arg.lower()
        name = validate_backend(arg)

    # Check if we've already set up a backend
    if 'matplotlib.backends' in sys.modules:
        # Warn only if called with a different name
        if (rcParams['backend'] != name) and warn:
            warnings.warn(_use_error_msg)

        # Unless we've been told to force it, just return
        if not force:
            return
        need_reload = True
    else:
        need_reload = False

    # Store the backend name
    rcParams['backend'] = name

    # If needed we reload here because a lot of setup code is triggered on
    # module import. See backends/__init__.py for more detail.
    if need_reload:
        reload(sys.modules['matplotlib.backends'])

def get_backend():
    """Return the name of the current backend."""
    return rcParams['backend']

def interactive(b):
    """
    Set interactive mode to boolean b.

    If b is True, then draw after every plotting command, e.g., after xlabel
    """
    rcParams['interactive'] = b

def is_interactive():
    'Return true if plot mode is interactive'
    return rcParams['interactive']

def tk_window_focus():
    """Return true if focus maintenance under TkAgg on win32 is on.
     This currently works only for python.exe and IPython.exe.
     Both IDLE and Pythonwin.exe fail badly when tk_window_focus is on."""
    if rcParams['backend'] != 'TkAgg':
        return False
    return rcParams['tk.window_focus']

# Now allow command line to override

# Allow command line access to the backend with -d (MATLAB compatible
# flag)

for s in sys.argv[1:]:
    # cast to str because we are using unicode_literals,
    # and argv is always str
    if s.startswith(str('-d')) and len(s) > 2:  # look for a -d flag
        try:
            use(s[2:])
        except (KeyError, ValueError):
            pass
        # we don't want to assume all -d flags are backends, e.g., -debug

default_test_modules = [
    'matplotlib.tests.test_agg',
    'matplotlib.tests.test_animation',
    'matplotlib.tests.test_arrow_patches',
    'matplotlib.tests.test_artist',
    'matplotlib.tests.test_axes',
    'matplotlib.tests.test_backend_bases',
    'matplotlib.tests.test_backend_pdf',
    'matplotlib.tests.test_backend_pgf',
    'matplotlib.tests.test_backend_ps',
    'matplotlib.tests.test_backend_qt4',
    'matplotlib.tests.test_backend_svg',
    'matplotlib.tests.test_basic',
    'matplotlib.tests.test_bbox_tight',
    'matplotlib.tests.test_cbook',
    'matplotlib.tests.test_coding_standards',
    'matplotlib.tests.test_collections',
    'matplotlib.tests.test_colorbar',
    'matplotlib.tests.test_colors',
    'matplotlib.tests.test_compare_images',
    'matplotlib.tests.test_contour',
    'matplotlib.tests.test_dates',
    'matplotlib.tests.test_delaunay',
    'matplotlib.tests.test_figure',
    'matplotlib.tests.test_font_manager',
    'matplotlib.tests.test_gridspec',
    'matplotlib.tests.test_image',
    'matplotlib.tests.test_legend',
    'matplotlib.tests.test_lines',
    'matplotlib.tests.test_mathtext',
    'matplotlib.tests.test_mlab',
    'matplotlib.tests.test_patches',
    'matplotlib.tests.test_path',
    'matplotlib.tests.test_patheffects',
    'matplotlib.tests.test_pickle',
    'matplotlib.tests.test_png',
    'matplotlib.tests.test_quiver',
    'matplotlib.tests.test_rcparams',
    'matplotlib.tests.test_scale',
    'matplotlib.tests.test_simplification',
    'matplotlib.tests.test_spines',
    'matplotlib.tests.test_streamplot',
    'matplotlib.tests.test_style',
    'matplotlib.tests.test_subplots',
    'matplotlib.tests.test_table',
    'matplotlib.tests.test_text',
    'matplotlib.tests.test_ticker',
    'matplotlib.tests.test_tightlayout',
    'matplotlib.tests.test_transforms',
    'matplotlib.tests.test_triangulation',
    'mpl_toolkits.tests.test_mplot3d',
    'mpl_toolkits.tests.test_axes_grid1',
    ]


def test(verbosity=1):
    """run the matplotlib test suite"""
    old_backend = rcParams['backend']
    try:
        use('agg')
        import nose
        import nose.plugins.builtin
        from .testing.noseclasses import KnownFailure
        from nose.plugins.manager import PluginManager
        from nose.plugins import multiprocess

        # store the old values before overriding
        plugins = []
        plugins.append( KnownFailure() )
        plugins.extend( [plugin() for plugin in nose.plugins.builtin.plugins] )

        manager = PluginManager(plugins=plugins)
        config = nose.config.Config(verbosity=verbosity, plugins=manager)

        # Nose doesn't automatically instantiate all of the plugins in the
        # child processes, so we have to provide the multiprocess plugin with
        # a list.
        multiprocess._instantiate_plugins = [KnownFailure]

        success = nose.run(
            defaultTest=default_test_modules,
            config=config,
        )
    finally:
        if old_backend.lower() != 'agg':
            use(old_backend)

    return success

test.__test__ = False # nose: this function is not a test

verbose.report('matplotlib version %s'%__version__)
verbose.report('verbose.level %s'%verbose.level)
verbose.report('interactive is %s'%is_interactive())
verbose.report('platform is %s'%sys.platform)
verbose.report('loaded modules: %s'%six.iterkeys(sys.modules), 'debug')
