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
        defines the :class:`~matplotlib.axes.Axes` class.  Most pyplot
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
        initialized by code which may be overridden by a matplotlibrc
        file.

    :func:`~matplotlib.rc`
        a function for setting groups of rcParams values

    :func:`~matplotlib.use`
        a function for setting the matplotlib backend.  If used, this
        function must be called immediately after importing matplotlib
        for the first time.  In particular, it must be called
        **before** importing pyplot (if pyplot is imported).

matplotlib was initially written by John D. Hunter (1968-2012) and is now
developed and maintained by a host of others.

Occasionally the internal documentation (python docstrings) will refer
to MATLAB&reg;, a registered trademark of The MathWorks, Inc.

"""
# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for existing editable installs.

import sys
if sys.version_info < (3, 5):  # noqa: E402
    raise ImportError("""
Matplotlib 3.0+ does not support Python 2.x, 3.0, 3.1, 3.2, 3.3, or 3.4.
Beginning with Matplotlib 3.0, Python 3.5 and above is required.

See Matplotlib `INSTALL.rst` file for more information:

    https://github.com/matplotlib/matplotlib/blob/master/INSTALL.rst

""")

import atexit
from collections.abc import MutableMapping
import contextlib
import distutils.version
import functools
import io
import importlib
import inspect
from inspect import Parameter
import locale
import logging
import os
from pathlib import Path
import pprint
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import warnings

# cbook must import matplotlib only within function
# definitions, so it is safe to import from it here.
from . import cbook, rcsetup
from matplotlib.cbook import (
    MatplotlibDeprecationWarning, dedent, get_label, sanitize_sequence)
from matplotlib.cbook import mplDeprecation  # deprecated
from matplotlib.rcsetup import defaultParams, validate_backend, cycler

import numpy

# Get the version from the _version.py versioneer file. For a git checkout,
# this is computed based on the number of commits since the last tag.
from ._version import get_versions
__version__ = str(get_versions()['version'])
del get_versions

_log = logging.getLogger(__name__)

__version__numpy__ = '1.10.0'  # minimum required numpy version

__bibtex__ = r"""@Article{Hunter:2007,
  Author    = {Hunter, J. D.},
  Title     = {Matplotlib: A 2D graphics environment},
  Journal   = {Computing In Science \& Engineering},
  Volume    = {9},
  Number    = {3},
  Pages     = {90--95},
  abstract  = {Matplotlib is a 2D graphics package used for Python
  for application development, interactive scripting, and
  publication-quality image generation across user
  interfaces and operating systems.},
  publisher = {IEEE COMPUTER SOC},
  year      = 2007
}"""


def compare_versions(a, b):
    "return True if a is greater than or equal to b"
    if isinstance(a, bytes):
        cbook.warn_deprecated(
            "3.0", "compare_version arguments should be strs.")
        a = a.decode('ascii')
    if isinstance(b, bytes):
        cbook.warn_deprecated(
            "3.0", "compare_version arguments should be strs.")
        b = b.decode('ascii')
    if a:
        a = distutils.version.LooseVersion(a)
        b = distutils.version.LooseVersion(b)
        return a >= b
    else:
        return False


try:
    import dateutil
except ImportError:
    raise ImportError("Matplotlib requires dateutil")


try:
    import pyparsing
except ImportError:
    raise ImportError("Matplotlib requires pyparsing")
else:
    if not compare_versions(pyparsing.__version__, '2.0.1'):
        raise ImportError(
            "Matplotlib requires pyparsing>=2.0.1; you have %s"
            % pyparsing.__version__)


if not compare_versions(numpy.__version__, __version__numpy__):
    raise ImportError(
        "Matplotlib requires numpy>=%s; you have %s" % (
            __version__numpy__, numpy.__version__))


if not hasattr(sys, 'argv'):  # for modpython
    sys.argv = ['modpython']


_verbose_msg = """\
matplotlib.verbose is deprecated;
Command line argument --verbose-LEVEL is deprecated.
This functionality is now provided by the standard
python logging library.  To get more (or less) logging output:
    import logging
    logger = logging.getLogger('matplotlib')
    logger.set_level(logging.INFO)"""


def _set_logger_verbose_level(level_str='silent', file_str='sys.stdout'):
    """
    Use a --verbose-LEVEL level to set the logging level:

    """
    levelmap = {'silent': logging.WARNING, 'helpful': logging.INFO,
                'debug': logging.DEBUG, 'debug-annoying': logging.DEBUG,
                'info': logging.INFO, 'warning': logging.WARNING}
    # Check that current state of logger isn't already more verbose
    # than the requested level.  If it is more verbose, then leave more
    # verbose.
    newlev = levelmap[level_str]
    oldlev = _log.getEffectiveLevel()
    if newlev < oldlev:
        _log.setLevel(newlev)
        std = {
            'sys.stdout': sys.stdout,
            'sys.stderr': sys.stderr,
        }
        if file_str in std:
            fileo = std[file_str]
        else:
            fileo = sys.stdout
            try:
                fileo = open(file_str, 'w')
                # if this fails, we will just write to stdout
            except IOError:
                warnings.warn('could not open log file "{0}"'
                              'for writing.  Check your '
                              'matplotlibrc'.format(file_str))
        console = logging.StreamHandler(fileo)
        console.setLevel(newlev)
        _log.addHandler(console)


def _parse_commandline():
    """
    Check for --verbose-LEVEL type command line arguments and
    set logging level appropriately.
    """

    levels = ('silent', 'helpful', 'debug', 'debug-annoying',
              'info', 'warning')

    for arg in sys.argv[1:]:
        if arg.startswith('--verbose-'):
            level_str = arg[10:]
            # If it doesn't match one of ours, then don't even
            # bother noting it, we are just a 3rd-party library
            # to somebody else's script.
            if level_str in levels:
                _set_logger_verbose_level(level_str)

_parse_commandline()


class Verbose(object):
    """
    A class to handle reporting.  Set the fileo attribute to any file
    instance to handle the output.  Default is sys.stdout
    """
    levels = ('silent', 'helpful', 'debug', 'debug-annoying')
    vald = {level: i for i, level in enumerate(levels)}

    # parse the verbosity from the command line; flags look like
    # --verbose-silent or --verbose-helpful
    _commandLineVerbose = None

    for arg in sys.argv[1:]:
        if not arg.startswith('--verbose-'):
            continue
        level_str = arg[10:]
        # If it doesn't match one of ours, then don't even
        # bother noting it, we are just a 3rd-party library
        # to somebody else's script.
        if level_str in levels:
            _commandLineVerbose = level_str

    @cbook.deprecated("2.2", message=_verbose_msg)
    def __init__(self):
        self.set_level('silent')
        self.fileo = sys.stdout

    @cbook.deprecated("2.2", message=_verbose_msg)
    def set_level(self, level):
        'set the verbosity to one of the Verbose.levels strings'

        if self._commandLineVerbose is not None:
            level = self._commandLineVerbose
        if level not in self.levels:
            warnings.warn('matplotlib: unrecognized --verbose-* string "%s".'
                          ' Legal values are %s' % (level, self.levels))
        else:
            self.level = level

    @cbook.deprecated("2.2", message=_verbose_msg)
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
                raise ValueError('Verbose object could not open log file "{0}"'
                                 ' for writing.\nCheck your matplotlibrc '
                                 'verbose.fileo setting'.format(fname))
            else:
                self.fileo = fileo

    @cbook.deprecated("2.2", message=_verbose_msg)
    def report(self, s, level='helpful'):
        """
        print message s to self.fileo if self.level>=level.  Return
        value indicates whether a message was issued

        """
        if self.ge(level):
            print(s, file=self.fileo)
            return True
        return False

    @cbook.deprecated("2.2", message=_verbose_msg)
    def wrap(self, fmt, func, level='helpful', always=True):
        """
        return a callable function that wraps func and reports it
        output through the verbose handler if current verbosity level
        is higher than level

        if always is True, the report will occur on every function
        call; otherwise only on the first time the function is called
        """
        assert callable(func)

        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)

            if (always or not wrapper._spoke):
                spoke = self.report(fmt % ret, level)
                if not wrapper._spoke:
                    wrapper._spoke = spoke
            return ret
        wrapper._spoke = False
        wrapper.__doc__ = func.__doc__
        return wrapper

    @cbook.deprecated("2.2", message=_verbose_msg)
    def ge(self, level):
        'return true if self.level is >= level'
        return self.vald[self.level] >= self.vald[level]


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    verbose = Verbose()


def _logged_cached(fmt, func=None):
    """
    Decorator that logs a function's return value, and memoizes that value.

    After ::

        @_logged_cached(fmt)
        def func(): ...

    the first call to *func* will log its return value at the DEBUG level using
    %-format string *fmt*, and memoize it; later calls to *func* will directly
    return that value.
    """
    if func is None:  # Return the actual decorator.
        return functools.partial(_logged_cached, fmt)

    called = False
    ret = None

    @functools.wraps(func)
    def wrapper():
        nonlocal called, ret
        if not called:
            ret = func()
            called = True
            _log.debug(fmt, ret)
        return ret

    return wrapper


def checkdep_dvipng():
    try:
        s = subprocess.Popen(['dvipng', '-version'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, stderr = s.communicate()
        line = stdout.decode('ascii').split('\n')[1]
        v = line.split()[-1]
        return v
    except (IndexError, ValueError, OSError):
        return None


def checkdep_ghostscript():
    if checkdep_ghostscript.executable is None:
        if sys.platform == 'win32':
            # mgs is the name in miktex
            gs_execs = ['gswin32c', 'gswin64c', 'mgs', 'gs']
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
                    checkdep_ghostscript.executable = gs_exec
                    checkdep_ghostscript.version = v
            except (IndexError, ValueError, OSError):
                pass
    return checkdep_ghostscript.executable, checkdep_ghostscript.version
checkdep_ghostscript.executable = None
checkdep_ghostscript.version = None


def checkdep_pdftops():
    try:
        s = subprocess.Popen(['pdftops', '-v'], stdout=subprocess.PIPE,
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
    if checkdep_inkscape.version is None:
        try:
            s = subprocess.Popen(['inkscape', '-V'],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            stdout, stderr = s.communicate()
            lines = stdout.decode('ascii').split('\n')
            for line in lines:
                if 'Inkscape' in line:
                    v = line.split()[1]
                    break
            checkdep_inkscape.version = v
        except (IndexError, ValueError, UnboundLocalError, OSError):
            pass
    return checkdep_inkscape.version
checkdep_inkscape.version = None


def checkdep_ps_distiller(s):
    if not s:
        return False

    flag = True
    gs_req = '8.60'
    gs_exec, gs_v = checkdep_ghostscript()
    if not compare_versions(gs_v, gs_req):
        flag = False
        warnings.warn(('matplotlibrc ps.usedistiller option can not be used '
                       'unless ghostscript-%s or later is installed on your '
                       'system') % gs_req)

    if s == 'xpdf':
        pdftops_req = '3.0'
        pdftops_req_alt = '0.9'  # poppler version numbers, ugh
        pdftops_v = checkdep_pdftops()
        if compare_versions(pdftops_v, pdftops_req):
            pass
        elif (compare_versions(pdftops_v, pdftops_req_alt) and not
              compare_versions(pdftops_v, '1.0')):
            pass
        else:
            flag = False
            warnings.warn(('matplotlibrc ps.usedistiller can not be set to '
                           'xpdf unless xpdf-%s or later is installed on '
                           'your system') % pdftops_req)

    if flag:
        return s
    else:
        return False


def checkdep_usetex(s):
    if not s:
        return False

    gs_req = '8.60'
    dvipng_req = '1.6'
    flag = True

    if shutil.which("tex") is None:
        flag = False
        warnings.warn('matplotlibrc text.usetex option can not be used unless '
                      'TeX is installed on your system')

    dvipng_v = checkdep_dvipng()
    if not compare_versions(dvipng_v, dvipng_req):
        flag = False
        warnings.warn('matplotlibrc text.usetex can not be used with *Agg '
                      'backend unless dvipng-%s or later is installed on '
                      'your system' % dvipng_req)

    gs_exec, gs_v = checkdep_ghostscript()
    if not compare_versions(gs_v, gs_req):
        flag = False
        warnings.warn('matplotlibrc text.usetex can not be used unless '
                      'ghostscript-%s or later is installed on your system'
                      % gs_req)

    return flag


@_logged_cached('$HOME=%s')
def get_home():
    """
    Return the user's home directory.

    If the user's home directory cannot be found, return None.
    """
    try:
        return str(Path.home())
    except Exception:
        return None


def _create_tmp_config_dir():
    """
    If the config directory can not be created, create a temporary directory.
    """
    configdir = os.environ['MPLCONFIGDIR'] = (
        tempfile.mkdtemp(prefix='matplotlib-'))
    atexit.register(shutil.rmtree, configdir)
    return configdir


def _get_xdg_config_dir():
    """
    Returns the XDG configuration directory, according to the `XDG
    base directory spec
    <http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html>`_.
    """
    return (os.environ.get('XDG_CONFIG_HOME')
            or (str(Path(get_home(), ".config"))
                if get_home()
                else None))


def _get_xdg_cache_dir():
    """
    Returns the XDG cache directory, according to the `XDG
    base directory spec
    <http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html>`_.
    """
    return (os.environ.get('XDG_CACHE_HOME')
            or (str(Path(get_home(), ".cache"))
                if get_home()
                else None))


def _get_config_or_cache_dir(xdg_base):
    configdir = os.environ.get('MPLCONFIGDIR')
    if configdir:
        configdir = Path(configdir).resolve()
    elif sys.platform.startswith(('linux', 'freebsd')) and xdg_base:
        configdir = Path(xdg_base, "matplotlib")
    elif get_home():
        configdir = Path(get_home(), ".matplotlib")
    else:
        configdir = None

    if configdir:
        try:
            configdir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        else:
            if os.access(str(configdir), os.W_OK) and configdir.is_dir():
                return str(configdir)

    return _create_tmp_config_dir()


@_logged_cached('CONFIGDIR=%s')
def get_configdir():
    """
    Return the string representing the configuration directory.

    The directory is chosen as follows:

    1. If the MPLCONFIGDIR environment variable is supplied, choose that.
    2a. On Linux, follow the XDG specification and look first in
        `$XDG_CONFIG_HOME`, if defined, or `$HOME/.config`.
    2b. On other platforms, choose `$HOME/.matplotlib`.
    3. If the chosen directory exists and is writable, use that as the
       configuration directory.
    4. If possible, create a temporary directory, and use it as the
       configuration directory.
    5. A writable directory could not be found or created; return None.
    """
    return _get_config_or_cache_dir(_get_xdg_config_dir())


@_logged_cached('CACHEDIR=%s')
def get_cachedir():
    """
    Return the location of the cache directory.

    The procedure used to find the directory is the same as for
    _get_config_dir, except using `$XDG_CACHE_HOME`/`~/.cache` instead.
    """
    return _get_config_or_cache_dir(_get_xdg_cache_dir())


def _get_data_path():
    'get the path to matplotlib data'

    if 'MATPLOTLIBDATA' in os.environ:
        path = os.environ['MATPLOTLIBDATA']
        if not os.path.isdir(path):
            raise RuntimeError('Path in environment MATPLOTLIBDATA not a '
                               'directory')
        return path

    def get_candidate_paths():
        yield Path(__file__).with_name('mpl-data')
        # setuptools' namespace_packages may highjack this init file
        # so need to try something known to be in Matplotlib, not basemap.
        import matplotlib.afm
        yield Path(matplotlib.afm.__file__).with_name('mpl-data')
        # py2exe zips pure python, so still need special check.
        if getattr(sys, 'frozen', None):
            yield Path(sys.executable).with_name('mpl-data')
            # Try again assuming we need to step up one more directory.
            yield Path(sys.executable).parent.with_name('mpl-data')
            # Try again assuming sys.path[0] is a dir not a exe.
            yield Path(sys.path[0]) / 'mpl-data'

    for path in get_candidate_paths():
        if path.is_dir():
            return str(path)

    raise RuntimeError('Could not find the matplotlib data files')


@_logged_cached('matplotlib data path: %s')
def get_data_path():
    if defaultParams['datapath'][0] is None:
        defaultParams['datapath'][0] = _get_data_path()
    return defaultParams['datapath'][0]


def get_py2exe_datafiles():
    data_path = Path(get_data_path())
    d = {}
    for path in filter(Path.is_file, data_path.glob("**/*")):
        (d.setdefault(str(path.parent.relative_to(data_path.parent)), [])
         .append(str(path)))
    return list(d.items())


def matplotlib_fname():
    """
    Get the location of the config file.

    The file location is determined in the following order

    - `$PWD/matplotlibrc`

    - `$MATPLOTLIBRC` if it is a file (or a named pipe, which can be created
      e.g. by process substitution)

    - `$MATPLOTLIBRC/matplotlibrc`

    - `$MPLCONFIGDIR/matplotlibrc`

    - On Linux,

          - `$XDG_CONFIG_HOME/matplotlib/matplotlibrc` (if
            $XDG_CONFIG_HOME is defined)

          - or `$HOME/.config/matplotlib/matplotlibrc` (if
            $XDG_CONFIG_HOME is not defined)

    - On other platforms,

         - `$HOME/.matplotlib/matplotlibrc` if `$HOME` is defined.

    - Lastly, it looks in `$MATPLOTLIBDATA/matplotlibrc` for a
      system-defined copy.
    """

    def gen_candidates():
        yield os.path.join(os.getcwd(), 'matplotlibrc')
        try:
            matplotlibrc = os.environ['MATPLOTLIBRC']
        except KeyError:
            pass
        else:
            yield matplotlibrc
            yield os.path.join(matplotlibrc, 'matplotlibrc')
        yield os.path.join(get_configdir(), 'matplotlibrc')
        yield os.path.join(get_data_path(), 'matplotlibrc')

    for fname in gen_candidates():
        if os.path.exists(fname):
            st_mode = os.stat(fname).st_mode
            if stat.S_ISREG(st_mode) or stat.S_ISFIFO(st_mode):
                break
    # Return first candidate that is a file, or last candidate if none is
    # valid (in that case, a warning is raised at startup by `rc_params`).
    return fname


# rcParams deprecated and automatically mapped to another key.
# Values are tuples of (version, new_name, f_old2new, f_new2old).
_deprecated_map = {}

# rcParams deprecated; some can manually be mapped to another key.
# Values are tuples of (version, new_name_or_None).
_deprecated_ignore_map = {
    'text.dvipnghack': ('2.1', None),
    'nbagg.transparent': ('2.2', 'figure.facecolor'),
    'plugins.directory': ('2.2', None),
    'pgf.debug': ('3.0', None),
}

# rcParams deprecated; can use None to suppress warnings; remain actually
# listed in the rcParams (not included in _all_deprecated).
# Values are typles of (version,)
_deprecated_remain_as_none = {
    'axes.hold': ('2.1',),
    'backend.qt4': ('2.2',),
    'backend.qt5': ('2.2',),
    'text.latex.unicode': ('3.0',),
}


_all_deprecated = {*_deprecated_map, *_deprecated_ignore_map}


class RcParams(MutableMapping, dict):

    """
    A dictionary object including validation

    validating functions are defined and associated with rc parameters in
    :mod:`matplotlib.rcsetup`
    """

    validate = {key: converter
                for key, (default, converter) in defaultParams.items()
                if key not in _all_deprecated}

    @property
    @cbook.deprecated("3.0")
    def msg_depr(self):
        return "%s is deprecated and replaced with %s; please use the latter."

    @property
    @cbook.deprecated("3.0")
    def msg_depr_ignore(self):
        return "%s is deprecated and ignored. Use %s instead."

    @property
    @cbook.deprecated("3.0")
    def msg_depr_set(self):
        return ("%s is deprecated. Please remove it from your matplotlibrc "
                "and/or style files.")

    @property
    @cbook.deprecated("3.0")
    def msg_obsolete(self):
        return ("%s is obsolete. Please remove it from your matplotlibrc "
                "and/or style files.")

    @property
    @cbook.deprecated("3.0")
    def msg_backend_obsolete(self):
        return ("The {} rcParam was deprecated in version 2.2.  In order to "
                "force the use of a specific Qt binding, either import that "
                "binding first, or set the QT_API environment variable.")

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        try:
            if key in _deprecated_map:
                version, alt_key, alt_val, inverse_alt = _deprecated_map[key]
                cbook.warn_deprecated(
                    version, key, obj_type="rcparam", alternative=alt_key)
                key = alt_key
                val = alt_val(val)
            elif key in _deprecated_remain_as_none and val is not None:
                version, = _deprecated_remain_as_none[key]
                addendum = ''
                if key.startswith('backend'):
                    addendum = (
                        "In order to force the use of a specific Qt binding, "
                        "either import that binding first, or set the QT_API "
                        "environment variable.")
                cbook.warn_deprecated(
                    "2.2", name=key, obj_type="rcparam", addendum=addendum)
            elif key in _deprecated_ignore_map:
                version, alt_key = _deprecated_ignore_map[key]
                cbook.warn_deprecated(
                    version, name=key, obj_type="rcparam", alternative=alt_key)
                return
            elif key == 'examples.directory':
                cbook.warn_deprecated(
                    "3.0", "{} is deprecated; in the future, examples will be "
                    "found relative to the 'datapath' directory.".format(key))
            elif key == 'backend':
                if val is rcsetup._auto_backend_sentinel:
                    if 'backend' in self:
                        return
            try:
                cval = self.validate[key](val)
            except ValueError as ve:
                raise ValueError("Key %s: %s" % (key, str(ve)))
            dict.__setitem__(self, key, cval)
        except KeyError:
            raise KeyError(
                '%s is not a valid rc parameter. See rcParams.keys() for a '
                'list of valid parameters.' % (key,))

    def __getitem__(self, key):
        if key in _deprecated_map:
            version, alt_key, alt_val, inverse_alt = _deprecated_map[key]
            cbook.warn_deprecated(
                version, key, obj_type="rcparam", alternative=alt_key)
            return inverse_alt(dict.__getitem__(self, alt_key))

        elif key in _deprecated_ignore_map:
            version, alt_key = _deprecated_ignore_map[key]
            cbook.warn_deprecated(
                version, key, obj_type="rcparam", alternative=alt_key)
            return dict.__getitem__(self, alt_key) if alt_key else None

        elif key == 'examples.directory':
            cbook.warn_deprecated(
                "3.0", "{} is deprecated; in the future, examples will be "
                "found relative to the 'datapath' directory.".format(key))

        elif key == "backend":
            val = dict.__getitem__(self, key)
            if val is rcsetup._auto_backend_sentinel:
                from matplotlib import pyplot as plt
                plt.switch_backend(rcsetup._auto_backend_sentinel)

        return dict.__getitem__(self, key)

    def __repr__(self):
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        repr_split = pprint.pformat(dict(self), indent=1,
                                    width=80 - indent).split('\n')
        repr_indented = ('\n' + ' ' * indent).join(repr_split)
        return '{}({})'.format(class_name, repr_indented)

    def __str__(self):
        return '\n'.join(map('{0[0]}: {0[1]}'.format, sorted(self.items())))

    def __iter__(self):
        """Yield sorted list of keys."""
        yield from sorted(dict.__iter__(self))

    def find_all(self, pattern):
        """
        Return the subset of this RcParams dictionary whose keys match,
        using :func:`re.search`, the given ``pattern``.

        .. note::

            Changes to the returned dictionary are *not* propagated to
            the parent RcParams dictionary.

        """
        pattern_re = re.compile(pattern)
        return RcParams((key, value)
                        for key, value in self.items()
                        if pattern_re.search(key))

    def copy(self):
        return {k: dict.__getitem__(self, k)
                for k in self}


def rc_params(fail_on_error=False):
    """Return a :class:`matplotlib.RcParams` instance from the
    default matplotlib rc file.
    """
    fname = matplotlib_fname()
    if not os.path.exists(fname):
        # this should never happen, default in mpl-data should always be found
        message = 'could not find rc file; returning defaults'
        ret = RcParams([(key, default) for key, (default, _) in
                        defaultParams.items()
                        if key not in _all_deprecated])
        warnings.warn(message)
        return ret

    return rc_params_from_file(fname, fail_on_error)


URL_REGEX = re.compile(r'http://|https://|ftp://|file://|file:\\')


def is_url(filename):
    """Return True if string is an http, ftp, or file URL path."""
    return URL_REGEX.match(filename) is not None


@contextlib.contextmanager
def _open_file_or_url(fname):
    if is_url(fname):
        with urllib.request.urlopen(fname) as f:
            yield (line.decode('utf-8') for line in f)
    else:
        fname = os.path.expanduser(fname)
        encoding = locale.getpreferredencoding(do_setlocale=False)
        if encoding is None:
            encoding = "utf-8"
        with open(fname, encoding=encoding) as f:
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
        try:
            for line in fd:
                cnt += 1
                strippedline = line.split('#', 1)[0].strip()
                if not strippedline:
                    continue
                tup = strippedline.split(':', 1)
                if len(tup) != 2:
                    error_details = _error_details_fmt % (cnt, line, fname)
                    warnings.warn('Illegal %s' % error_details)
                    continue
                key, val = tup
                key = key.strip()
                val = val.strip()
                if key in rc_temp:
                    warnings.warn('Duplicate key in file "%s", line #%d' %
                                  (fname, cnt))
                rc_temp[key] = (val, line, cnt)
        except UnicodeDecodeError:
            warnings.warn(
                ('Cannot decode configuration file %s with '
                 'encoding %s, check LANG and LC_* variables')
                % (fname, locale.getpreferredencoding(do_setlocale=False) or
                   'utf-8 (default)'))
            raise

    config = RcParams()

    for key in ('verbose.level', 'verbose.fileo'):
        if key in rc_temp:
            val, line, cnt = rc_temp.pop(key)
            if fail_on_error:
                config[key] = val  # try to convert to proper type or raise
            else:
                try:
                    config[key] = val  # try to convert to proper type or skip
                except Exception as msg:
                    error_details = _error_details_fmt % (cnt, line, fname)
                    warnings.warn('Bad val "%s" on %s\n\t%s' %
                                  (val, error_details, msg))

    for key, (val, line, cnt) in rc_temp.items():
        if key in defaultParams:
            if fail_on_error:
                config[key] = val  # try to convert to proper type or raise
            else:
                try:
                    config[key] = val  # try to convert to proper type or skip
                except Exception as msg:
                    error_details = _error_details_fmt % (cnt, line, fname)
                    warnings.warn('Bad val "%s" on %s\n\t%s' %
                                  (val, error_details, msg))
        elif key in _deprecated_ignore_map:
            version, alt_key = _deprecated_ignore_map[key]
            cbook.warn_deprecated(
                version, key, alternative=alt_key,
                addendum="Please update your matplotlibrc.")
        else:
            print("""
Bad key "%s" on line %d in
%s.
You probably need to get an updated matplotlibrc file from
http://github.com/matplotlib/matplotlib/blob/master/matplotlibrc.template
or from the matplotlib source distribution""" % (key, cnt, fname),
                  file=sys.stderr)

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

    iter_params = defaultParams.items()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
        config = RcParams([(key, default) for key, (default, _) in iter_params
                           if key not in _all_deprecated])
    config.update(config_from_file)

    if config['datapath'] is None:
        config['datapath'] = get_data_path()

    if "".join(config['text.latex.preamble']):
        _log.info("""
*****************************************************************
You have the following UNSUPPORTED LaTeX preamble customizations:
%s
Please do not ask for support with these customizations active.
*****************************************************************
""", '\n'.join(config['text.latex.preamble']))
    _log.debug('loaded rc file %s', fname)

    return config


# this is the instance used by the matplotlib classes
rcParams = rc_params()

# Don't trigger deprecation warning when just fetching.
if dict.__getitem__(rcParams, 'examples.directory'):
    # paths that are intended to be relative to matplotlib_fname()
    # are allowed for the examples.directory parameter.
    # However, we will need to fully qualify the path because
    # Sphinx requires absolute paths.
    if not os.path.isabs(rcParams['examples.directory']):
        _basedir, _fname = os.path.split(matplotlib_fname())
        # Sometimes matplotlib_fname() can return relative paths,
        # Also, using realpath() guarantees that Sphinx will use
        # the same path that matplotlib sees (in case of weird symlinks).
        _basedir = os.path.realpath(_basedir)
        _fullpath = os.path.join(_basedir, rcParams['examples.directory'])
        rcParams['examples.directory'] = _fullpath


with warnings.catch_warnings():
    warnings.simplefilter("ignore", MatplotlibDeprecationWarning)
    rcParamsOrig = RcParams(rcParams.copy())
    rcParamsDefault = RcParams([(key, default) for key, (default, converter) in
                                defaultParams.items()
                                if key not in _all_deprecated])

rcParams['ps.usedistiller'] = checkdep_ps_distiller(
                      rcParams['ps.usedistiller'])

rcParams['text.usetex'] = checkdep_usetex(rcParams['text.usetex'])

if rcParams['axes.formatter.use_locale']:
    locale.setlocale(locale.LC_ALL, '')


def rc(group, **kwargs):
    """
    Set the current rc params.  *group* is the grouping for the rc, e.g.,
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

    This enables you to easily switch between several configurations.  Use
    ``matplotlib.style.use('default')`` or :func:`~matplotlib.rcdefaults` to
    restore the default rc params after changes.
    """

    aliases = {
        'lw':  'linewidth',
        'ls':  'linestyle',
        'c':   'color',
        'fc':  'facecolor',
        'ec':  'edgecolor',
        'mew': 'markeredgewidth',
        'aa':  'antialiased',
        }

    if isinstance(group, str):
        group = (group,)
    for g in group:
        for k, v in kwargs.items():
            name = aliases.get(k) or k
            key = '%s.%s' % (g, name)
            try:
                rcParams[key] = v
            except KeyError:
                raise KeyError(('Unrecognized key "%s" for group "%s" and '
                                'name "%s"') % (key, g, name))


def rcdefaults():
    """
    Restore the rc params from Matplotlib's internal default style.

    Style-blacklisted rc params (defined in
    `matplotlib.style.core.STYLE_BLACKLIST`) are not updated.

    See Also
    --------
    rc_file_defaults :
        Restore the rc params from the rc file originally loaded by Matplotlib.
    matplotlib.style.use :
        Use a specific style file.  Call ``style.use('default')`` to restore
        the default style.
    """
    # Deprecation warnings were already handled when creating rcParamsDefault,
    # no need to reemit them here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mplDeprecation)
        from .style.core import STYLE_BLACKLIST
        rcParams.clear()
        rcParams.update({k: v for k, v in rcParamsDefault.items()
                         if k not in STYLE_BLACKLIST})


def rc_file_defaults():
    """
    Restore the rc params from the original rc file loaded by Matplotlib.

    Style-blacklisted rc params (defined in
    `matplotlib.style.core.STYLE_BLACKLIST`) are not updated.
    """
    # Deprecation warnings were already handled when creating rcParamsOrig, no
    # need to reemit them here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mplDeprecation)
        from .style.core import STYLE_BLACKLIST
        rcParams.update({k: rcParamsOrig[k] for k in rcParamsOrig
                         if k not in STYLE_BLACKLIST})


def rc_file(fname):
    """
    Update rc params from file.

    Style-blacklisted rc params (defined in
    `matplotlib.style.core.STYLE_BLACKLIST`) are not updated.
    """
    # Deprecation warnings were already handled in rc_params_from_file, no need
    # to reemit them here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mplDeprecation)
        from .style.core import STYLE_BLACKLIST
        rc_from_file = rc_params_from_file(fname)
        rcParams.update({k: rc_from_file[k] for k in rc_from_file
                         if k not in STYLE_BLACKLIST})


class rc_context:
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
    'fname'.  Passing a dictionary only is also valid. For example a
    common usage is::

        with mpl.rc_context(rc={'interactive': False}):
            fig, ax = plt.subplots()
            ax.plot(range(3), range(3))
            fig.savefig('A.png', format='png')
            plt.close(fig)
    """
    # While it may seem natural to implement rc_context using
    # contextlib.contextmanager, that would entail always calling the finally:
    # clause of the contextmanager (which restores the original rcs) including
    # during garbage collection; as a result, something like `plt.xkcd();
    # gc.collect()` would result in the style being lost (as `xkcd()` is
    # implemented on top of rc_context, and nothing is holding onto context
    # manager except possibly circular references.

    def __init__(self, rc=None, fname=None):
        self._orig = rcParams.copy()
        try:
            if fname:
                rc_file(fname)
            if rc:
                rcParams.update(rc)
        except Exception:
            self.__fallback()
            raise

    def __fallback(self):
        # If anything goes wrong, revert to the original rcs.
        updated_backend = self._orig['backend']
        dict.update(rcParams, self._orig)
        # except for the backend.  If the context block triggered resloving
        # the auto backend resolution keep that value around
        if self._orig['backend'] is rcsetup._auto_backend_sentinel:
            rcParams['backend'] = updated_backend

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.__fallback()


def use(arg, warn=True, force=False):
    """
    Set the matplotlib backend to one of the known backends.

    Parameters
    ----------
    arg : str
        The backend to switch to.  This can either be one of the
        'standard' backend names:

        - interactive backends:
          GTK3Agg, GTK3Cairo, MacOSX, nbAgg,
          Qt4Agg, Qt4Cairo, Qt5Agg, Qt5Cairo,
          TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo

        - non-interactive backends:
          agg, cairo, pdf, pgf, ps, svg, template

        or a string of the form: ``module://my.module.name``.

        Note: Standard backend names are case-insensitive here.

    warn : bool, optional
        If True, warn if this is called after pyplot has been imported
        and a backend is set up.

        defaults to True

    force : bool, optional
        If True, attempt to switch the backend.  This defaults to
        False.

    See Also
    --------
    :ref:`backends`
    matplotlib.get_backend
    """
    name = validate_backend(arg)

    # if setting back to the same thing, do nothing
    if (dict.__getitem__(rcParams, 'backend') == name):
        pass

    # Check if we have already imported pyplot and triggered
    # backend selection, do a bit more work
    elif 'matplotlib.pyplot' in sys.modules:
        # If we are here then the requested is different than the current.
        # If we are going to force the switch, never warn, else, if warn
        # is True, then direct users to `plt.switch_backend`
        if (not force) and warn:
            warnings.warn(
                ("matplotlib.pyplot as already been imported, "
                 "this call will have no effect."),
                stacklevel=2)

        # if we are going to force switching the backend, pull in
        # `switch_backend` from pyplot.  This will only happen if
        # pyplot is already imported.
        if force:
            from matplotlib.pyplot import switch_backend
            switch_backend(name)
    # Finally if pyplot is not imported update both rcParams and
    # rcDefaults so restoring the defaults later with rcdefaults
    # won't change the backend.  This is a bit of overkill as 'backend'
    # is already in style.core.STYLE_BLACKLIST, but better to be safe.
    else:
        rcParams['backend'] = rcParamsDefault['backend'] = name


if os.environ.get('MPLBACKEND'):
    rcParams['backend'] = os.environ.get('MPLBACKEND')


def get_backend():
    """
    Return the name of the current backend.

    See Also
    --------
    matplotlib.use
    """
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


default_test_modules = [
    'matplotlib.tests',
    'matplotlib.sphinxext.tests',
    'mpl_toolkits.tests',
]


def _init_tests():
    # CPython's faulthandler since v3.6 handles exceptions on Windows
    # https://bugs.python.org/issue23848 but until v3.6.4 it was printing
    # non-fatal exceptions https://bugs.python.org/issue30557
    import platform
    if not (sys.platform == 'win32' and
            (3, 6) < sys.version_info < (3, 6, 4) and
            platform.python_implementation() == 'CPython'):
        import faulthandler
        faulthandler.enable()

    # The version of FreeType to install locally for running the
    # tests.  This must match the value in `setupext.py`
    LOCAL_FREETYPE_VERSION = '2.6.1'

    from matplotlib import ft2font
    if (ft2font.__freetype_version__ != LOCAL_FREETYPE_VERSION or
        ft2font.__freetype_build_type__ != 'local'):
        warnings.warn(
            "Matplotlib is not built with the correct FreeType version to run "
            "tests.  Set local_freetype=True in setup.cfg and rebuild. "
            "Expect many image comparison failures below. "
            "Expected freetype version {0}. "
            "Found freetype version {1}. "
            "Freetype build type is {2}local".format(
                LOCAL_FREETYPE_VERSION,
                ft2font.__freetype_version__,
                "" if ft2font.__freetype_build_type__ == 'local' else "not "
            )
        )

    try:
        import pytest
    except ImportError:
        print("matplotlib.test requires pytest to run.")
        raise


def test(verbosity=None, coverage=False, switch_backend_warn=True,
         recursionlimit=0, **kwargs):
    """run the matplotlib test suite"""
    _init_tests()
    if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'tests')):
        raise ImportError("Matplotlib test data is not installed")

    old_backend = get_backend()
    old_recursionlimit = sys.getrecursionlimit()
    try:
        use('agg')
        if recursionlimit:
            sys.setrecursionlimit(recursionlimit)
        import pytest

        args = kwargs.pop('argv', [])
        provide_default_modules = True
        use_pyargs = True
        for arg in args:
            if any(arg.startswith(module_path)
                   for module_path in default_test_modules):
                provide_default_modules = False
                break
            if os.path.exists(arg):
                provide_default_modules = False
                use_pyargs = False
                break
        if use_pyargs:
            args += ['--pyargs']
        if provide_default_modules:
            args += default_test_modules

        if coverage:
            args += ['--cov']

        if verbosity:
            args += ['-' + 'v' * verbosity]

        retcode = pytest.main(args, **kwargs)
    finally:
        if old_backend.lower() != 'agg':
            use(old_backend, warn=switch_backend_warn)
        if recursionlimit:
            sys.setrecursionlimit(old_recursionlimit)

    return retcode


test.__test__ = False  # pytest: this function is not a test


def _replacer(data, key):
    """Either returns data[key] or passes data back. Also
    converts input data to a sequence as needed.
    """
    # if key isn't a string don't bother
    if not isinstance(key, str):
        return key
    # try to use __getitem__
    try:
        return sanitize_sequence(data[key])
    # key does not exist, silently fall back to key
    except KeyError:
        return key


_DATA_DOC_APPENDIX = """

.. note::
    In addition to the above described arguments, this function can take a
    **data** keyword argument. If such a **data** argument is given, the
    following arguments are replaced by **data[<arg>]**:

    {replaced}

    Objects passed as **data** must support item access (``data[<arg>]``) and
    membership test (``<arg> in data``).
"""


def _add_data_doc(docstring, replace_names, replace_all_args):
    """Add documentation for a *data* field to the given docstring.

    Parameters
    ----------
    docstring : str
        The input docstring.
    replace_names : list of strings or None
        The list of parameter names which arguments should be replaced by
        `data[name]`. If None, all arguments are replaced if they are
        included in `data`.
    replace_all_args : bool
        If True, all arguments in *args get replaced, even if they are not
        in replace_names.

    Returns
    -------
        The augmented docstring.
    """
    if docstring is None:
        docstring = ''
    else:
        docstring = dedent(docstring)
    _repl = ""
    if replace_names is None:
        _repl = "* All positional and all keyword arguments."
    else:
        if len(replace_names) != 0:
            _repl = "* All arguments with the following names: '{names}'."
        if replace_all_args:
            _repl += "\n    * All positional arguments."
        _repl = _repl.format(names="', '".join(sorted(replace_names)))
    return docstring + _DATA_DOC_APPENDIX.format(replaced=_repl)


def _preprocess_data(replace_names=None, replace_all_args=False,
                     label_namer=None, positional_parameter_names=None):
    """
    A decorator to add a 'data' kwarg to any a function.  The signature
    of the input function must include the ax argument at the first position ::

       def foo(ax, *args, **kwargs)

    so this is suitable for use with Axes methods.

    Parameters
    ----------
    replace_names : list of strings, optional, default: None
        The list of parameter names which arguments should be replaced by
        `data[name]`. If None, all arguments are replaced if they are
        included in `data`.
    replace_all_args : bool, default: False
        If True, all arguments in *args get replaced, even if they are not
        in replace_names.
    label_namer : string, optional, default: None
        The name of the parameter which argument should be used as label, if
        label is not set. If None, the label keyword argument is not set.
    positional_parameter_names : list of strings or callable, optional
        The full list of positional parameter names (excluding an explicit
        `ax`/'self' argument at the first place and including all possible
        positional parameter in `*args`), in the right order. Can also include
        all other keyword parameter. Only needed if the wrapped function does
        contain `*args` and (replace_names is not None or replace_all_args is
        False). If it is a callable, it will be called with the actual
        tuple of *args and the data and should return a list like
        above.
        NOTE: callables should only be used when the names and order of *args
        can only be determined at runtime. Please use list of names
        when the order and names of *args is clear before runtime!

    .. note:: decorator also converts MappingView input data to list.
    """
    if replace_names is not None:
        replace_names = set(replace_names)

    def param(func):
        sig = inspect.signature(func)
        _has_varargs = False
        _has_varkwargs = False
        _arg_names = []
        params = list(sig.parameters.values())
        for p in params:
            if p.kind is Parameter.VAR_POSITIONAL:
                _has_varargs = True
            elif p.kind is Parameter.VAR_KEYWORD:
                _has_varkwargs = True
            else:
                _arg_names.append(p.name)
        data_param = Parameter('data', Parameter.KEYWORD_ONLY, default=None)
        if _has_varkwargs:
            params.insert(-1, data_param)
        else:
            params.append(data_param)
        new_sig = sig.replace(parameters=params)
        # Import-time check: do we have enough information to replace *args?
        arg_names_at_runtime = False
        # there can't be any positional arguments behind *args and no
        # positional args can end up in **kwargs, so only *varargs make
        # problems.
        # http://stupidpythonideas.blogspot.de/2013/08/arguments-and-parameters.html
        if not _has_varargs:
            # all args are "named", so no problem
            # remove the first "ax" / self arg
            arg_names = _arg_names[1:]
        else:
            # Here we have "unnamed" variables and we need a way to determine
            # whether to replace a arg or not
            if replace_names is None:
                # all argnames should be replaced
                arg_names = None
            elif len(replace_names) == 0:
                # No argnames should be replaced
                arg_names = []
            elif len(_arg_names) > 1 and (positional_parameter_names is None):
                # we got no manual parameter names but more than an 'ax' ...
                if len(replace_names - set(_arg_names[1:])) == 0:
                    # all to be replaced arguments are in the list
                    arg_names = _arg_names[1:]
                else:
                    raise AssertionError(
                        "Got unknown 'replace_names' and wrapped function "
                        "{!r} uses '*args', need 'positional_parameter_names'"
                        .format(func.__name__))
            else:
                if positional_parameter_names is not None:
                    if callable(positional_parameter_names):
                        # determined by the function at runtime
                        arg_names_at_runtime = True
                        # so that we don't compute the label_pos at import time
                        arg_names = []
                    else:
                        arg_names = positional_parameter_names
                else:
                    if replace_all_args:
                        arg_names = []
                    else:
                        raise AssertionError(
                            "Got 'replace_names' and wrapped function {!r} "
                            "uses *args, need 'positional_parameter_names' or "
                            "'replace_all_args'".format(func.__name__))

        # compute the possible label_namer and label position in positional
        # arguments
        label_pos = 9999  # bigger than all "possible" argument lists
        label_namer_pos = 9999  # bigger than all "possible" argument lists
        if (label_namer and  # we actually want a label here ...
                arg_names and  # and we can determine a label in *args ...
                label_namer in arg_names):  # and it is in *args
            label_namer_pos = arg_names.index(label_namer)
            if "label" in arg_names:
                label_pos = arg_names.index("label")

        # Check the case we know a label_namer but we can't find it the
        # arg_names... Unfortunately the label_namer can be in **kwargs,
        # which we can't detect here and which results in a non-set label
        # which might surprise the user :-(
        if label_namer and not arg_names_at_runtime and not _has_varkwargs:
            if not arg_names:
                raise AssertionError(
                    "label_namer {!r} can't be found as the parameter without "
                    "'positional_parameter_names'".format(label_namer))
            elif label_namer not in arg_names:
                raise AssertionError(
                    "label_namer {!r} can't be found in the parameter names "
                    "(known argnames: %s).".format(label_namer, arg_names))
            else:
                # this is the case when the name is in arg_names
                pass

        @functools.wraps(func)
        def inner(ax, *args, data=None, **kwargs):
            # this is needed because we want to change these values if
            # arg_names_at_runtime==True, but python does not allow assigning
            # to a variable in a outer scope. So use some new local ones and
            # set them to the already computed values.
            _label_pos = label_pos
            _label_namer_pos = label_namer_pos
            _arg_names = arg_names

            label = None

            if data is None:  # data validation
                args = tuple(sanitize_sequence(a) for a in args)
            else:
                if arg_names_at_runtime:
                    # update the information about replace names and
                    # label position
                    _arg_names = positional_parameter_names(args, data)
                    if (label_namer and  # we actually want a label here ...
                            _arg_names and  # and we can find a label in *args
                            (label_namer in _arg_names)):  # and it is in *args
                        _label_namer_pos = _arg_names.index(label_namer)
                        if "label" in _arg_names:
                            _label_pos = arg_names.index("label")

                # save the current label_namer value so that it can be used as
                # a label
                if _label_namer_pos < len(args):
                    label = args[_label_namer_pos]
                else:
                    label = kwargs.get(label_namer, None)
                # ensure a string, as label can't be anything else
                if not isinstance(label, str):
                    label = None

                if (replace_names is None) or (replace_all_args is True):
                    # all should be replaced
                    args = tuple(_replacer(data, a) for
                                 j, a in enumerate(args))
                else:
                    # An arg is replaced if the arg_name of that position is
                    #   in replace_names ...
                    if len(_arg_names) < len(args):
                        raise RuntimeError(
                            "Got more args than function expects")
                    args = tuple(_replacer(data, a)
                                 if _arg_names[j] in replace_names else a
                                 for j, a in enumerate(args))

                if replace_names is None:
                    # replace all kwargs ...
                    kwargs = {k: _replacer(data, v) for k, v in kwargs.items()}
                else:
                    # ... or only if a kwarg of that name is in replace_names
                    kwargs = {
                        k: _replacer(data, v) if k in replace_names else v
                        for k, v in kwargs.items()}

            # replace the label if this func "wants" a label arg and the user
            # didn't set one. Note: if the user puts in "label=None", it does
            # *NOT* get replaced!
            user_supplied_label = (
                len(args) >= _label_pos or  # label is included in args
                'label' in kwargs  # ... or in kwargs
            )
            if label_namer and not user_supplied_label:
                if _label_namer_pos < len(args):
                    kwargs['label'] = get_label(args[_label_namer_pos], label)
                elif label_namer in kwargs:
                    kwargs['label'] = get_label(kwargs[label_namer], label)
                else:
                    warnings.warn(
                        "Tried to set a label via parameter %r in func %r but "
                        "couldn't find such an argument.\n"
                        "(This is a programming error, please report to "
                        "the Matplotlib list!)" % (label_namer, func.__name__),
                        RuntimeWarning, stacklevel=2)
            return func(ax, *args, **kwargs)

        inner.__doc__ = _add_data_doc(inner.__doc__,
                                      replace_names, replace_all_args)
        inner.__signature__ = new_sig
        return inner

    return param

_log.debug('matplotlib version %s', __version__)
_log.debug('interactive is %s', is_interactive())
_log.debug('platform is %s', sys.platform)
_log.debug('loaded modules: %s', list(sys.modules))
