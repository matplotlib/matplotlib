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
from collections import namedtuple
from collections.abc import MutableMapping
import contextlib
from distutils.version import LooseVersion
import functools
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
import subprocess
import tempfile

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

__bibtex__ = r"""@Article{Hunter:2007,
  Author    = {Hunter, J. D.},
  Title     = {Matplotlib: A 2D graphics environment},
  Journal   = {Computing in Science \& Engineering},
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


@cbook.deprecated("3.2")
def compare_versions(a, b):
    "Return whether version *a* is greater than or equal to version *b*."
    if isinstance(a, bytes):
        cbook.warn_deprecated(
            "3.0", message="compare_versions arguments should be strs.")
        a = a.decode('ascii')
    if isinstance(b, bytes):
        cbook.warn_deprecated(
            "3.0", message="compare_versions arguments should be strs.")
        b = b.decode('ascii')
    if a:
        return LooseVersion(a) >= LooseVersion(b)
    else:
        return False


def _check_versions():

    # Quickfix to ensure Microsoft Visual C++ redistributable
    # DLLs are loaded before importing kiwisolver
    from . import ft2font

    for modname, minver in [
            ("cycler", "0.10"),
            ("dateutil", "2.1"),
            ("kiwisolver", "1.0.1"),
            ("numpy", "1.11"),
            ("pyparsing", "2.0.1"),
    ]:
        module = importlib.import_module(modname)
        if LooseVersion(module.__version__) < minver:
            raise ImportError("Matplotlib requires {}>={}; you have {}"
                              .format(modname, minver, module.__version__))


_check_versions()


if not hasattr(sys, 'argv'):  # for modpython
    sys.argv = ['modpython']


# The decorator ensures this always returns the same handler (and it is only
# attached once).
@functools.lru_cache()
def _ensure_handler():
    """
    The first time this function is called, attach a `StreamHandler` using the
    same format as `logging.basicConfig` to the Matplotlib root logger.

    Return this handler every time this function is called.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    _log.addHandler(handler)
    return handler


def set_loglevel(level):
    """
    Sets the Matplotlib's root logger and root logger handler level, creating
    the handler if it does not exist yet.

    Typically, one should call ``set_loglevel("info")`` or
    ``set_loglevel("debug")`` to get additional debugging information.

    Parameters
    ----------
    level : {"notset", "debug", "info", "warning", "error", "critical"}
        The log level of the handler.

    Notes
    -----
    The first time this function is called, an additional handler is attached
    to Matplotlib's root handler; this handler is reused every time and this
    function simply manipulates the logger and handler's level.
    """
    _log.setLevel(level.upper())
    _ensure_handler().setLevel(level.upper())


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


_ExecInfo = namedtuple("_ExecInfo", "executable version")


class ExecutableNotFoundError(FileNotFoundError):
    """
    Error raised when an executable that Matplotlib optionally
    depends on can't be found.
    """
    pass


@functools.lru_cache()
def _get_executable_info(name):
    """
    Get the version of some executable that Matplotlib optionally depends on.

    .. warning:
       The list of executables that this function supports is set according to
       Matplotlib's internal needs, and may change without notice.

    Parameters
    ----------
    name : str
        The executable to query.  The following values are currently supported:
        "dvipng", "gs", "inkscape", "magick", "pdftops".  This list is subject
        to change without notice.

    Returns
    -------
    If the executable is found, a namedtuple with fields ``executable`` (`str`)
    and ``version`` (`distutils.version.LooseVersion`, or ``None`` if the
    version cannot be determined).

    Raises
    ------
    ExecutableNotFoundError
        If the executable is not found or older than the oldest version
        supported by Matplotlib.
    ValueError
        If the executable is not one that we know how to query.
    """

    def impl(args, regex, min_ver=None, ignore_exit_code=False):
        # Execute the subprocess specified by args; capture stdout and stderr.
        # Search for a regex match in the output; if the match succeeds, the
        # first group of the match is the version.
        # Return an _ExecInfo if the executable exists, and has a version of
        # at least min_ver (if set); else, raise ExecutableNotFoundError.
        try:
            output = subprocess.check_output(
                args, stderr=subprocess.STDOUT,
                universal_newlines=True, errors="replace")
        except subprocess.CalledProcessError as _cpe:
            if ignore_exit_code:
                output = _cpe.output
            else:
                raise ExecutableNotFoundError(str(_cpe)) from _cpe
        except OSError as _ose:
            raise ExecutableNotFoundError(str(_ose)) from _ose
        match = re.search(regex, output)
        if match:
            version = LooseVersion(match.group(1))
            if min_ver is not None and version < min_ver:
                raise ExecutableNotFoundError(
                    f"You have {args[0]} version {version} but the minimum "
                    f"version supported by Matplotlib is {min_ver}")
            return _ExecInfo(args[0], version)
        else:
            raise ExecutableNotFoundError(
                f"Failed to determine the version of {args[0]} from "
                f"{' '.join(args)}, which output {output}")

    if name == "dvipng":
        return impl(["dvipng", "-version"], "(?m)^dvipng(?: .*)? (.+)", "1.6")
    elif name == "gs":
        execs = (["gswin32c", "gswin64c", "mgs", "gs"]  # "mgs" for miktex.
                 if sys.platform == "win32" else
                 ["gs"])
        for e in execs:
            try:
                return impl([e, "--version"], "(.*)", "9")
            except ExecutableNotFoundError:
                pass
        message = "Failed to find a Ghostscript installation"
        raise ExecutableNotFoundError(message)
    elif name == "inkscape":
        info = impl(["inkscape", "-V"], "^Inkscape ([^ ]*)")
        if info and info.version >= "1.0":
            raise ExecutableNotFoundError(
                f"You have Inkscape version {info.version} but Matplotlib "
                f"only supports Inkscape<1.0")
        return info
    elif name == "magick":
        path = None
        if sys.platform == "win32":
            # Check the registry to avoid confusing ImageMagick's convert with
            # Windows's builtin convert.exe.
            import winreg
            binpath = ""
            for flag in [0, winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY]:
                try:
                    with winreg.OpenKeyEx(
                            winreg.HKEY_LOCAL_MACHINE,
                            r"Software\Imagemagick\Current",
                            0, winreg.KEY_QUERY_VALUE | flag) as hkey:
                        binpath = winreg.QueryValueEx(hkey, "BinPath")[0]
                except OSError:
                    pass
            if binpath:
                for name in ["convert.exe", "magick.exe"]:
                    candidate = Path(binpath, name)
                    if candidate.exists():
                        path = str(candidate)
                        break
        else:
            path = "convert"
        if path is None:
            raise ExecutableNotFoundError(
                "Failed to find an ImageMagick installation")
        return impl([path, "--version"], r"^Version: ImageMagick (\S*)")
    elif name == "pdftops":
        info = impl(["pdftops", "-v"], "^pdftops version (.*)",
                    ignore_exit_code=True)
        if info and not ("3.0" <= info.version
                         # poppler version numbers.
                         or "0.9" <= info.version <= "1.0"):
            raise ExecutableNotFoundError(
                f"You have pdftops version {info.version} but the minimum "
                f"version supported by Matplotlib is 3.0")
        return info
    else:
        raise ValueError("Unknown executable: {!r}".format(name))


@cbook.deprecated("3.1")
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


@cbook.deprecated("3.1")
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
                    if compare_versions(v, '9.0'):
                        checkdep_ghostscript.executable = gs_exec
                        checkdep_ghostscript.version = v
            except (IndexError, ValueError, OSError):
                pass
    return checkdep_ghostscript.executable, checkdep_ghostscript.version
checkdep_ghostscript.executable = None
checkdep_ghostscript.version = None


@cbook.deprecated("3.1")
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


@cbook.deprecated("3.1")
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


@cbook.deprecated("3.2")
def checkdep_ps_distiller(s):
    if not s:
        return False
    try:
        _get_executable_info("gs")
    except ExecutableNotFoundError:
        _log.warning(
            "Setting rcParams['ps.usedistiller'] requires ghostscript.")
        return False
    if s == "xpdf":
        try:
            _get_executable_info("pdftops")
        except ExecutableNotFoundError:
            _log.warning(
                "Setting rcParams['ps.usedistiller'] to 'xpdf' requires xpdf.")
            return False
    return s


def checkdep_usetex(s):
    if not s:
        return False
    if not shutil.which("tex"):
        _log.warning("usetex mode requires TeX.")
        return False
    try:
        _get_executable_info("dvipng")
    except ExecutableNotFoundError:
        _log.warning("usetex mode requires dvipng.")
        return False
    try:
        _get_executable_info("gs")
    except ExecutableNotFoundError:
        _log.warning("usetex mode requires ghostscript.")
        return False
    return True


@cbook.deprecated("3.2", alternative="os.path.expanduser('~')")
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


def _create_tmp_config_or_cache_dir():
    """
    If the config or cache directory cannot be created, create a temporary one.
    """
    configdir = os.environ['MPLCONFIGDIR'] = (
        tempfile.mkdtemp(prefix='matplotlib-'))
    atexit.register(shutil.rmtree, configdir)
    return configdir


def _get_xdg_config_dir():
    """
    Return the XDG configuration directory, according to the `XDG
    base directory spec
    <http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html>`_.
    """
    return os.environ.get('XDG_CONFIG_HOME') or str(Path.home() / ".config")


def _get_xdg_cache_dir():
    """
    Return the XDG cache directory, according to the `XDG
    base directory spec
    <http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html>`_.
    """
    return os.environ.get('XDG_CACHE_HOME') or str(Path.home() / ".cache")


def _get_config_or_cache_dir(xdg_base):
    configdir = os.environ.get('MPLCONFIGDIR')
    if configdir:
        configdir = Path(configdir).resolve()
    elif sys.platform.startswith(('linux', 'freebsd')) and xdg_base:
        configdir = Path(xdg_base, "matplotlib")
    else:
        configdir = Path.home() / ".matplotlib"
    try:
        configdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    else:
        if os.access(str(configdir), os.W_OK) and configdir.is_dir():
            return str(configdir)
    return _create_tmp_config_or_cache_dir()


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
    """Return the path to matplotlib data."""

    if 'MATPLOTLIBDATA' in os.environ:
        path = os.environ['MATPLOTLIBDATA']
        if not os.path.isdir(path):
            raise RuntimeError('Path in environment MATPLOTLIBDATA not a '
                               'directory')
        cbook.warn_deprecated(
            "3.1", name="MATPLOTLIBDATA", obj_type="environment variable")
        return path

    path = Path(__file__).with_name("mpl-data")
    if path.is_dir():
        return str(path)

    cbook.warn_deprecated(
        "3.2", message="Matplotlib installs where the data is not in the "
        "mpl-data subdirectory of the package are deprecated since %(since)s "
        "and support for them will be removed %(removal)s.")

    def get_candidate_paths():
        # setuptools' namespace_packages may hijack this init file
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


@cbook.deprecated("3.1")
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

    - ``$PWD/matplotlibrc``
    - ``$MATPLOTLIBRC`` if it is not a directory
    - ``$MATPLOTLIBRC/matplotlibrc``
    - ``$MPLCONFIGDIR/matplotlibrc``
    - On Linux,
        - ``$XDG_CONFIG_HOME/matplotlib/matplotlibrc`` (if ``$XDG_CONFIG_HOME``
          is defined)
        - or ``$HOME/.config/matplotlib/matplotlibrc`` (if ``$XDG_CONFIG_HOME``
          is not defined)
    - On other platforms,
      - ``$HOME/.matplotlib/matplotlibrc`` if ``$HOME`` is defined
    - Lastly, it looks in ``$MATPLOTLIBDATA/matplotlibrc``, which should always
      exist.
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
        if os.path.exists(fname) and not os.path.isdir(fname):
            return fname

    raise RuntimeError("Could not find matplotlibrc file; your Matplotlib "
                       "install is broken")


# rcParams deprecated and automatically mapped to another key.
# Values are tuples of (version, new_name, f_old2new, f_new2old).
_deprecated_map = {}

# rcParams deprecated; some can manually be mapped to another key.
# Values are tuples of (version, new_name_or_None).
_deprecated_ignore_map = {
    'pgf.debug': ('3.0', None),
}

# rcParams deprecated; can use None to suppress warnings; remain actually
# listed in the rcParams (not included in _all_deprecated).
# Values are tuples of (version,)
_deprecated_remain_as_none = {
    'text.latex.unicode': ('3.0',),
    'savefig.frameon': ('3.1',),
    'verbose.fileo': ('3.1',),
    'verbose.level': ('3.1',),
}


_all_deprecated = {*_deprecated_map, *_deprecated_ignore_map}


class RcParams(MutableMapping, dict):
    """
    A dictionary object including validation.

    Validating functions are defined and associated with rc parameters in
    :mod:`matplotlib.rcsetup`.

    See Also
    --------
    :ref:`customizing-with-matplotlibrc-files`
    """

    validate = {key: converter
                for key, (default, converter) in defaultParams.items()
                if key not in _all_deprecated}

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        try:
            if key in _deprecated_map:
                version, alt_key, alt_val, inverse_alt = _deprecated_map[key]
                cbook.warn_deprecated(
                    version, name=key, obj_type="rcparam", alternative=alt_key)
                key = alt_key
                val = alt_val(val)
            elif key in _deprecated_remain_as_none and val is not None:
                version, = _deprecated_remain_as_none[key]
                cbook.warn_deprecated(
                    version, name=key, obj_type="rcparam")
            elif key in _deprecated_ignore_map:
                version, alt_key = _deprecated_ignore_map[key]
                cbook.warn_deprecated(
                    version, name=key, obj_type="rcparam", alternative=alt_key)
                return
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
                f"{key} is not a valid rc parameter (see rcParams.keys() for "
                f"a list of valid parameters)")

    def __getitem__(self, key):
        if key in _deprecated_map:
            version, alt_key, alt_val, inverse_alt = _deprecated_map[key]
            cbook.warn_deprecated(
                version, name=key, obj_type="rcparam", alternative=alt_key)
            return inverse_alt(dict.__getitem__(self, alt_key))

        elif key in _deprecated_ignore_map:
            version, alt_key = _deprecated_ignore_map[key]
            cbook.warn_deprecated(
                version, name=key, obj_type="rcparam", alternative=alt_key)
            return dict.__getitem__(self, alt_key) if alt_key else None

        elif key == "backend":
            val = dict.__getitem__(self, key)
            if val is rcsetup._auto_backend_sentinel:
                from matplotlib import pyplot as plt
                plt.switch_backend(rcsetup._auto_backend_sentinel)

        return dict.__getitem__(self, key)

    def __repr__(self):
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        with cbook._suppress_matplotlib_deprecation_warning():
            repr_split = pprint.pformat(dict(self), indent=1,
                                        width=80 - indent).split('\n')
        repr_indented = ('\n' + ' ' * indent).join(repr_split)
        return '{}({})'.format(class_name, repr_indented)

    def __str__(self):
        return '\n'.join(map('{0[0]}: {0[1]}'.format, sorted(self.items())))

    def __iter__(self):
        """Yield sorted list of keys."""
        with cbook._suppress_matplotlib_deprecation_warning():
            yield from sorted(dict.__iter__(self))

    def __len__(self):
        return dict.__len__(self)

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
        return {k: dict.__getitem__(self, k) for k in self}


def rc_params(fail_on_error=False):
    """Construct a `RcParams` instance from the default Matplotlib rc file."""
    return rc_params_from_file(matplotlib_fname(), fail_on_error)


URL_REGEX = re.compile(r'^http://|^https://|^ftp://|^file:')


def is_url(filename):
    """Return True if string is an http, ftp, or file URL path."""
    return URL_REGEX.match(filename) is not None


@contextlib.contextmanager
def _open_file_or_url(fname):
    if not isinstance(fname, Path) and is_url(fname):
        import urllib.request
        with urllib.request.urlopen(fname) as f:
            yield (line.decode('utf-8') for line in f)
    else:
        fname = os.path.expanduser(fname)
        encoding = locale.getpreferredencoding(do_setlocale=False)
        if encoding is None:
            encoding = "utf-8"
        with open(fname, encoding=encoding) as f:
            yield f


def _rc_params_in_file(fname, fail_on_error=False):
    """
    Construct a `RcParams` instance from file *fname*.

    Unlike `rc_params_from_file`, the configuration class only contains the
    parameters specified in the file (i.e. default values are not filled in).
    """
    _error_details_fmt = 'line #%d\n\t"%s"\n\tin file "%s"'

    rc_temp = {}
    with _open_file_or_url(fname) as fd:
        try:
            for line_no, line in enumerate(fd, 1):
                strippedline = line.split('#', 1)[0].strip()
                if not strippedline:
                    continue
                tup = strippedline.split(':', 1)
                if len(tup) != 2:
                    error_details = _error_details_fmt % (line_no, line, fname)
                    _log.warning('Illegal %s', error_details)
                    continue
                key, val = tup
                key = key.strip()
                val = val.strip()
                if key in rc_temp:
                    _log.warning('Duplicate key in file %r line #%d.',
                                 fname, line_no)
                rc_temp[key] = (val, line, line_no)
        except UnicodeDecodeError:
            _log.warning('Cannot decode configuration file %s with encoding '
                         '%s, check LANG and LC_* variables.',
                         fname,
                         locale.getpreferredencoding(do_setlocale=False)
                         or 'utf-8 (default)')
            raise

    config = RcParams()

    for key, (val, line, line_no) in rc_temp.items():
        if key in defaultParams:
            if fail_on_error:
                config[key] = val  # try to convert to proper type or raise
            else:
                try:
                    config[key] = val  # try to convert to proper type or skip
                except Exception as msg:
                    error_details = _error_details_fmt % (line_no, line, fname)
                    _log.warning('Bad val %r on %s\n\t%s',
                                 val, error_details, msg)
        elif key in _deprecated_ignore_map:
            version, alt_key = _deprecated_ignore_map[key]
            cbook.warn_deprecated(
                version, name=key, alternative=alt_key,
                addendum="Please update your matplotlibrc.")
        else:
            version = 'master' if '.post' in __version__ else f'v{__version__}'
            print(f"""
Bad key "{key}" on line {line_no} in
{fname}.
You probably need to get an updated matplotlibrc file from
https://github.com/matplotlib/matplotlib/blob/{version}/matplotlibrc.template
or from the matplotlib source distribution""", file=sys.stderr)
    return config


def rc_params_from_file(fname, fail_on_error=False, use_default_template=True):
    """
    Construct a `RcParams` from file *fname*.

    Parameters
    ----------
    fname : str or path-like
        Name of file parsed for Matplotlib settings.
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
    with cbook._suppress_matplotlib_deprecation_warning():
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


with cbook._suppress_matplotlib_deprecation_warning():
    rcParamsOrig = RcParams(rcParams.copy())
    rcParamsDefault = RcParams([(key, default) for key, (default, converter) in
                                defaultParams.items()
                                if key not in _all_deprecated])

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

    The following aliases are available to save typing for interactive users:

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

    Notes
    -----
    Similar functionality is available by using the normal dict interface, i.e.
    ``rcParams.update({"lines.linewidth": 2, ...})`` (but ``rcParams.update``
    does not support abbreviations or grouping).
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
    rc_file_defaults
        Restore the rc params from the rc file originally loaded by Matplotlib.
    matplotlib.style.use :
        Use a specific style file.  Call ``style.use('default')`` to restore
        the default style.
    """
    # Deprecation warnings were already handled when creating rcParamsDefault,
    # no need to reemit them here.
    with cbook._suppress_matplotlib_deprecation_warning():
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
    with cbook._suppress_matplotlib_deprecation_warning():
        from .style.core import STYLE_BLACKLIST
        rcParams.update({k: rcParamsOrig[k] for k in rcParamsOrig
                         if k not in STYLE_BLACKLIST})


def rc_file(fname, *, use_default_template=True):
    """
    Update rc params from file.

    Style-blacklisted rc params (defined in
    `matplotlib.style.core.STYLE_BLACKLIST`) are not updated.

    Parameters
    ----------
    fname : str
        Name of file parsed for matplotlib settings.

    use_default_template : bool
        If True, initialize with default parameters before updating with those
        in the given file. If False, the current configuration persists
        and only the parameters specified in the file are updated.
    """
    # Deprecation warnings were already handled in rc_params_from_file, no need
    # to reemit them here.
    with cbook._suppress_matplotlib_deprecation_warning():
        from .style.core import STYLE_BLACKLIST
        rc_from_file = rc_params_from_file(
            fname, use_default_template=use_default_template)
        rcParams.update({k: rc_from_file[k] for k in rc_from_file
                         if k not in STYLE_BLACKLIST})


class rc_context:
    """
    Return a context manager for managing rc settings.

    This allows one to do::

        with mpl.rc_context(fname='screen.rc'):
            plt.plot(x, a)  # uses 'screen.rc'
            with mpl.rc_context(fname='print.rc'):
                plt.plot(x, b)  # uses 'print.rc'
            plt.plot(x, c)  # uses 'screen.rc'

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
        # except for the backend.  If the context block triggered resolving
        # the auto backend resolution keep that value around
        if self._orig['backend'] is rcsetup._auto_backend_sentinel:
            rcParams['backend'] = updated_backend

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.__fallback()


@cbook._rename_parameter("3.1", "arg", "backend")
@cbook._delete_parameter("3.1", "warn")
def use(backend, warn=False, force=True):
    """
    Select the backend used for rendering and GUI integration.

    Parameters
    ----------
    backend : str
        The backend to switch to.  This can either be one of the standard
        backend names, which are case-insensitive:

        - interactive backends:
          GTK3Agg, GTK3Cairo, MacOSX, nbAgg,
          Qt4Agg, Qt4Cairo, Qt5Agg, Qt5Cairo,
          TkAgg, TkCairo, WebAgg, WX, WXAgg, WXCairo

        - non-interactive backends:
          agg, cairo, pdf, pgf, ps, svg, template

        or a string of the form: ``module://my.module.name``.

    warn : bool, optional, default: False
        If True and not *force*, emit a warning if a failure-to-switch
        `ImportError` has been suppressed.  This parameter is deprecated.

    force : bool, optional, default: True
        If True (the default), raise an `ImportError` if the backend cannot be
        set up (either because it fails to import, or because an incompatible
        GUI interactive framework is already running); if False, ignore the
        failure.

    See Also
    --------
    :ref:`backends`
    matplotlib.get_backend
    """
    name = validate_backend(backend)
    if dict.__getitem__(rcParams, 'backend') == name:
        # Nothing to do if the requested backend is already set
        pass
    else:
        # Update both rcParams and rcDefaults so restoring the defaults later
        # with rcdefaults won't change the backend.  This is a bit of overkill
        # as 'backend' is already in style.core.STYLE_BLACKLIST, but better to
        # be safe.
        rcParams['backend'] = rcParamsDefault['backend'] = name
        try:
            from matplotlib import pyplot as plt
            plt.switch_backend(name)
        except ImportError as exc:
            if force:
                raise
            if warn:
                cbook._warn_external(
                    f"Failed to switch backend to {backend}: {exc}")


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
    Set whether to redraw after every plotting command (e.g. `.pyplot.xlabel`).
    """
    rcParams['interactive'] = b


def is_interactive():
    """Return whether to redraw after every plotting command."""
    return rcParams['interactive']


@cbook.deprecated("3.1", alternative="rcParams['tk.window_focus']")
def tk_window_focus():
    """
    Return true if focus maintenance under TkAgg on win32 is on.

    This currently works only for python.exe and IPython.exe.
    Both IDLE and Pythonwin.exe fail badly when tk_window_focus is on.
    """
    if rcParams['backend'] != 'TkAgg':
        return False
    return rcParams['tk.window_focus']


default_test_modules = [
    'matplotlib.tests',
    'mpl_toolkits.tests',
]


def _init_tests():
    # The version of FreeType to install locally for running the
    # tests.  This must match the value in `setupext.py`
    LOCAL_FREETYPE_VERSION = '2.6.1'

    from matplotlib import ft2font
    if (ft2font.__freetype_version__ != LOCAL_FREETYPE_VERSION or
        ft2font.__freetype_build_type__ != 'local'):
        _log.warning(
            "Matplotlib is not built with the correct FreeType version to run "
            "tests.  Set local_freetype=True in setup.cfg and rebuild. "
            "Expect many image comparison failures below. "
            "Expected freetype version {0}. "
            "Found freetype version {1}. "
            "Freetype build type is {2}local".format(
                LOCAL_FREETYPE_VERSION,
                ft2font.__freetype_version__,
                "" if ft2font.__freetype_build_type__ == 'local' else "not "))

    try:
        import pytest
    except ImportError:
        print("matplotlib.test requires pytest to run.")
        raise


@cbook._delete_parameter("3.2", "switch_backend_warn")
def test(verbosity=None, coverage=False, switch_backend_warn=True,
         recursionlimit=0, **kwargs):
    """Run the matplotlib test suite."""
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
            use(old_backend)
        if recursionlimit:
            sys.setrecursionlimit(old_recursionlimit)

    return retcode


test.__test__ = False  # pytest: this function is not a test


def _replacer(data, value):
    """
    Either returns ``data[value]`` or passes ``data`` back, converts either to
    a sequence.
    """
    try:
        # if key isn't a string don't bother
        if isinstance(value, str):
            # try to use __getitem__
            value = data[value]
    except Exception:
        # key does not exist, silently fall back to key
        pass
    return sanitize_sequence(value)


def _label_from_arg(y, default_name):
    try:
        return y.name
    except AttributeError:
        if isinstance(default_name, str):
            return default_name
    return None


_DATA_DOC_TITLE = """

Notes
-----
"""

_DATA_DOC_APPENDIX = """

.. note::
    In addition to the above described arguments, this function can take a
    **data** keyword argument. If such a **data** argument is given, the
    following arguments are replaced by **data[<arg>]**:

    {replaced}

    Objects passed as **data** must support item access (``data[<arg>]``) and
    membership test (``<arg> in data``).
"""


def _add_data_doc(docstring, replace_names):
    """Add documentation for a *data* field to the given docstring.

    Parameters
    ----------
    docstring : str
        The input docstring.
    replace_names : list of str or None
        The list of parameter names which arguments should be replaced by
        ``data[name]`` (if ``data[name]`` does not throw an exception).  If
        None, replacement is attempted for all arguments.

    Returns
    -------
        The augmented docstring.
    """
    docstring = inspect.cleandoc(docstring) if docstring is not None else ""
    repl = ("* All positional and all keyword arguments."
            if replace_names is None else
            ""
            if len(replace_names) == 0 else
            "* All arguments with the following names: {}.".format(
                ", ".join(map(repr, sorted(replace_names)))))
    addendum = _DATA_DOC_APPENDIX.format(replaced=repl)
    if _DATA_DOC_TITLE not in docstring:
        addendum = _DATA_DOC_TITLE + addendum
    return docstring + addendum


def _preprocess_data(func=None, *, replace_names=None, label_namer=None):
    """
    A decorator to add a 'data' kwarg to a function.

    When applied::

        @_preprocess_data()
        def func(ax, *args, **kwargs): ...

    the signature is modified to ``decorated(ax, *args, data=None, **kwargs)``
    with the following behavior:

    - if called with ``data=None``, forward the other arguments to ``func``;
    - otherwise, *data* must be a mapping; for any argument passed in as a
      string ``name``, replace the argument by ``data[name]`` (if this does not
      throw an exception), then forward the arguments to ``func``.

    In either case, any argument that is a `MappingView` is also converted to a
    list.

    Parameters
    ----------
    replace_names : list of str or None, optional, default: None
        The list of parameter names for which lookup into *data* should be
        attempted. If None, replacement is attempted for all arguments.
    label_namer : str, optional, default: None
        If set e.g. to "namer" (which must be a kwarg in the function's
        signature -- not as ``**kwargs``), if the *namer* argument passed in is
        a (string) key of *data* and no *label* kwarg is passed, then use the
        (string) value of the *namer* as *label*. ::

            @_preprocess_data(label_namer="foo")
            def func(foo, label=None): ...

            func("key", data={"key": value})
            # is equivalent to
            func.__wrapped__(value, label="key")
    """

    if func is None:  # Return the actual decorator.
        return functools.partial(
            _preprocess_data,
            replace_names=replace_names, label_namer=label_namer)

    sig = inspect.signature(func)
    varargs_name = None
    varkwargs_name = None
    arg_names = []
    params = list(sig.parameters.values())
    for p in params:
        if p.kind is Parameter.VAR_POSITIONAL:
            varargs_name = p.name
        elif p.kind is Parameter.VAR_KEYWORD:
            varkwargs_name = p.name
        else:
            arg_names.append(p.name)
    data_param = Parameter("data", Parameter.KEYWORD_ONLY, default=None)
    if varkwargs_name:
        params.insert(-1, data_param)
    else:
        params.append(data_param)
    new_sig = sig.replace(parameters=params)
    arg_names = arg_names[1:]  # remove the first "ax" / self arg

    if replace_names is not None:
        replace_names = set(replace_names)

    assert (replace_names or set()) <= set(arg_names) or varkwargs_name, (
        "Matplotlib internal error: invalid replace_names ({!r}) for {!r}"
        .format(replace_names, func.__name__))
    assert label_namer is None or label_namer in arg_names, (
        "Matplotlib internal error: invalid label_namer ({!r}) for {!r}"
            .format(label_namer, func.__name__))

    @functools.wraps(func)
    def inner(ax, *args, data=None, **kwargs):
        if data is None:
            return func(ax, *map(sanitize_sequence, args), **kwargs)

        bound = new_sig.bind(ax, *args, **kwargs)
        auto_label = (bound.arguments.get(label_namer)
                      or bound.kwargs.get(label_namer))

        for k, v in bound.arguments.items():
            if k == varkwargs_name:
                for k1, v1 in v.items():
                    if replace_names is None or k1 in replace_names:
                        v[k1] = _replacer(data, v1)
            elif k == varargs_name:
                if replace_names is None:
                    bound.arguments[k] = tuple(_replacer(data, v1) for v1 in v)
            else:
                if replace_names is None or k in replace_names:
                    bound.arguments[k] = _replacer(data, v)

        new_args = bound.args
        new_kwargs = bound.kwargs

        args_and_kwargs = {**bound.arguments, **bound.kwargs}
        if label_namer and "label" not in args_and_kwargs:
            new_kwargs["label"] = _label_from_arg(
                args_and_kwargs.get(label_namer), auto_label)

        return func(*new_args, **new_kwargs)

    inner.__doc__ = _add_data_doc(inner.__doc__, replace_names)
    inner.__signature__ = new_sig
    return inner


_log.debug('matplotlib version %s', __version__)
_log.debug('interactive is %s', is_interactive())
_log.debug('platform is %s', sys.platform)
_log.debug('loaded modules: %s', list(sys.modules))
