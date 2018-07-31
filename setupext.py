import builtins
import configparser
from distutils import sysconfig, version
from distutils.core import Extension
import distutils.command.build_ext
import glob
import hashlib
import importlib
import multiprocessing
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import textwrap
import urllib.request
import warnings

import setuptools
import versioneer


def _get_xdg_cache_dir():
    """
    Return the XDG cache directory.

    See https://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
    """
    cache_dir = os.environ.get('XDG_CACHE_HOME')
    if not cache_dir:
        cache_dir = os.path.expanduser('~/.cache')
        if cache_dir.startswith('~/'):  # Expansion failed.
            return None
    return os.path.join(cache_dir, 'matplotlib')


# SHA256 hashes of the FreeType tarballs
_freetype_hashes = {
    '2.6.1': '0a3c7dfbda6da1e8fce29232e8e96d987ababbbf71ebc8c75659e4132c367014',
    '2.6.2': '8da42fc4904e600be4b692555ae1dcbf532897da9c5b9fb5ebd3758c77e5c2d4',
    '2.6.3': '7942096c40ee6fea882bd4207667ad3f24bff568b96b10fd3885e11a7baad9a3',
    '2.6.4': '27f0e38347a1850ad57f84fc4dfed68ba0bc30c96a6fa6138ef84d485dd9a8d7',
    '2.6.5': '3bb24add9b9ec53636a63ea8e867ed978c4f8fdd8f1fa5ccfd41171163d4249a',
    '2.7': '7b657d5f872b0ab56461f3bd310bd1c5ec64619bd15f0d8e08282d494d9cfea4',
    '2.7.1': '162ef25aa64480b1189cdb261228e6c5c44f212aac4b4621e28cf2157efb59f5',
    '2.8': '33a28fabac471891d0523033e99c0005b95e5618dc8ffa7fa47f9dadcacb1c9b',
    '2.8.1': '876711d064a6a1bd74beb18dd37f219af26100f72daaebd2d86cb493d7cd7ec6',
}
# This is the version of FreeType to use when building a local
# version.  It must match the value in
# lib/matplotlib.__init__.py and also needs to be changed below in the
# embedded windows build script (grep for "REMINDER" in this file)
LOCAL_FREETYPE_VERSION = '2.6.1'
LOCAL_FREETYPE_HASH = _freetype_hashes.get(LOCAL_FREETYPE_VERSION, 'unknown')


# matplotlib build options, which can be altered using setup.cfg
options = {
    'display_status': True,
    'backend': None,
    'basedirlist': None
    }


setup_cfg = os.environ.get('MPLSETUPCFG', 'setup.cfg')
if os.path.exists(setup_cfg):
    config = configparser.ConfigParser()
    config.read(setup_cfg)

    if config.has_option('status', 'suppress'):
        options['display_status'] = not config.getboolean("status", "suppress")

    if config.has_option('rc_options', 'backend'):
        options['backend'] = config.get("rc_options", "backend")

    if config.has_option('directories', 'basedirlist'):
        options['basedirlist'] = [
            x.strip() for x in
            config.get("directories", "basedirlist").split(',')]

    if config.has_option('test', 'local_freetype'):
        options['local_freetype'] = config.getboolean("test", "local_freetype")
else:
    config = None

lft = bool(os.environ.get('MPLLOCALFREETYPE', False))
options['local_freetype'] = lft or options.get('local_freetype', False)


def extract_versions():
    """
    Extracts version values from the main matplotlib __init__.py and
    returns them as a dictionary.
    """
    with open('lib/matplotlib/__init__.py') as fd:
        for line in fd.readlines():
            if line.startswith('__version__numpy__'):
                exec(line.strip())
    return locals()


def has_include_file(include_dirs, filename):
    """
    Returns `True` if *filename* can be found in one of the
    directories in *include_dirs*.
    """
    if sys.platform == 'win32':
        include_dirs = [*include_dirs,  # Don't modify it in-place.
                        *os.environ.get('INCLUDE', '.').split(os.pathsep)]
    return any(pathlib.Path(dir, filename).exists() for dir in include_dirs)


def check_include_file(include_dirs, filename, package):
    """
    Raises an exception if the given include file can not be found.
    """
    if not has_include_file(include_dirs, filename):
        raise CheckFailed(
            "The C/C++ header for %s (%s) could not be found.  You "
            "may need to install the development package." %
            (package, filename))


def get_base_dirs():
    """
    Returns a list of standard base directories on this platform.
    """
    if options['basedirlist']:
        return options['basedirlist']

    if os.environ.get('MPLBASEDIRLIST'):
        return os.environ.get('MPLBASEDIRLIST').split(os.pathsep)

    win_bases = ['win32_static']
    # on conda windows, we also add the <conda_env_dir>\Library,
    # as conda installs libs/includes there
    # env var names mess: https://github.com/conda/conda/issues/2312
    conda_env_path = os.getenv('CONDA_PREFIX')  # conda >= 4.1
    if not conda_env_path:
        conda_env_path = os.getenv('CONDA_DEFAULT_ENV')  # conda < 4.1
    if conda_env_path and os.path.isdir(conda_env_path):
        win_bases.append(os.path.join(conda_env_path, "Library"))

    basedir_map = {
        'win32': win_bases,
        'darwin': ['/usr/local/', '/usr', '/usr/X11',
                   '/opt/X11', '/opt/local'],
        'sunos5': [os.getenv('MPLIB_BASE') or '/usr/local', ],
        'gnu0': ['/usr'],
        'aix5': ['/usr/local'],
        }
    return basedir_map.get(sys.platform, ['/usr/local', '/usr'])


def get_include_dirs():
    """
    Returns a list of standard include directories on this platform.
    """
    include_dirs = [os.path.join(d, 'include') for d in get_base_dirs()]
    if sys.platform != 'win32':
        # gcc includes these dirs automatically, so also look for headers in
        # these dirs
        include_dirs.extend(
            os.environ.get('CPATH', '').split(os.pathsep))
        include_dirs.extend(
            os.environ.get('CPLUS_INCLUDE_PATH', '').split(os.pathsep))
    return include_dirs


def is_min_version(found, minversion):
    """
    Returns whether *found* is a version at least as high as *minversion*.
    """
    return version.LooseVersion(found) >= version.LooseVersion(minversion)


# Define the display functions only if display_status is True.
if options['display_status']:
    def print_line(char='='):
        print(char * 76)

    def print_status(package, status):
        initial_indent = "%22s: " % package
        indent = ' ' * 24
        print(textwrap.fill(str(status), width=76,
                            initial_indent=initial_indent,
                            subsequent_indent=indent))

    def print_message(message):
        indent = ' ' * 24 + "* "
        print(textwrap.fill(str(message), width=76,
                            initial_indent=indent,
                            subsequent_indent=indent))

    def print_raw(section):
        print(section)
else:
    def print_line(*args, **kwargs):
        pass
    print_status = print_message = print_raw = print_line


# Remove the -Wstrict-prototypes option, is it's not valid for C++
customize_compiler = distutils.command.build_ext.customize_compiler


def my_customize_compiler(compiler):
    retval = customize_compiler(compiler)
    try:
        compiler.compiler_so.remove('-Wstrict-prototypes')
    except (ValueError, AttributeError):
        pass
    return retval

distutils.command.build_ext.customize_compiler = my_customize_compiler


def make_extension(name, files, *args, **kwargs):
    """
    Make a new extension.  Automatically sets include_dirs and
    library_dirs to the base directories appropriate for this
    platform.

    `name` is the name of the extension.

    `files` is a list of source files.

    Any additional arguments are passed to the
    `distutils.core.Extension` constructor.
    """
    ext = DelayedExtension(name, files, *args, **kwargs)
    for dir in get_base_dirs():
        include_dir = os.path.join(dir, 'include')
        if os.path.exists(include_dir):
            ext.include_dirs.append(include_dir)
        for lib in ('lib', 'lib64'):
            lib_dir = os.path.join(dir, lib)
            if os.path.exists(lib_dir):
                ext.library_dirs.append(lib_dir)
    ext.include_dirs.append('.')

    return ext


def get_file_hash(filename):
    """
    Get the SHA256 hash of a given filename.
    """
    BLOCKSIZE = 1 << 16
    hasher = hashlib.sha256()
    with open(filename, 'rb') as fd:
        buf = fd.read(BLOCKSIZE)
        while buf:
            hasher.update(buf)
            buf = fd.read(BLOCKSIZE)
    return hasher.hexdigest()


class PkgConfig(object):
    """
    This is a class for communicating with pkg-config.
    """
    def __init__(self):
        """
        Determines whether pkg-config exists on this machine.
        """
        if sys.platform == 'win32':
            self.has_pkgconfig = False
        else:
            self.pkg_config = os.environ.get('PKG_CONFIG', 'pkg-config')
            self.set_pkgconfig_path()
            self.has_pkgconfig = shutil.which(self.pkg_config) is not None
            if not self.has_pkgconfig:
                print("IMPORTANT WARNING:\n"
                      "    pkg-config is not installed.\n"
                      "    matplotlib may not be able to find some of its dependencies")

    def set_pkgconfig_path(self):
        pkgconfig_path = sysconfig.get_config_var('LIBDIR')
        if pkgconfig_path is None:
            return

        pkgconfig_path = os.path.join(pkgconfig_path, 'pkgconfig')
        if not os.path.isdir(pkgconfig_path):
            return

        try:
            os.environ['PKG_CONFIG_PATH'] += ':' + pkgconfig_path
        except KeyError:
            os.environ['PKG_CONFIG_PATH'] = pkgconfig_path

    def setup_extension(self, ext, package, default_include_dirs=[],
                        default_library_dirs=[], default_libraries=[],
                        alt_exec=None):
        """
        Add parameters to the given `ext` for the given `package`.
        """
        flag_map = {
            '-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}

        executable = alt_exec
        if self.has_pkgconfig:
            executable = (self.pkg_config + ' {0}').format(package)

        use_defaults = True

        if executable is not None:
            command = "{0} --libs --cflags ".format(executable)

            try:
                output = subprocess.check_output(
                    command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                pass
            else:
                output = output.decode(sys.getfilesystemencoding())
                use_defaults = False
                for token in output.split():
                    attr = flag_map.get(token[:2])
                    if attr is not None:
                        getattr(ext, attr).insert(0, token[2:])

        if use_defaults:
            basedirs = get_base_dirs()
            for base in basedirs:
                for include in default_include_dirs:
                    dir = os.path.join(base, include)
                    if os.path.exists(dir):
                        ext.include_dirs.append(dir)
                for lib in default_library_dirs:
                    dir = os.path.join(base, lib)
                    if os.path.exists(dir):
                        ext.library_dirs.append(dir)
            ext.libraries.extend(default_libraries)
            return True

        return False

    def get_version(self, package):
        """
        Get the version of the package from pkg-config.
        """
        if not self.has_pkgconfig:
            return None

        status, output = subprocess.getstatusoutput(
            self.pkg_config + " %s --modversion" % (package))
        if status == 0:
            return output
        return None


# The PkgConfig class should be used through this singleton
pkg_config = PkgConfig()


class CheckFailed(Exception):
    """
    Exception thrown when a `SetupPackage.check` method fails.
    """
    pass


class SetupPackage(object):
    optional = False
    pkg_names = {
        "apt-get": None,
        "yum": None,
        "dnf": None,
        "brew": None,
        "port": None,
        "windows_url": None
        }

    def check(self):
        """
        Checks whether the build dependencies are met.  Should raise a
        `CheckFailed` exception if the dependency could not be met, otherwise
        return a string indicating a version number or some other message
        indicating what was found.
        """
        pass

    def runtime_check(self):
        """
        True if the runtime dependencies of the backend are met.  Assumes that
        the build-time dependencies are met.
        """
        return True

    def get_packages(self):
        """
        Get a list of package names to add to the configuration.
        These are added to the `packages` list passed to
        `distutils.setup`.
        """
        return []

    def get_namespace_packages(self):
        """
        Get a list of namespace package names to add to the configuration.
        These are added to the `namespace_packages` list passed to
        `distutils.setup`.
        """
        return []

    def get_py_modules(self):
        """
        Get a list of top-level modules to add to the configuration.
        These are added to the `py_modules` list passed to
        `distutils.setup`.
        """
        return []

    def get_package_data(self):
        """
        Get a package data dictionary to add to the configuration.
        These are merged into to the `package_data` list passed to
        `distutils.setup`.
        """
        return {}

    def get_extension(self):
        """
        Get a list of C extensions (`distutils.core.Extension`
        objects) to add to the configuration.  These are added to the
        `extensions` list passed to `distutils.setup`.
        """
        return None

    def get_install_requires(self):
        """
        Get a list of Python packages that we require.
        pip/easy_install will attempt to download and install this
        package if it is not installed.
        """
        return []

    def get_setup_requires(self):
        """
        Get a list of Python packages that we require at build time.
        pip/easy_install will attempt to download and install this
        package if it is not installed.
        """
        return []

    def _check_for_pkg_config(self, package, include_file, min_version=None,
                              version=None):
        """
        A convenience function for writing checks for a
        pkg_config-defined dependency.

        `package` is the pkg_config package name.

        `include_file` is a top-level include file we expect to find.

        `min_version` is the minimum version required.

        `version` will override the found version if this package
        requires an alternate method for that. Set version='unknown'
        if the version is not known but you still want to disabled
        pkg_config version check.
        """
        if version is None:
            version = pkg_config.get_version(package)

            if version is None:
                raise CheckFailed(
                    "pkg-config information for '%s' could not be found." %
                    package)

        if min_version == 'PATCH':
            raise CheckFailed(
                "Requires patches that have not been merged upstream.")

        if min_version and version != 'unknown':
            if not is_min_version(version, min_version):
                raise CheckFailed(
                    "Requires %s %s or later.  Found %s." %
                    (package, min_version, version))

        ext = self.get_extension()
        if ext is None:
            ext = make_extension('test', [])
            pkg_config.setup_extension(ext, package)

        check_include_file(
            ext.include_dirs + get_include_dirs(), include_file, package)

        return 'version %s' % version

    def do_custom_build(self):
        """
        If a package needs to do extra custom things, such as building a
        third-party library, before building an extension, it should
        override this method.
        """
        pass

    def install_help_msg(self):
        """
        Do not override this method !

        Generate the help message to show if the package is not installed.
        To use this in subclasses, simply add the dictionary `pkg_names` as
        a class variable:

        pkg_names = {
            "apt-get": <Name of the apt-get package>,
            "yum": <Name of the yum package>,
            "dnf": <Name of the dnf package>,
            "brew": <Name of the brew package>,
            "port": <Name of the port package>,
            "windows_url": <The url which has installation instructions>
            }

        All the dictionary keys are optional. If a key is not present or has
        the value `None` no message is provided for that platform.
        """
        def _try_managers(*managers):
            for manager in managers:
                pkg_name = self.pkg_names.get(manager, None)
                if pkg_name:
                    if shutil.which(manager) is not None:
                        if manager == 'port':
                            pkgconfig = 'pkgconfig'
                        else:
                            pkgconfig = 'pkg-config'
                        return ('Try installing {0} with `{1} install {2}` '
                                'and pkg-config with `{1} install {3}`'
                                .format(self.name, manager, pkg_name,
                                        pkgconfig))

        message = None
        if sys.platform == "win32":
            url = self.pkg_names.get("windows_url", None)
            if url:
                message = ('Please check {0} for instructions to install {1}'
                           .format(url, self.name))
        elif sys.platform == "darwin":
            message = _try_managers("brew", "port")
        elif sys.platform == "linux":
            release = platform.linux_distribution()[0].lower()
            if release in ('debian', 'ubuntu'):
                message = _try_managers('apt-get')
            elif release in ('centos', 'redhat', 'fedora'):
                message = _try_managers('dnf', 'yum')
        return message


class OptionalPackage(SetupPackage):
    optional = True
    force = False
    config_category = "packages"
    default_config = "auto"

    @classmethod
    def get_config(cls):
        """
        Look at `setup.cfg` and return one of ["auto", True, False] indicating
        if the package is at default state ("auto"), forced by the user (case
        insensitively defined as 1, true, yes, on for True) or opted-out (case
        insensitively defined as 0, false, no, off for False).
        """
        conf = cls.default_config
        if config is not None and config.has_option(cls.config_category, cls.name):
            try:
                conf = config.getboolean(cls.config_category, cls.name)
            except ValueError:
                conf = config.get(cls.config_category, cls.name)
        return conf

    def check(self):
        """
        Do not override this method!

        For custom dependency checks override self.check_requirements().
        Two things are checked: Configuration file and requirements.
        """
        # Check configuration file
        conf = self.get_config()
        # Default "auto" state or install forced by user
        if conf in [True, 'auto']:
            message = "installing"
            # Set non-optional if user sets `True` in config
            if conf is True:
                self.optional = False
        # Configuration opt-out by user
        else:
            # Some backend extensions (e.g. Agg) need to be built for certain
            # other GUI backends (e.g. TkAgg) even when manually disabled
            if self.force is True:
                message = "installing forced (config override)"
            else:
                raise CheckFailed("skipping due to configuration")

        # Check requirements and add extra information (if any) to message.
        # If requirements are not met a CheckFailed should be raised in there.
        additional_info = self.check_requirements()
        if additional_info:
            message += ", " + additional_info

        # No CheckFailed raised until now, return install message.
        return message

    def check_requirements(self):
        """
        Override this method to do custom dependency checks.

         - Raise CheckFailed() if requirements are not met.
         - Return message with additional information, or an empty string
           (or None) for no additional information.
        """
        return ""


class OptionalBackendPackage(OptionalPackage):
    config_category = "gui_support"


class Platform(SetupPackage):
    name = "platform"

    def check(self):
        return sys.platform


class Python(SetupPackage):
    name = "python"

    def check(self):
        if sys.version_info < (3, 5):
            error = """
Matplotlib 3.0+ does not support Python 2.x, 3.0, 3.1, 3.2, 3.3, or 3.4.
Beginning with Matplotlib 3.0, Python 3.5 and above is required.

This may be due to an out of date pip.

Make sure you have pip >= 9.0.1.
"""
            raise CheckFailed(error)
        return sys.version


class Matplotlib(SetupPackage):
    name = "matplotlib"

    def check(self):
        return versioneer.get_version()

    def get_packages(self):
        return setuptools.find_packages(
            "lib",
            include=["matplotlib", "matplotlib.*"],
            exclude=["matplotlib.tests", "matplotlib.*.tests"])

    def get_py_modules(self):
        return ['pylab']

    def get_package_data(self):

        def iter_dir(base):
            return [
                str(path.relative_to('lib/matplotlib'))
                for path in pathlib.Path('lib/matplotlib', base).rglob('*')]

        return {
            'matplotlib':
            [
                'mpl-data/matplotlibrc',
                *iter_dir('mpl-data/fonts'),
                *iter_dir('mpl-data/images'),
                *iter_dir('mpl-data/stylelib'),
                *iter_dir('backends/web_backend'),
            ]}


class SampleData(OptionalPackage):
    """
    This handles the sample data that ships with matplotlib.  It is
    technically optional, though most often will be desired.
    """
    name = "sample_data"

    def get_package_data(self):

        def iter_dir(base):
            return [
                str(path.relative_to('lib/matplotlib'))
                for path in pathlib.Path('lib/matplotlib', base).rglob('*')]

        return {
            'matplotlib':
            [
                *iter_dir('mpl-data/sample_data'),
            ]}


class Toolkits(OptionalPackage):
    name = "toolkits"

    def get_packages(self):
        return [
            'mpl_toolkits',
            'mpl_toolkits.mplot3d',
            'mpl_toolkits.axes_grid',
            'mpl_toolkits.axes_grid1',
            'mpl_toolkits.axisartist',
            ]

    def get_namespace_packages(self):
        return ['mpl_toolkits']


class Tests(OptionalPackage):
    name = "tests"
    pytest_min_version = '3.4'
    default_config = False

    def check(self):
        super().check()

        msgs = []
        msg_template = ('{package} is required to run the Matplotlib test '
                        'suite. Please install it with pip or your preferred '
                        'tool to run the test suite')

        bad_pytest = msg_template.format(
            package='pytest %s or later' % self.pytest_min_version
        )
        try:
            import pytest
            if is_min_version(pytest.__version__, self.pytest_min_version):
                msgs += ['using pytest version %s' % pytest.__version__]
            else:
                msgs += [bad_pytest]
        except ImportError:
            msgs += [bad_pytest]

        return ' / '.join(msgs)

    def get_packages(self):
        return [
            'matplotlib.tests',
            'matplotlib.sphinxext.tests',
            ]

    def get_package_data(self):
        baseline_images = [
            'tests/baseline_images/%s/*' % x
            for x in os.listdir('lib/matplotlib/tests/baseline_images')]

        return {
            'matplotlib':
            baseline_images +
            [
                'tests/cmr10.pfb',
                'tests/mpltest.ttf',
                'tests/test_rcparams.rc',
                'tests/test_utf32_be_rcparams.rc',
                'sphinxext/tests/tinypages/*.rst',
                'sphinxext/tests/tinypages/*.py',
                'sphinxext/tests/tinypages/_static/*',
            ]}


class Toolkits_Tests(Tests):
    name = "toolkits_tests"

    def check_requirements(self):
        conf = self.get_config()
        toolkits_conf = Toolkits.get_config()
        tests_conf = Tests.get_config()

        if conf is True:
            Tests.force = True
            Toolkits.force = True
        elif conf == "auto" and not (toolkits_conf and tests_conf):
            # Only auto-install if both toolkits and tests are set
            # to be installed
            raise CheckFailed("toolkits_tests needs 'toolkits' and 'tests'")
        return ""

    def get_packages(self):
        return [
            'mpl_toolkits.tests',
            ]

    def get_package_data(self):
        baseline_images = [
            'tests/baseline_images/%s/*' % x
            for x in os.listdir('lib/mpl_toolkits/tests/baseline_images')]

        return {'mpl_toolkits': baseline_images}

    def get_namespace_packages(self):
        return ['mpl_toolkits']


class DelayedExtension(Extension, object):
    """
    A distutils Extension subclass where some of its members
    may have delayed computation until reaching the build phase.

    This is so we can, for example, get the Numpy include dirs
    after pip has installed Numpy for us if it wasn't already
    on the system.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._finalized = False
        self._hooks = {}

    def add_hook(self, member, func):
        """
        Add a hook to dynamically compute a member.

        Parameters
        ----------
        member : string
            The name of the member

        func : callable
            The function to call to get dynamically-computed values
            for the member.
        """
        self._hooks[member] = func

    def finalize(self):
        self._finalized = True

    class DelayedMember(property):
        def __init__(self, name):
            self._name = name

        def __get__(self, obj, objtype=None):
            result = getattr(obj, '_' + self._name, [])

            if obj._finalized:
                if self._name in obj._hooks:
                    result = obj._hooks[self._name]() + result

            return result

        def __set__(self, obj, value):
            setattr(obj, '_' + self._name, value)

    include_dirs = DelayedMember('include_dirs')


class Numpy(SetupPackage):
    name = "numpy"

    @staticmethod
    def include_dirs_hook():
        if hasattr(builtins, '__NUMPY_SETUP__'):
            del builtins.__NUMPY_SETUP__
        import numpy
        importlib.reload(numpy)

        ext = Extension('test', [])
        ext.include_dirs.append(numpy.get_include())
        if not has_include_file(
                ext.include_dirs, os.path.join("numpy", "arrayobject.h")):
            warnings.warn(
                "The C headers for numpy could not be found. "
                "You may need to install the development package")

        return [numpy.get_include()]

    def check(self):
        min_version = extract_versions()['__version__numpy__']
        try:
            import numpy
        except ImportError:
            return 'not found. pip may install it below.'

        if not is_min_version(numpy.__version__, min_version):
            raise SystemExit(
                "Requires numpy %s or later to build.  (Found %s)" %
                (min_version, numpy.__version__))

        return 'version %s' % numpy.__version__

    def add_flags(self, ext):
        # Ensure that PY_ARRAY_UNIQUE_SYMBOL is uniquely defined for
        # each extension
        array_api_name = 'MPL_' + ext.name.replace('.', '_') + '_ARRAY_API'

        ext.define_macros.append(('PY_ARRAY_UNIQUE_SYMBOL', array_api_name))
        ext.add_hook('include_dirs', self.include_dirs_hook)

        ext.define_macros.append(('NPY_NO_DEPRECATED_API',
                                  'NPY_1_7_API_VERSION'))

        # Allow NumPy's printf format specifiers in C++.
        ext.define_macros.append(('__STDC_FORMAT_MACROS', 1))

    def get_setup_requires(self):
        return ['numpy>=1.10.0']

    def get_install_requires(self):
        return ['numpy>=1.10.0']


class LibAgg(SetupPackage):
    name = 'libagg'

    def check(self):
        self.__class__.found_external = True
        try:
            return self._check_for_pkg_config(
                'libagg', 'agg2/agg_basics.h', min_version='PATCH')
        except CheckFailed as e:
            self.__class__.found_external = False
            return str(e) + ' Using local copy.'

    def add_flags(self, ext, add_sources=True):
        if self.found_external:
            pkg_config.setup_extension(ext, 'libagg')
        else:
            ext.include_dirs.insert(0, 'extern/agg24-svn/include')
            if add_sources:
                agg_sources = [
                    'agg_bezier_arc.cpp',
                    'agg_curves.cpp',
                    'agg_image_filters.cpp',
                    'agg_trans_affine.cpp',
                    'agg_vcgen_contour.cpp',
                    'agg_vcgen_dash.cpp',
                    'agg_vcgen_stroke.cpp',
                    'agg_vpgen_segmentator.cpp'
                    ]
                ext.sources.extend(
                    os.path.join('extern', 'agg24-svn', 'src', x) for x in agg_sources)


class FreeType(SetupPackage):
    name = "freetype"
    pkg_names = {
        "apt-get": "libfreetype6-dev",
        "yum": "freetype-devel",
        "dnf": "freetype-devel",
        "brew": "freetype",
        "port": "freetype",
        "windows_url": "http://gnuwin32.sourceforge.net/packages/freetype.htm"
        }

    def check(self):
        if options.get('local_freetype'):
            return "Using local version for testing"

        if sys.platform == 'win32':
            try:
                check_include_file(get_include_dirs(), 'ft2build.h', 'freetype')
            except CheckFailed:
                check_include_file(get_include_dirs(), 'freetype2\\ft2build.h', 'freetype')
            return 'Using unknown version found on system.'

        status, output = subprocess.getstatusoutput(
            "freetype-config --ftversion")
        if status == 0:
            version = output
        else:
            version = None

        # Early versions of freetype grep badly inside freetype-config,
        # so catch those cases. (tested with 2.5.3).
        if version is None or 'No such file or directory\ngrep:' in version:
            version = self.version_from_header()

        # pkg_config returns the libtool version rather than the
        # freetype version so we need to explicitly pass the version
        # to _check_for_pkg_config
        return self._check_for_pkg_config(
            'freetype2', 'ft2build.h',
            min_version='2.3', version=version)

    def version_from_header(self):
        version = 'unknown'
        ext = self.get_extension()
        if ext is None:
            return version
        # Return the first version found in the include dirs.
        for include_dir in ext.include_dirs:
            header_fname = os.path.join(include_dir, 'freetype.h')
            if os.path.exists(header_fname):
                major, minor, patch = 0, 0, 0
                with open(header_fname, 'r') as fh:
                    for line in fh:
                        if line.startswith('#define FREETYPE_'):
                            value = line.rsplit(' ', 1)[1].strip()
                            if 'MAJOR' in line:
                                major = value
                            elif 'MINOR' in line:
                                minor = value
                            else:
                                patch = value
                return '.'.join([major, minor, patch])

    def add_flags(self, ext):
        if options.get('local_freetype'):
            src_path = os.path.join(
                'build', 'freetype-{0}'.format(LOCAL_FREETYPE_VERSION))
            # Statically link to the locally-built freetype.
            # This is certainly broken on Windows.
            ext.include_dirs.insert(0, os.path.join(src_path, 'include'))
            if sys.platform == 'win32':
                libfreetype = 'libfreetype.lib'
            else:
                libfreetype = 'libfreetype.a'
            ext.extra_objects.insert(
                0, os.path.join(src_path, 'objs', '.libs', libfreetype))
            ext.define_macros.append(('FREETYPE_BUILD_TYPE', 'local'))
        else:
            pkg_config.setup_extension(
                ext, 'freetype2',
                default_include_dirs=[
                    'include/freetype2', 'freetype2',
                    'lib/freetype2/include',
                    'lib/freetype2/include/freetype2'],
                default_library_dirs=[
                    'freetype2/lib'],
                default_libraries=['freetype', 'z'])
            ext.define_macros.append(('FREETYPE_BUILD_TYPE', 'system'))

    def do_custom_build(self):
        from pathlib import Path

        # We're using a system freetype
        if not options.get('local_freetype'):
            return

        src_path = os.path.join(
            'build', 'freetype-{0}'.format(LOCAL_FREETYPE_VERSION))

        # We've already built freetype
        if sys.platform == 'win32':
            libfreetype = 'libfreetype.lib'
        else:
            libfreetype = 'libfreetype.a'

        if os.path.isfile(os.path.join(src_path, 'objs', '.libs', libfreetype)):
            return

        tarball = 'freetype-{0}.tar.gz'.format(LOCAL_FREETYPE_VERSION)
        tarball_path = os.path.join('build', tarball)
        try:
            tarball_cache_dir = _get_xdg_cache_dir()
            tarball_cache_path = os.path.join(tarball_cache_dir, tarball)
        except:
            # again, do not really care if this fails
            tarball_cache_dir = None
            tarball_cache_path = None
        if not os.path.isfile(tarball_path):
            if (tarball_cache_path is not None and
                    os.path.isfile(tarball_cache_path)):
                if get_file_hash(tarball_cache_path) == LOCAL_FREETYPE_HASH:
                    os.makedirs('build', exist_ok=True)
                    try:
                        shutil.copy(tarball_cache_path, tarball_path)
                        print('Using cached tarball: {}'
                              .format(tarball_cache_path))
                    except OSError:
                        # If this fails, oh well just re-download
                        pass

            if not os.path.isfile(tarball_path):
                if not os.path.exists('build'):
                    os.makedirs('build')

                url_fmts = [
                    'https://downloads.sourceforge.net/project/freetype'
                    '/freetype2/{version}/{tarball}',
                    'https://download.savannah.gnu.org/releases/freetype'
                    '/{tarball}'
                ]
                for url_fmt in url_fmts:
                    tarball_url = url_fmt.format(
                        version=LOCAL_FREETYPE_VERSION, tarball=tarball)

                    print("Downloading {}".format(tarball_url))
                    try:
                        urllib.request.urlretrieve(tarball_url, tarball_path)
                    except IOError:  # URLError (a subclass) on Py3.
                        print("Failed to download {}".format(tarball_url))
                    else:
                        if get_file_hash(tarball_path) != LOCAL_FREETYPE_HASH:
                            print("Invalid hash.")
                        else:
                            break
                else:
                    raise IOError("Failed to download FreeType. You can "
                                  "download the file by alternative means and "
                                  "copy it to {}".format(tarball_path))
                os.makedirs(tarball_cache_dir, exist_ok=True)
                try:
                    shutil.copy(tarball_path, tarball_cache_path)
                    print('Cached tarball at {}'.format(tarball_cache_path))
                except OSError:
                    # If this fails, we can always re-download.
                    pass

            if get_file_hash(tarball_path) != LOCAL_FREETYPE_HASH:
                raise IOError(
                    "{} does not match expected hash.".format(tarball))

        print("Building {}".format(tarball))
        with tarfile.open(tarball_path, "r:gz") as tgz:
            tgz.extractall("build")

        if sys.platform != 'win32':
            # compilation on all other platforms than windows
            env = {**os.environ,
                   "CFLAGS": "{} -fPIC".format(os.environ.get("CFLAGS", ""))}
            subprocess.check_call(
                ["./configure", "--with-zlib=no", "--with-bzip2=no",
                 "--with-png=no", "--with-harfbuzz=no"],
                env=env, cwd=src_path)
            subprocess.check_call(["make"], env=env, cwd=src_path)
        else:
            # compilation on windows
            shutil.rmtree(str(Path(src_path, "objs")), ignore_errors=True)
            FREETYPE_BUILD_CMD = r"""
call "%ProgramFiles%\Microsoft SDKs\Windows\v7.0\Bin\SetEnv.Cmd" ^
    /Release /{xXX} /xp
call "{vcvarsall}" {xXX}
set MSBUILD=C:\Windows\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe
%MSBUILD% "builds\windows\{vc20xx}\freetype.sln" ^
    /t:Clean;Build /p:Configuration="Release";Platform={WinXX}
"""
            import distutils.msvc9compiler as msvc
            # Note: freetype has no build profile for 2014, so we don't bother...
            vc = 'vc2010'
            WinXX = 'x64' if platform.architecture()[0] == '64bit' else 'Win32'
            xXX = 'x64' if platform.architecture()[0] == '64bit' else 'x86'
            vcvarsall = msvc.find_vcvarsall(10.0)
            if vcvarsall is None:
                raise RuntimeError('Microsoft VS 2010 required')
            cmdfile = Path("build/build_freetype.cmd")
            cmdfile.write_text(FREETYPE_BUILD_CMD.format(
                vc20xx=vc, WinXX=WinXX, xXX=xXX, vcvarsall=vcvarsall))
            subprocess.check_call([str(cmdfile.resolve())],
                                  shell=True, cwd=src_path)
            # Move to the corresponding Unix build path.
            Path(src_path, "objs/.libs").mkdir()
            # Be robust against change of FreeType version.
            lib_path, = (Path(src_path, "objs", vc, xXX)
                         .glob("freetype*.lib"))
            shutil.copy2(str(lib_path),
                         str(Path(src_path, "objs/.libs/libfreetype.lib")))


class FT2Font(SetupPackage):
    name = 'ft2font'

    def get_extension(self):
        sources = [
            'src/ft2font.cpp',
            'src/ft2font_wrapper.cpp',
            'src/mplutils.cpp',
            'src/py_converters.cpp',
            ]
        ext = make_extension('matplotlib.ft2font', sources)
        FreeType().add_flags(ext)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext, add_sources=False)
        return ext


class Png(SetupPackage):
    name = "png"
    pkg_names = {
        "apt-get": "libpng12-dev",
        "yum": "libpng-devel",
        "dnf": "libpng-devel",
        "brew": "libpng",
        "port": "libpng",
        "windows_url": "http://gnuwin32.sourceforge.net/packages/libpng.htm"
        }

    def check(self):
        if sys.platform == 'win32':
            check_include_file(get_include_dirs(), 'png.h', 'png')
            return 'Using unknown version found on system.'

        status, output = subprocess.getstatusoutput("libpng-config --version")
        if status == 0:
            version = output
        else:
            version = None

        try:
            return self._check_for_pkg_config(
                'libpng', 'png.h',
                min_version='1.2', version=version)
        except CheckFailed as e:
            if has_include_file(get_include_dirs(), 'png.h'):
                return str(e) + ' Using unknown version found on system.'
            raise

    def get_extension(self):
        sources = [
            'src/_png.cpp',
            'src/mplutils.cpp'
            ]
        ext = make_extension('matplotlib._png', sources)
        pkg_config.setup_extension(
            ext, 'libpng', default_libraries=['png', 'z'],
            alt_exec='libpng-config --ldflags')
        Numpy().add_flags(ext)
        return ext


class Qhull(SetupPackage):
    name = "qhull"

    def check(self):
        self.__class__.found_external = True
        try:
            return self._check_for_pkg_config(
                'libqhull', 'libqhull/qhull_a.h', min_version='2015.2')
        except CheckFailed as e:
            self.__class__.found_pkgconfig = False
            self.__class__.found_external = False
            return str(e) + ' Using local copy.'

    def add_flags(self, ext):
        if self.found_external:
            pkg_config.setup_extension(ext, 'qhull',
                                       default_libraries=['qhull'])
        else:
            ext.include_dirs.insert(0, 'extern')
            ext.sources.extend(sorted(glob.glob('extern/libqhull/*.c')))
            if sysconfig.get_config_var('LIBM') == '-lm':
                ext.libraries.extend('m')


class TTConv(SetupPackage):
    name = "ttconv"

    def get_extension(self):
        sources = [
            'src/_ttconv.cpp',
            'extern/ttconv/pprdrv_tt.cpp',
            'extern/ttconv/pprdrv_tt2.cpp',
            'extern/ttconv/ttutil.cpp'
            ]
        ext = make_extension('matplotlib.ttconv', sources)
        Numpy().add_flags(ext)
        ext.include_dirs.insert(0, 'extern')
        return ext


class Path(SetupPackage):
    name = "path"

    def get_extension(self):
        sources = [
            'src/py_converters.cpp',
            'src/_path_wrapper.cpp'
            ]

        ext = make_extension('matplotlib._path', sources)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext)
        return ext


class Image(SetupPackage):
    name = "image"

    def get_extension(self):
        sources = [
            'src/_image.cpp',
            'src/mplutils.cpp',
            'src/_image_wrapper.cpp',
            'src/py_converters.cpp'
            ]
        ext = make_extension('matplotlib._image', sources)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext)

        return ext


class Contour(SetupPackage):
    name = "contour"

    def get_extension(self):
        sources = [
            "src/_contour.cpp",
            "src/_contour_wrapper.cpp",
            'src/py_converters.cpp',
            ]
        ext = make_extension('matplotlib._contour', sources)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext, add_sources=False)
        return ext


class QhullWrap(SetupPackage):
    name = "qhull_wrap"

    def get_extension(self):
        sources = ['src/qhull_wrap.c']
        ext = make_extension('matplotlib._qhull', sources,
                             define_macros=[('MPL_DEVNULL', os.devnull)])
        Numpy().add_flags(ext)
        Qhull().add_flags(ext)
        return ext


class Tri(SetupPackage):
    name = "tri"

    def get_extension(self):
        sources = [
            "lib/matplotlib/tri/_tri.cpp",
            "lib/matplotlib/tri/_tri_wrapper.cpp",
            "src/mplutils.cpp"
            ]
        ext = make_extension('matplotlib._tri', sources)
        Numpy().add_flags(ext)
        return ext


class InstallRequires(SetupPackage):
    name = "install_requires"

    def check(self):
        return "handled by setuptools"

    def get_install_requires(self):
        return [
            "cycler>=0.10",
            "kiwisolver>=1.0.1",
            "pyparsing>=2.0.1,!=2.0.4,!=2.1.2,!=2.1.6",
            "python-dateutil>=2.1",
        ]


class BackendAgg(OptionalBackendPackage):
    name = "agg"
    force = True

    def get_extension(self):
        sources = [
            "src/mplutils.cpp",
            "src/py_converters.cpp",
            "src/_backend_agg.cpp",
            "src/_backend_agg_wrapper.cpp"
            ]
        ext = make_extension('matplotlib.backends._backend_agg', sources)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext)
        FreeType().add_flags(ext)
        return ext


class BackendTkAgg(OptionalBackendPackage):
    name = "tkagg"
    force = True

    def check(self):
        return "installing; run-time loading from Python Tcl / Tk"

    def runtime_check(self):
        """Checks whether TkAgg runtime dependencies are met."""
        return importlib.util.find_spec("tkinter") is not None

    def get_extension(self):
        sources = [
            'src/_tkagg.cpp'
            ]

        ext = make_extension('matplotlib.backends._tkagg', sources)
        self.add_flags(ext)
        LibAgg().add_flags(ext, add_sources=False)
        return ext

    def add_flags(self, ext):
        ext.include_dirs.insert(0, 'src')
        if sys.platform == 'win32':
            # PSAPI library needed for finding Tcl / Tk at run time
            ext.libraries.extend(['psapi'])
        elif sys.platform == 'linux':
            ext.libraries.extend(['dl'])


class BackendGtk3Agg(OptionalBackendPackage):
    name = "gtk3agg"

    def check_requirements(self):
        if not any(map(importlib.util.find_spec, ["cairocffi", "cairo"])):
            raise CheckFailed("Requires cairocffi or pycairo to be installed.")

        try:
            import gi
        except ImportError:
            raise CheckFailed("Requires pygobject to be installed.")

        try:
            gi.require_version("Gtk", "3.0")
        except ValueError:
            raise CheckFailed(
                "Requires gtk3 development files to be installed.")
        except AttributeError:
            raise CheckFailed("pygobject version too old.")

        try:
            from gi.repository import Gtk, Gdk, GObject
        except (ImportError, RuntimeError):
            raise CheckFailed("Requires pygobject to be installed.")

        return "version {}.{}.{}".format(
            Gtk.get_major_version(),
            Gtk.get_minor_version(),
            Gtk.get_micro_version())

    def get_package_data(self):
        return {'matplotlib': ['mpl-data/*.glade']}


class BackendGtk3Cairo(BackendGtk3Agg):
    name = "gtk3cairo"


class BackendWxAgg(OptionalBackendPackage):
    name = "wxagg"

    def check_requirements(self):
        try:
            import wx
            backend_version = wx.VERSION_STRING
        except ImportError:
            raise CheckFailed("requires wxPython")

        return "version %s" % backend_version


class BackendMacOSX(OptionalBackendPackage):
    name = 'macosx'

    def check_requirements(self):
        if sys.platform != 'darwin':
            raise CheckFailed("Mac OS-X only")

        return 'darwin'

    def get_extension(self):
        sources = [
            'src/_macosx.m'
            ]

        ext = make_extension('matplotlib.backends._macosx', sources)
        ext.extra_link_args.extend(['-framework', 'Cocoa'])
        return ext


class Windowing(OptionalBackendPackage):
    """
    Builds the windowing extension.
    """
    name = "windowing"

    def check_requirements(self):
        if sys.platform != 'win32':
            raise CheckFailed("Microsoft Windows only")
        config = self.get_config()
        if config is False:
            raise CheckFailed("skipping due to configuration")
        return ""

    def get_extension(self):
        sources = [
            "src/_windowing.cpp"
            ]
        ext = make_extension('matplotlib._windowing', sources)
        ext.include_dirs.extend(['C:/include'])
        ext.libraries.extend(['user32'])
        ext.library_dirs.extend(['C:/lib'])
        ext.extra_link_args.append("-mwindows")
        return ext


class BackendQtBase(OptionalBackendPackage):

    def convert_qt_version(self, version):
        version = '%x' % version
        temp = []
        while len(version) > 0:
            version, chunk = version[:-2], version[-2:]
            temp.insert(0, str(int(chunk, 16)))
        return '.'.join(temp)

    def check_requirements(self):
        """
        If PyQt4/PyQt5 is already imported, importing PyQt5/PyQt4 will fail
        so we need to test in a subprocess (as for Gtk3).
        """
        try:
            p = multiprocessing.Pool()

        except:
            # Can't do multiprocessing, fall back to normal approach
            # (this will fail if importing both PyQt4 and PyQt5).
            try:
                # Try in-process
                msg = self.callback(self)
            except RuntimeError:
                raise CheckFailed(
                    "Could not import: are PyQt4 & PyQt5 both installed?")

        else:
            # Multiprocessing OK
            try:
                res = p.map_async(self.callback, [self])
                msg = res.get(timeout=10)[0]
            except multiprocessing.TimeoutError:
                p.terminate()
                # No result returned. Probably hanging, terminate the process.
                raise CheckFailed("Check timed out")
            except:
                # Some other error.
                p.close()
                raise
            else:
                # Clean exit
                p.close()
            finally:
                # Tidy up multiprocessing
                p.join()

        return msg


def backend_pyside_internal_check(self):
    try:
        from PySide import __version__
        from PySide import QtCore
    except ImportError:
        raise CheckFailed("PySide not found")
    else:
        return ("Qt: %s, PySide: %s" %
                (QtCore.__version__, __version__))


def backend_pyqt4_internal_check(self):
    try:
        from PyQt4 import QtCore
    except ImportError:
        raise CheckFailed("PyQt4 not found")

    try:
        qt_version = QtCore.QT_VERSION
        pyqt_version_str = QtCore.PYQT_VERSION_STR
    except AttributeError:
        raise CheckFailed('PyQt4 not correctly imported')
    else:
        return ("Qt: %s, PyQt: %s" % (self.convert_qt_version(qt_version), pyqt_version_str))


def backend_qt4_internal_check(self):
    successes = []
    failures = []
    try:
        successes.append(backend_pyside_internal_check(self))
    except CheckFailed as e:
        failures.append(str(e))

    try:
        successes.append(backend_pyqt4_internal_check(self))
    except CheckFailed as e:
        failures.append(str(e))

    if len(successes) == 0:
        raise CheckFailed('; '.join(failures))
    return '; '.join(successes + failures)


class BackendQt4(BackendQtBase):
    name = "qt4agg"

    def __init__(self, *args, **kwargs):
        BackendQtBase.__init__(self, *args, **kwargs)
        self.callback = backend_qt4_internal_check

def backend_pyside2_internal_check(self):
    try:
        from PySide2 import __version__
        from PySide2 import QtCore
    except ImportError:
        raise CheckFailed("PySide2 not found")
    else:
        return ("Qt: %s, PySide2: %s" %
                (QtCore.__version__, __version__))

def backend_pyqt5_internal_check(self):
    try:
        from PyQt5 import QtCore
    except ImportError:
        raise CheckFailed("PyQt5 not found")

    try:
        qt_version = QtCore.QT_VERSION
        pyqt_version_str = QtCore.PYQT_VERSION_STR
    except AttributeError:
        raise CheckFailed('PyQt5 not correctly imported')
    else:
        return ("Qt: %s, PyQt: %s" % (self.convert_qt_version(qt_version), pyqt_version_str))

def backend_qt5_internal_check(self):
    successes = []
    failures = []
    try:
        successes.append(backend_pyside2_internal_check(self))
    except CheckFailed as e:
        failures.append(str(e))

    try:
        successes.append(backend_pyqt5_internal_check(self))
    except CheckFailed as e:
        failures.append(str(e))

    if len(successes) == 0:
        raise CheckFailed('; '.join(failures))
    return '; '.join(successes + failures)

class BackendQt5(BackendQtBase):
    name = "qt5agg"

    def __init__(self, *args, **kwargs):
        BackendQtBase.__init__(self, *args, **kwargs)
        self.callback = backend_qt5_internal_check


class BackendCairo(OptionalBackendPackage):
    name = "cairo"

    def check_requirements(self):
        try:
            import cairocffi
        except ImportError:
            try:
                import cairo
            except ImportError:
                raise CheckFailed("cairocffi or pycairo not found")
            else:
                return "pycairo version %s" % cairo.version
        else:
            return "cairocffi version %s" % cairocffi.version


class OptionalPackageData(OptionalPackage):
    config_category = "package_data"


class Dlls(OptionalPackageData):
    """
    On Windows, this packages any DLL files that can be found in the
    lib/matplotlib/* directories.
    """
    name = "dlls"

    def check_requirements(self):
        if sys.platform != 'win32':
            raise CheckFailed("Microsoft Windows only")

    def get_package_data(self):
        return {'': ['*.dll']}

    @classmethod
    def get_config(cls):
        """
        Look at `setup.cfg` and return one of ["auto", True, False] indicating
        if the package is at default state ("auto"), forced by the user (True)
        or opted-out (False).
        """
        try:
            return config.getboolean(cls.config_category, cls.name)
        except:
            return False  # <-- default
