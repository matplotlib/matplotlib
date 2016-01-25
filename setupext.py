from __future__ import print_function, absolute_import

from distutils import sysconfig
from distutils import version
from distutils.core import Extension
import glob
import io
import multiprocessing
import os
import re
import subprocess
from subprocess import check_output
import sys
import warnings
from textwrap import fill

import versioneer


PY3min = (sys.version_info[0] >= 3)
PY32min = (PY3min and sys.version_info[1] >= 2 or sys.version_info[0] > 3)


# This is the version of FreeType to use when building a local
# version.  It must match the value in
# lib/matplotlib.__init__.py and also needs to be changed below in the
# embedded windows build script (grep for "REMINDER" in this file)
LOCAL_FREETYPE_VERSION = '2.6.1'
# md5 hash of the freetype tarball
LOCAL_FREETYPE_HASH = '348e667d728c597360e4a87c16556597'

if sys.platform != 'win32':
    if not PY3min:
        from commands import getstatusoutput
    else:
        from subprocess import getstatusoutput


if PY3min:
    import configparser
else:
    import ConfigParser as configparser


# matplotlib build options, which can be altered using setup.cfg
options = {
    'display_status': True,
    'verbose': False,
    'backend': None,
    'basedirlist': None
    }


setup_cfg = os.environ.get('MPLSETUPCFG', 'setup.cfg')
if os.path.exists(setup_cfg):
    if PY32min:
        config = configparser.ConfigParser()
    else:
        config = configparser.SafeConfigParser()
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
        options['local_freetype'] = config.get("test", "local_freetype")
else:
    config = None


def get_win32_compiler():
    """
    Determine the compiler being used on win32.
    """
    # Used to determine mingw32 or msvc
    # This is pretty bad logic, someone know a better way?
    for v in sys.argv:
        if 'mingw32' in v:
            return 'mingw32'
    return 'msvc'
win32_compiler = get_win32_compiler()


def extract_versions():
    """
    Extracts version values from the main matplotlib __init__.py and
    returns them as a dictionary.
    """
    with open('lib/matplotlib/__init__.py') as fd:
        for line in fd.readlines():
            if (line.startswith('__version__numpy__')):
                exec(line.strip())
    return locals()


def has_include_file(include_dirs, filename):
    """
    Returns `True` if `filename` can be found in one of the
    directories in `include_dirs`.
    """
    if sys.platform == 'win32':
        include_dirs += os.environ.get('INCLUDE', '.').split(';')
    for dir in include_dirs:
        if os.path.exists(os.path.join(dir, filename)):
            return True
    return False


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

    win_bases = ['win32_static', ]
    # on conda windows, we also add the <installdir>\Library of the local interperter,
    # as conda installs libs/includes there
    if os.getenv('CONDA_DEFAULT_ENV'):
        win_bases.append(os.path.join(os.getenv('CONDA_DEFAULT_ENV'), "Library"))

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
        # gcc includes this dir automatically, so also look for headers in
        # these dirs
        include_dirs.extend(
            os.environ.get('CPLUS_INCLUDE_PATH', '').split(os.pathsep))
    return include_dirs


def is_min_version(found, minversion):
    """
    Returns `True` if `found` is at least as high a version as
    `minversion`.
    """
    expected_version = version.LooseVersion(minversion)
    found_version = version.LooseVersion(found)
    return found_version >= expected_version


# Define the display functions only if display_status is True.
if options['display_status']:
    def print_line(char='='):
        print(char * 76)

    def print_status(package, status):
        initial_indent = "%22s: " % package
        indent = ' ' * 24
        print(fill(str(status), width=76,
                   initial_indent=initial_indent,
                   subsequent_indent=indent))

    def print_message(message):
        indent = ' ' * 24 + "* "
        print(fill(str(message), width=76,
                   initial_indent=indent,
                   subsequent_indent=indent))

    def print_raw(section):
        print(section)
else:
    def print_line(*args, **kwargs):
        pass
    print_status = print_message = print_raw = print_line


# Remove the -Wstrict-prototypesoption, is it's not valid for C++
customize_compiler = sysconfig.customize_compiler


def my_customize_compiler(compiler):
    retval = customize_compiler(compiler)
    try:
        compiler.compiler_so.remove('-Wstrict-prototypes')
    except (ValueError, AttributeError):
        pass
    return retval

sysconfig.customize_compiler = my_customize_compiler


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
    Get the MD5 hash of a given filename.
    """
    import hashlib
    BLOCKSIZE = 1 << 16
    hasher = hashlib.md5()
    with open(filename, 'rb') as fd:
        buf = fd.read(BLOCKSIZE)
        while len(buf) > 0:
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
            try:
                self.pkg_config = os.environ['PKG_CONFIG']
            except KeyError:
                self.pkg_config = 'pkg-config'

            self.set_pkgconfig_path()
            status, output = getstatusoutput(self.pkg_config + " --help")
            self.has_pkgconfig = (status == 0)
            if not self.has_pkgconfig:
                print("IMPORTANT WARNING:")
                print(
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
                output = check_output(command, shell=True,
                                      stderr=subprocess.STDOUT)
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

        status, output = getstatusoutput(
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

    def check(self):
        """
        Checks whether the dependencies are met.  Should raise a
        `CheckFailed` exception if the dependency could not be met,
        otherwise return a string indicating a version number or some
        other message indicating what was found.
        """
        pass

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
            if (not is_min_version(version, min_version)):
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


class OptionalPackage(SetupPackage):
    optional = True
    force = False
    config_category = "packages"

    @classmethod
    def get_config(cls):
        """
        Look at `setup.cfg` and return one of ["auto", True, False] indicating
        if the package is at default state ("auto"), forced by the user (case
        insensitively defined as 1, true, yes, on for True) or opted-out (case
        insensitively defined as 0, false, no, off for False).
        """
        conf = "auto"
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
        major, minor1, minor2, s, tmp = sys.version_info

        if major < 2:
            raise CheckFailed(
                "Requires Python 2.7 or later")
        elif major == 2 and minor1 < 7:
            raise CheckFailed(
                "Requires Python 2.7 or later (in the 2.x series)")
        elif major == 3 and minor1 < 4:
            raise CheckFailed(
                "Requires Python 3.4 or later (in the 3.x series)")

        return sys.version


class Matplotlib(SetupPackage):
    name = "matplotlib"

    def check(self):
        return versioneer.get_version()

    def get_packages(self):
        return [
            'matplotlib',
            'matplotlib.backends',
            'matplotlib.backends.qt_editor',
            'matplotlib.compat',
            'matplotlib.projections',
            'matplotlib.axes',
            'matplotlib.sphinxext',
            'matplotlib.style',
            'matplotlib.testing',
            'matplotlib.testing.jpl_units',
            'matplotlib.tri',
            ]

    def get_py_modules(self):
        return ['pylab']

    def get_package_data(self):
        return {
            'matplotlib':
            [
                'mpl-data/fonts/afm/*.afm',
                'mpl-data/fonts/pdfcorefonts/*.afm',
                'mpl-data/fonts/pdfcorefonts/*.txt',
                'mpl-data/fonts/ttf/*.ttf',
                'mpl-data/fonts/ttf/LICENSE_STIX',
                'mpl-data/fonts/ttf/COPYRIGHT.TXT',
                'mpl-data/fonts/ttf/README.TXT',
                'mpl-data/fonts/ttf/RELEASENOTES.TXT',
                'mpl-data/images/*.xpm',
                'mpl-data/images/*.svg',
                'mpl-data/images/*.gif',
                'mpl-data/images/*.pdf',
                'mpl-data/images/*.png',
                'mpl-data/images/*.ppm',
                'mpl-data/example/*.npy',
                'mpl-data/matplotlibrc',
                'backends/web_backend/*.*',
                'backends/web_backend/jquery/js/*.min.js',
                'backends/web_backend/jquery/css/themes/base/*.min.css',
                'backends/web_backend/jquery/css/themes/base/images/*',
                'backends/web_backend/css/*.*',
                'backends/Matplotlib.nib/*',
                'mpl-data/stylelib/*.mplstyle',
             ]}


class SampleData(OptionalPackage):
    """
    This handles the sample data that ships with matplotlib.  It is
    technically optional, though most often will be desired.
    """
    name = "sample_data"

    def get_package_data(self):
        return {
            'matplotlib':
            [
                'mpl-data/sample_data/*.*',
                'mpl-data/sample_data/axes_grid/*.*',
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
    nose_min_version = '0.11.1'

    def check(self):
        super(Tests, self).check()

        msgs = []
        msg_template = ('{package} is required to run the matplotlib test '
                        'suite. Please install it with pip or your preferred'
                        ' tool to run the test suite')

        bad_nose = msg_template.format(
            package='nose %s or later' % self.nose_min_version
        )
        try:
            import nose
            if is_min_version(nose.__version__, self.nose_min_version):
                msgs += ['using nose version %s' % nose.__version__]
            else:
                msgs += [bad_nose]
        except ImportError:
            msgs += [bad_nose]

        if sys.version_info >= (3, 3):
            msgs += ['using unittest.mock']
        else:
            try:
                import mock
                msgs += ['using mock %s' % mock.__version__]
            except ImportError:
                msgs += [msg_template.format(package='mock')]

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
        super(DelayedExtension, self).__init__(*args, **kwargs)
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
        if sys.version_info[0] >= 3:
            import builtins
            if hasattr(builtins, '__NUMPY_SETUP__'):
                del builtins.__NUMPY_SETUP__
            import imp
            import numpy
            imp.reload(numpy)
        else:
            import __builtin__
            if hasattr(__builtin__, '__NUMPY_SETUP__'):
                del __builtin__.__NUMPY_SETUP__
            import numpy
            reload(numpy)

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

    def get_setup_requires(self):
        return ['numpy>=1.6']

    def get_install_requires(self):
        return ['numpy>=1.6']


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
            ext.include_dirs.append('extern/agg24-svn/include')
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

    def check(self):
        if options.get('local_freetype'):
            return "Using local version for testing"

        if sys.platform == 'win32':
            try:
                check_include_file(get_include_dirs(), 'ft2build.h', 'freetype')
            except CheckFailed:
                check_include_file(get_include_dirs(), 'freetype2\\ft2build.h', 'freetype')
            return 'Using unknown version found on system.'

        status, output = getstatusoutput("freetype-config --ftversion")
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
        if not os.path.isfile(tarball_path):
            tarball_url = 'http://download.savannah.gnu.org/releases/freetype/{0}'.format(tarball)

            print("Downloading {0}".format(tarball_url))
            if sys.version_info[0] == 2:
                from urllib import urlretrieve
            else:
                from urllib.request import urlretrieve

            if not os.path.exists('build'):
                os.makedirs('build')
            urlretrieve(tarball_url, tarball_path)

            if get_file_hash(tarball_path) != LOCAL_FREETYPE_HASH:
                raise IOError("{0} does not match expected hash.".format(tarball))

        print("Building {0}".format(tarball))
        if sys.platform != 'win32':
            # compilation on all other platforms than windows
            cflags = 'CFLAGS="{0} -fPIC" '.format(os.environ.get('CFLAGS', ''))

            subprocess.check_call(
                ['tar', 'zxf', tarball], cwd='build')
            subprocess.check_call(
                [cflags + './configure --with-zlib=no --with-bzip2=no '
                 '--with-png=no --with-harfbuzz=no'], shell=True, cwd=src_path)
            subprocess.check_call(
                [cflags + 'make'], shell=True, cwd=src_path)
        else:
            # compilation on windows
            FREETYPE_BUILD_CMD = """\
call "%ProgramFiles%\\Microsoft SDKs\\Windows\\v7.0\\Bin\\SetEnv.Cmd" /Release /{xXX} /xp
call "{vcvarsall}" {xXX}
set MSBUILD=C:\\Windows\\Microsoft.NET\\Framework\\v4.0.30319\\MSBuild.exe
rd /S /Q %FREETYPE%\\objs
%MSBUILD% %FREETYPE%\\builds\\windows\\{vc20xx}\\freetype.sln /t:Clean;Build /p:Configuration="{config}";Platform={WinXX}
echo Build completed, moving result"
:: move to the "normal" path for the unix builds...
mkdir %FREETYPE%\\objs\\.libs
:: REMINDER: fix when changing the version
copy %FREETYPE%\\objs\\{vc20xx}\\{xXX}\\freetype261.lib %FREETYPE%\\objs\\.libs\\libfreetype.lib 
if errorlevel 1 (
  rem This is a py27 version, which has a different location for the lib file :-/
  copy %FREETYPE%\\objs\\win32\\{vc20xx}\\freetype261.lib %FREETYPE%\\objs\\.libs\\libfreetype.lib
)
"""
            from setup_external_compile import fixproj, prepare_build_cmd, VS2010, X64, tar_extract
            # Note: freetype has no build profile for 2014, so we don't bother...
            vc = 'vc2010' if VS2010 else 'vc2008'
            WinXX = 'x64' if X64 else 'Win32'
            tar_extract(tarball_path, "build")
            # This is only false for py2.7, even on py3.5...
            if not VS2010:
                fixproj(os.path.join(src_path, 'builds', 'windows', vc, 'freetype.sln'), WinXX)
                fixproj(os.path.join(src_path, 'builds', 'windows', vc, 'freetype.vcproj'), WinXX)

            cmdfile = os.path.join("build", 'build_freetype.cmd')
            with open(cmdfile, 'w') as cmd:
                cmd.write(prepare_build_cmd(FREETYPE_BUILD_CMD, vc20xx=vc, WinXX=WinXX,
                                            config='Release' if VS2010 else 'LIB Release'))

            os.environ['FREETYPE'] = src_path
            subprocess.check_call([cmdfile], shell=True)


class FT2Font(SetupPackage):
    name = 'ft2font'

    def get_extension(self):
        sources = [
            'src/ft2font.cpp',
            'src/ft2font_wrapper.cpp',
            'src/mplutils.cpp'
            ]
        ext = make_extension('matplotlib.ft2font', sources)
        FreeType().add_flags(ext)
        Numpy().add_flags(ext)
        return ext


class Png(SetupPackage):
    name = "png"

    def check(self):
        if sys.platform == 'win32':
            check_include_file(get_include_dirs(), 'png.h', 'png')
            return 'Using unknown version found on system.'

        status, output = getstatusoutput("libpng-config --version")
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
                'qhull', 'qhull/qhull_a.h', min_version='2003.1')
        except CheckFailed as e:
            self.__class__.found_pkgconfig = False
            # Qhull may not be in the pkg-config system but may still be
            # present on this system, so check if the header files can be
            # found.
            include_dirs = [
                os.path.join(x, 'qhull') for x in get_include_dirs()]
            if has_include_file(include_dirs, 'qhull_a.h'):
                return 'Using system Qhull (version unknown, no pkg-config info)'
            else:
                self.__class__.found_external = False
                return str(e) + ' Using local copy.'

    def add_flags(self, ext):
        if self.found_external:
            pkg_config.setup_extension(ext, 'qhull',
                                       default_libraries=['qhull'])
        else:
            ext.include_dirs.append('extern')
            ext.sources.extend(glob.glob('extern/qhull/*.c'))


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
        ext.include_dirs.append('extern')
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
            'src/_image_wrapper.cpp'
            ]
        ext = make_extension('matplotlib._image', sources)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext)
        return ext


class ContourLegacy(SetupPackage):
    name = "contour_legacy"

    def get_extension(self):
        sources = [
            "src/cntr.c"
            ]
        ext = make_extension('matplotlib._cntr', sources)
        Numpy().add_flags(ext)
        return ext


class Contour(SetupPackage):
    name = "contour"

    def get_extension(self):
        sources = [
            "src/_contour.cpp",
            "src/_contour_wrapper.cpp",
            ]
        ext = make_extension('matplotlib._contour', sources)
        Numpy().add_flags(ext)
        return ext


class Delaunay(SetupPackage):
    name = "delaunay"

    def get_packages(self):
        return ['matplotlib.delaunay']

    def get_extension(self):
        sources = ["_delaunay.cpp", "VoronoiDiagramGenerator.cpp",
                   "delaunay_utils.cpp", "natneighbors.cpp"]
        sources = [os.path.join('lib/matplotlib/delaunay', s) for s in sources]
        ext = make_extension('matplotlib._delaunay', sources)
        Numpy().add_flags(ext)
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


class Externals(SetupPackage):
    name = "externals"

    def get_packages(self):
        return ['matplotlib.externals']


class Pytz(SetupPackage):
    name = "pytz"

    def check(self):
        try:
            import pytz
        except ImportError:
            return (
                "pytz was not found. "
                "pip/easy_install may attempt to install it "
                "after matplotlib.")

        return "using pytz version %s" % pytz.__version__

    def get_install_requires(self):
        return ['pytz']


class Cycler(SetupPackage):
    name = "cycler"

    def check(self):
        try:
            import cycler
        except ImportError:
            return (
                "cycler was not found. "
                "pip/easy_install may attempt to install it "
                "after matplotlib.")

        return "using cycler version %s" % cycler.__version__

    def get_install_requires(self):
        return ['cycler']


class Dateutil(SetupPackage):
    name = "dateutil"

    def __init__(self, version=None):
        self.version = version

    def check(self):
        try:
            import dateutil
        except ImportError:
            # dateutil 2.1 has a file encoding bug that breaks installation on
            # python 3.3
            # https://github.com/matplotlib/matplotlib/issues/2373
            # hack around the problem by installing the (working) v2.0
            major, minor1, _, _, _ = sys.version_info
            if self.version is None and (major, minor1) == (3, 3):
                self.version = '!=2.1'

            return (
                "dateutil was not found. It is required for date axis "
                "support. pip/easy_install may attempt to install it "
                "after matplotlib.")

        return "using dateutil version %s" % dateutil.__version__

    def get_install_requires(self):
        dateutil = 'python-dateutil'
        if self.version is not None:
            dateutil += self.version
        return [dateutil]


class FuncTools32(SetupPackage):
    name = "functools32"

    def check(self):
        if sys.version_info[:2] < (3, 2):
            try:
                import functools32
            except ImportError:
                return (
                    "functools32 was not found. It is required for for"
                    "python versions prior to 3.2 "
                    "pip/easy_install may attempt to install it "
                    "after matplotlib.")

            return "using functools32"
        else:
            return "Not required"

    def get_install_requires(self):
        if sys.version_info[:2] < (3, 2):
            return ['functools32']
        else:
            return []


class Tornado(OptionalPackage):
    name = "tornado"

    def check(self):
        try:
            import tornado
        except ImportError:
            return (
                "tornado was not found. It is required for the WebAgg "
                "backend. pip/easy_install may attempt to install it "
                "after matplotlib.")

        return "using tornado version %s" % tornado.version


class Pyparsing(SetupPackage):
    name = "pyparsing"
    # pyparsing 2.0.4 has broken python 3 support.
    def is_ok(self):
        # pyparsing 2.0.0 bug, but it may be patched in distributions
        try:
            import pyparsing
            f = pyparsing.Forward()
            f <<= pyparsing.Literal('a')
            return f is not None
        except (ImportError, TypeError):
            return False

    def check(self):
        try:
            import pyparsing
        except ImportError:
            return (
                "pyparsing was not found. It is required for mathtext "
                "support. pip/easy_install may attempt to install it "
                "after matplotlib.")

        required = [1, 5, 6]
        if [int(x) for x in pyparsing.__version__.split('.')] < required:
            return (
                "matplotlib requires pyparsing >= {0}".format(
                    '.'.join(str(x) for x in required)))

        if not self.is_ok():
            return (
                "Your pyparsing contains a bug that will be monkey-patched by "
                "matplotlib.  For best results, upgrade to pyparsing 2.0.1 or "
                "later.")

        return "using pyparsing version %s" % pyparsing.__version__

    def get_install_requires(self):
        if self.is_ok():
            return ['pyparsing>=1.5.6,!=2.0.4']
        else:
            return ['pyparsing>=1.5.6,!=2.0.0,!=2.0.4']


class BackendAgg(OptionalBackendPackage):
    name = "agg"

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

    def __init__(self):
        self.tcl_tk_cache = None

    def check_requirements(self):
        try:
            if PY3min:
                import tkinter as Tkinter
            else:
                import Tkinter
        except ImportError:
            raise CheckFailed('TKAgg requires Tkinter.')
        except RuntimeError:
            raise CheckFailed('Tkinter present but import failed.')
        else:
            if Tkinter.TkVersion < 8.3:
                raise CheckFailed("Tcl/Tk v8.3 or later required.")

        ext = self.get_extension()
        check_include_file(ext.include_dirs, "tk.h", "Tk")

        try:
            tk_v = Tkinter.__version__.split()[-2]
        except (AttributeError, IndexError):
            # Tkinter.__version__ has been removed in python 3
            tk_v = 'not identified'

        BackendAgg.force = True

        return "version %s" % tk_v

    def get_extension(self):
        sources = [
            'src/py_converters.cpp',
            'src/_tkagg.cpp'
            ]

        ext = make_extension('matplotlib.backends._tkagg', sources)
        self.add_flags(ext)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext, add_sources=False)
        return ext

    def query_tcltk(self):
        """
        Tries to open a Tk window in order to query the Tk object
        about its library paths.  This should never be called more
        than once by the same process, as Tk intricacies may cause the
        Python interpreter to hang. The function also has a workaround
        if no X server is running (useful for autobuild systems).
        """
        # Use cached values if they exist, which ensures this function
        # only executes once
        if self.tcl_tk_cache is not None:
            return self.tcl_tk_cache

        # By this point, we already know that Tkinter imports correctly
        if PY3min:
            import tkinter as Tkinter
        else:
            import Tkinter
        tcl_lib_dir = ''
        tk_lib_dir = ''
        # First try to open a Tk window (requires a running X server)
        try:
            tk = Tkinter.Tk()
        except Tkinter.TclError:
            # Next, start Tcl interpreter without opening a Tk window
            # (no need for X server) This feature is available in
            # python version 2.4 and up
            try:
                tcl = Tkinter.Tcl()
            except AttributeError:    # Python version not high enough
                pass
            except Tkinter.TclError:  # Something went wrong while opening Tcl
                pass
            else:
                tcl_lib_dir = str(tcl.getvar('tcl_library'))
                # Guess Tk location based on Tcl location
                (head, tail) = os.path.split(tcl_lib_dir)
                tail = tail.replace('Tcl', 'Tk').replace('tcl', 'tk')
                tk_lib_dir = os.path.join(head, tail)
                if not os.path.exists(tk_lib_dir):
                    tk_lib_dir = tcl_lib_dir.replace(
                        'Tcl', 'Tk').replace('tcl', 'tk')
        else:
            # Obtain Tcl and Tk locations from Tk widget
            tk.withdraw()
            tcl_lib_dir = str(tk.getvar('tcl_library'))
            tk_lib_dir = str(tk.getvar('tk_library'))
            tk.destroy()

        # Save directories and version string to cache
        self.tcl_tk_cache = tcl_lib_dir, tk_lib_dir, str(Tkinter.TkVersion)[:3]
        return self.tcl_tk_cache

    def parse_tcl_config(self, tcl_lib_dir, tk_lib_dir):
        try:
            if PY3min:
                import tkinter as Tkinter
            else:
                import Tkinter
        except ImportError:
            return None

        tcl_poss = [tcl_lib_dir,
                    os.path.normpath(os.path.join(tcl_lib_dir, '..')),
                    "/usr/lib/tcl" + str(Tkinter.TclVersion),
                    "/usr/lib"]
        tk_poss = [tk_lib_dir,
                    os.path.normpath(os.path.join(tk_lib_dir, '..')),
                   "/usr/lib/tk" + str(Tkinter.TkVersion),
                   "/usr/lib"]
        for ptcl, ptk in zip(tcl_poss, tk_poss):
            tcl_config = os.path.join(ptcl, "tclConfig.sh")
            tk_config = os.path.join(ptk, "tkConfig.sh")
            if (os.path.exists(tcl_config) and os.path.exists(tk_config)):
                break
        if not (os.path.exists(tcl_config) and os.path.exists(tk_config)):
            return None

        def get_var(file, varname):
            p = subprocess.Popen(
                '. %s ; eval echo ${%s}' % (file, varname),
                shell=True,
                executable="/bin/sh",
                stdout=subprocess.PIPE)
            result = p.communicate()[0]
            return result.decode('ascii')

        tcl_lib_dir = get_var(
            tcl_config, 'TCL_LIB_SPEC').split()[0][2:].strip()
        tcl_inc_dir = get_var(
            tcl_config, 'TCL_INCLUDE_SPEC')[2:].strip()
        tcl_lib = get_var(tcl_config, 'TCL_LIB_FLAG')[2:].strip()

        tk_lib_dir = get_var(tk_config, 'TK_LIB_SPEC').split()[0][2:].strip()
        tk_inc_dir = get_var(tk_config, 'TK_INCLUDE_SPEC').strip()
        if tk_inc_dir == '':
            tk_inc_dir = tcl_inc_dir
        else:
            tk_inc_dir = tk_inc_dir[2:]
        tk_lib = get_var(tk_config, 'TK_LIB_FLAG')[2:].strip()

        if not os.path.exists(os.path.join(tk_inc_dir, 'tk.h')):
            return None

        return (tcl_lib_dir, tcl_inc_dir, tcl_lib,
                tk_lib_dir, tk_inc_dir, tk_lib)

    def guess_tcl_config(self, tcl_lib_dir, tk_lib_dir, tk_ver):
        if not (os.path.exists(tcl_lib_dir) and os.path.exists(tk_lib_dir)):
            return None

        tcl_lib = os.path.normpath(os.path.join(tcl_lib_dir, '../'))
        tk_lib = os.path.normpath(os.path.join(tk_lib_dir, '../'))

        tcl_inc = os.path.normpath(
            os.path.join(tcl_lib_dir,
                         '../../include/tcl' + tk_ver))
        if not os.path.exists(tcl_inc):
            tcl_inc = os.path.normpath(
                os.path.join(tcl_lib_dir,
                             '../../include'))

        tk_inc = os.path.normpath(os.path.join(
            tk_lib_dir,
            '../../include/tk' + tk_ver))
        if not os.path.exists(tk_inc):
            tk_inc = os.path.normpath(os.path.join(
                tk_lib_dir,
                '../../include'))

        if not os.path.exists(os.path.join(tk_inc, 'tk.h')):
            tk_inc = tcl_inc

        if not os.path.exists(tcl_inc):
            # this is a hack for suse linux, which is broken
            if (sys.platform.startswith('linux') and
                os.path.exists('/usr/include/tcl.h') and
                os.path.exists('/usr/include/tk.h')):
                tcl_inc = '/usr/include'
                tk_inc = '/usr/include'

        if not os.path.exists(os.path.join(tk_inc, 'tk.h')):
            return None

        return tcl_lib, tcl_inc, 'tcl' + tk_ver, tk_lib, tk_inc, 'tk' + tk_ver

    def hardcoded_tcl_config(self):
        tcl_inc = "/usr/local/include"
        tk_inc = "/usr/local/include"
        tcl_lib = "/usr/local/lib"
        tk_lib = "/usr/local/lib"
        return tcl_lib, tcl_inc, 'tcl', tk_lib, tk_inc, 'tk'

    def add_flags(self, ext):
        if sys.platform == 'win32':
            if os.getenv('CONDA_DEFAULT_ENV'):
                # We are in conda and conda builds against tcl85 for all versions
                # includes are directly in the conda\library\include dir and
                # libs in DLL or lib
                ext.include_dirs.extend(['include'])
                ext.libraries.extend(['tk85', 'tcl85'])
                ext.library_dirs.extend(['dlls']) # or lib?
            else:
                major, minor1, minor2, s, tmp = sys.version_info
                if sys.version_info[0:2] < (3, 4):
                    ext.include_dirs.extend(['win32_static/include/tcl85'])
                    ext.libraries.extend(['tk85', 'tcl85'])
                else:
                    ext.include_dirs.extend(['win32_static/include/tcl86'])
                    ext.libraries.extend(['tk86t', 'tcl86t'])
                ext.library_dirs.extend([os.path.join(sys.prefix, 'dlls')])

        elif sys.platform == 'darwin':
            # this config section lifted directly from Imaging - thanks to
            # the effbot!

            # First test for a MacOSX/darwin framework install
            from os.path import join, exists
            framework_dirs = [
                join(os.getenv('HOME'), '/Library/Frameworks'),
                '/Library/Frameworks',
                '/System/Library/Frameworks/',
            ]

            # Find the directory that contains the Tcl.framework and
            # Tk.framework bundles.
            tk_framework_found = 0
            for F in framework_dirs:
                # both Tcl.framework and Tk.framework should be present
                for fw in 'Tcl', 'Tk':
                    if not exists(join(F, fw + '.framework')):
                        break
                else:
                    # ok, F is now directory with both frameworks. Continure
                    # building
                    tk_framework_found = 1
                    break
            if tk_framework_found:
                # For 8.4a2, we must add -I options that point inside
                # the Tcl and Tk frameworks. In later release we
                # should hopefully be able to pass the -F option to
                # gcc, which specifies a framework lookup path.

                tk_include_dirs = [
                    join(F, fw + '.framework', H)
                    for fw in ('Tcl', 'Tk')
                    for H in ('Headers', 'Versions/Current/PrivateHeaders')
                ]

                # For 8.4a2, the X11 headers are not included. Rather
                # than include a complicated search, this is a
                # hard-coded path. It could bail out if X11 libs are
                # not found...

                # tk_include_dirs.append('/usr/X11R6/include')
                frameworks = ['-framework', 'Tcl', '-framework', 'Tk']
                ext.include_dirs.extend(tk_include_dirs)
                ext.extra_link_args.extend(frameworks)
                ext.extra_compile_args.extend(frameworks)

        # you're still here? ok we'll try it this way...
        else:
            # There are 3 methods to try, in decreasing order of "smartness"
            #
            #   1. Parse the tclConfig.sh and tkConfig.sh files that have
            #      all the information we need
            #
            #   2. Guess the include and lib dirs based on the location of
            #      Tkinter's 'tcl_library' and 'tk_library' variables.
            #
            #   3. Use some hardcoded locations that seem to work on a lot
            #      of distros.

            # Query Tcl/Tk system for library paths and version string
            try:
                tcl_lib_dir, tk_lib_dir, tk_ver = self.query_tcltk()
            except:
                tk_ver = ''
                result = self.hardcoded_tcl_config()
            else:
                result = self.parse_tcl_config(tcl_lib_dir, tk_lib_dir)
                if result is None:
                    result = self.guess_tcl_config(
                        tcl_lib_dir, tk_lib_dir, tk_ver)
                    if result is None:
                        result = self.hardcoded_tcl_config()

            # Add final versions of directories and libraries to ext lists
            (tcl_lib_dir, tcl_inc_dir, tcl_lib,
             tk_lib_dir, tk_inc_dir, tk_lib) = result
            ext.include_dirs.extend([tcl_inc_dir, tk_inc_dir])
            ext.library_dirs.extend([tcl_lib_dir, tk_lib_dir])
            ext.libraries.extend([tcl_lib, tk_lib])


class BackendGtk(OptionalBackendPackage):
    name = "gtk"

    def check_requirements(self):
        try:
            import gtk
        except ImportError:
            raise CheckFailed("Requires pygtk")
        except RuntimeError:
            raise CheckFailed('pygtk present, but import failed.')
        else:
            version = (2, 2, 0)
            if gtk.pygtk_version < version:
                raise CheckFailed(
                    "Requires pygtk %d.%d.%d or later. "
                    "Found %d.%d.%d" % (version + gtk.pygtk_version))

        ext = self.get_extension()
        self.add_flags(ext)
        check_include_file(ext.include_dirs,
                           os.path.join("gtk", "gtk.h"),
                           'gtk')
        check_include_file(ext.include_dirs,
                           os.path.join("pygtk", "pygtk.h"),
                           'pygtk')

        return 'Gtk: %s pygtk: %s' % (
            ".".join(str(x) for x in gtk.gtk_version),
            ".".join(str(x) for x in gtk.pygtk_version))

    def get_package_data(self):
        return {'matplotlib': ['mpl-data/*.glade']}

    def get_extension(self):
        sources = [
            'src/_backend_gdk.c'
            ]
        ext = make_extension('matplotlib.backends._backend_gdk', sources)
        self.add_flags(ext)
        Numpy().add_flags(ext)
        return ext

    def add_flags(self, ext):
        if sys.platform == 'win32':
            def getoutput(s):
                ret = os.popen(s).read().strip()
                return ret

            if 'PKG_CONFIG_PATH' not in os.environ:
                # If Gtk+ is installed, pkg-config is required to be installed
                os.environ['PKG_CONFIG_PATH'] = 'C:\\GTK\\lib\\pkgconfig'

                # popen broken on my win32 plaform so I can't use pkgconfig
                ext.library_dirs.extend(
                    ['C:/GTK/bin', 'C:/GTK/lib'])

                ext.include_dirs.extend(
                    ['win32_static/include/pygtk-2.0',
                     'C:/GTK/include',
                     'C:/GTK/include/gobject',
                     'C:/GTK/include/gext',
                     'C:/GTK/include/glib',
                     'C:/GTK/include/pango',
                     'C:/GTK/include/atk',
                     'C:/GTK/include/X11',
                     'C:/GTK/include/cairo',
                     'C:/GTK/include/gdk',
                     'C:/GTK/include/gdk-pixbuf',
                     'C:/GTK/include/gtk',
                     ])

            pygtkIncludes = getoutput(
                'pkg-config --cflags-only-I pygtk-2.0').split()
            gtkIncludes = getoutput(
                'pkg-config --cflags-only-I gtk+-2.0').split()
            includes = pygtkIncludes + gtkIncludes
            ext.include_dirs.extend([include[2:] for include in includes])

            pygtkLinker = getoutput('pkg-config --libs pygtk-2.0').split()
            gtkLinker = getoutput('pkg-config --libs gtk+-2.0').split()
            linkerFlags = pygtkLinker + gtkLinker

            ext.libraries.extend(
                [flag[2:] for flag in linkerFlags if flag.startswith('-l')])

            ext.library_dirs.extend(
                [flag[2:] for flag in linkerFlags if flag.startswith('-L')])

            ext.extra_link_args.extend(
                [flag for flag in linkerFlags if not
                 (flag.startswith('-l') or flag.startswith('-L'))])

            # visual studio doesn't need the math library
            if (sys.platform == 'win32' and
                win32_compiler == 'msvc' and
                'm' in ext.libraries):
                ext.libraries.remove('m')

        elif sys.platform != 'win32':
            pkg_config.setup_extension(ext, 'pygtk-2.0')
            pkg_config.setup_extension(ext, 'gtk+-2.0')


class BackendGtkAgg(BackendGtk):
    name = "gtkagg"

    def check(self):
        try:
            return super(BackendGtkAgg, self).check()
        except:
            raise
        else:
            BackendAgg.force = True

    def get_package_data(self):
        return {'matplotlib': ['mpl-data/*.glade']}

    def get_extension(self):
        sources = [
            'src/py_converters.cpp',
            'src/_gtkagg.cpp',
            'src/mplutils.cpp'
            ]
        ext = make_extension('matplotlib.backends._gtkagg', sources)
        self.add_flags(ext)
        LibAgg().add_flags(ext)
        Numpy().add_flags(ext)
        return ext


def backend_gtk3agg_internal_check(x):
    try:
        import gi
    except ImportError:
        return (False, "Requires pygobject to be installed.")

    try:
        gi.require_version("Gtk", "3.0")
    except ValueError:
        return (False, "Requires gtk3 development files to be installed.")
    except AttributeError:
        return (False, "pygobject version too old.")

    try:
        from gi.repository import Gtk, Gdk, GObject
    except (ImportError, RuntimeError):
        return (False, "Requires pygobject to be installed.")

    return (True, "version %s.%s.%s" % (
        Gtk.get_major_version(),
        Gtk.get_micro_version(),
        Gtk.get_minor_version()))


class BackendGtk3Agg(OptionalBackendPackage):
    name = "gtk3agg"

    def check_requirements(self):
        if 'TRAVIS' in os.environ:
            raise CheckFailed("Can't build with Travis")

        # This check needs to be performed out-of-process, because
        # importing gi and then importing regular old pygtk afterward
        # segfaults the interpreter.
        try:
            p = multiprocessing.Pool()
        except:
            return "unknown (can not use multiprocessing to determine)"
        try:
            res = p.map_async(backend_gtk3agg_internal_check, [0])
            success, msg = res.get(timeout=10)[0]
        except multiprocessing.TimeoutError:
            p.terminate()
            # No result returned. Probaly hanging, terminate the process.
            success = False
            raise CheckFailed("Check timed out")
        except:
            p.close()
            # Some other error.
            success = False
            msg = "Could not determine"
            raise
        else:
            p.close()
        finally:
            p.join()

        if success:
            BackendAgg.force = True
            return msg
        else:
            raise CheckFailed(msg)

    def get_package_data(self):
        return {'matplotlib': ['mpl-data/*.glade']}


def backend_gtk3cairo_internal_check(x):
    try:
        import cairocffi
    except ImportError:
        try:
            import cairo
        except ImportError:
            return (False, "Requires cairocffi or pycairo to be installed.")

    try:
        import gi
    except ImportError:
        return (False, "Requires pygobject to be installed.")

    try:
        gi.require_version("Gtk", "3.0")
    except ValueError:
        return (False, "Requires gtk3 development files to be installed.")
    except AttributeError:
        return (False, "pygobject version too old.")

    try:
        from gi.repository import Gtk, Gdk, GObject
    except (RuntimeError, ImportError):
        return (False, "Requires pygobject to be installed.")

    return (True, "version %s.%s.%s" % (
        Gtk.get_major_version(),
        Gtk.get_micro_version(),
        Gtk.get_minor_version()))


class BackendGtk3Cairo(OptionalBackendPackage):
    name = "gtk3cairo"

    def check_requirements(self):
        if 'TRAVIS' in os.environ:
            raise CheckFailed("Can't build with Travis")

        # This check needs to be performed out-of-process, because
        # importing gi and then importing regular old pygtk afterward
        # segfaults the interpreter.
        try:
            p = multiprocessing.Pool()
        except:
            return "unknown (can not use multiprocessing to determine)"
        try:
            res = p.map_async(backend_gtk3cairo_internal_check, [0])
            success, msg = res.get(timeout=10)[0]
        except multiprocessing.TimeoutError:
            p.terminate()
            # No result returned. Probaly hanging, terminate the process.
            success = False
            raise CheckFailed("Check timed out")
        except:
            p.close()
            success = False
            raise
        else:
            p.close()
        finally:
            p.join()

        if success:
            BackendAgg.force = True
            return msg
        else:
            raise CheckFailed(msg)

    def get_package_data(self):
        return {'matplotlib': ['mpl-data/*.glade']}


class BackendWxAgg(OptionalBackendPackage):
    name = "wxagg"

    def check_requirements(self):
        wxversioninstalled = True
        try:
            import wxversion
        except ImportError:
            wxversioninstalled = False

        if wxversioninstalled:
            try:
                _wx_ensure_failed = wxversion.AlreadyImportedError
            except AttributeError:
                _wx_ensure_failed = wxversion.VersionError

            try:
                wxversion.ensureMinimal('2.8')
            except _wx_ensure_failed:
                pass

        try:
            import wx
            backend_version = wx.VERSION_STRING
        except ImportError:
            raise CheckFailed("requires wxPython")

        # Extra version check in case wxversion lacks AlreadyImportedError;
        # then VersionError might have been raised and ignored when
        # there really *is* a problem with the version.
        major, minor = [int(n) for n in backend_version.split('.')[:2]]
        if major < 2 or (major < 3 and minor < 8):
            raise CheckFailed(
                "Requires wxPython 2.8, found %s" % backend_version)

        BackendAgg.force = True

        return "version %s" % backend_version


class BackendMacOSX(OptionalBackendPackage):
    name = 'macosx'

    def check_requirements(self):
        if sys.platform != 'darwin':
            raise CheckFailed("Mac OS-X only")

        return 'darwin'

    def get_extension(self):
        sources = [
            'src/_macosx.m',
            'src/py_converters.cpp',
            'src/path_cleanup.cpp'
            ]

        ext = make_extension('matplotlib.backends._macosx', sources)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext)
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
        return "installing"

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
        '''
        If PyQt4/PyQt5 is already imported, importing PyQt5/PyQt4 will fail
        so we need to test in a subprocess (as for Gtk3).
        '''
        try:
            p = multiprocessing.Pool()

        except:
            # Can't do multiprocessing, fall back to normal approach ( this will fail if importing both PyQt4 and PyQt5 )
            try:
                # Try in-process
                msg = self.callback(self)

            except RuntimeError:
                raise CheckFailed("Could not import: are PyQt4 & PyQt5 both installed?")

            except:
                # Raise any other exceptions
                raise

        else:
            # Multiprocessing OK
            try:
                res = p.map_async(self.callback, [self])
                msg = res.get(timeout=10)[0]
            except multiprocessing.TimeoutError:
                p.terminate()
                # No result returned. Probaly hanging, terminate the process.
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
        BackendAgg.force = True
        return ("Qt: %s, PySide: %s" %
                (QtCore.__version__, __version__))


def backend_pyqt4_internal_check(self):
    try:
        from PyQt4 import QtCore
    except ImportError:
        raise CheckFailed("PyQt4 not found")

    try:
        qt_version = QtCore.QT_VERSION
        pyqt_version_str = QtCore.QT_VERSION_STR
    except AttributeError:
        raise CheckFailed('PyQt4 not correctly imported')
    else:
        BackendAgg.force = True
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


def backend_qt5_internal_check(self):
    try:
        from PyQt5 import QtCore
    except ImportError:
        raise CheckFailed("PyQt5 not found")

    try:
        qt_version = QtCore.QT_VERSION
        pyqt_version_str = QtCore.QT_VERSION_STR
    except AttributeError:
        raise CheckFailed('PyQt5 not correctly imported')
    else:
        BackendAgg.force = True
        return ("Qt: %s, PyQt: %s" % (self.convert_qt_version(qt_version), pyqt_version_str))


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


class DviPng(SetupPackage):
    name = "dvipng"
    optional = True

    def check(self):
        try:
            output = check_output('dvipng -version', shell=True,
                                  stderr=subprocess.STDOUT)
            return "version %s" % output.splitlines()[1].decode().split()[-1]
        except (IndexError, ValueError, subprocess.CalledProcessError):
            raise CheckFailed()


class Ghostscript(SetupPackage):
    name = "ghostscript"
    optional = True

    def check(self):
        try:
            if sys.platform == 'win32':
                command = 'gswin32c --version'
                try:
                    output = check_output(command, shell=True,
                                          stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError:
                    command = 'gswin64c --version'
                    output = check_output(command, shell=True,
                                          stderr=subprocess.STDOUT)
            else:
                command = 'gs --version'
                output = check_output(command, shell=True,
                                      stderr=subprocess.STDOUT)
            return "version %s" % output.decode()[:-1]
        except (IndexError, ValueError, subprocess.CalledProcessError):
            raise CheckFailed()


class LaTeX(SetupPackage):
    name = "latex"
    optional = True

    def check(self):
        try:
            output = check_output('latex -version', shell=True,
                                  stderr=subprocess.STDOUT)
            line = output.splitlines()[0].decode()
            pattern = '(3\.1\d+)|(MiKTeX \d+.\d+)'
            match = re.search(pattern, line)
            return "version %s" % match.group(0)
        except (IndexError, ValueError, AttributeError, subprocess.CalledProcessError):
            raise CheckFailed()


class PdfToPs(SetupPackage):
    name = "pdftops"
    optional = True

    def check(self):
        try:
            output = check_output('pdftops -v', shell=True,
                                  stderr=subprocess.STDOUT)
            for line in output.splitlines():
                line = line.decode()
                if 'version' in line:
                    return "version %s" % line.split()[2]
        except (IndexError, ValueError, subprocess.CalledProcessError):
            pass

        raise CheckFailed()


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
