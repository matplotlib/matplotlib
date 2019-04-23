import builtins
import configparser
from distutils import sysconfig, version
from distutils.core import Extension
from io import BytesIO
import glob
import hashlib
import importlib
import logging
import os
import pathlib
import platform
import setuptools
import shlex
import shutil
import subprocess
import sys
import tarfile
import textwrap
import urllib.request
from urllib.request import Request
import versioneer
import warnings

_log = logging.getLogger(__name__)


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


def get_fd_hash(fd):
    """
    Compute the sha256 hash of the bytes in a file-like
    """
    BLOCKSIZE = 1 << 16
    hasher = hashlib.sha256()
    old_pos = fd.tell()
    fd.seek(0)
    buf = fd.read(BLOCKSIZE)
    while buf:
        hasher.update(buf)
        buf = fd.read(BLOCKSIZE)
    fd.seek(old_pos)
    return hasher.hexdigest()


def download_or_cache(url, sha):
    """
    Get bytes from the given url or local cache.

    Parameters
    ----------
    url : str
        The url to download

    sha : str
        The sha256 of the file

    Returns
    -------
    BytesIO
        The file loaded into memory.
    """
    cache_dir = _get_xdg_cache_dir()

    def get_from_cache(local_fn):
        if cache_dir is None:
            raise Exception("no cache dir")
        buf = BytesIO(pathlib.Path(cache_dir, local_fn).read_bytes())
        if get_fd_hash(buf) != sha:
            return None
        buf.seek(0)
        return buf

    def write_cache(local_fn, data):
        if cache_dir is None:
            raise Exception("no cache dir")
        cache_filename = os.path.join(cache_dir, local_fn)
        os.makedirs(cache_dir, exist_ok=True)
        old_pos = data.tell()
        data.seek(0)
        with open(cache_filename, "xb") as fout:
            fout.write(data.read())
        data.seek(old_pos)

    try:
        return get_from_cache(sha)
    except Exception:
        pass

    # jQueryUI's website blocks direct downloads from urllib.request's
    # default User-Agent, but not (for example) wget; so I don't feel too
    # bad passing in an empty User-Agent.
    with urllib.request.urlopen(
            Request(url, headers={"User-Agent": ""})) as req:
        file_contents = BytesIO(req.read())
        file_contents.seek(0)

    file_sha = get_fd_hash(file_contents)

    if file_sha != sha:
        raise Exception(("The download file does not match the "
                         "expected sha.  {url} was expected to have "
                         "{sha} but it had {file_sha}").format(
                             sha=sha, file_sha=file_sha, url=url))

    try:
        write_cache(sha, file_contents)
    except Exception:
        pass

    file_contents.seek(0)
    return file_contents


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
    }


setup_cfg = os.environ.get('MPLSETUPCFG', 'setup.cfg')
if os.path.exists(setup_cfg):
    config = configparser.ConfigParser()
    config.read(setup_cfg)

    if config.has_option('status', 'suppress'):
        options['display_status'] = not config.getboolean("status", "suppress")

    if config.has_option('rc_options', 'backend'):
        options['backend'] = config.get("rc_options", "backend")

    if config.has_option('test', 'local_freetype'):
        options['local_freetype'] = config.getboolean("test", "local_freetype")
else:
    config = None

lft = bool(os.environ.get('MPLLOCALFREETYPE', False))
options['local_freetype'] = lft or options.get('local_freetype', False)


def is_min_version(found, minversion):
    """
    Returns whether *found* is a version at least as high as *minversion*.
    """
    return version.LooseVersion(found) >= version.LooseVersion(minversion)


# Define the display functions only if display_status is True.
if options['display_status']:
    def print_line(char='='):
        print(char * 80)

    def print_status(package, status):
        initial_indent = "%12s: " % package
        indent = ' ' * 18
        print(textwrap.fill(str(status), width=80,
                            initial_indent=initial_indent,
                            subsequent_indent=indent))

    def print_message(message):
        indent = ' ' * 18 + "* "
        print(textwrap.fill(str(message), width=80,
                            initial_indent=indent,
                            subsequent_indent=indent))

    def print_raw(section):
        print(section)
else:
    def print_line(*args, **kwargs):
        pass
    print_status = print_message = print_raw = print_line


def get_buffer_hash(fd):
    BLOCKSIZE = 1 << 16
    hasher = hashlib.sha256()
    buf = fd.read(BLOCKSIZE)
    while buf:
        hasher.update(buf)
        buf = fd.read(BLOCKSIZE)
    return hasher.hexdigest()


class PkgConfig(object):
    """This is a class for communicating with pkg-config."""

    def __init__(self):
        """Determines whether pkg-config exists on this machine."""
        self.pkg_config = None
        if sys.platform != 'win32':
            pkg_config = os.environ.get('PKG_CONFIG', 'pkg-config')
            if shutil.which(pkg_config) is not None:
                self.pkg_config = pkg_config
                self.set_pkgconfig_path()
            else:
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

    def setup_extension(
            self, ext, package,
            atleast_version=None, alt_exec=None, default_libraries=()):
        """Add parameters to the given *ext* for the given *package*."""

        # First, try to get the flags from pkg-config.

        cmd = ([self.pkg_config, package] if self.pkg_config else alt_exec)
        if cmd is not None:
            try:
                if self.pkg_config and atleast_version:
                    subprocess.check_call(
                        [*cmd, f"--atleast-version={atleast_version}"])
                # Use sys.getfilesystemencoding() to allow round-tripping
                # when passed back to later subprocess calls; do not use
                # locale.getpreferredencoding() which universal_newlines=True
                # would do.
                cflags = shlex.split(
                    os.fsdecode(subprocess.check_output([*cmd, "--cflags"])))
                libs = shlex.split(
                    os.fsdecode(subprocess.check_output([*cmd, "--libs"])))
            except (OSError, subprocess.CalledProcessError):
                pass
            else:
                ext.extra_compile_args.extend(cflags)
                ext.extra_link_args.extend(libs)
                return

        # If that fails, fall back on the defaults.

        # conda Windows header and library paths.
        # https://github.com/conda/conda/issues/2312 re: getting the env dir.
        if sys.platform == 'win32':
            conda_env_path = (os.getenv('CONDA_PREFIX')  # conda >= 4.1
                              or os.getenv('CONDA_DEFAULT_ENV'))  # conda < 4.1
            if conda_env_path and os.path.isdir(conda_env_path):
                ext.include_dirs.append(os.fspath(
                    pathlib.Path(conda_env_path, "Library/include")))
                ext.library_dirs.append(os.fspath(
                    pathlib.Path(conda_env_path, "Library/lib")))

        # Default linked libs.
        ext.libraries.extend(default_libraries)


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
        return sys.version


def _pkg_data_helper(pkg, subdir):
    """Glob "lib/$pkg/$subdir/**/*", returning paths relative to "lib/$pkg"."""
    base = pathlib.Path("lib", pkg)
    return [str(path.relative_to(base)) for path in (base / subdir).rglob("*")]


class Matplotlib(SetupPackage):
    name = "matplotlib"

    def check(self):
        return versioneer.get_version()

    def get_packages(self):
        return setuptools.find_packages("lib", exclude=["*.tests"])

    def get_namespace_packages(self):
        return ['mpl_toolkits']

    def get_py_modules(self):
        return ['pylab']

    def get_package_data(self):
        return {
            'matplotlib': [
                'mpl-data/matplotlibrc',
                *_pkg_data_helper('matplotlib', 'mpl-data/fonts'),
                *_pkg_data_helper('matplotlib', 'mpl-data/images'),
                *_pkg_data_helper('matplotlib', 'mpl-data/stylelib'),
                *_pkg_data_helper('matplotlib', 'backends/web_backend'),
            ],
        }

    def get_install_requires(self):
        return [
            "cycler>=0.10",
            "kiwisolver>=1.0.1",
            "pyparsing>=2.0.1,!=2.0.4,!=2.1.2,!=2.1.6",
            "python-dateutil>=2.1",
        ]


class SampleData(OptionalPackage):
    """
    This handles the sample data that ships with matplotlib.  It is
    technically optional, though most often will be desired.
    """
    name = "sample_data"

    def get_package_data(self):
        return {
            'matplotlib': [
                *_pkg_data_helper('matplotlib', 'mpl-data/sample_data'),
            ],
        }


class Tests(OptionalPackage):
    name = "tests"
    default_config = False

    def get_packages(self):
        return setuptools.find_packages("lib", include=["*.tests"])

    def get_package_data(self):
        return {
            'matplotlib': [
                *_pkg_data_helper('matplotlib', 'tests/baseline_images'),
                'tests/cmr10.pfb',
                'tests/mpltest.ttf',
                'tests/test_rcparams.rc',
                'tests/test_utf32_be_rcparams.rc',
                'sphinxext/tests/tinypages/*.rst',
                'sphinxext/tests/tinypages/*.py',
                'sphinxext/tests/tinypages/_static/*',
            ],
            'mpl_toolkits': [
                *_pkg_data_helper('mpl_toolkits', 'tests/baseline_images'),
            ]
        }


class Numpy(SetupPackage):
    name = "numpy"

    def add_flags(self, ext):
        import numpy as np
        ext.include_dirs.append(np.get_include())
        ext.define_macros.extend([
            # Ensure that PY_ARRAY_UNIQUE_SYMBOL is uniquely defined for each
            # extension.
            ('PY_ARRAY_UNIQUE_SYMBOL',
             'MPL_' + ext.name.replace('.', '_') + '_ARRAY_API'),
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
            # Allow NumPy's printf format specifiers in C++.
            ('__STDC_FORMAT_MACROS', 1),
        ])

    def get_setup_requires(self):
        return ['numpy>=1.11']

    def get_install_requires(self):
        return ['numpy>=1.11']


class LibAgg(SetupPackage):
    name = 'libagg'

    def add_flags(self, ext, add_sources=True):
        # We need a patched Agg not available elsewhere, so always use the
        # vendored version.
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
            ext.sources.extend(os.path.join('extern', 'agg24-svn', 'src', x)
                               for x in agg_sources)


# For FreeType2 and libpng, we add a separate checkdep_foo.c source to at the
# top of the extension sources.  This file is compiled first and immediately
# aborts the compilation either with "foo.h: No such file or directory" if the
# header is not found, or an appropriate error message if the header indicates
# a too-old version.


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

    def add_flags(self, ext):
        ext.sources.insert(0, 'src/checkdep_freetype2.c')
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
                # FreeType 2.3 has libtool version 9.11.3 as can be checked
                # from the tarball.  For FreeType>=2.4, there is a conversion
                # table in docs/VERSIONS.txt in the FreeType source tree.
                ext, 'freetype2',
                atleast_version='9.11.3',
                alt_exec=['freetype-config'],
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

        # bailing because it is already built
        if os.path.isfile(os.path.join(
                src_path, 'objs', '.libs', libfreetype)):
            return

        # do we need to download / load the source from cache?
        if not os.path.exists(src_path):
            os.makedirs('build', exist_ok=True)

            url_fmts = [
                ('https://downloads.sourceforge.net/project/freetype'
                 '/freetype2/{version}/{tarball}'),
                ('https://download.savannah.gnu.org/releases/freetype'
                 '/{tarball}')
            ]
            tarball = 'freetype-{0}.tar.gz'.format(LOCAL_FREETYPE_VERSION)

            target_urls = [
                url_fmt.format(version=LOCAL_FREETYPE_VERSION,
                               tarball=tarball)
                for url_fmt in url_fmts]

            for tarball_url in target_urls:
                try:
                    tar_contents = download_or_cache(tarball_url,
                                                     LOCAL_FREETYPE_HASH)
                    break
                except Exception:
                    pass
            else:
                raise IOError("Failed to download FreeType.  Please download "
                              "one of {target_urls} and extract it into "
                              "{src_path} at the top-level of the source "
                              "repository".format(
                                  target_urls=target_urls, src_path=src_path))

            print("Extracting {}".format(tarball))
            # just to be sure
            tar_contents.seek(0)
            with tarfile.open(tarball, mode="r:gz",
                              fileobj=tar_contents) as tgz:
                tgz.extractall("build")

        print("Building freetype in {}".format(src_path))
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
            shutil.rmtree(str(pathlib.Path(src_path, "objs")),
                          ignore_errors=True)
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
            cmdfile = pathlib.Path("build/build_freetype.cmd")
            cmdfile.write_text(FREETYPE_BUILD_CMD.format(
                vc20xx=vc, WinXX=WinXX, xXX=xXX, vcvarsall=vcvarsall))
            subprocess.check_call([str(cmdfile.resolve())],
                                  shell=True, cwd=src_path)
            # Move to the corresponding Unix build path.
            pathlib.Path(src_path, "objs/.libs").mkdir()
            # Be robust against change of FreeType version.
            lib_path, = (pathlib.Path(src_path, "objs", vc, xXX)
                         .glob("freetype*.lib"))
            shutil.copy2(
                str(lib_path),
                str(pathlib.Path(src_path, "objs/.libs/libfreetype.lib")))


class FT2Font(SetupPackage):
    name = 'ft2font'

    def get_extension(self):
        sources = [
            'src/ft2font.cpp',
            'src/ft2font_wrapper.cpp',
            'src/mplutils.cpp',
            'src/py_converters.cpp',
            ]
        ext = Extension('matplotlib.ft2font', sources)
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

    def get_extension(self):
        sources = [
            'src/checkdep_libpng.c',
            'src/_png.cpp',
            'src/mplutils.cpp',
            ]
        ext = Extension('matplotlib._png', sources)
        pkg_config.setup_extension(
            ext, 'libpng',
            atleast_version='1.2',
            alt_exec=['libpng-config', '--ldflags'],
            default_libraries=['png', 'z'])
        Numpy().add_flags(ext)
        return ext


class Qhull(SetupPackage):
    name = "qhull"

    def add_flags(self, ext):
        # Qhull doesn't distribute pkg-config info, so we have no way of
        # knowing whether a system install is recent enough.  Thus, always use
        # the vendored version.
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
        ext = Extension('matplotlib.ttconv', sources)
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

        ext = Extension('matplotlib._path', sources)
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
        ext = Extension('matplotlib._image', sources)
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
        ext = Extension('matplotlib._contour', sources)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext, add_sources=False)
        return ext


class QhullWrap(SetupPackage):
    name = "qhull_wrap"

    def get_extension(self):
        sources = ['src/qhull_wrap.c']
        ext = Extension('matplotlib._qhull', sources,
                        define_macros=[('MPL_DEVNULL', os.devnull)])
        Numpy().add_flags(ext)
        Qhull().add_flags(ext)
        return ext


class Tri(SetupPackage):
    name = "tri"

    def get_extension(self):
        sources = [
            "src/tri/_tri.cpp",
            "src/tri/_tri_wrapper.cpp",
            "src/mplutils.cpp"
            ]
        ext = Extension('matplotlib._tri', sources)
        Numpy().add_flags(ext)
        return ext


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
        ext = Extension('matplotlib.backends._backend_agg', sources)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext)
        FreeType().add_flags(ext)
        return ext


class BackendTkAgg(OptionalBackendPackage):
    name = "tkagg"
    force = True

    def check(self):
        return "installing; run-time loading from Python Tcl/Tk"

    def get_extension(self):
        sources = [
            'src/_tkagg.cpp',
            'src/py_converters.cpp',
            ]

        ext = Extension('matplotlib.backends._tkagg', sources)
        self.add_flags(ext)
        Numpy().add_flags(ext)
        LibAgg().add_flags(ext, add_sources=False)
        return ext

    def add_flags(self, ext):
        ext.include_dirs.insert(0, 'src')
        if sys.platform == 'win32':
            # psapi library needed for finding Tcl/Tk at run time.
            # user32 library needed for window manipulation functions.
            ext.libraries.extend(['psapi', 'user32'])
            ext.extra_link_args.extend(["-mwindows"])
        elif sys.platform == 'linux':
            ext.libraries.extend(['dl'])


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

        ext = Extension('matplotlib.backends._macosx', sources)
        ext.extra_link_args.extend(['-framework', 'Cocoa'])
        if platform.python_implementation().lower() == 'pypy':
            ext.extra_compile_args.append('-DPYPY=1')
        return ext


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
        except Exception:
            return False  # <-- default
