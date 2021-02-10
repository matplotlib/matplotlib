import configparser
from distutils import ccompiler, sysconfig
from distutils.core import Extension
import functools
import hashlib
from io import BytesIO
import logging
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import textwrap
import urllib.request
import versioneer

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
    return Path(cache_dir, 'matplotlib')


def _get_hash(data):
    """Compute the sha256 hash of *data*."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


@functools.lru_cache()
def _get_ssl_context():
    import certifi
    import ssl
    return ssl.create_default_context(cafile=certifi.where())


def get_from_cache_or_download(url, sha):
    """
    Get bytes from the given url or local cache.

    Parameters
    ----------
    url : str
        The url to download.
    sha : str
        The sha256 of the file.

    Returns
    -------
    BytesIO
        The file loaded into memory.
    """
    cache_dir = _get_xdg_cache_dir()

    if cache_dir is not None:  # Try to read from cache.
        try:
            data = (cache_dir / sha).read_bytes()
        except IOError:
            pass
        else:
            if _get_hash(data) == sha:
                return BytesIO(data)

    # jQueryUI's website blocks direct downloads from urllib.request's
    # default User-Agent, but not (for example) wget; so I don't feel too
    # bad passing in an empty User-Agent.
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": ""}),
            context=_get_ssl_context()) as req:
        data = req.read()

    file_sha = _get_hash(data)
    if file_sha != sha:
        raise Exception(
            f"The downloaded file does not match the expected sha.  {url} was "
            f"expected to have {sha} but it had {file_sha}")

    if cache_dir is not None:  # Try to cache the downloaded file.
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_dir / sha, "xb") as fout:
                fout.write(data)
        except IOError:
            pass

    return BytesIO(data)


def get_and_extract_tarball(urls, sha, dirname):
    """
    Obtain a tarball (from cache or download) and extract it.

    Parameters
    ----------
    urls : list[str]
        URLs from which download is attempted (in order of attempt), if the
        tarball is not in the cache yet.
    sha : str
        SHA256 hash of the tarball; used both as a cache key (by
        `get_from_cache_or_download`) and to validate a downloaded tarball.
    dirname : path-like
        Directory where the tarball is extracted.
    """
    toplevel = Path("build", dirname)
    if not toplevel.exists():  # Download it or load it from cache.
        Path("build").mkdir(exist_ok=True)
        for url in urls:
            try:
                tar_contents = get_from_cache_or_download(url, sha)
                break
            except Exception:
                pass
        else:
            raise IOError(
                f"Failed to download any of the following: {urls}.  "
                f"Please download one of these urls and extract it into "
                f"'build/' at the top-level of the source repository.")
        print("Extracting {}".format(urllib.parse.urlparse(url).path))
        with tarfile.open(fileobj=tar_contents, mode="r:gz") as tgz:
            if os.path.commonpath(tgz.getnames()) != dirname:
                raise IOError(
                    f"The downloaded tgz file was expected to have {dirname} "
                    f"as sole top-level directory, but that is not the case")
            tgz.extractall("build")
    return toplevel


# SHA256 hashes of the FreeType tarballs
_freetype_hashes = {
    '2.6.1':
        '0a3c7dfbda6da1e8fce29232e8e96d987ababbbf71ebc8c75659e4132c367014',
    '2.6.2':
        '8da42fc4904e600be4b692555ae1dcbf532897da9c5b9fb5ebd3758c77e5c2d4',
    '2.6.3':
        '7942096c40ee6fea882bd4207667ad3f24bff568b96b10fd3885e11a7baad9a3',
    '2.6.4':
        '27f0e38347a1850ad57f84fc4dfed68ba0bc30c96a6fa6138ef84d485dd9a8d7',
    '2.6.5':
        '3bb24add9b9ec53636a63ea8e867ed978c4f8fdd8f1fa5ccfd41171163d4249a',
    '2.7':
        '7b657d5f872b0ab56461f3bd310bd1c5ec64619bd15f0d8e08282d494d9cfea4',
    '2.7.1':
        '162ef25aa64480b1189cdb261228e6c5c44f212aac4b4621e28cf2157efb59f5',
    '2.8':
        '33a28fabac471891d0523033e99c0005b95e5618dc8ffa7fa47f9dadcacb1c9b',
    '2.8.1':
        '876711d064a6a1bd74beb18dd37f219af26100f72daaebd2d86cb493d7cd7ec6',
    '2.9':
        'bf380e4d7c4f3b5b1c1a7b2bf3abb967bda5e9ab480d0df656e0e08c5019c5e6',
    '2.9.1':
        'ec391504e55498adceb30baceebd147a6e963f636eb617424bcfc47a169898ce',
    '2.10.0':
        '955e17244e9b38adb0c98df66abb50467312e6bb70eac07e49ce6bd1a20e809a',
    '2.10.1':
        '3a60d391fd579440561bf0e7f31af2222bc610ad6ce4d9d7bd2165bca8669110',
}
# This is the version of FreeType to use when building a local
# version.  It must match the value in
# lib/matplotlib.__init__.py and also needs to be changed below in the
# embedded windows build script (grep for "REMINDER" in this file)
LOCAL_FREETYPE_VERSION = '2.6.1'
LOCAL_FREETYPE_HASH = _freetype_hashes.get(LOCAL_FREETYPE_VERSION, 'unknown')

LOCAL_QHULL_VERSION = '2020.2'


# matplotlib build options, which can be altered using setup.cfg
setup_cfg = os.environ.get('MPLSETUPCFG') or 'setup.cfg'
config = configparser.ConfigParser()
if os.path.exists(setup_cfg):
    config.read(setup_cfg)
options = {
    'backend': config.get('rc_options', 'backend', fallback=None),
    'system_freetype': config.getboolean(
        'libs', 'system_freetype', fallback=sys.platform.startswith('aix')),
    'system_qhull': config.getboolean('libs', 'system_qhull',
                                      fallback=False),
}


if '-q' in sys.argv or '--quiet' in sys.argv:
    def print_raw(*args, **kwargs): pass  # Suppress our own output.
else:
    print_raw = print


def print_status(package, status):
    initial_indent = "%12s: " % package
    indent = ' ' * 18
    print_raw(textwrap.fill(str(status), width=80,
                            initial_indent=initial_indent,
                            subsequent_indent=indent))


@functools.lru_cache(1)  # We only need to compute this once.
def get_pkg_config():
    """
    Get path to pkg-config and set up the PKG_CONFIG environment variable.
    """
    if sys.platform == 'win32':
        return None
    pkg_config = os.environ.get('PKG_CONFIG') or 'pkg-config'
    if shutil.which(pkg_config) is None:
        print(
            "IMPORTANT WARNING:\n"
            "    pkg-config is not installed.\n"
            "    Matplotlib may not be able to find some of its dependencies.")
        return None
    pkg_config_path = sysconfig.get_config_var('LIBDIR')
    if pkg_config_path is not None:
        pkg_config_path = os.path.join(pkg_config_path, 'pkgconfig')
        try:
            os.environ['PKG_CONFIG_PATH'] += ':' + pkg_config_path
        except KeyError:
            os.environ['PKG_CONFIG_PATH'] = pkg_config_path
    return pkg_config


def pkg_config_setup_extension(
        ext, package,
        atleast_version=None, alt_exec=None, default_libraries=()):
    """Add parameters to the given *ext* for the given *package*."""

    # First, try to get the flags from pkg-config.

    pkg_config = get_pkg_config()
    cmd = [pkg_config, package] if pkg_config else alt_exec
    if cmd is not None:
        try:
            if pkg_config and atleast_version:
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
            conda_env_path = Path(conda_env_path)
            ext.include_dirs.append(str(conda_env_path / "Library/include"))
            ext.library_dirs.append(str(conda_env_path / "Library/lib"))

    # Default linked libs.
    ext.libraries.extend(default_libraries)


class Skipped(Exception):
    """
    Exception thrown by `SetupPackage.check` to indicate that a package should
    be skipped.
    """


class SetupPackage:

    def check(self):
        """
        If the package should be installed, return an informative string, or
        None if no information should be displayed at all.

        If the package should be skipped, raise a `Skipped` exception.

        If a missing build dependency is fatal, call `sys.exit`.
        """

    def get_package_data(self):
        """
        Get a package data dictionary to add to the configuration.
        These are merged into to the *package_data* list passed to
        `setuptools.setup`.
        """
        return {}

    def get_extensions(self):
        """
        Return or yield a list of C extensions (`distutils.core.Extension`
        objects) to add to the configuration.  These are added to the
        *extensions* list passed to `setuptools.setup`.
        """
        return []

    def do_custom_build(self, env):
        """
        If a package needs to do extra custom things, such as building a
        third-party library, before building an extension, it should
        override this method.
        """


class OptionalPackage(SetupPackage):
    config_category = "packages"
    default_config = True

    def check(self):
        """
        Check whether ``setup.cfg`` requests this package to be installed.

        May be overridden by subclasses for additional checks.
        """
        if config.getboolean(self.config_category, self.name,
                             fallback=self.default_config):
            return "installing"
        else:  # Configuration opt-out by user
            raise Skipped("skipping due to configuration")


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
    base = Path("lib", pkg)
    return [str(path.relative_to(base)) for path in (base / subdir).rglob("*")]


class Matplotlib(SetupPackage):
    name = "matplotlib"

    def check(self):
        return versioneer.get_version()

    def get_package_data(self):
        return {
            'matplotlib': [
                'mpl-data/matplotlibrc',
                *_pkg_data_helper('matplotlib', 'mpl-data'),
                *_pkg_data_helper('matplotlib', 'backends/web_backend'),
                '*.dll',  # Only actually matters on Windows.
            ],
        }

    def get_extensions(self):
        # agg
        ext = Extension(
            "matplotlib.backends._backend_agg", [
                "src/mplutils.cpp",
                "src/py_converters.cpp",
                "src/_backend_agg.cpp",
                "src/_backend_agg_wrapper.cpp",
            ])
        add_numpy_flags(ext)
        add_libagg_flags_and_sources(ext)
        FreeType.add_flags(ext)
        yield ext
        # c_internal_utils
        ext = Extension(
            "matplotlib._c_internal_utils", ["src/_c_internal_utils.c"],
            libraries=({
                "linux": ["dl"],
                "win32": ["ole32", "shell32", "user32"],
            }.get(sys.platform, [])))
        yield ext
        # contour
        ext = Extension(
            "matplotlib._contour", [
                "src/_contour.cpp",
                "src/_contour_wrapper.cpp",
                "src/py_converters.cpp",
            ])
        add_numpy_flags(ext)
        add_libagg_flags(ext)
        yield ext
        # ft2font
        ext = Extension(
            "matplotlib.ft2font", [
                "src/ft2font.cpp",
                "src/ft2font_wrapper.cpp",
                "src/mplutils.cpp",
                "src/py_converters.cpp",
            ])
        FreeType.add_flags(ext)
        add_numpy_flags(ext)
        add_libagg_flags(ext)
        yield ext
        # image
        ext = Extension(
            "matplotlib._image", [
                "src/_image.cpp",
                "src/mplutils.cpp",
                "src/_image_wrapper.cpp",
                "src/py_converters.cpp",
            ])
        add_numpy_flags(ext)
        add_libagg_flags_and_sources(ext)
        yield ext
        # path
        ext = Extension(
            "matplotlib._path", [
                "src/py_converters.cpp",
                "src/_path_wrapper.cpp",
            ])
        add_numpy_flags(ext)
        add_libagg_flags_and_sources(ext)
        yield ext
        # qhull
        ext = Extension(
            "matplotlib._qhull", ["src/qhull_wrap.c"],
            define_macros=[("MPL_DEVNULL", os.devnull)])
        add_numpy_flags(ext)
        Qhull.add_flags(ext)
        yield ext
        # tkagg
        ext = Extension(
            "matplotlib.backends._tkagg", [
                "src/_tkagg.cpp",
            ],
            include_dirs=["src"],
            # psapi library needed for finding Tcl/Tk at run time.
            libraries=({"linux": ["dl"], "win32": ["psapi"],
                        "cygwin": ["psapi"]}.get(sys.platform, [])),
            extra_link_args={"win32": ["-mwindows"]}.get(sys.platform, []))
        add_numpy_flags(ext)
        add_libagg_flags(ext)
        yield ext
        # tri
        ext = Extension(
            "matplotlib._tri", [
                "src/tri/_tri.cpp",
                "src/tri/_tri_wrapper.cpp",
                "src/mplutils.cpp",
            ])
        add_numpy_flags(ext)
        yield ext
        # ttconv
        ext = Extension(
            "matplotlib._ttconv", [
                "src/_ttconv.cpp",
                "extern/ttconv/pprdrv_tt.cpp",
                "extern/ttconv/pprdrv_tt2.cpp",
                "extern/ttconv/ttutil.cpp",
            ],
            include_dirs=["extern"])
        add_numpy_flags(ext)
        yield ext


class Tests(OptionalPackage):
    name = "tests"
    default_config = False

    def get_package_data(self):
        return {
            'matplotlib': [
                *_pkg_data_helper('matplotlib', 'tests/baseline_images'),
                *_pkg_data_helper('matplotlib', 'tests/tinypages'),
                'tests/cmr10.pfb',
                'tests/mpltest.ttf',
            ],
            'mpl_toolkits': [
                *_pkg_data_helper('mpl_toolkits', 'tests/baseline_images'),
            ]
        }


def add_numpy_flags(ext):
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


def add_libagg_flags(ext):
    # We need a patched Agg not available elsewhere, so always use the vendored
    # version.
    ext.include_dirs.insert(0, "extern/agg24-svn/include")


def add_libagg_flags_and_sources(ext):
    # We need a patched Agg not available elsewhere, so always use the vendored
    # version.
    ext.include_dirs.insert(0, "extern/agg24-svn/include")
    agg_sources = [
        "agg_bezier_arc.cpp",
        "agg_curves.cpp",
        "agg_image_filters.cpp",
        "agg_trans_affine.cpp",
        "agg_vcgen_contour.cpp",
        "agg_vcgen_dash.cpp",
        "agg_vcgen_stroke.cpp",
        "agg_vpgen_segmentator.cpp",
    ]
    ext.sources.extend(
        os.path.join("extern", "agg24-svn", "src", x) for x in agg_sources)


# First compile checkdep_freetype2.c, which aborts the compilation either
# with "foo.h: No such file or directory" if the header is not found, or an
# appropriate error message if the header indicates a too-old version.


class FreeType(SetupPackage):
    name = "freetype"

    @classmethod
    def add_flags(cls, ext):
        ext.sources.insert(0, 'src/checkdep_freetype2.c')
        if options.get('system_freetype'):
            pkg_config_setup_extension(
                # FreeType 2.3 has libtool version 9.11.3 as can be checked
                # from the tarball.  For FreeType>=2.4, there is a conversion
                # table in docs/VERSIONS.txt in the FreeType source tree.
                ext, 'freetype2',
                atleast_version='9.11.3',
                alt_exec=['freetype-config'],
                default_libraries=['freetype'])
            ext.define_macros.append(('FREETYPE_BUILD_TYPE', 'system'))
        else:
            src_path = Path('build', f'freetype-{LOCAL_FREETYPE_VERSION}')
            # Statically link to the locally-built freetype.
            # This is certainly broken on Windows.
            ext.include_dirs.insert(0, str(src_path / 'include'))
            if sys.platform == 'win32':
                libfreetype = 'libfreetype.lib'
            else:
                libfreetype = 'libfreetype.a'
            ext.extra_objects.insert(
                0, str(src_path / 'objs' / '.libs' / libfreetype))
            ext.define_macros.append(('FREETYPE_BUILD_TYPE', 'local'))

    def do_custom_build(self, env):
        # We're using a system freetype
        if options.get('system_freetype'):
            return

        tarball = f'freetype-{LOCAL_FREETYPE_VERSION}.tar.gz'
        src_path = get_and_extract_tarball(
            urls=[
                (f'https://downloads.sourceforge.net/project/freetype'
                 f'/freetype2/{LOCAL_FREETYPE_VERSION}/{tarball}'),
                (f'https://download.savannah.gnu.org/releases/freetype'
                 f'/{tarball}')
            ],
            sha=LOCAL_FREETYPE_HASH,
            dirname=f'freetype-{LOCAL_FREETYPE_VERSION}',
        )

        if sys.platform == 'win32':
            libfreetype = 'libfreetype.lib'
        else:
            libfreetype = 'libfreetype.a'
        if (src_path / 'objs' / '.libs' / libfreetype).is_file():
            return  # Bail out because we have already built FreeType.

        print(f"Building freetype in {src_path}")
        if sys.platform != 'win32':  # compilation on non-windows
            env = {**env, "CFLAGS": "{} -fPIC".format(env.get("CFLAGS", ""))}
            subprocess.check_call(
                ["./configure", "--with-zlib=no", "--with-bzip2=no",
                 "--with-png=no", "--with-harfbuzz=no", "--enable-static",
                 "--disable-shared"],
                env=env, cwd=src_path)
            if 'GNUMAKE' in env:
                make = env['GNUMAKE']
            elif 'MAKE' in env:
                make = env['MAKE']
            else:
                try:
                    output = subprocess.check_output(['make', '-v'],
                                                     stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError:
                    output = b''
                if b'GNU' not in output and b'makepp' not in output:
                    make = 'gmake'
                else:
                    make = 'make'
            subprocess.check_call([make], env=env, cwd=src_path)
        else:  # compilation on windows
            shutil.rmtree(src_path / "objs", ignore_errors=True)
            msbuild_platform = (
                'x64' if platform.architecture()[0] == '64bit' else 'Win32')
            base_path = Path("build/freetype-2.6.1/builds/windows")
            vc = 'vc2010'
            sln_path = (
                base_path / vc / "freetype.sln"
            )
            # https://developercommunity.visualstudio.com/comments/190992/view.html
            (sln_path.parent / "Directory.Build.props").write_text("""
<Project>
 <PropertyGroup>
  <!-- The following line *cannot* be split over multiple lines. -->
  <WindowsTargetPlatformVersion>$([Microsoft.Build.Utilities.ToolLocationHelper]::GetLatestSDKTargetPlatformVersion('Windows', '10.0'))</WindowsTargetPlatformVersion>
 </PropertyGroup>
</Project>
""")
            # It is not a trivial task to determine PlatformToolset to plug it
            # into msbuild command, and Directory.Build.props will not override
            # the value in the project file.
            # The DefaultPlatformToolset is from Microsoft.Cpp.Default.props
            with open(base_path / vc / "freetype.vcxproj", 'r+b') as f:
                toolset_repl = b'PlatformToolset>$(DefaultPlatformToolset)<'
                vcxproj = f.read().replace(b'PlatformToolset>v100<',
                                           toolset_repl)
                assert toolset_repl in vcxproj, (
                   'Upgrading Freetype might break this')
                f.seek(0)
                f.truncate()
                f.write(vcxproj)

            cc = ccompiler.new_compiler()
            cc.initialize()  # Get msbuild in the %PATH% of cc.spawn.
            cc.spawn(["msbuild", str(sln_path),
                      "/t:Clean;Build",
                      f"/p:Configuration=Release;Platform={msbuild_platform}"])
            # Move to the corresponding Unix build path.
            (src_path / "objs" / ".libs").mkdir()
            # Be robust against change of FreeType version.
            lib_path, = (src_path / "objs" / vc / msbuild_platform).glob(
                "freetype*.lib")
            shutil.copy2(lib_path, src_path / "objs/.libs/libfreetype.lib")


class Qhull(SetupPackage):
    name = "qhull"
    _extensions_to_update = []

    @classmethod
    def add_flags(cls, ext):
        if options.get("system_qhull"):
            ext.libraries.append("qhull_r")
        else:
            cls._extensions_to_update.append(ext)

    def do_custom_build(self, env):
        if options.get('system_qhull'):
            return

        toplevel = get_and_extract_tarball(
            urls=["http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz"],
            sha="b5c2d7eb833278881b952c8a52d20179eab87766b00b865000469a45c1838b7e",
            dirname=f"qhull-{LOCAL_QHULL_VERSION}",
        )
        shutil.copyfile(toplevel / "COPYING.txt", "LICENSE/LICENSE_QHULL")

        for ext in self._extensions_to_update:
            qhull_path = Path(f'build/qhull-{LOCAL_QHULL_VERSION}/src')
            ext.include_dirs.insert(0, str(qhull_path))
            ext.sources.extend(map(str, sorted(qhull_path.glob('libqhull_r/*.c'))))
            if sysconfig.get_config_var("LIBM") == "-lm":
                ext.libraries.extend("m")


class BackendMacOSX(OptionalPackage):
    config_category = 'gui_support'
    name = 'macosx'

    def check(self):
        if sys.platform != 'darwin':
            raise Skipped("Mac OS-X only")
        return super().check()

    def get_extensions(self):
        sources = [
            'src/_macosx.m'
            ]
        ext = Extension('matplotlib.backends._macosx', sources)
        ext.extra_compile_args.extend(['-Werror=unguarded-availability'])
        ext.extra_link_args.extend(['-framework', 'Cocoa'])
        if platform.python_implementation().lower() == 'pypy':
            ext.extra_compile_args.append('-DPYPY=1')
        yield ext
