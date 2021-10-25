"""
The Matplotlib build options can be modified with a mplsetup.cfg file. See
mplsetup.cfg.template for more information.
"""

# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
import sys

py_min_version = (3, 7)  # minimal supported python version
since_mpl_version = (3, 4)  # py_min_version is required since this mpl version

if sys.version_info < py_min_version:
    error = """
Beginning with Matplotlib {0}, Python {1} or above is required.
You are using Python {2}.

This may be due to an out of date pip.

Make sure you have pip >= 9.0.1.
""".format('.'.join(str(n) for n in since_mpl_version),
           '.'.join(str(n) for n in py_min_version),
           '.'.join(str(n) for n in sys.version_info[:3]))
    sys.exit(error)

import os
from pathlib import Path
import shutil
import subprocess

from setuptools import setup, find_packages, Distribution, Extension
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.test
import setuptools.command.sdist

import setupext
from setupext import print_raw, print_status


# These are the packages in the order we want to display them.
mpl_packages = [
    setupext.Matplotlib(),
    setupext.Python(),
    setupext.Platform(),
    setupext.FreeType(),
    setupext.Qhull(),
    setupext.Tests(),
    setupext.BackendMacOSX(),
    ]


# From https://bugs.python.org/issue26689
def has_flag(self, flagname):
    """Return whether a flag name is supported on the specified compiler."""
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            self.compile([f.name], extra_postargs=[flagname])
        except Exception as exc:
            # https://github.com/pypa/setuptools/issues/2698
            if type(exc).__name__ != "CompileError":
                raise
            return False
    return True


class NoopTestCommand(setuptools.command.test.test):
    def __init__(self, dist):
        print("Matplotlib does not support running tests with "
              "'python setup.py test'. Please run 'pytest'.")


class BuildExtraLibraries(setuptools.command.build_ext.build_ext):
    def finalize_options(self):
        self.distribution.ext_modules[:] = [
            ext
            for package in good_packages
            for ext in package.get_extensions()
        ]
        super().finalize_options()

    def add_optimization_flags(self):
        """
        Add optional optimization flags to extension.

        This adds flags for LTO and hidden visibility to both compiled
        extensions, and to the environment variables so that vendored libraries
        will also use them. If the compiler does not support these flags, then
        none are added.
        """

        env = os.environ.copy()
        if sys.platform == 'win32':
            return env
        enable_lto = setupext.config.getboolean('libs', 'enable_lto',
                                                fallback=None)

        def prepare_flags(name, enable_lto):
            """
            Prepare *FLAGS from the environment.

            If set, return them, and also check whether LTO is disabled in each
            one, raising an error if Matplotlib config explicitly enabled LTO.
            """
            if name in os.environ:
                if '-fno-lto' in os.environ[name]:
                    if enable_lto is True:
                        raise ValueError('Configuration enable_lto=True, but '
                                         '{0} contains -fno-lto'.format(name))
                    enable_lto = False
                return [os.environ[name]], enable_lto
            return [], enable_lto

        _, enable_lto = prepare_flags('CFLAGS', enable_lto)  # Only check lto.
        cppflags, enable_lto = prepare_flags('CPPFLAGS', enable_lto)
        cxxflags, enable_lto = prepare_flags('CXXFLAGS', enable_lto)
        ldflags, enable_lto = prepare_flags('LDFLAGS', enable_lto)

        if enable_lto is False:
            return env

        if has_flag(self.compiler, '-fvisibility=hidden'):
            for ext in self.extensions:
                ext.extra_compile_args.append('-fvisibility=hidden')
            cppflags.append('-fvisibility=hidden')
        if has_flag(self.compiler, '-fvisibility-inlines-hidden'):
            for ext in self.extensions:
                if self.compiler.detect_language(ext.sources) != 'cpp':
                    continue
                ext.extra_compile_args.append('-fvisibility-inlines-hidden')
            cxxflags.append('-fvisibility-inlines-hidden')
        ranlib = 'RANLIB' in env
        if not ranlib and self.compiler.compiler_type == 'unix':
            try:
                result = subprocess.run(self.compiler.compiler +
                                        ['--version'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
            except Exception as e:
                pass
            else:
                version = result.stdout.lower()
                if 'gcc' in version:
                    ranlib = shutil.which('gcc-ranlib')
                elif 'clang' in version:
                    if sys.platform == 'darwin':
                        ranlib = True
                    else:
                        ranlib = shutil.which('llvm-ranlib')
        if ranlib and has_flag(self.compiler, '-flto'):
            for ext in self.extensions:
                ext.extra_compile_args.append('-flto')
            cppflags.append('-flto')
            ldflags.append('-flto')
            # Needed so FreeType static library doesn't lose its LTO objects.
            if isinstance(ranlib, str):
                env['RANLIB'] = ranlib

        env['CPPFLAGS'] = ' '.join(cppflags)
        env['CXXFLAGS'] = ' '.join(cxxflags)
        env['LDFLAGS'] = ' '.join(ldflags)

        return env

    def build_extensions(self):
        # Remove the -Wstrict-prototypes option, it's not valid for C++.  Fixed
        # in Py3.7 as bpo-5755.
        try:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except (ValueError, AttributeError):
            pass
        if (self.compiler.compiler_type == 'msvc' and
                os.environ.get('MPL_DISABLE_FH4')):
            # Disable FH4 Exception Handling implementation so that we don't
            # require VCRUNTIME140_1.dll. For more details, see:
            # https://devblogs.microsoft.com/cppblog/making-cpp-exception-handling-smaller-x64/
            # https://github.com/joerick/cibuildwheel/issues/423#issuecomment-677763904
            for ext in self.extensions:
                ext.extra_compile_args.append('/d2FH4-')

        env = self.add_optimization_flags()
        for package in good_packages:
            package.do_custom_build(env)
        return super().build_extensions()

    def build_extension(self, ext):
        # When C coverage is enabled, the path to the object file is saved.
        # Since we re-use source files in multiple extensions, libgcov will
        # complain at runtime that it is trying to save coverage for the same
        # object file at different timestamps (since each source is compiled
        # again for each extension). Thus, we need to use unique temporary
        # build directories to store object files for each extension.
        orig_build_temp = self.build_temp
        self.build_temp = os.path.join(self.build_temp, ext.name)
        try:
            super().build_extension(ext)
        finally:
            self.build_temp = orig_build_temp


def update_matplotlibrc(path):
    # If packagers want to change the default backend, insert a `#backend: ...`
    # line.  Otherwise, use the default `##backend: Agg` which has no effect
    # even after decommenting, which allows _auto_backend_sentinel to be filled
    # in at import time.
    template_lines = path.read_text().splitlines(True)
    backend_line_idx, = [  # Also asserts that there is a single such line.
        idx for idx, line in enumerate(template_lines)
        if "#backend:" in line]
    template_lines[backend_line_idx] = (
        "#backend: {}".format(setupext.options["backend"])
        if setupext.options["backend"]
        else "##backend: Agg")
    path.write_text("".join(template_lines))


class BuildPy(setuptools.command.build_py.build_py):
    def run(self):
        super().run()
        update_matplotlibrc(
            Path(self.build_lib, "matplotlib/mpl-data/matplotlibrc"))


class Sdist(setuptools.command.sdist.sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        update_matplotlibrc(
            Path(base_dir, "lib/matplotlib/mpl-data/matplotlibrc"))


package_data = {}  # Will be filled below by the various components.

# If the user just queries for information, don't bother figuring out which
# packages to build or install.
if not (any('--' + opt in sys.argv
            for opt in Distribution.display_option_names + ['help'])
        or 'clean' in sys.argv):
    # Go through all of the packages and figure out which ones we are
    # going to build/install.
    print_raw()
    print_raw("Edit mplsetup.cfg to change the build options; "
              "suppress output with --quiet.")
    print_raw()
    print_raw("BUILDING MATPLOTLIB")

    good_packages = []
    for package in mpl_packages:
        try:
            message = package.check()
        except setupext.Skipped as e:
            print_status(package.name, "no  [{e}]".format(e=e))
            continue
        if message is not None:
            print_status(package.name,
                         "yes [{message}]".format(message=message))
        good_packages.append(package)

    print_raw()

    # Now collect all of the information we need to build all of the packages.
    for package in good_packages:
        # Extension modules only get added in build_ext, as numpy will have
        # been installed (as setup_requires) at that point.
        data = package.get_package_data()
        for key, val in data.items():
            package_data.setdefault(key, [])
            package_data[key] = list(set(val + package_data[key]))

setup(  # Finally, pass this all along to setuptools to do the heavy lifting.
    name="matplotlib",
    description="Python plotting package",
    author="John D. Hunter, Michael Droettboom",
    author_email="matplotlib-users@python.org",
    url="https://matplotlib.org",
    download_url="https://matplotlib.org/users/installing.html",
    project_urls={
        'Documentation': 'https://matplotlib.org',
        'Source Code': 'https://github.com/matplotlib/matplotlib',
        'Bug Tracker': 'https://github.com/matplotlib/matplotlib/issues',
        'Forum': 'https://discourse.matplotlib.org/',
        'Donate': 'https://numfocus.org/donate-to-matplotlib'
    },
    long_description=Path("README.rst").read_text(encoding="utf-8"),
    long_description_content_type="text/x-rst",
    license="PSF",
    platforms="any",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Framework :: Matplotlib',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: Python Software Foundation License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Visualization',
    ],

    package_dir={"": "lib"},
    packages=find_packages("lib"),
    namespace_packages=["mpl_toolkits"],
    py_modules=["pylab"],
    # Dummy extension to trigger build_ext, which will swap it out with
    # real extensions that can depend on numpy for the build.
    ext_modules=[Extension("", [])],
    package_data=package_data,

    python_requires='>={}'.format('.'.join(str(n) for n in py_min_version)),
    setup_requires=[
        "certifi>=2020.06.20",
        "numpy>=1.17",
        "setuptools_scm>=4",
        "setuptools_scm_git_archive",
    ],
    install_requires=[
        "cycler>=0.10",
        "fonttools>=4.22.0",
        "kiwisolver>=1.0.1",
        "numpy>=1.17",
        "packaging>=20.0",
        "pillow>=6.2.0",
        "pyparsing>=2.2.1",
        "python-dateutil>=2.7",
    ] + (
        # Installing from a git checkout.
        ["setuptools_scm>=4"] if Path(__file__).with_name(".git").exists()
        else []
    ),
    use_scm_version={
        "version_scheme": "release-branch-semver",
        "local_scheme": "node-and-date",
        "write_to": "lib/matplotlib/_version.py",
        "parentdir_prefix_version": "matplotlib-",
        "fallback_version": "0.0+UNKNOWN",
    },
    cmdclass={
        "test": NoopTestCommand,
        "build_ext": BuildExtraLibraries,
        "build_py": BuildPy,
        "sdist": Sdist,
    },
)
