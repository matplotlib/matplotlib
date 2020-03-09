"""
The matplotlib build options can be modified with a setup.cfg file. See
setup.cfg.template for more information.
"""

# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
import sys

min_version = (3, 6)

if sys.version_info < min_version:
    error = """
Beginning with Matplotlib 3.1, Python {0} or above is required.

This may be due to an out of date pip.

Make sure you have pip >= 9.0.1.
""".format('.'.join(str(n) for n in min_version)),
    sys.exit(error)

from pathlib import Path
import shutil
from zipfile import ZipFile

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as BuildExtCommand
from setuptools.command.develop import develop as DevelopCommand
from setuptools.command.install_lib import install_lib as InstallLibCommand
from setuptools.command.test import test as TestCommand

# The setuptools version of sdist adds a setup.cfg file to the tree.
# We don't want that, so we simply remove it, and it will fall back to
# vanilla distutils.
try:
    from setuptools.command import sdist
except ImportError:
    pass
else:
    del sdist.sdist.make_release_tree

from distutils.dist import Distribution

import setupext
from setupext import print_raw, print_status, download_or_cache

# Get the version from versioneer
import versioneer
__version__ = versioneer.get_version()


# These are the packages in the order we want to display them.
mpl_packages = [
    setupext.Matplotlib(),
    setupext.Python(),
    setupext.Platform(),
    setupext.LibAgg(),
    setupext.FreeType(),
    setupext.FT2Font(),
    setupext.Qhull(),
    setupext.Image(),
    setupext.TTConv(),
    setupext.Path(),
    setupext.Contour(),
    setupext.QhullWrap(),
    setupext.Tri(),
    setupext.SampleData(),
    setupext.Tests(),
    setupext.BackendAgg(),
    setupext.BackendTkAgg(),
    setupext.BackendMacOSX(),
    ]


classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Framework :: Matplotlib',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Education',
    'License :: OSI Approved :: Python Software Foundation License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering :: Visualization',
    ]


class NoopTestCommand(TestCommand):
    def __init__(self, dist):
        print("Matplotlib does not support running tests with "
              "'python setup.py test'. Please run 'pytest'.")


class BuildExtraLibraries(BuildExtCommand):
    def finalize_options(self):
        self.distribution.ext_modules[:] = filter(
            None, (package.get_extension() for package in good_packages))
        super().finalize_options()

    def build_extensions(self):
        # Remove the -Wstrict-prototypes option, it's not valid for C++.  Fixed
        # in Py3.7 as bpo-5755.
        try:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except (ValueError, AttributeError):
            pass
        for package in good_packages:
            package.do_custom_build()
        return super().build_extensions()


cmdclass = versioneer.get_cmdclass()
cmdclass['test'] = NoopTestCommand
cmdclass['build_ext'] = BuildExtraLibraries


def _download_jquery_to(dest):
    # Note: When bumping the jquery-ui version, also update the versions in
    # single_figure.html and all_figures.html.
    url = "https://jqueryui.com/resources/download/jquery-ui-1.12.1.zip"
    sha = "f8233674366ab36b2c34c577ec77a3d70cac75d2e387d8587f3836345c0f624d"
    name = Path(url).stem
    if (dest / name).exists():
        return
    # If we are installing from an sdist, use the already downloaded jquery-ui.
    sdist_src = Path("lib/matplotlib/backends/web_backend", name)
    if sdist_src.exists():
        shutil.copytree(sdist_src, dest / name)
        return
    if not (dest / name).exists():
        dest.mkdir(parents=True, exist_ok=True)
        try:
            buff = download_or_cache(url, sha)
        except Exception:
            raise IOError(f"Failed to download jquery-ui.  Please download "
                          f"{url} and extract it to {dest}.")
        with ZipFile(buff) as zf:
            zf.extractall(dest)


# Relying on versioneer's implementation detail.
class sdist_with_jquery(cmdclass['sdist']):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        _download_jquery_to(
            Path(base_dir, "lib/matplotlib/backends/web_backend/"))


# Affects install and bdist_wheel.
class install_lib_with_jquery(InstallLibCommand):
    def run(self):
        super().run()
        _download_jquery_to(
            Path(self.install_dir, "matplotlib/backends/web_backend/"))


class develop_with_jquery(DevelopCommand):
    def run(self):
        super().run()
        _download_jquery_to(Path("lib/matplotlib/backends/web_backend/"))


cmdclass['sdist'] = sdist_with_jquery
cmdclass['install_lib'] = install_lib_with_jquery
cmdclass['develop'] = develop_with_jquery


# One doesn't normally see `if __name__ == '__main__'` blocks in a setup.py,
# however, this is needed on Windows to avoid creating infinite subprocesses
# when using multiprocessing.
if __name__ == '__main__':
    package_data = {}  # Will be filled below by the various components.

    # If the user just queries for information, don't bother figuring out which
    # packages to build or install.
    if not (any('--' + opt in sys.argv
                for opt in [*Distribution.display_option_names, 'help'])
            or 'clean' in sys.argv):
        # Go through all of the packages and figure out which ones we are
        # going to build/install.
        print_raw()
        print_raw("Edit setup.cfg to change the build options; "
                  "suppress output with --quiet.")
        print_raw()
        print_raw("BUILDING MATPLOTLIB")

        good_packages = []
        for package in mpl_packages:
            try:
                message = package.check()
            except setupext.Skipped as e:
                print_status(package.name, f"no  [{e}]")
                continue
            if message is not None:
                print_status(package.name, f"yes [{message}]")
            good_packages.append(package)

        print_raw()

        # Now collect all of the information we need to build all of the
        # packages.
        for package in good_packages:
            # Extension modules only get added in build_ext, as numpy will have
            # been installed (as setup_requires) at that point.
            data = package.get_package_data()
            for key, val in data.items():
                package_data.setdefault(key, [])
                package_data[key] = list(set(val + package_data[key]))

        # Write the default matplotlibrc file
        with open('matplotlibrc.template') as fd:
            template_lines = fd.read().splitlines(True)
        backend_line_idx, = [  # Also asserts that there is a single such line.
            idx for idx, line in enumerate(template_lines)
            if line.startswith('#backend:')]
        if setupext.options['backend']:
            template_lines[backend_line_idx] = (
                'backend: {}'.format(setupext.options['backend']))
        with open('lib/matplotlib/mpl-data/matplotlibrc', 'w') as fd:
            fd.write(''.join(template_lines))

    # Use Readme as long description
    with open('README.rst', encoding='utf-8') as fd:
        long_description = fd.read()

    # Finally, pass this all along to distutils to do the heavy lifting.
    setup(
        name="matplotlib",
        version=__version__,
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
        long_description=long_description,
        long_description_content_type="text/x-rst",
        license="PSF",
        platforms="any",
        package_dir={"": "lib"},
        packages=find_packages("lib"),
        namespace_packages=["mpl_toolkits"],
        py_modules=["pylab"],
        # Dummy extension to trigger build_ext, which will swap it out with
        # real extensions that can depend on numpy for the build.
        ext_modules=[Extension("", [])],
        package_data=package_data,
        classifiers=classifiers,

        python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
        setup_requires=[
            "numpy>=1.15",
        ],
        install_requires=[
            "cycler>=0.10",
            "kiwisolver>=1.0.1",
            "numpy>=1.15",
            "pillow>=6.2.0",
            "pyparsing>=2.0.3,!=2.0.4,!=2.1.2,!=2.1.6",
            "python-dateutil>=2.1",
        ],

        cmdclass=cmdclass,
    )
