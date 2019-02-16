"""
The matplotlib build options can be modified with a setup.cfg file. See
setup.cfg.template for more information.
"""

# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
from string import Template
import sys

if sys.version_info < (3, 5):
    error = """
Matplotlib 3.0+ does not support Python 2.x, 3.0, 3.1, 3.2, 3.3, or 3.4.
Beginning with Matplotlib 3.0, Python 3.5 and above is required.

This may be due to an out of date pip.

Make sure you have pip >= 9.0.1.
"""
    sys.exit(error)

from io import BytesIO
import os
from string import Template
import urllib.request
from zipfile import ZipFile

from setuptools import setup
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
from setupext import (print_line, print_raw, print_message, print_status,
                      download_or_cache, makedirs as _makedirs)

# Get the version from versioneer
import versioneer
__version__ = versioneer.get_version()


# These are the packages in the order we want to display them.  This
# list may contain strings to create section headers for the display.
mpl_packages = [
    'Building Matplotlib',
    setupext.Matplotlib(),
    setupext.Python(),
    setupext.Platform(),
    'Required dependencies and extensions',
    setupext.Numpy(),
    setupext.InstallRequires(),
    setupext.LibAgg(),
    setupext.FreeType(),
    setupext.FT2Font(),
    setupext.Png(),
    setupext.Qhull(),
    setupext.Image(),
    setupext.TTConv(),
    setupext.Path(),
    setupext.Contour(),
    setupext.QhullWrap(),
    setupext.Tri(),
    'Optional subpackages',
    setupext.SampleData(),
    setupext.Toolkits(),
    setupext.Tests(),
    setupext.Toolkits_Tests(),
    'Optional backend extensions',
    setupext.BackendAgg(),
    setupext.BackendTkAgg(),
    setupext.BackendMacOSX(),
    setupext.Windowing(),
    'Optional package data',
    setupext.Dlls(),
    ]


classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Python Software Foundation License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering :: Visualization',
    ]


class NoopTestCommand(TestCommand):
    def __init__(self, dist):
        print("Matplotlib does not support running tests with "
              "'python setup.py test'. Please run 'pytest'.")


class BuildExtraLibraries(BuildExtCommand):
    def run(self):
        for package in good_packages:
            package.do_custom_build()

        return BuildExtCommand.run(self)


cmdclass = versioneer.get_cmdclass()
cmdclass['test'] = NoopTestCommand
cmdclass['build_ext'] = BuildExtraLibraries


def _download_jquery_to(dest):
    # Note: When bumping the jquery-ui version, also update the versions in
    # single_figure.html and all_figures.html.
    url = "https://jqueryui.com/resources/download/jquery-ui-1.12.1.zip"
    sha = 'f8233674366ab36b2c34c577ec77a3d70cac75d2e387d8587f3836345c0f624d'
    if not os.path.exists(os.path.join(dest, "jquery-ui-1.12.1")):
        _makedirs(dest, exist_ok=True)
        try:
            buff = download_or_cache(url, sha)
        except Exception:
            raise IOError("Failed to download jquery-ui.  Please download " +
                          "{url} and extract it to {dest}.".format(
                              url=url, dest=dest))
        with ZipFile(buff) as zf:
            zf.extractall(dest)


# Relying on versioneer's implementation detail.
class sdist_with_jquery(cmdclass['sdist']):
    def make_release_tree(self, base_dir, files):
        super(sdist_with_jquery, self).make_release_tree(base_dir, files)
        _download_jquery_to(
            os.path.join(base_dir, "lib/matplotlib/backends/web_backend/"))


# Affects install and bdist_wheel.
class install_lib_with_jquery(InstallLibCommand):
    def run(self):
        super(install_lib_with_jquery, self).run()
        _download_jquery_to(
            os.path.join(self.install_dir, "matplotlib/backends/web_backend/"))


class develop_with_jquery(DevelopCommand):
    def run(self):
        super(develop_with_jquery, self).run()
        _download_jquery_to("lib/matplotlib/backends/web_backend/")


cmdclass['sdist'] = sdist_with_jquery
cmdclass['install_lib'] = install_lib_with_jquery
cmdclass['develop'] = develop_with_jquery


# One doesn't normally see `if __name__ == '__main__'` blocks in a setup.py,
# however, this is needed on Windows to avoid creating infinite subprocesses
# when using multiprocessing.
if __name__ == '__main__':
    # These are distutils.setup parameters that the various packages add
    # things to.
    packages = []
    namespace_packages = []
    py_modules = []
    ext_modules = []
    package_data = {}
    package_dir = {'': 'lib'}
    install_requires = []
    setup_requires = []

    # If the user just queries for information, don't bother figuring out which
    # packages to build or install.
    if (any('--' + opt in sys.argv for opt in
            Distribution.display_option_names + ['help']) or
            'clean' in sys.argv):
        setup_requires = []
    else:
        # Go through all of the packages and figure out which ones we are
        # going to build/install.
        print_line()
        print_raw("Edit setup.cfg to change the build options")

        required_failed = []
        good_packages = []
        for package in mpl_packages:
            if isinstance(package, str):
                print_raw('')
                print_raw(package.upper())
            else:
                try:
                    result = package.check()
                    if result is not None:
                        message = 'yes [%s]' % result
                        print_status(package.name, message)
                except setupext.CheckFailed as e:
                    msg = str(e).strip()
                    if len(msg):
                        print_status(package.name, 'no  [%s]' % msg)
                    else:
                        print_status(package.name, 'no')
                    if not package.optional:
                        required_failed.append(package)
                else:
                    good_packages.append(package)
        print_raw('')

        # Abort if any of the required packages can not be built.
        if required_failed:
            print_line()
            print_message("The following required packages can not be built: "
                          "%s" % ", ".join(x.name for x in required_failed))
            for pkg in required_failed:
                msg = pkg.install_help_msg()
                if msg:
                    print_message(msg)
            sys.exit(1)

        # Now collect all of the information we need to build all of the
        # packages.
        for package in good_packages:
            packages.extend(package.get_packages())
            namespace_packages.extend(package.get_namespace_packages())
            py_modules.extend(package.get_py_modules())
            ext = package.get_extension()
            if ext is not None:
                ext_modules.append(ext)
            data = package.get_package_data()
            for key, val in data.items():
                package_data.setdefault(key, [])
                package_data[key] = list(set(val + package_data[key]))
            install_requires.extend(package.get_install_requires())
            setup_requires.extend(package.get_setup_requires())

        # Write the default matplotlibrc file
        with open('matplotlibrc.template') as fd:
            template_lines = fd.read().splitlines(True)
        backend_line_idx, = [  # Also asserts that there is a single such line.
            idx for idx, line in enumerate(template_lines)
            if line.startswith('#backend ')]
        if setupext.options['backend']:
            template_lines[backend_line_idx] = (
                'backend: {}'.format(setupext.options['backend']))
        with open('lib/matplotlib/mpl-data/matplotlibrc', 'w') as fd:
            fd.write(''.join(template_lines))

        # Finalize the extension modules so they can get the Numpy include
        # dirs
        for mod in ext_modules:
            mod.finalize()

    # Finally, pass this all along to distutils to do the heavy lifting.
    setup(
        name="matplotlib",
        version=__version__,
        description="Python plotting package",
        author="John D. Hunter, Michael Droettboom",
        author_email="matplotlib-users@python.org",
        url="http://matplotlib.org",
        long_description="""
        Matplotlib strives to produce publication quality 2D graphics
        for interactive graphing, scientific publishing, user interface
        development and web application servers targeting multiple user
        interfaces and hardcopy output formats.
        """,
        license="PSF",
        packages=packages,
        namespace_packages=namespace_packages,
        platforms='any',
        py_modules=py_modules,
        ext_modules=ext_modules,
        package_dir=package_dir,
        package_data=package_data,
        classifiers=classifiers,
        download_url="http://matplotlib.org/users/installing.html",

        python_requires='>=3.5',
        # List third-party Python packages that we require
        install_requires=install_requires,
        setup_requires=setup_requires,

        # matplotlib has C/C++ extensions, so it's not zip safe.
        # Telling setuptools this prevents it from doing an automatic
        # check for zip safety.
        zip_safe=False,
        cmdclass=cmdclass,
    )
