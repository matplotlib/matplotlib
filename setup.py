"""
The matplotlib build options can be modified with a setup.cfg file. See
setup.cfg.template for more information.
"""

from __future__ import print_function, absolute_import

# This needs to be the very first thing to use distribute
from distribute_setup import use_setuptools
use_setuptools()
from setuptools.command.test import test as TestCommand

import sys

# distutils is breaking our sdists for files in symlinked dirs.
# distutils will copy if os.link is not available, so this is a hack
# to force copying
import os
try:
    del os.link
except AttributeError:
    pass

# This 'if' statement is needed to prevent spawning infinite processes
# on Windows
if __name__ == '__main__':
    # BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
    # update it when the contents of directories change.
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

try:
    from setuptools import setup
except ImportError:
    try:
        from setuptools.core import setup
    except ImportError:
        from distutils.core import setup

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
from setupext import print_line, print_raw, print_message, print_status

# Get the version from the source code
__version__ = setupext.Matplotlib().check()


# These are the packages in the order we want to display them.  This
# list may contain strings to create section headers for the display.
mpl_packages = [
    'Building Matplotlib',
    setupext.Matplotlib(),
    setupext.Python(),
    setupext.Platform(),
    'Required dependencies and extensions',
    setupext.Numpy(),
    setupext.Dateutil(),
    setupext.Pytz(),
    setupext.Tornado(),
    setupext.Pyparsing(),
    setupext.LibAgg(),
    setupext.FreeType(),
    setupext.FT2Font(),
    setupext.Png(),
    setupext.Qhull(),
    setupext.Image(),
    setupext.TTConv(),
    setupext.Path(),
    setupext.ContourLegacy(),
    setupext.Contour(),
    setupext.Delaunay(),
    setupext.QhullWrap(),
    setupext.Tri(),
    setupext.Externals(),
    'Optional subpackages',
    setupext.SampleData(),
    setupext.Toolkits(),
    setupext.Tests(),
    setupext.Toolkits_Tests(),
    'Optional backend extensions',
    # These backends are listed in order of preference, the first
    # being the most preferred.  The first one that looks like it will
    # work will be selected as the default backend.
    setupext.BackendMacOSX(),
    setupext.BackendQt5(),
    setupext.BackendQt4(),
    setupext.BackendGtk3Agg(),
    setupext.BackendGtk3Cairo(),
    setupext.BackendGtkAgg(),
    setupext.BackendTkAgg(),
    setupext.BackendWxAgg(),
    setupext.BackendGtk(),
    setupext.BackendAgg(),
    setupext.BackendCairo(),
    setupext.Windowing(),
    'Optional LaTeX dependencies',
    setupext.DviPng(),
    setupext.Ghostscript(),
    setupext.LaTeX(),
    setupext.PdfToPs()
    ]


classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Python Software Foundation License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Visualization',
    ]


class NoseTestCommand(TestCommand):
    """Invoke unit tests using nose after an in-place build."""

    description = "Invoke unit tests using nose after an in-place build."
    user_options = [
        ("pep8-only", None, "pep8 checks"),
        ("omit-pep8", None, "Do not perform pep8 checks"),
        ("nocapture", None, "do not capture stdout (nosetests)"),
        ("nose-verbose", None, "be verbose (nosetests)"),
        ("processes=", None, "number of processes (nosetests)"),
        ("process-timeout=", None, "process timeout (nosetests)"),
        ("with-coverage", None, "with coverage"),
        ("detailed-error-msg", None, "detailed error message (nosetest)"),
        ("tests=", None, "comma separated selection of tests (nosetest)"),
    ]

    def initialize_options(self):
        self.pep8_only = None
        self.omit_pep8 = None

        # parameters passed to nose tests
        self.processes = None
        self.process_timeout = None
        self.nose_verbose = None
        self.nocapture = None
        self.with_coverage = None
        self.detailed_error_msg = None
        self.tests = None

    def finalize_options(self):
        self.test_args = []
        if self.pep8_only:
            self.pep8_only = True
        if self.omit_pep8:
            self.omit_pep8 = True

        if self.pep8_only and self.omit_pep8:
            from distutils.errors import DistutilsOptionError
            raise DistutilsOptionError(
                "You are using several options for the test command in an "
                "incompatible manner. Please use either --pep8-only or "
                "--omit-pep8"
            )

        if self.processes:
            self.test_args.append("--processes={prc}".format(
                prc=self.processes))

        if self.process_timeout:
            self.test_args.append("--process-timeout={tout}".format(
                tout=self.process_timeout))

        if self.nose_verbose:
            self.test_args.append("--verbose")

        if self.nocapture:
            self.test_args.append("--nocapture")

        if self.with_coverage:
            self.test_args.append("--with-coverage")

        if self.detailed_error_msg:
            self.test_args.append("-d")

        if self.tests:
            self.test_args.append("--tests={names}".format(names=self.tests))

    def run(self):
        if self.distribution.install_requires:
            self.distribution.fetch_build_eggs(
                self.distribution.install_requires)
        if self.distribution.tests_require:
            self.distribution.fetch_build_eggs(self.distribution.tests_require)

        self.announce('running unittests with nose')
        self.with_project_on_sys_path(self.run_tests)

    def run_tests(self):
        import matplotlib
        matplotlib.use('agg')
        import nose
        from matplotlib.testing.noseclasses import KnownFailure
        from matplotlib import default_test_modules as testmodules
        from matplotlib import font_manager
        import time
        # Make sure the font caches are created before starting any possibly
        # parallel tests
        if font_manager._fmcache is not None:
            while not os.path.exists(font_manager._fmcache):
                time.sleep(0.5)
        plugins = [KnownFailure]

        # Nose doesn't automatically instantiate all of the plugins in the
        # child processes, so we have to provide the multiprocess plugin
        # with a list.
        from nose.plugins import multiprocess
        multiprocess._instantiate_plugins = plugins

        if self.omit_pep8:
            testmodules.remove('matplotlib.tests.test_coding_standards')
        elif self.pep8_only:
            testmodules = ['matplotlib.tests.test_coding_standards']

        nose.main(addplugins=[x() for x in plugins],
                  defaultTest=testmodules,
                  argv=['nosetests'] + self.test_args,
                  exit=True)


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
    tests_require = []
    default_backend = None

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
                if isinstance(package, setupext.OptionalBackendPackage):
                    if default_backend is None:
                        default_backend = package.name
    print_raw('')

    # Abort if any of the required packages can not be built.
    if required_failed:
        print_line()
        print_message(
            "The following required packages can not "
            "be built: %s" %
            ', '.join(x.name for x in required_failed))
        sys.exit(1)

    # Now collect all of the information we need to build all of the
    # packages.
    for package in good_packages:
        if isinstance(package, str):
            continue
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
        tests_require.extend(package.get_tests_require())

    # Write the default matplotlibrc file
    if default_backend is None:
        default_backend = 'svg'
    if setupext.options['backend']:
        default_backend = setupext.options['backend']
    with open('matplotlibrc.template') as fd:
        template = fd.read()
    with open('lib/matplotlib/mpl-data/matplotlibrc', 'w') as fd:
        fd.write(template % {'backend': default_backend})

    # Build in verbose mode if requested
    if setupext.options['verbose']:
        for mod in ext_modules:
            mod.extra_compile_args.append('-DVERBOSE')

    # Finalize the extension modules so they can get the Numpy include
    # dirs
    for mod in ext_modules:
        mod.finalize()

    extra_args = {}

    # Avoid installing setup_requires dependencies if the user just
    # queries for information
    if (any('--' + opt in sys.argv for opt in
           Distribution.display_option_names + ['help']) or
        'clean' in sys.argv):
        setup_requires = []

    # Finally, pass this all along to distutils to do the heavy lifting.
    distrib = setup(
        name="matplotlib",
        version=__version__,
        description="Python plotting package",
        author="John D. Hunter, Michael Droettboom",
        author_email="mdroe@stsci.edu",
        url="http://matplotlib.org",
        long_description="""
        matplotlib strives to produce publication quality 2D graphics
        for interactive graphing, scientific publishing, user interface
        development and web application servers targeting multiple user
        interfaces and hardcopy output formats.  There is a 'pylab' mode
        which emulates matlab graphics.
        """,
        license="BSD",
        packages=packages,
        namespace_packages=namespace_packages,
        platforms='any',
        py_modules=py_modules,
        ext_modules=ext_modules,
        package_dir=package_dir,
        package_data=package_data,
        classifiers=classifiers,
        download_url="https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-{0}/matplotlib-{0}.tar.gz".format(__version__),

        # List third-party Python packages that we require
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,

        # matplotlib has C/C++ extensions, so it's not zip safe.
        # Telling setuptools this prevents it from doing an automatic
        # check for zip safety.
        zip_safe=False,
        cmdclass={'test': NoseTestCommand},

        **extra_args
    )
