"""
You will need to have freetype, libpng and zlib installed to compile
matplotlib, inlcuding the *-devel versions of these libraries if you
are using a package manager like RPM or debian.

The matplotlib build options can be modified with a setup.cfg file. See
setup.cfg.template for more information.
"""
from __future__ import print_function

# distutils is breaking our sdists for files in symlinked dirs.
# distutils will copy if os.link is not available, so this is a hack
# to force copying
import os
try:
    del os.link
except AttributeError:
    pass

# This dict will be updated as we try to select the best option during
# the build process. However, values in setup.cfg will be used, if
# defined.
rc = {'backend':'Agg'}

# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

import sys
major, minor1, minor2, s, tmp = sys.version_info

if major==2 and minor1<6 or major<2:
    raise SystemExit("""matplotlib requires Python 2.6 or later.""")

import glob
from distutils.core import setup
try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py
from setupext import build_agg, build_gtkagg, build_tkagg,\
     build_macosx, build_ft2font, build_image, build_windowing, build_path, \
     build_contour, build_delaunay, build_gdk, \
     build_ttconv, print_line, print_status, print_message, \
     print_raw, check_for_freetype, check_for_libpng, check_for_gtk, \
     check_for_tk, check_for_macosx, check_for_numpy, \
     check_for_qt, check_for_qt4, check_for_pyside, check_for_cairo, \
     check_provide_pytz, check_provide_dateutil,\
     check_for_dvipng, check_for_ghostscript, check_for_latex, \
     check_for_pdftops, options, build_png, build_tri, check_provide_six


packages = [
    'matplotlib',
    'matplotlib.backends',
    'matplotlib.backends.qt4_editor',
    'matplotlib.projections',
    'matplotlib.testing',
    'matplotlib.testing.jpl_units',
    'matplotlib.tests',
    'mpl_toolkits',
    'mpl_toolkits.mplot3d',
    'mpl_toolkits.axes_grid',
    'mpl_toolkits.axes_grid1',
    'mpl_toolkits.axisartist',
    'matplotlib.sphinxext',
    'matplotlib.tri',

    ]

py_modules = ['pylab']

ext_modules = []

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Python Software Foundation License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Visualization',
    ]

for line in open('lib/matplotlib/__init__.py').readlines():
    if (line.startswith('__version__')):
        exec(line.strip())

print_line()
print_raw("BUILDING MATPLOTLIB")
print_status('matplotlib', __version__)
print_status('python', sys.version)
print_status('platform', sys.platform)
if sys.platform == 'win32':
    print_status('Windows version', sys.getwindowsversion())
print_raw("")
print_raw("REQUIRED DEPENDENCIES")

# Specify all the required mpl data
package_data = {'matplotlib':['mpl-data/fonts/afm/*.afm',
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
                              'mpl-data/images/*.png',
                              'mpl-data/images/*.ppm',
                              'mpl-data/example/*.npy',
                              'mpl-data/matplotlibrc',
                              'mpl-data/*.glade',
                              'mpl-data/sample_data/*.*',
                              'mpl-data/sample_data/axes_grid/*.*',
                              'backends/Matplotlib.nib/*',
                              ]}

package_dir = {'': 'lib'}

if 1:
    # TODO: exclude these when making release?
    baseline_images = glob.glob(os.path.join('lib','matplotlib','tests',
                                             'baseline_images','*','*'))
    def chop_package(fname):
        badstr = os.path.join('lib','matplotlib','')
        assert fname.startswith(badstr)
        result = fname[ len(badstr): ]
        return result
    baseline_images = [chop_package(f) for f in baseline_images]
    package_data['matplotlib'].extend(baseline_images)
    package_data['matplotlib'].append('tests/mpltest.ttf')
    package_data['matplotlib'].append('tests/test_rcparams.rc')

if not check_for_numpy(__version__numpy__):
    sys.exit(1)

if not check_for_freetype():
    sys.exit(1)

build_ft2font(ext_modules, packages)
build_ttconv(ext_modules, packages)
build_contour(ext_modules, packages)
build_delaunay(ext_modules, packages)
build_path(ext_modules, packages)
build_tri(ext_modules, packages)

print_raw("")
print_raw("OPTIONAL BACKEND DEPENDENCIES")
has_libpng = check_for_libpng()

if has_libpng and options['build_agg']:
    build_agg(ext_modules, packages)
    rc['backend'] = 'Agg'
else:
    rc['backend'] = 'SVG'

if has_libpng and options['build_image']:
    build_image(ext_modules, packages)

if has_libpng and options['build_agg'] or options['build_image']:
    build_png(ext_modules, packages)

if options['build_windowing'] and sys.platform=='win32':
   build_windowing(ext_modules, packages)

# the options can be True, False, or 'auto'. If True, try to build
# regardless of the lack of dependencies. If auto, silently skip
# when dependencies are missing.
if options['build_tkagg']:
    if check_for_tk() or (options['build_tkagg'] is True):
        options['build_agg'] = 1
        build_tkagg(ext_modules, packages)
        rc['backend'] = 'TkAgg'

hasgtk = check_for_gtk()
if options['build_gtk']:
    if hasgtk or (options['build_gtk'] is True):
        build_gdk(ext_modules, packages)
if options['build_gtkagg']:
    if hasgtk or (options['build_gtkagg'] is True):
        options['build_agg'] = 1
        build_gtkagg(ext_modules, packages)
        rc['backend'] = 'GTKAgg'

if options['build_macosx']:
    if check_for_macosx() or (options['build_macosx'] is True):
        build_macosx(ext_modules, packages)
        rc['backend'] = 'MacOSX'

# These are informational only.  We don't build any extensions for them.
check_for_qt()
check_for_qt4()
check_for_pyside()
check_for_cairo()

print_raw("")
print_raw("OPTIONAL DATE/TIMEZONE DEPENDENCIES")

provide_dateutil = check_provide_dateutil()
provide_pytz = check_provide_pytz()
provide_six = check_provide_six()

def add_pytz():
    packages.append('pytz')

    resources = ['zone.tab', 'locales/pytz.pot']
    for dirpath, dirnames, filenames in os.walk(
        os.path.join('lib', 'pytz', 'zoneinfo')
        ):

        # remove the 'pytz' part of the path
        basepath = os.path.join(*dirpath.split(os.path.sep)[2:])
        #print dirpath, basepath
        resources.extend([os.path.join(basepath, filename)
                          for filename in filenames])
    package_data['pytz'] = resources
    #print resources
    assert len(resources) > 10, 'zoneinfo files not found!'

def add_dateutil():
    packages.append('dateutil')
    packages.append('dateutil.zoneinfo')
    package_data['dateutil'] = ['zoneinfo/*.tar.gz']
    if sys.version_info[0] >= 3:
        package_dir['dateutil'] = 'lib/dateutil_py3'
    else:
        package_dir['dateutil'] = 'lib/dateutil_py2'

def add_six():
    py_modules.append('six')

if sys.platform=='win32':
    # always add these to the win32 installer
    add_pytz()
    add_dateutil()
    add_six()
else:
    # only add them if we need them
    if provide_pytz:
        add_pytz()
    if provide_dateutil:
        add_dateutil()
    if provide_six:
        add_six()

print_raw("")
print_raw("OPTIONAL USETEX DEPENDENCIES")
check_for_dvipng()
check_for_ghostscript()
check_for_latex()
check_for_pdftops()

print_raw("")
print_raw("[Edit setup.cfg to suppress the above messages]")
print_line()

# Write the default matplotlibrc file
if options['backend']: rc['backend'] = options['backend']
template = open('matplotlibrc.template').read()
open('lib/matplotlib/mpl-data/matplotlibrc', 'w').write(template%rc)

try: additional_params # has setupegg.py provided
except NameError: additional_params = {}

for mod in ext_modules:
    if options['verbose']:
        mod.extra_compile_args.append('-DVERBOSE')

if sys.version_info[0] >= 3:
    def should_2to3(file, root):
        file = os.path.abspath(file)[len(os.path.abspath(root))+1:]
        if ('py3' in file or
            file.startswith('pytz') or
            file.startswith('dateutil') or
            file.startswith('six')):
            return False
        return True

    import multiprocessing
    def refactor(x):
        from lib2to3.refactor import RefactoringTool, get_fixers_from_package
        class DistutilsRefactoringTool(RefactoringTool):
            def ignore(self, msg, *args, **kw):
                pass
            log_error = log_message = log_debug = ignore
        fixer_names = get_fixers_from_package('lib2to3.fixes')
        r = DistutilsRefactoringTool(fixer_names, options=None)
        r.refactor([x], write=True)

    original_build_py = build_py
    class build_py(original_build_py):
        def run_2to3(self, files):
            # We need to skip certain files that have already been
            # converted to Python 3.x
            filtered = [x for x in files if should_2to3(x, self.build_lib)]
            if sys.platform.startswith('win'):
                # doing this in parallel on windows may crash your computer
                [refactor(f) for f in filtered]
            else:
                p = multiprocessing.Pool()
                for i, x in enumerate(p.imap_unordered(refactor, filtered)):
                    print("Running 2to3... %.02f%%" %
                          (float(i) / len(filtered) * 100.0), end='\r')
            print()

print_raw("pymods %s" % py_modules)
print_raw("packages %s" % packages)
distrib = setup(name="matplotlib",
      version= __version__,
      description = "Python plotting package",
      author = "John D. Hunter, Michael Droettboom",
      author_email="mdroe@stsci.edu",
      url = "http://matplotlib.org",
      long_description = """
      matplotlib strives to produce publication quality 2D graphics
      for interactive graphing, scientific publishing, user interface
      development and web application servers targeting multiple user
      interfaces and hardcopy output formats.  There is a 'pylab' mode
      which emulates matlab graphics
      """,
      packages = packages,
      platforms='any',
      py_modules = py_modules,
      ext_modules = ext_modules,
      package_dir = package_dir,
      package_data = package_data,
      classifiers = classifiers,
      cmdclass = {'build_py': build_py},
      **additional_params
      )
