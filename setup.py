"""
You will need to have freetype, libpng and zlib installed to comile
matplotlib, inlcuding the *-devel versions of these libraries if you
are using a package manager like RPM or debian.

The matplotlib build options can be modified with a setup.cfg file. See
setup.cfg.template for more information.
"""
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

if major==2 and minor1<4 or major<2:
    raise SystemExit("""matplotlib requires Python 2.4 or later.""")

import glob
from distutils.core import setup
from setupext import build_agg, build_gtkagg, build_tkagg,\
     build_macosx, build_ft2font, build_image, build_windowing, build_path, \
     build_contour, build_delaunay, build_nxutils, build_gdk, \
     build_ttconv, print_line, print_status, print_message, \
     print_raw, check_for_freetype, check_for_libpng, check_for_gtk, \
     check_for_tk, check_for_macosx, check_for_numpy, \
     check_for_qt, check_for_qt4, check_for_pyside, check_for_cairo, \
     check_provide_pytz, check_provide_dateutil,\
     check_for_dvipng, check_for_ghostscript, check_for_latex, \
     check_for_pdftops, check_for_datetime, options, build_png, build_tri

# jdh
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
                              'mpl-data/images/*.png',
                              'mpl-data/images/*.ppm',
                              'mpl-data/example/*.npy',
                              'mpl-data/matplotlibrc',
                              'mpl-data/matplotlib.conf',
                              'mpl-data/*.glade',
                              'backends/Matplotlib.nib/*',
                              ]}

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

if not check_for_numpy(__version__numpy__):
    sys.exit(1)

if not check_for_freetype():
    sys.exit(1)

build_ft2font(ext_modules, packages)
build_ttconv(ext_modules, packages)
build_contour(ext_modules, packages)
build_delaunay(ext_modules, packages)
build_nxutils(ext_modules, packages)
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

hasdatetime = check_for_datetime()
provide_dateutil = check_provide_dateutil(hasdatetime)
provide_pytz = check_provide_pytz(hasdatetime)

if hasdatetime: # dates require python23 datetime
    # only install pytz and dateutil if the user hasn't got them

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
        package_data['dateutil'] = ['zoneinfo/zoneinfo*.tar.*']

    if sys.platform=='win32':
        # always add these to the win32 installer
        add_pytz()
        add_dateutil()
    else:
        # only add them if we need them
        if provide_pytz:
            add_pytz()
            print_raw("adding pytz")
        if provide_dateutil: add_dateutil()

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

# Write the default matplotlib.conf file
template = open('lib/matplotlib/mpl-data/matplotlib.conf.template').read()
template = template.replace("datapath = ", "#datapath = ")
template = template.replace("    use = 'Agg'", "    use = '%s'"%rc['backend'])
open('lib/matplotlib/mpl-data/matplotlib.conf', 'w').write(template)

try: additional_params # has setupegg.py provided
except NameError: additional_params = {}

for mod in ext_modules:
    if options['verbose']:
        mod.extra_compile_args.append('-DVERBOSE')

print_raw("pymods %s" % py_modules)
print_raw("packages %s" % packages)
distrib = setup(name="matplotlib",
      version= __version__,
      description = "Python plotting package",
      author = "John D. Hunter",
      author_email="jdh2358@gmail.com",
      url = "http://matplotlib.sourceforge.net",
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
      package_dir = {'': 'lib'},
      package_data = package_data,
      **additional_params
      )
