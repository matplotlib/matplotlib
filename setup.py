"""
You will need to have freetype, libpng and zlib installed to compile
matplotlib, inlcuding the *-devel versions of these libraries if you
are using a package manager like RPM or debian.

The matplotlib build options can be modified with a setup.cfg file. See
setup.cfg.template for more information.
"""

# This dict will be updated as we try to select the best option during
# the build process. However, values in setup.cfg will be used, if 
# defined.
rc = {'backend':'Agg', 'numerix':'numpy'}


# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

import sys
major, minor1, minor2, s, tmp = sys.version_info

if major==2 and minor1<=3:
    # setuptools monkeypatches distutils.core.Distribution to support
    # package_data
    try: import setuptools
    except ImportError:
        raise SystemExit("""\
matplotlib requires setuptools for installation with python-2.3.  Visit:
http://cheeseshop.python.org/pypi/setuptools
for installation instructions for the proper version of setuptools for your
system.  If this is your first time upgrading matplotlib with the new
setuptools requirement, you must delete the old matplotlib install
directory.""")

if major==2 and minor1<3 or major<2:
    raise SystemExit("""matplotlib requires Python 2.3 or later.""")
    
import glob
from distutils.core import setup
from setupext import build_agg, build_gtkagg, build_tkagg, build_wxagg,\
     build_ft2font, build_image, build_windowing, build_transforms, \
     build_contour, build_nxutils, build_traits, build_swigagg, build_gdk, \
     build_subprocess, build_ttconv, print_line, print_status, print_message, \
     print_raw, check_for_freetype, check_for_libpng, check_for_gtk, \
     check_for_tk, check_for_wx, check_for_numpy, check_for_qt, check_for_qt4, \
     check_for_cairo, check_provide_traits, check_provide_pytz, \
     check_provide_dateutil, check_provide_configobj, check_for_dvipng, \
     check_for_ghostscript, check_for_latex, check_for_pdftops, \
     check_for_datetime, options
#import distutils.sysconfig

# jdh
packages = [
    'matplotlib',
    'matplotlib.backends',
    'matplotlib.toolkits',
    'matplotlib.numerix',
    'matplotlib.numerix.mlab',
    'matplotlib.numerix.ma',
    'matplotlib.numerix.npyma',
    'matplotlib.numerix.linear_algebra',
    'matplotlib.numerix.random_array',
    'matplotlib.numerix.fft'
    ]

py_modules = ['pylab']

ext_modules = []

for line in file('lib/matplotlib/__init__.py').readlines():
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
                              'mpl-data/images/*.xpm',
                              'mpl-data/images/*.svg',
                              'mpl-data/images/*.png',
                              'mpl-data/images/*.ppm',
                              'mpl-data/matplotlibrc',
                              'mpl-data/matplotlib.conf',
                              'mpl-data/*.glade',
                              'backends/Matplotlib.nib/*',
                              ]}

if not check_for_numpy():
    sys.exit()

try: import subprocess
except ImportError: havesubprocess = False
else: havesubprocess = True

if havesubprocess and sys.version < '2.4':
    # Python didn't come with subprocess, so let's make sure it's
    # not in some Python egg (e.g. an older version of matplotlib)
    # that may get removed.
    subprocess_dir = os.path.dirname(subprocess.__file__)
    if subprocess_dir.endswith('.egg/subprocess'):
        havesubprocess = False
        
if not havesubprocess:
    packages.append('subprocess')
    if sys.platform == 'win32':
        build_subprocess(ext_modules, packages)

if not check_for_freetype():
    sys.exit(1)
    
build_ft2font(ext_modules, packages)
build_ttconv(ext_modules, packages)
build_contour(ext_modules, packages)
build_nxutils(ext_modules, packages)

build_swigagg(ext_modules, packages)
build_transforms(ext_modules, packages)

print_raw("")
print_raw("OPTIONAL BACKEND DEPENDENCIES")

if check_for_libpng() and options['build_agg']:
    build_agg(ext_modules, packages)
    rc['backend'] = 'Agg'
else:
    rc['backend'] = 'SVG'

if options['build_image']:
    build_image(ext_modules, packages)

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

if options['build_wxagg']:
    if check_for_wx() or (options['build_wxagg'] is True):
        options['build_agg'] = 1
        import wx
        if getattr(wx, '__version__', '0.0')[0:3] < '2.8' :
            build_wxagg(ext_modules, packages)
            wxagg_backend_status = "yes"
        else:
            print_message("WxAgg extension not required for wxPython >= 2.8")
        rc['backend'] = 'WXAgg'

hasgtk = check_for_gtk()
if options['build_gtk']:
    if hasgtk or (options['build_gtk'] is True):
        build_gdk(ext_modules, packages)
if options['build_gtkagg']:
    if hasgtk or (options['build_gtkagg'] is True):
        options['build_agg'] = 1
        build_gtkagg(ext_modules, packages)
        rc['backend'] = 'GTKAgg'

# These are informational only.  We don't build any extensions for them.
check_for_qt()
check_for_qt4()
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
        # install pytz subdirs
        for dirpath, dirname, filenames in os.walk(os.path.join('lib', 'pytz', 
                                                                'zoneinfo')):
            if '.svn' not in dirpath:
                # remove the 'lib/pytz' part of the path
                basepath = dirpath.split(os.path.sep, 2)[2]
                resources.extend([os.path.join(basepath, filename)
                                  for filename in filenames])
        package_data['pytz'] = resources
        assert len(resources) > 10, 'pytz zoneinfo files not found!'
#                packages.append('/'.join(dirpath.split(os.sep)[1:]))

    def add_dateutil():
        packages.append('dateutil')
        packages.append('dateutil/zoneinfo')
        package_data['dateutil'] = ['zoneinfo/zoneinfo*.tar.*']

    if sys.platform=='win32':
        # always add these to the win32 installer
        add_pytz()
        add_dateutil()
    else:
        # only add them if we need them
        if provide_pytz: add_pytz()
        if provide_dateutil: add_dateutil()

print_raw("")
print_raw("OPTIONAL USETEX DEPENDENCIES")
check_for_dvipng()
check_for_ghostscript()
check_for_latex()
check_for_pdftops()

# TODO: comment out for mpl release:
print_raw("")
print_raw("EXPERIMENTAL CONFIG PACKAGE DEPENDENCIES")
packages.append('matplotlib.config')
if check_provide_configobj(): py_modules.append('configobj')
if check_provide_traits(): build_traits(ext_modules, packages)

print_raw("")
print_raw("[Edit setup.cfg to suppress the above messages]")
print_line()

# Write the default matplotlibrc file
if sys.platform=='win32':
    rc['backend'] = 'TkAgg'
    rc['numerix'] = 'numpy'
else:
    if options['backend']: rc['backend'] = options['backend']
    if options['numerix']: rc['numerix'] = options['numerix']
template = file('matplotlibrc.template').read()
file('lib/matplotlib/mpl-data/matplotlibrc', 'w').write(template%rc)

# Write the default matplotlib.conf file
template = file('lib/matplotlib/mpl-data/matplotlib.conf.template').read()
template = template.replace("datapath = ", "#datapath = ")
template = template.replace("numerix = 'numpy'", "numerix = '%s'"%rc['numerix'])
template = template.replace("    use = 'Agg'", "    use = '%s'"%rc['backend'])
file('lib/matplotlib/mpl-data/matplotlib.conf', 'w').write(template)

try: additional_params # has setupegg.py provided
except NameError: additional_params = {}

for mod in ext_modules:
    if options['verbose']:
        mod.extra_compile_args.append('-DVERBOSE')

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
