"""
Note! If you are building for python2.2, you must comment out the
py_modules line below and manually copy lib/pylab.py to
site-packages/pylab.py

You will need to have freetype, libpng and zlib installed to compile
matplotlib, inlcuding the *-devel versions of these libraries if you
are using a package manager like RPM or debian.

matplotlib has added some extension module code which can optionally
be built by setting the appropriate flag below.

The GTKAgg and TkAgg will try to build if they detect pygtk or Tkinter
respectively; set them to 0 if you do not want to build them
"""


rc = dict({'backend':'PS', 'numerix':'Numeric'})

# build the image support module - requires agg and Numeric or
# numarray.  You can build the image module with either Numeric or
# numarray or both.  By default, matplotlib will build support for
# whatever array packages you have installed.
BUILD_IMAGE = 1

# Build the antigrain geometry toolkit.  Agg makes heavy use of
# templates, so it probably requires a fairly recent compiler to build
# it.  It makes very nice antialiased output and also supports alpha
# blending
BUILD_AGG          = 1
BUILD_GTKAGG       = 'auto'
BUILD_GTK          = 'auto'

# build TK GUI with Agg renderer ; requires Tkinter Python extension
# and Tk includes
# Use False or 0 if you don't want to build
BUILD_TKAGG        = 'auto'

# build wxPython extension code to efficiently blit agg into wx.  Only
# needed for wxpython <2.8 if you plan on doing animations
BUILD_WXAGG        = 0


# build a small extension to manage the focus on win32 platforms.
#BUILD_WINDOWING        = 0
BUILD_WINDOWING        = 'auto'


VERBOSE = False # insert lots of diagnostic prints in extension code




## You shouldn't need to customize below this point

# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

import sys
major, minor1, minor2, s, tmp = sys.version_info

if major==2 and minor1==2:
    print >> sys.stderr, "***\n\nWARNING, see build info for python2.2 in the header of setup.py\n\n***"
if major==2 and minor1<=3:
    # setuptools monkeypatches distutils.core.Distribution to support
    # package_data
    try: import setuptools 
    except ImportError:
        raise SystemExit("""\
matplotlib requires setuptools for installation.  Visit:
http://cheeseshop.python.org/pypi/setuptools
for installation instructions for the proper version of setuptools for your
system.  If this is your first time upgrading matplotlib with the new
setuptools requirement, you must delete the old matplotlib install
directory.""")
    
import glob
from distutils.core import Extension, setup
from setupext import build_agg, build_gtkagg, build_tkagg, build_wxagg,\
     build_ft2font, build_image, build_windowing, build_transforms, \
     build_contour, build_nxutils, build_enthought, build_swigagg, build_gdk, \
     build_subprocess, build_isnan
import distutils.sysconfig

for line in file('lib/matplotlib/__init__.py').readlines():
    if line[:11] == '__version__':
        exec(line.strip())

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
                              'backends/Matplotlib.nib/*',
                              ]}

# Figure out which array packages to provide binary support for
# and append to the NUMERIX list.
NUMERIX = []

try:
    import Numeric
    NUMERIX.append('Numeric')
except ImportError:
    pass
try:
    import numarray
    NUMERIX.append('numarray')
except ImportError:
    pass
try:
     import numpy
     NUMERIX.append('numpy')
except ImportError:
     pass

if not NUMERIX:
    raise RuntimeError("You must install one or more of numpy, Numeric, and numarray to build matplotlib")


#NUMERIX = ['numpy']
rc['numerix'] = NUMERIX[-1]

ext_modules = []

# these are not optional
BUILD_FT2FONT = 1
BUILD_CONTOUR = 1
BUILD_NXUTILS = 1

# jdh
packages = [
    'matplotlib',
    'matplotlib.backends',
    'matplotlib.toolkits',
    'matplotlib.numerix',
    'matplotlib.numerix.mlab',
    'matplotlib.numerix.ma',
    'matplotlib.numerix.linear_algebra',
    'matplotlib.numerix.random_array',
    'matplotlib.numerix.fft',
    ]


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

build_isnan(ext_modules, packages)

try: import datetime
except ImportError: havedate = False
else: havedate = True

if havedate: # dates require python23 datetime
    # only install pytz and dateutil if the user hasn't got them
    def add_pytz():
        packages.append('pytz')
        # install pytz subdirs
        for dirpath, dirname, filenames in os.walk(os.path.join('lib', 'pytz','zoneinfo')):
            if '.svn' not in dirpath:
                packages.append('/'.join(dirpath.split(os.sep)[1:]))

    def add_dateutil():
        packages.append('dateutil')

    if sys.platform=='win32':
        # always add these to the win32 installer
        add_pytz()
        add_dateutil()
    else:
        # only add them if we need them
        try: import dateutil
        except ImportError:
            add_dateutil()

        try: import pytz
        except ImportError:
            add_pytz()

build_swigagg(ext_modules, packages)
build_transforms(ext_modules, packages, NUMERIX)
build_enthought(ext_modules, packages)

def havegtk():
    'check for the presence of pygtk'
    if havegtk.gotit is not None: return havegtk.gotit
    try:
        import gtk
    except ImportError:
        print 'building for GTK requires pygtk; you must be able to "import gtk" in your build/install environment'
        havegtk.gotit = False
    except RuntimeError:
        print 'pygtk present but import failed'
        havegtk.gotit = False
    else:
        version = (2,2,0)
        if gtk.pygtk_version < version:
            print "Error: GTK backend requires PyGTK %d.%d.%d (or later), " \
                  "%d.%d.%d was detected." % (
                version + gtk.pygtk_version)
            havegtk.gotit = False
        else:
            havegtk.gotit = True
    return havegtk.gotit

havegtk.gotit = None

if BUILD_GTK and havegtk():
    build_gdk(ext_modules, packages, NUMERIX)
    rc['backend'] = 'GTK'

if BUILD_GTKAGG and havegtk():
    BUILD_AGG = 1
    build_gtkagg(ext_modules, packages, NUMERIX)
    rc['backend'] = 'GTKAgg'

if BUILD_TKAGG:
    try:
        import Tkinter
    except ImportError:
        print 'TKAgg requires TkInter'
        BUILD_TKAGG = 0
    except RuntimeError:
        print 'Tkinter present but import failed'
        BUILD_TKAGG = 0
    else:
        try:
            tk = Tkinter.Tk()
            tk.withdraw()
        except Tkinter.TclError:
            print 'Tkinter present, but window failed to open'
            BUILD_TKAGG = 0
        else:
            BUILD_AGG = 1
            build_tkagg(ext_modules, packages, NUMERIX)
            rc['backend'] = 'TkAgg'

if BUILD_WXAGG:
    try:
        import wx
    except ImportError:
        if BUILD_WXAGG != 'auto':
            print 'WXAgg\'s accelerator requires wxPython'
        BUILD_WXAGG = 0
    else:
        if getattr(wx, '__version__', '0.0')[0:3] < '2.8':
            BUILD_AGG = 1
            build_wxagg(ext_modules, packages, NUMERIX,
                not (isinstance(BUILD_WXAGG, str) # don't abort if BUILD_WXAGG
                     and BUILD_WXAGG.lower() == 'auto')) # is "auto"
        rc['backend'] = 'WXAgg'

if BUILD_AGG:
    build_agg(ext_modules, packages, NUMERIX)
    if rc['backend'] == 'PS': rc['backend'] = 'Agg'



if BUILD_FT2FONT:
    build_ft2font(ext_modules, packages)

if BUILD_WINDOWING and sys.platform=='win32':
   build_windowing(ext_modules, packages)

if BUILD_IMAGE:
    build_image(ext_modules, packages, NUMERIX)

if 1:  # I don't think we need to make these optional
    build_contour(ext_modules, packages, NUMERIX)
    build_nxutils(ext_modules, packages, NUMERIX)

for mod in ext_modules:
    if VERBOSE:
        mod.extra_compile_args.append('-DVERBOSE')



# packagers: set rc['numerix'] and rc['backend'] here to override the auto
# defaults, eg
#rc['numerix'] = numpy
#rc['backend'] = GTKAgg
if sys.platform=='win32':
    rc = dict(backend='TkAgg', numerix='numpy')
template = file('matplotlibrc.template').read()
file('lib/matplotlib/mpl-data/matplotlibrc', 'w').write(template%rc)

try: additional_params # has setupegg.py provided
except NameError: additional_params = {}

distrib = setup(name="matplotlib",
      version= __version__,
      description = "Matlab(TM) style python plotting package",
      author = "John D. Hunter",
      author_email="jdhunter@ace.bsd.uchicago.edu",
      url = "http://matplotlib.sourceforge.net",
      long_description = """
      matplotlib strives to produce publication quality 2D graphics
      using matlab plotting for inspiration.  Although the main lib is
      object oriented, there is a functional interface "pylab"
      for people coming from Matlab.
      """,
      packages = packages,
      platforms='any',
      py_modules = ['pylab'],
      ext_modules = ext_modules,
      package_dir = {'': 'lib'},
      package_data = package_data,
      **additional_params
      )
