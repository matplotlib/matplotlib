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

# build the image support module - requires agg and Numeric or
# numarray.  You can build the image module with either Numeric or
# numarray or both.  By default, matplotlib will build support for
# whatever array packages you have installed.
BUILD_IMAGE = 1

# Build the antigrain geometry toolkit.  Agg makes heavy use of
# templates, so it probably requires a fairly recent compiler to build
# it.  It makes very nice antialiased output and also supports alpha
# blending
BUILD_AGG = 1

# Render Agg to the GTK canvas
#BUILD_GTKAGG       = 0
BUILD_GTKAGG       = 'auto'

BUILD_GTK          = 'auto'

# build TK GUI with Agg renderer ; requires Tkinter Python extension
# and Tk includes
# Use False or 0 if you don't want to build
#BUILD_TKAGG        = 0
BUILD_TKAGG        = 'auto'

# build wxPython GUI with Agg renderer ; requires wxPython package
BUILD_WXAGG        = 'auto'


# build a small extension to manage the focus on win32 platforms.
#BUILD_WINDOWING        = 0
BUILD_WINDOWING        = 'auto'


VERBOSE = False # insert lots of diagnostic prints in extension code




## You shouldn't need to customize below this point

# BEFORE importing disutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

from distutils.core import setup
import sys,os
import glob
from distutils.core import Extension
from setupext import build_agg, build_gtkagg, build_tkagg, build_wxagg,\
     build_ft2font, build_image, build_windowing, build_transforms, \
     build_contour, build_enthought, build_swigagg, build_gdk
import distutils.sysconfig

major, minor1, minor2, s, tmp = sys.version_info

if major==2 and minor1==2:
    print >> sys.stderr, "***\n\nWARNING, see build info for python2.2 in the header of setup.py\n\n***"
    
for line in file('lib/matplotlib/__init__.py').readlines():
    if line[:11] == '__version__':
        exec(line)
        break

data = []

data.extend(glob.glob('fonts/afm/*.afm'))
data.extend(glob.glob('fonts/ttf/*.ttf'))
data.extend(glob.glob('images/*.xpm'))
data.extend(glob.glob('images/*.svg'))
data.extend(glob.glob('images/*.png'))
data.extend(glob.glob('images/*.ppm'))
data.append('matplotlibrc')

data_files=[('share/matplotlib', data),]

# Needed for CocoaAgg
data_files.append(('share/matplotlib/Matplotlib.nib',
		   glob.glob('lib/matplotlib/backends/Matplotlib.nib/*.nib')))

# Figure out which array packages to provide binary support for
# and define the NUMERIX value: Numeric, numarray, or both.
try:
    import Numeric
    HAVE_NUMERIC=1
except ImportError:
    HAVE_NUMERIC=0
try:
    import numarray
    HAVE_NUMARRAY=1
except ImportError:
    HAVE_NUMARRAY=0
    
NUMERIX=["neither", "Numeric","numarray","both"][HAVE_NUMARRAY*2+HAVE_NUMERIC]


if NUMERIX == "neither":
    raise RuntimeError("You must install Numeric, numarray, or both to build matplotlib")

# This print interers with --version, which license depends on
#print "Compiling matplotlib for:", NUMERIX

ext_modules = []

# these are not optional
BUILD_FT2FONT = 1
BUILD_CONTOUR = 1

packages = [
    'matplotlib',
    'matplotlib/backends',
    'matplotlib/numerix',
    'matplotlib/numerix/mlab',
    'matplotlib/numerix/ma',
    'matplotlib/numerix/linear_algebra',
    'matplotlib/numerix/random_array',
    'matplotlib/numerix/fft',                    
    ]




try: import datetime
except ImportError: havedate = False
else: havedate = True

if havedate: # dates require python23 datetime
    # only install pytz and dateutil if the user hasn't got them
    def add_pytz():
        packages.append('pytz')
        # install pytz subdirs
        for dirpath, dirname, filenames in os.walk(os.path.join('lib', 'pytz','zoneinfo')):
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

if BUILD_GTK:
    try:
        import gtk
    except ImportError:
        print 'GTK requires pygtk'
        BUILD_GTK=0
    except RuntimeError:
        print 'pygtk present but import failed'
    

if BUILD_GTK:
    build_gdk(ext_modules, packages, NUMERIX)
    
if BUILD_GTKAGG:
    try:
        import gtk
    except ImportError:
        print 'GTKAgg requires pygtk'
        BUILD_GTKAGG=0
    except RuntimeError:
        print 'pygtk present but import failed'
if BUILD_GTKAGG:
    BUILD_AGG = 1
    build_gtkagg(ext_modules, packages, NUMERIX)

if BUILD_TKAGG:
    try: import Tkinter
    except ImportError: print 'TKAgg requires TkInter'
    else:
        BUILD_AGG = 1
        build_tkagg(ext_modules, packages, NUMERIX)

if BUILD_WXAGG:
    try: import wxPython
    except ImportError: print 'WXAgg\'s accelerator requires wxPython'
    else:
        BUILD_AGG = 1
        build_wxagg(ext_modules, packages, NUMERIX,
            not (isinstance(BUILD_WXAGG, str) # don't about if BUILD_WXAGG
                 and BUILD_WXAGG.lower() == 'auto')) # is "auto"

if BUILD_AGG:
    build_agg(ext_modules, packages, NUMERIX)

if BUILD_FT2FONT:
    build_ft2font(ext_modules, packages)

if BUILD_WINDOWING and sys.platform=='win32':
   build_windowing(ext_modules, packages)

if BUILD_IMAGE:
    build_image(ext_modules, packages, NUMERIX)
    
if 1:  # I don't think we need to make these optional
    build_contour(ext_modules, packages, NUMERIX)
    
for mod in ext_modules:
    if VERBOSE:
        mod.extra_compile_args.append('-DVERBOSE')
    
setup(name="matplotlib",
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
      data_files = data_files,
      package_dir = {'': 'lib'},
      )
