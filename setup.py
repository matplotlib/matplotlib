"""
matplotlib has added some extension module code which can optionally
be built by setting the appropriate flag below.

The GTKAgg and TkAgg will try to build if they detect pygtk or Tkinter
respectively; set them to 0 if you do not want to build them
"""

# build the freetype2 interface - this is required for mathtext
# Requires freetype2, and libz
BUILD_FT2FONT = 1

# build the image support module - requires agg and Numeric or
# numarray.  You can build the image module with either Numeric or
# numarray.  Whichever way you build it, the code will work with both
# Numeric or numarray arrays, but you will get the best performance if
# you build with the type of array you use most
#BUILD_IMAGE    = 0           # no image support
BUILD_IMAGE    = 'Numeric'    # use Numeric 
#BUILD_IMAGE    = 'numarray'  # use numarray 

# Build the antigrain geometry toolkit.  Agg makes heavy use of
# templates, so it probably requires a fairly recent compiler to build
# it.  It makes very nice antialiased output and also supports alpha
# blending
BUILD_AGG = 1

# The builds below are alpha.  They use an image backend (eg GD or
# Agg) to render to the GTK canvas.  The idea is to could use a single
# high quality image renderer to render to all the GUI windows

# build GTK GUI with Agg renderer ; requires pygtk src distros installed
# Use False or 0 if you don't want to build
BUILD_GTKAGG       = 'auto'

# build TK GUI with Agg renderer ; requires Tkinter Python extension
# and Tk includes
# Use False or 0 if you don't want to build
BUILD_TKAGG        = 'auto'

# build a small extension to manage the focus on win32 platforms.
BUILD_WINDOWING        = 'auto'

## You shouldn't need to customize below this point


from distutils.core import setup
import sys,os
import glob
from setupext import build_agg, build_gtkagg, build_tkagg, \
     build_ft2font, build_image, build_windowing
import distutils.sysconfig

data = []
data.extend(glob.glob('fonts/afm/*.afm'))
data.extend(glob.glob('fonts/ttf/*.ttf'))
data.extend(glob.glob('images/*.xpm'))
data.extend(glob.glob('images/*.ppm'))
data.append('.matplotlibrc')

data_files=[('share/matplotlib', data),]

ext_modules = []
packages = [
    'matplotlib',
    'matplotlib/backends',
    ]

    
if BUILD_GTKAGG:
    
    try: import gtk
    except ImportError: print 'GTKAgg requires pygtk'
    else:
        BUILD_AGG = 1
        build_gtkagg(ext_modules, packages)

if BUILD_TKAGG:
    try: import Tkinter
    except ImportError: print 'GTKAgg requires pygtk'
    else:
        BUILD_AGG = 1
        build_tkagg(ext_modules, packages)


if BUILD_AGG:
    BUILD_FT2FONT = 1
    build_agg(ext_modules, packages)

if BUILD_FT2FONT:
    build_ft2font(ext_modules, packages)

if BUILD_WINDOWING and sys.platform=='win32':
   build_windowing(ext_modules, packages)

if BUILD_IMAGE:
    build_image(ext_modules, packages, BUILD_IMAGE)

setup(name="matplotlib",
      version= '0.53',
      description = "Matlab style python plotting package",
      author = "John D. Hunter",
      author_email="jdhunter@ace.bsd.uchicago.edu",
      url = "http://matplotlib.sourceforge.net",
      long_description = """
      matplotlib strives to produce publication quality 2D graphics
      using matlab plotting for inspiration.  Although the main lib is
      object oriented, there is a functional matlab style interface
      for people coming from matlab.
      """,
      packages = packages,
      platforms='any',
      ext_modules = ext_modules, 
      data_files = data_files,
      )
