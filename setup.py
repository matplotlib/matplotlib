"""
matplotlib has added some extension module code which can optionally
be built by setting the appropriate flag below.
"""

# Build the fonttools and TTFQuery packages, required by the Paint,
# Agg and GD backends.
BUILD_FONTTOOLS = 1

# AGG is a new backend that wraps the antigrain geometry toolkit; set
# BUILD_AGG to 1 to build it.  Agg makes heavy use of templates, so it
# probably requires a fairly recent compiler to build it.  It makes
# very nice antialiased output and also supports alpha blending
BUILD_AGG          = 1

# The builds below are experimental.  They use an image backend (eg GD
# or Agg) to render to the GTK canvas.  The idea is that we could use
# a single high quality image renderer to render to all the GUI
# windows

# build GTK GUI with Agg renderer ; requires pygtk src distros installed
BUILD_GTKAGG       = 1

# build GTK GUI with GD renderer ; requires pygtk and GD src distros installed
BUILD_GTKGD        = 0

# build the freetype2 interface - highly experimental and broken!
# Unless your name is John Hunter, you probably don't want this
BUILD_FT2FONT      = 1  




## You shouldn't need to customize below this point

from distutils.core import setup
import sys,os
import glob
from setupext import build_gtkgd, build_agg, build_fonttools, build_gtkagg, \
     build_ft2font
import distutils.sysconfig

data = []
data.extend(glob.glob('fonts/afm/*.afm'))
data.extend(glob.glob('fonts/ttf/*.ttf'))
data.extend(glob.glob('images/*.xpm'))

data_files=[('share/matplotlib', data),]

ext_modules = []
packages = [
    'matplotlib',
    'matplotlib/backends',
    ]


if (BUILD_FONTTOOLS or BUILD_AGG or BUILD_GTKAGG or
    BUILD_GTKGD or BUILD_FT2FONT):
    build_fonttools(ext_modules, packages)
    # we need to manually install FontTools.pth since we can't use
    # extra_path which puts all packages -- matplotlib, ttfquery and
    # FontTools -- in the FontTools subdir
    sitep = distutils.sysconfig.get_python_lib()
    data_files.append( (sitep, ['FontTools.pth']) )

if BUILD_GTKGD:
    build_fonttools(ext_modules, packages)
    build_gtkgd(ext_modules, packages)

if BUILD_AGG:
    build_fonttools(ext_modules, packages)
    build_agg(ext_modules, packages)

if BUILD_GTKAGG:
    build_fonttools(ext_modules, packages)
    build_agg(ext_modules, packages)
    build_gtkagg(ext_modules, packages)

if BUILD_FT2FONT:
    build_fonttools(ext_modules, packages)
    build_ft2font(ext_modules, packages)


        

setup(name="matplotlib",
      version= '0.51a',
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


        
