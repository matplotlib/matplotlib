"""
matplotlib has added some extension module code and now includes the
TTF modules fonttools and ttfquery (for GD, Paint, and Agg).  To build
the ttf modules, set the BUILD_FONTTOOLS flag

AGG is a new backend that wraps the antigrain geometry toolkit; set
BUILD_AGG to 1 to build it.  Agg makes heavy use of templates, so it
probably requires a fairly recent compiler to build it.


"""

from distutils.core import setup
import sys,os
import glob
from setupext import build_gtkgd, build_agg, build_fonttools, build_gtkagg

# set these flags to build the optional extension modules
BUILD_FONTTOOLS    = 0   # includes TTFQuey
BUILD_GTKGD        = 0   # requires pygtk and GD src distros installed
BUILD_AGG          = 0
BUILD_GTKAGG       = 0

data = []
data.extend(glob.glob('fonts/afm/*.afm'))
data.extend(glob.glob('fonts/ttf/*.ttf'))
data.extend(glob.glob('images/*.xpm'))

ext_modules = []
packages = [
    'matplotlib',
    'matplotlib/backends',
    ]


if BUILD_FONTTOOLS:
    build_fonttools(ext_modules, packages)

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


setup(name="matplotlib",
      version= '0.50q',
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
      data_files=[('share/matplotlib', data)],
      )


