from distutils.core import setup
import sys,os

import glob
from setupext import build_gtkgd, build_agg


data = []
data.extend(glob.glob('fonts/afm/*.afm'))
data.extend(glob.glob('fonts/ttf/*.ttf'))
data.extend(glob.glob('images/*.xpm'))

ext_modules = []

if 1: # how do I add '--with-gtkgd' flag checking?
    build_gtkgd(ext_modules)

if 1: 
    build_agg(ext_modules)

setup(name="matplotlib",
      version= '0.50j',
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
      packages=['matplotlib', 'matplotlib/backends'],
      platforms='any',
      ext_modules = ext_modules, 
      data_files=[('share/matplotlib', data)],
      )
