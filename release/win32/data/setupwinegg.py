from __future__ import print_function
from distutils import cygwinccompiler

try:
    # Python 2.6
    # Replace the msvcr func to return an empty list
    cygwinccompiler.get_msvcr
    cygwinccompiler.get_msvcr = lambda: []

except AttributeError:
    pass

from setuptools import setup
execfile('setup.py',
         {'additional_params' :
         {'namespace_packages' : ['mpl_toolkits']}})
