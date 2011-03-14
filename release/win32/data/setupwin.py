from __future__ import print_function
from distutils import cygwinccompiler

try:
    # Python 2.6
    # Replace the msvcr func to return an []
    cygwinccompiler.get_msvcr
    cygwinccompiler.get_msvcr = lambda: []

except AttributeError:
    pass

execfile('setup.py')
