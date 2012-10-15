"""
Poor man's setuptools script...
"""

import os
import sys
from setuptools import setup

# Setupegg assumes the install tree and source tree are exactly the same. Since
# this is not the case, symlink the correct dateutil dir depending on which
# version of python is used
if not os.path.isdir('lib/dateutil'):
    if sys.version_info[0] >= 3:
        os.symlink('dateutil_py3', 'lib/dateutil')
    else:
        os.symlink('dateutil_py2', 'lib/dateutil')

execfile('setup.py',
         {'additional_params' :
          {'namespace_packages' : ['mpl_toolkits'],
           #'entry_points': {'nose.plugins':  ['KnownFailure =  matplotlib.testing.noseclasses:KnownFailure', ] }
           }})
