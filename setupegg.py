"""
Poor man's setuptools script...
"""

from setuptools import setup
execfile('setup.py',
         {'additional_params' :
          {'namespace_packages' : ['mpl_toolkits'],
           #'entry_points': {'nose.plugins':  ['KnownFailure =  matplotlib.testing.noseclasses:KnownFailure', ] }
           }})
