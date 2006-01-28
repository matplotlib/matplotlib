"""
Poor man's setuptools script...
"""

from setuptools import setup
execfile('setup.py',
         {'additional_params' :
          {'namespace_packages' : ['matplotlib.toolkits']}})
