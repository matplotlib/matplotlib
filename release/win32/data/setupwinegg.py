from distutils import cygwinccompiler

try:
	# Python 2.6
	# Replace the msvcr func to return an empty list
	cygwinccompiler.get_msvcr
	cygwinccompiler.get_msvcr = lambda: []

except AttributeError:
	# Before Python 2.6
	# Wrap the init func to clear to dll libs
	def new_init(self, **kwargs):
		cygwinccompiler.CygwinCCompiler.__init__(self, **kwargs)
		self.dll_libraries = []
	cygwinccompiler.CygwinCCompiler.__init__ = new_init

from setuptools import setup
execfile('setup.py',
         {'additional_params' :
         {'namespace_packages' : ['mpl_toolkits']}})
