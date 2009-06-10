from distutils import cygwinccompiler

try:
	# Python 2.6
	# Replace the msvcr func to return an 'msvcr71'
	cygwinccompiler.get_msvcr
	cygwinccompiler.get_msvcr = lambda: ['msvcr71']

except AttributeError:
	pass

execfile('setup.py')
