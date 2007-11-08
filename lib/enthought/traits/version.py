# Wrapped in a try/except in those situations where someone hasn't installed
# as an egg.  What do we do then?  For now, we just punt since we don't want
# to define the version number in two places.
#try:
#    import pkg_resources
#    version = pkg_resources.require('enthought.traits')[0].version
#except:
#    version = ''
version = '2.6b1-mpl'


