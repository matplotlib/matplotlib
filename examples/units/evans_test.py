import matplotlib
matplotlib.rcParams['units'] = True

from matplotlib.cbook import iterable
import matplotlib.units as units
import matplotlib.ticker as ticker
from pylab import figure, show, nx

class Foo:
    def __init__( self, val, unit=1.0 ):
        self.unit = unit
        self._val = val * unit

    def value( self, unit ):
        if unit is None: unit = self.unit
        return self._val / unit

class FooConverter:

    def tickers(x, unit):
        'return (majorloc, minorloc, majorfmt, minorfmt) or None to accept defaults'
        if unit==1.0 or unit==2.0:
            majloc = ticker.IndexLocator( 4, 0 )
            majfmt = ticker.FormatStrFormatter("VAL: %s")
            minloc = ticker.NullLocator()
            minfmt = ticker.NullFormatter()
            return majloc, minloc, majfmt, minfmt
        else: return None
    tickers = staticmethod(tickers)

    def convert_to_value(obj, unit):
        """
        convert obj using unit.  If obj is a sequence, return the
        converted sequence
        """
        print 'convert to value', unit
        if iterable(obj):
            return [o.value(unit) for o in obj]
        else:
            return obj.value(unit)
    convert_to_value = staticmethod(convert_to_value)

units.manager.converters[Foo] = FooConverter()

# create some Foos
x = []
for val in range( 0, 50, 2 ):
    x.append( Foo( val, 1.0 ) )

# and some arbitrary y data
y = [i for i in range( len(x) ) ]


# plot specifying units
fig = figure()
fig.subplots_adjust(bottom=0.2)
ax = fig.add_subplot(111)
ax.plot( x, y, 'o', xunits=2.0 )
for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_ha('right')

#fig.savefig('plot1.png')

# plot without specifying units; will use the None branch for tickers
fig2 = figure()
ax = fig2.add_subplot(111)
ax.plot( x, y )
#p.savefig('plot2.png')
show()
