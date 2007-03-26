"""
A mockup "Foo" units class which supports
conversion and different tick formatting depending on the "unit".
Here the "unit" is just a scalar conversion factor, but this example shows mpl is
entirely agnostic to what kind of units client packages use
"""

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

    def axisinfo(unit):
        'return the Foo AxisInfo'
        if unit==1.0 or unit==2.0:
            return units.AxisInfo(
                majloc = ticker.IndexLocator( 4, 0 ),
                majfmt = ticker.FormatStrFormatter("VAL: %s"),
                label='foo',
                )

        else:
            return None
    axisinfo = staticmethod(axisinfo)

    def convert(obj, unit):
        """
        convert obj using unit.  If obj is a sequence, return the
        converted sequence
        """
        if units.ConversionInterface.is_numlike(obj):
            return obj

        if iterable(obj):
            return [o.value(unit) for o in obj]
        else:
            return obj.value(unit)
    convert = staticmethod(convert)

    def default_units(x):
        'return the default unit for x or None'
        if iterable(x):
            for thisx in x:
                return thisx.unit
        else:
            return x.unit
    default_units = staticmethod(default_units)

units.registry[Foo] = FooConverter()

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

# plot without specifying units; will use the None branch for axisinfo
fig2 = figure()
ax = fig2.add_subplot(111)
ax.plot( x, y ) # uses default units
#p.savefig('plot2.png')

fig3 = figure()
ax = fig3.add_subplot(111)
ax.plot( x, y, xunits=0.5)

show()
