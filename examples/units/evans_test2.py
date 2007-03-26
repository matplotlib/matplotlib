from basic_units import radians, degrees
from pylab import figure, show, nx
from matplotlib.cbook import iterable
import math

def cos( x ):
   if ( iterable(x) ):
      result = []
      for val in x:
         result.append( math.cos( val.convert_to( radians ).get_value() ) )
      return result
   else:
      return math.cos( x.convert_to( radians ).get_value() )

# the following command strips away the units and
# therefore demonstrates nothing.  The valeus are
# the same for both graphs.  In order to really
# use the units, the list of values passed into
# the plot function must be a unitized type.

# x = nx.arange(0, 15, 0.01) * radians

x = []
for i in range(0, 1500):
   x.append( i*0.01*radians )

fig = figure()

ax = fig.add_subplot(211)
ax.plot(x, cos(x), xunits=radians)

ax = fig.add_subplot(212)
ax.plot(x, cos(x), xunits=degrees)

show()

