"""
Plot with radians from the basic_units mockup example package
This example shows how the unit class can determine the tick locating,
formatting and axis labeling
"""
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


x = nx.arange(0, 15, 0.01) * radians


fig = figure()

ax = fig.add_subplot(211)
ax.plot(x, cos(x), xunits=radians)

ax = fig.add_subplot(212)
ax.plot(x, cos(x), xunits=degrees)

show()

