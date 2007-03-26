from basic_units import cm, inch
from pylab import figure, show, nx

cms = cm *nx.arange(0, 10, 2)

fig = figure()

ax1 = fig.add_subplot(2,2,1)
ax1.plot(cms, cms, xunits=cm, yunits=cm)
    
ax2 = fig.add_subplot(2,2,2)
ax2.plot(cms, cms, xunits=cm, yunits=inch)

ax3 = fig.add_subplot(2,2,3)
ax3.plot(cms, cms, xunits=inch, yunits=cm)

ax4 = fig.add_subplot(2,2,4)
ax4.plot(cms, cms, xunits=inch, yunits=inch)
#fig.savefig('simple_conversion_plot.png')

show()
