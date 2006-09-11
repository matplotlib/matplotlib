import matplotlib.basic_units as bu
from matplotlib.pylab import figure, show, nx

cm = bu.BasicUnit('cm', 'centimeters')
inch = bu.BasicUnit('inch', 'inches')

inch.add_conversion_factor(cm, 2.54)
cm.add_conversion_factor(inch, 1/2.54)

lengths_cm = cm*nx.arange(0, 10, 0.5)
fig = figure()
ax = fig.add_subplot(211)
ax.plot(lengths_cm, 2.0*lengths_cm, xunits=cm, yunits=cm)
ax.set_xlabel('in centimeters')
ax = fig.add_subplot(212)
ax.plot(lengths_cm, 2.0*lengths_cm, xunits=inch, yunits=cm)
ax.set_xlabel('in inches')
#fig.savefig('simple_conversion.png')
show()
