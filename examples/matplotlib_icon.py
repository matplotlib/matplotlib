"""
make the matplotlib svg minimization icon
"""
import matplotlib
matplotlib.use('SVG')
from matplotlib.matlab import *

fig = figure(figsize=(.5, .5), dpi=72)
subplot(111)
t = arange(0.0, 2.0, 0.05)
s = sin(2*pi*t)
plot(t,s)
#axis('off')
set(gca(), xticklabels=[], yticklabels=[])
savefig('../images/matplotlib.svg', facecolor=0.75)

