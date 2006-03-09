"""
Hatching (pattern filled polygons) is supported currently on PS
backend only.  See the set_patch method in
http://matplotlib.sf.net/matplotlib.patches.html#Patch
for details
"""
import matplotlib
matplotlib.use('PS')
from pylab import figure

fig = figure()
ax = fig.add_subplot(111)
bars = ax.bar(range(1,5), range(1,5), color='gray', ecolor='black')

patterns = ('/', '+', 'x', '\\')
for bar, pattern in zip(bars, patterns):
     bar.set_hatch(pattern)
fig.savefig('hatch4.ps')
