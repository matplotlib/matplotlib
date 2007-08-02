#!/usr/bin/env python
"""
Use matplotlib's internal LaTex parser and layout engine.  For true
latex rendering, see the text.usetex option
"""
import numpy as npy
from pylab import figure, show
fig = figure()
fig.subplots_adjust(bottom=0.2)

ax = fig.add_subplot(111, axisbg='y')
ax.plot([1,2,3], 'r')
x = npy.arange(0.0, 3.0, 0.1)

ax.grid(True)
ax.set_xlabel(r'$\Delta_i^j$', fontsize=20)
ax.set_ylabel(r'$\Delta_{i+1}^j$', fontsize=20)
tex = r'$\mathcal{R}\prod_{i=\alpha_{i+1}}^\infty a_i\sin(2 \pi f x_i)$'

ax.text(1, 1.6, tex, fontsize=20, va='bottom', markup="tex")

#title(r'$\Delta_i^j \hspace{0.4} \rm{versus} \hspace{0.4} \Delta_{i+1}^j$', fontsize=20)
fig.savefig('mathtext_demo')

show()
