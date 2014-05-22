'''
This example shows how to use MaxNPiLocator and PiFormatter in order to have 
automatic pretty looking axes tick labels for PI numbres. This demo contains two subplots
that have PI axes with different options.
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import *

# rendering area
plt.figure(figsize=(8,10), dpi=100)

###########
#example 1#
###########
ax1 = plt.subplot(211)

# range
x = np.linspace(0, (2 * np.pi), 256)

# formulas to graph: sine, cosine, tangent, cotangent, cosecant, secant
funcs = [['sine', np.sin],['cosine', np.cos],['tangent', np.tan],
        ['cotangent', lambda x: 1/np.tan(x)],['cosecant', lambda x: 1/np.sin(x)],['secant', lambda x: 1/np.cos(x)]]
for lab, fun in funcs:
    ax1.plot(x,fun(x),linewidth=2.5,label=lab)

# tick spines
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.xaxis.set_ticks_position('bottom')
ax1.spines['bottom'].set_position(('data',0))
ax1.yaxis.set_ticks_position('left')
ax1.spines['left'].set_position(('data',0))

# set up a trig axes
# use PiFormatter to obtain labels with pi fractions
ax1.xaxis.set_major_formatter(mpl.ticker.PiFormatter())
# Pi locator - no more than 12 ticks, only 1/4 pi step!
ax1.xaxis.set_major_locator(mpl.ticker.MaxNPiLocator(nbins=12,trig_steps = [10.0/4.0]))

ax1.yaxis.set_major_formatter(mpl.ticker.PiFormatter())
# no more than 6 ticks
ax1.yaxis.set_major_locator(mpl.ticker.MaxNPiLocator(nbins=6))

ax1.set_ylim(-4, 4)

# legend
ax1.legend(loc='upper left', framealpha = 0.5)

# make tick labels font larger
for label in ax1.get_xticklabels()+ax1.get_yticklabels():
  label.set_fontsize(20)

###########
#example 2#
###########
ax2 = plt.subplot(212)
x = np.linspace(-1*np.pi,3*np.pi,1000)
y =np.sin(x)+np.sin(2*x)

# use_tex = False - do not use TeX formatting for output
ax2.yaxis.set_major_formatter(mpl.ticker.PiFormatter(use_tex = False))
# mannually set up 1/2pi step
ax2.yaxis.set_major_locator(mpl.ticker.MaxNPiLocator(nbins=5,trig_steps = [10.0/2.0]))

ax2.xaxis.set_major_formatter(mpl.ticker.PiFormatter(use_tex = False))
ax2.xaxis.set_major_locator(mpl.ticker.MaxNPiLocator(nbins=8))

ax2.plot(x,y)
ax2.grid()
ax2.legend(['PiFormatter(use_tex = False)'])
plt.tight_layout()
plt.show()
