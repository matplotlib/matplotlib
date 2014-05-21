#!/usr/bin/python
import numpy as np
import matplotlib as mpl
from pylab import *

# This example shows how to use MaxNPiLocator and PiFormatter in order to have 
# automatic pretty looking axes tick labels for PI numbres. This demo contains two subplots
# that have PI axes with different options.

# rendering area
figure(figsize=(8,10), dpi=100)

###########
#example 1#
###########
subplot(211)

# range
x = np.linspace(0, (2 * np.pi), 256,endpoint=True)
# formulas to graph
sine = np.sin(x)
cosine = np.cos(x)
tangent = np.tan(x)
cotangent = 1/np.tan(x)
cosecant = 1/np.sin(x)
secant = 1/np.cos(x)

# line styles and labels
plot(x, sine, color="red", linewidth=2.5, linestyle="-", label="sin")
plot(x, cosine, color="blue", linewidth=2.5, linestyle="-", label="cos")
plot(x, tangent, color="orange", linewidth=2.5, linestyle="-", label="tan")
plot(x, cotangent, color="purple", linewidth=2.5, linestyle="-", label="cot")
plot(x, cosecant, color="green", linewidth=2.5, linestyle="-", label="csc")
plot(x, secant, color="yellow", linewidth=2.5, linestyle="-", label="sec")

# tick spines
ax = gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

# set up a trig axes
# use PiFormatter to obtain labels with pi fractions
ax.xaxis.set_major_formatter(mpl.ticker.PiFormatter())
# Pi locator - no more than 12 ticks, only 1/4 pi step!
ax.xaxis.set_major_locator(mpl.ticker.MaxNPiLocator(nbins=12,trig_steps = [10.0/4.0]))

ax.yaxis.set_major_formatter(mpl.ticker.PiFormatter())
# no more than 6 ticks
ax.yaxis.set_major_locator(mpl.ticker.MaxNPiLocator(nbins=6))

ylim(-4, 4)

# legend
legend(loc='upper left', framealpha = 0.5)

for label in ax.get_xticklabels()+ax.get_yticklabels():
  label.set_fontsize(20)

###########
#example 2#
###########
subplot(212)
x = linspace(-1*math.pi,3*math.pi,1000)
y =sin(x)+sin(2*x)

ax = gca()
#use_tex = False - do not use TeX formatting for output
ax.yaxis.set_major_formatter(mpl.ticker.PiFormatter(use_tex = False))
# mannually set up 1/2pi step
ax.yaxis.set_major_locator(mpl.ticker.MaxNPiLocator(nbins=5,trig_steps = [10.0/2.0]))

ax.xaxis.set_major_formatter(mpl.ticker.PiFormatter(use_tex = False))
ax.xaxis.set_major_locator(mpl.ticker.MaxNPiLocator(nbins=8))

ax.plot(x,y)
ax.grid()
ax.legend(['PiFormatter(use_tex = False)'])
tight_layout()
plt.savefig('trig_demo.png')
#show()
