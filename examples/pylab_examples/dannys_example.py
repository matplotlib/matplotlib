import matplotlib
matplotlib.rc('text', usetex = True)
import pylab
import numpy as np

## interface tracking profiles
N = 500
delta = 0.6
X = -1 + 2. * np.arange(N) / (N - 1)
pylab.plot(X, (1 - np.tanh(4. * X / delta)) / 2,     ## phase field tanh profiles
           X, (X + 1) / 2,                           ## level set distance function
           X, (1.4 + np.tanh(4. * X / delta)) / 4,   ## composition profile
           X, X < 0, 'k--',                               ## sharp interface
           linewidth = 5)

## legend
pylab.legend((r'phase field', r'level set', r'composition', r'sharp interface'), shadow = True, loc = (0.01, 0.55))
ltext = pylab.gca().get_legend().get_texts()
pylab.setp(ltext[0], fontsize = 20, color = 'b')
pylab.setp(ltext[1], fontsize = 20, color = 'g')
pylab.setp(ltext[2], fontsize = 20, color = 'r')
pylab.setp(ltext[3], fontsize = 20, color = 'k')

## the arrow
height = 0.1
offset = 0.02
pylab.plot((-delta / 2., delta / 2), (height, height), 'k', linewidth = 2)
pylab.plot((-delta / 2, -delta / 2 + offset * 2), (height, height - offset), 'k', linewidth = 2)
pylab.plot((-delta / 2, -delta / 2 + offset * 2), (height, height + offset), 'k', linewidth = 2)
pylab.plot((delta / 2, delta / 2 - offset * 2), (height, height - offset), 'k', linewidth = 2)
pylab.plot((delta / 2, delta / 2 - offset * 2), (height, height + offset), 'k', linewidth = 2)
pylab.text(-0.06, height - 0.06, r'$\delta$', {'color' : 'k', 'fontsize' : 24})

## X-axis label
pylab.xticks((-1, 0,  1), ('-1', '0',  '1'), color = 'k', size = 20)

## Left Y-axis labels
pylab.ylabel(r'\bf{phase field} $\phi$', {'color'    : 'b',
                                          'fontsize'   : 20 })
pylab.yticks((0, 0.5, 1), ('0', '.5', '1'), color = 'k', size = 20)

## Right Y-axis labels
pylab.text(1.05, 0.5, r"\bf{level set} $\phi$", {'color' : 'g', 'fontsize' : 20},
           horizontalalignment = 'left',
           verticalalignment = 'center',
           rotation = 90,
           clip_on = False)
pylab.text(1.01, -0.02, "-1", {'color' : 'k', 'fontsize' : 20})
pylab.text(1.01, 0.98, "1", {'color' : 'k', 'fontsize' : 20})
pylab.text(1.01, 0.48, "0", {'color' : 'k', 'fontsize' : 20})

## level set equations
pylab.text(0.1, 0.85, r'$|\nabla\phi| = 1,$ \newline $ \frac{\partial \phi}{\partial t} + U|\nabla \phi| = 0$', {'color' : 'g', 'fontsize' : 20})

## phase field equations
pylab.text(0.2, 0.15, r'$\mathcal{F} = \int f\left( \phi, c \right) dV,$ \newline $ \frac{ \partial \phi } { \partial t } = -M_{ \phi } \frac{ \delta \mathcal{F} } { \delta \phi }$',
           {'color' : 'b', 'fontsize' : 20})

pylab.show()
