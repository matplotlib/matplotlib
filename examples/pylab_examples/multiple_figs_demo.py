#!/usr/bin/env python
# Working with multiple figure windows and subplots

import matplotlib
matplotlib.use('gtk3agg')
matplotlib.rcParams['backend.gtk3.tabbed'] = True
from pylab import *

from matplotlib.backend_bases import ToolBase
class SampleNonGuiTool(ToolBase):
    def set_figures(self, *figures):
        #stupid routine that says how many axes and lines are in each
        #figure
        for figure in figures:
            title = figure.canvas.get_window_title()
            print(title)
            lines = [line for ax in figure.axes for line in ax.lines]
            print('Axes: %d Lines: %d' % (len(figure.axes), len(lines)))


t = arange(0.0, 2.0, 0.01)
s1 = sin(2 * pi * t)
s2 = sin(4 * pi * t)

figure(1)
subplot(211)
plot(t, s1)
subplot(212)
plot(t, 2 * s1)

figure(2)
plot(t, s2)

# now switch back to figure 1 and make some changes
figure(1)
subplot(211)
plot(t, s2, 'gs')
setp(gca(), 'xticklabels', [])

figure(1)
savefig('fig1')
figure(2)
savefig('fig2')

figure(2).canvas.toolbar.add_tool(SampleNonGuiTool, text='Stats')



show()
