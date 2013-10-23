import matplotlib
matplotlib.use('GTK3Agg')
matplotlib.rcParams['backend.single_window'] = True
import matplotlib.pyplot as plt
from matplotlib.backend_bases import ToolBase
import numpy as np

x = np.arange(100)

#Create 4 figures
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(x, x ** 2, marker='o', label='hey', picker=5)
ax1.legend(loc='lower left')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x, np.sqrt(x))
ax2.set_xlabel('x')
ax2.set_ylabel('y')
#In the axes control tool,
#there is a second axes for this subplot, check it out :)
ax22 = ax2.twinx()
ax22.plot(x, -np.sqrt(x), picker=5, marker='x', label='in second axis')
ax22.set_ylabel('Minus x')

d = 5
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(x[::d], (x ** 3)[::d], 'ro-', label='Line label')

fig4 = plt.figure()
ax41 = fig4.add_subplot(211)
ax41.plot(x, x + 5, label='add 5')

ax42 = fig4.add_subplot(212)
ax42.plot(x, np.log(x + 15), label='add 15')

###################
#Figure management
#Change the figure1 tab label
fig1.canvas.manager.set_window_title('My first Figure')

#Change the figure manager window title
fig1.canvas.manager.set_mainwindow_title('The powerful window manager')

#Detach figure3 from the rest
fig3.canvas.manager.detach()

#Put the figure4 in the same manager as fig3
fig4.canvas.manager.reparent(fig3)

#Control the parent from the figure instantiation with the parent argument
#To place it in the same parent as fig1 we have several options
#parent=fig1
#parent=fig1.canvas.manager
#parent=fig2.canvas.manager.parent
fig5 = plt.figure(parent=fig1)
ax5 = fig5.add_subplot(111)
ax5.plot(x, x**4)
#if we want it in a separate window
#parent=False


###################
#Toolbar management
class SampleNonGuiTool(ToolBase):
    text = 'Stats'

    def set_figures(self, *figures):
        #stupid routine that says how many axes are in each figure
        for figure in figures:
            title = figure.canvas.get_window_title()
            print('Figure "%s": Has %d axes' % (title, len(figure.axes)))

#Add simple SampleNonGuiTool to the toolbar of fig1-fig2
fig1.canvas.manager.toolbar.add_tool(SampleNonGuiTool)

#Lets reorder the buttons in the fig3-fig4 toolbar
#Back? who needs back? my mom always told me, don't look back,
fig3.canvas.manager.toolbar.remove_tool(1)

#Move home somewhere nicer, I always wanted to live close to a rainbow
fig3.canvas.manager.toolbar.move_tool(0, 8)

plt.show()
