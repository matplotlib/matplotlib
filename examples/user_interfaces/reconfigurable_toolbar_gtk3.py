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
ax1.plot(x, x)


###################
#Toolbar management

#Lets reorder the buttons in the fig3-fig4 toolbar
#Back? who needs back? my mom always told me, don't look back,
fig1.canvas.manager.toolbar.remove_tool(1)

#Move home somewhere nicer
fig1.canvas.manager.toolbar.move_tool(0, 8)


class SampleNonGuiTool(ToolBase):
    text = 'Stats'

    def set_figures(self, *figures):
        #stupid routine that says how many axes are in each figure
        for figure in figures:
            title = figure.canvas.get_window_title()
            print('Figure "%s": Has %d axes' % (title, len(figure.axes)))

#Add simple SampleNonGuiTool to the toolbar of fig1-fig2
fig1.canvas.manager.toolbar.add_tool(SampleNonGuiTool)

plt.show()
