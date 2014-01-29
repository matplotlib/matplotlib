import matplotlib
matplotlib.use('GTK3Agg')
#matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([0, 1])

#Lets play with the buttons in the fig toolbar
#
#Back? who needs back? my mom always told me, don't look back,
fig.canvas.manager.toolbar.remove_tool(1)

#Move home somewhere else
fig.canvas.manager.toolbar.move_tool(0, 6)

plt.show()
