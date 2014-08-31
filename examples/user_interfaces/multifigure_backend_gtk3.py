import matplotlib
matplotlib.use('GTK3Agg')
matplotlib.rcParams['backend.single_window'] = True
import matplotlib.pyplot as plt


import numpy as np

x = np.arange(100)

#Create 4 figures
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(x, x)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x, np.sqrt(x))


fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(x, x ** 2)

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.plot(x, x ** 3)


###################
#Figure management
#Change the figure1 tab label
fig1.canvas.manager.set_window_title('Just a line')

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


plt.show()
