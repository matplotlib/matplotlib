import matplotlib
matplotlib.use('GTK3AGG')
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1, 2, 3])

fig.canvas.manager.navigation.list_tools()
plt.show()