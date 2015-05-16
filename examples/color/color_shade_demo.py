"""
Demo of shade_color utility.

"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fig = plt.figure()
ax = fig.add_subplot(111)

lightened_color = mcolors.shade_color('blue', -50)
darkened_color = mcolors.shade_color('blue', +50)

ax.fill_between([0,1], [0,0], [1,1], facecolor=darkened_color)
ax.fill_between([0,1], [0,0], [.66, .66], facecolor='blue')
ax.fill_between([0,1], [0,0], [.33, .33], facecolor=lightened_color)

plt.xlim([0, 1])
plt.ylim([0, 1])

plt.show()