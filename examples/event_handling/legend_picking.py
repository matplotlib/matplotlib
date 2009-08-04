"""
Enable picking on the legend to toggle the legended line on and off
"""
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0.0, 0.2, 0.1)
y1 = 2*np.sin(2*np.pi*t)
y2 = 4*np.sin(2*np.pi*2*t)

fig = plt.figure()
ax = fig.add_subplot(111)

line1, = ax.plot(t, y1, lw=2, color='red', label='1 hz')
line2, = ax.plot(t, y2, lw=2, color='blue', label='2 hz')

leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
leg.get_frame().set_alpha(0.4)


lines = [line1, line2]
lined = dict()
for legline, realine in zip(leg.get_lines(), lines):
    legline.set_picker(5)  # 5 pts tolerance
    lined[legline] = realine

def onpick(event):
    legline = event.artist
    realline = lined[legline]
    vis = realline.get_visible()
    realline.set_visible(not vis)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
