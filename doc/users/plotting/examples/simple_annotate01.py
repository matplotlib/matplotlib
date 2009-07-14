
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

x1, y1 = 0.3, 0.3
x2, y2 = 0.7, 0.7

fig = plt.figure(1)
fig.clf()
from mpl_toolkits.axes_grid.axes_grid import Grid
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

from matplotlib.font_manager import FontProperties

def add_at(ax, t, loc=2):
    fp = dict(size=10)
    _at = AnchoredText(t, loc=loc, prop=fp)
    ax.add_artist(_at)
    return _at


grid = Grid(fig, 111, (4, 4), label_mode="1", share_all=True)

grid[0].set_autoscale_on(False)

ax = grid[0]
ax.plot([x1, x2], [y1, y2], "o")
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="->"))

add_at(ax, "A $->$ B", loc=2)

ax = grid[1]
ax.plot([x1, x2], [y1, y2], "o")
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.3"))

add_at(ax, "connectionstyle=arc3", loc=2)


ax = grid[2]
ax.plot([x1, x2], [y1, y2], "o")
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.3",
                            shrinkB=5,
                            )
            )

add_at(ax, "shrinkB=5", loc=2)


ax = grid[3]
ax.plot([x1, x2], [y1, y2], "o")
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.5)
ax.add_artist(el)
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.2",
                            )
            )


ax = grid[4]
ax.plot([x1, x2], [y1, y2], "o")
el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.5)
ax.add_artist(el)
ax.annotate("",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.2",
                            patchB=el,
                            )
            )


add_at(ax, "patchB", loc=2)



ax = grid[5]
ax.plot([x1], [y1], "o")
ax.annotate("Test",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                      fc="w",
                      ),
            arrowprops=dict(arrowstyle="->",
                            #connectionstyle="arc3,rad=0.2",
                            )
            )


add_at(ax, "annotate", loc=2)


ax = grid[6]
ax.plot([x1], [y1], "o")
ax.annotate("Test",
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            ha="center", va="center",
            bbox=dict(boxstyle="round",
                      fc="w",
                      ),
            arrowprops=dict(arrowstyle="->",
                            #connectionstyle="arc3,rad=0.2",
                            relpos=(0., 0.)
                            )
            )


add_at(ax, "relpos=(0,0)", loc=2)



#ax.set_xlim(0, 1)
#ax.set_ylim(0, 1)
plt.draw()
