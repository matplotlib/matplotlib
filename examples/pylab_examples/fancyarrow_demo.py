import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

styles = mpatches.ArrowStyle.get_styles()

figheight = (len(styles)+.5)
fig1 = plt.figure(1, (4, figheight))
fontsize = 0.3 * fig1.dpi


ax = fig1.add_axes([0, 0, 1, 1], frameon=False, aspect=1.)

ax.set_xlim(0, 4)
ax.set_ylim(0, figheight)

for i, (stylename, styleclass) in enumerate(sorted(styles.items())):
    y = (float(len(styles)) -0.25 - i) # /figheight
    p = mpatches.Circle((3.2, y), 0.2, fc="w")
    ax.add_patch(p)

    ax.annotate(stylename, (3.2, y),
                (2., y),
                #xycoords="figure fraction", textcoords="figure fraction",
                ha="right", va="center",
                size=fontsize,
                arrowprops=dict(arrowstyle=stylename,
                                patchB=p,
                                shrinkA=5,
                                shrinkB=5,
                                fc="w", ec="k",
                                connectionstyle="arc3,rad=-0.05",
                                ),
                bbox=dict(boxstyle="square", fc="w"))

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)



plt.draw()
plt.show()
