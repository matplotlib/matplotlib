import matplotlib.pyplot as plt


fig, axs = plt.subplots(2, 2, figsize=(6.5, 4))
fig.set_facecolor('lightblue')
fig.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.4, 0.4)

overlay = fig.add_axes([0, 0, 1, 1], zorder=100)
overlay.axis("off")
xycoords = 'figure fraction'
arrowprops = dict(arrowstyle="<->", shrinkA=0, shrinkB=0)

for ax in axs.flat:
    ax.set(xticks=[], yticks=[])

overlay.annotate("", (0, 0.75), (0.1, 0.75),
                 xycoords=xycoords, arrowprops=arrowprops)  # left
overlay.annotate("", (0.435, 0.25), (0.565, 0.25),
                 xycoords=xycoords, arrowprops=arrowprops)  # wspace
overlay.annotate("", (0, 0.8), (0.9, 0.8),
                 xycoords=xycoords, arrowprops=arrowprops)  # right
fig.text(0.05, 0.7, "left", ha="center")
fig.text(0.5, 0.3, "wspace", ha="center")
fig.text(0.05, 0.83, "right", ha="center")

overlay.annotate("", (0.75, 0), (0.75, 0.1),
                 xycoords=xycoords, arrowprops=arrowprops)  # bottom
overlay.annotate("", (0.25, 0.435), (0.25, 0.565),
                 xycoords=xycoords, arrowprops=arrowprops)  # hspace
overlay.annotate("", (0.8, 0), (0.8, 0.9),
                 xycoords=xycoords, arrowprops=arrowprops)  # top
fig.text(0.65, 0.05, "bottom", va="center")
fig.text(0.28, 0.5, "hspace", va="center")
fig.text(0.82, 0.05, "top", va="center")
