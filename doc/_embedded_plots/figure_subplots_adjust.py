import matplotlib.pyplot as plt

def arrow(p1, p2, **props):
    overlay.annotate(
        "", p1, p2, xycoords='figure fraction',
        arrowprops=dict(arrowstyle="<->", shrinkA=0, shrinkB=0, **props))

fig, axs = plt.subplots(2, 2, figsize=(6.5, 4))
fig.set_facecolor('lightblue')
fig.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0.4, 0.4)

overlay = fig.add_axes([0, 0, 1, 1], zorder=100)
overlay.axis("off")

for ax in axs.flat:
    ax.set(xticks=[], yticks=[])

arrow((0, 0.75), (0.1, 0.75))  # left
arrow((0.435, 0.25), (0.565, 0.25))  # wspace
arrow((0.1, 0.8), (1, 0.8))  # right
fig.text(0.05, 0.7, "left", ha="center")
fig.text(0.5, 0.3, "wspace", ha="center")
fig.text(0.95, 0.83, "right", ha="center")

arrow((0.75, 0), (0.75, 0.1))  # bottom
arrow((0.25, 0.435), (0.25, 0.565))  # hspace
arrow((0.80, 0.1), (0.8, 1))  # top
fig.text(0.65, 0.05, "bottom", va="center")
fig.text(0.28, 0.5, "hspace", va="center")
fig.text(0.75, 0.95, "top", va="center")
