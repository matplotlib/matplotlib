from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea

fig=plt.figure(1, figsize=(3,3))
ax = plt.subplot(111)


ada = AnchoredDrawingArea(40, 20, 0, 0,
                          loc=1, pad=0., frameon=False)
p1 = Circle((10, 10), 10)
ada.drawing_area.add_artist(p1)
p2 = Circle((30, 10), 5, fc="r")
ada.drawing_area.add_artist(p2)

ax.add_artist(ada)

plt.show()
