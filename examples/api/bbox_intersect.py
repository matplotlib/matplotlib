from pylab import *
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.path import Path
from matplotlib.patches import Rectangle

rect = Rectangle((-1, -1), 2, 2, facecolor="#aaaaaa")
gca().add_patch(rect)
bbox = Bbox.from_bounds(-1, -1, 2, 2)

for i in range(12):
    vertices = (np.random.random((4, 2)) - 0.5) * 6.0
    vertices = np.ma.masked_array(vertices, [[False, False], [True, True], [False, False], [False, False]])
    path = Path(vertices)
    if path.intersects_bbox(bbox):
        color = 'r'
    else:
        color = 'b'
    plot(vertices[:,0], vertices[:,1], color=color)

show()
