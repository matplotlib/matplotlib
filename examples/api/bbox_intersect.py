import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.path import Path

rect = plt.Rectangle((-1, -1), 2, 2, facecolor="#aaaaaa")
plt.gca().add_patch(rect)
bbox = Bbox.from_bounds(-1, -1, 2, 2)

for i in range(12):
    vertices = (np.random.random((2, 2)) - 0.5) * 6.0
    path = Path(vertices)
    if path.intersects_bbox(bbox):
        color = 'r'
    else:
        color = 'b'
    plt.plot(vertices[:, 0], vertices[:, 1], color=color)

plt.show()
