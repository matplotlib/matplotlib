"""
=====
Fig X
=====

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines


fig = plt.figure()

l1 = lines.Line2D([0, 1], [0, 1], transform=fig.transFigure, figure=fig)

l2 = lines.Line2D([0, 1], [1, 0], transform=fig.transFigure, figure=fig)

fig.lines.extend([l1, l2])

plt.show()
