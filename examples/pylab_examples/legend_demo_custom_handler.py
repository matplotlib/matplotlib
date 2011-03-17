import matplotlib.pyplot as plt
import numpy as np

ax = plt.subplot(111, aspect=1)
x, y = np.random.randn(2, 20)
#[1.1, 2, 2.8], [1.1, 2, 1.8]
l1, = ax.plot(x,y, "k+", mew=3, ms=12)
l2, = ax.plot(x,y, "w+", mew=1, ms=10)

import matplotlib.patches as mpatches
c = mpatches.Circle((0, 0), 1, fc="g", ec="r", lw=3)
ax.add_patch(c)



from matplotlib.legend_handler import HandlerPatch

def make_legend_ellipse(legend, orig_handle,
                        xdescent, ydescent,
                        width, height, fontsize):
    p = mpatches.Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent),
                         width = width+xdescent, height=(height+ydescent))

    return p

plt.legend([c, (l1, l2)], ["Label 1", "Label 2"],
           handler_map={mpatches.Circle:HandlerPatch(patch_func=make_legend_ellipse),
                        })

plt.show()
