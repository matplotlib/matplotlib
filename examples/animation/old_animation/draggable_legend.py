import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1,2,3], label="test")

l = ax.legend()
d1 = l.draggable()

xy = 1, 2
txt = ax.annotate("Test", xy, xytext=(-30, 30),
                  textcoords="offset points",
                  bbox=dict(boxstyle="round",fc=(0.2, 1, 1)),
                  arrowprops=dict(arrowstyle="->"))
d2 = txt.draggable()


from matplotlib._png import read_png
from matplotlib.cbook import get_sample_data

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fn = get_sample_data("ada.png", asfileobj=False)
arr_ada = read_png(fn)

imagebox = OffsetImage(arr_ada, zoom=0.2)

ab = AnnotationBbox(imagebox, xy,
                    xybox=(120., -80.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.5,
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=3")
                    )


ax.add_artist(ab)

d3 = ab.draggable(use_blit=True)
    
    
plt.show()
