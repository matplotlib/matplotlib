import random
from matplotlib.collections import RegularPolyCollection
import matplotlib.cm as cm
from matplotlib.pyplot import figure, show
from numpy.random import rand

fig = figure()
ax = fig.add_subplot(111, xlim=(0,1), ylim=(0,1), autoscale_on=False)
ax.set_title("Press 'a' to add a point, 'd' to delete one")
# a single point
offsets = [(0.5,0.5)]
facecolors = [cm.jet(0.5)]

collection = RegularPolyCollection(
    #fig.dpi,
    5, # a pentagon
    rotation=0,
    sizes=(50,),
    facecolors = facecolors,
    edgecolors = 'black',
    linewidths = (1,),
    offsets = offsets,
    transOffset = ax.transData,
    )

ax.add_collection(collection)

def onpress(event):
    """
    press 'a' to add a random point from the collection, 'd' to delete one
    """
    if event.key=='a':
        x,y = rand(2)
        color = cm.jet(rand())
        offsets.append((x,y))
        facecolors.append(color)
        collection.set_offsets(offsets)
        collection.set_facecolors(facecolors)
        fig.canvas.draw()
    elif event.key=='d':
        N = len(offsets)
        if N>0:
            ind = random.randint(0,N-1)
            offsets.pop(ind)
            facecolors.pop(ind)
            collection.set_offsets(offsets)
            collection.set_facecolors(facecolors)
            fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', onpress)

show()
