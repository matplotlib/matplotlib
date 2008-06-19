import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(10))
ax.set_yticks((2,5,7))
labels = ax.set_yticklabels(('really, really, really', 'long', 'labels'))

def on_draw(event):
   for label in labels:
       bbox = label.get_window_extent()
       if bbox.xmin<0:
           fig.subplots_adjust(left=1.1*fig.subplotpars.left)
           fig.canvas.draw()
           break

fig.canvas.mpl_connect('draw_event', on_draw)

plt.show()
