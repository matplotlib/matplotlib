'''
Make a colorbar as a separate figure.
'''

import pylab
import matplotlib as mpl

# Make a figure and axes with dimensions as desired.
fig = pylab.figure(figsize=(8,1.5))
ax = fig.add_axes([0.05, 0.4, 0.9, 0.5])

# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=5, vmax=10)

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')
cb.set_label('Some Units')

pylab.show()

