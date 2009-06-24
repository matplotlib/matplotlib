import numpy as np
import matplotlib.pyplot as plt

# the random data
x = np.random.randn(1000)
y = np.random.randn(1000)


fig = plt.figure(1, figsize=(5.5,5.5))

from mpl_toolkits.axes_grid import make_axes_locatable

axScatter = plt.subplot(111)
divider = make_axes_locatable(axScatter)

# create a new axes with a height of 1.2 inch above the axScatter
axHistx = divider.new_vertical(1.2, pad=0.1, sharex=axScatter)

# create a new axes with a width of 1.2 inch on the right side of the
# axScatter
axHisty = divider.new_horizontal(1.2, pad=0.1, sharey=axScatter)

fig.add_axes(axHistx)
fig.add_axes(axHisty)


# make some labels invisible
plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
         visible=False)

# the scatter plot:
axScatter.scatter(x, y)
axScatter.set_aspect(1.)

# now determine nice limits by hand:
binwidth = 0.25
xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
lim = ( int(xymax/binwidth) + 1) * binwidth

bins = np.arange(-lim, lim + binwidth, binwidth)
axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=bins, orientation='horizontal')

# the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
# thus there is no need to manually adjust the xlim and ylim of these
# axis.

#axHistx.axis["bottom"].major_ticklabels.set_visible(False)
for tl in axHistx.get_xticklabels():
    tl.set_visible(False)
axHistx.set_yticks([0, 50, 100])

#axHisty.axis["left"].major_ticklabels.set_visible(False)
for tl in axHisty.get_yticklabels():
    tl.set_visible(False)
axHisty.set_xticks([0, 50, 100])

plt.draw()
plt.show()
#plt.savefig("a.pdf")
