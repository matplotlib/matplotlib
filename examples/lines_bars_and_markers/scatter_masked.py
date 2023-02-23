"""
==============
Scatter Masked
==============

A NumPy masked array (see `numpy.ma`) can be passed to `.Axes.scatter` or
`.pyplot.scatter` as the value of the *s* parameter in order to exclude certain
data points from the plot.

This example uses this technique so that data points within a particular radius
are not included in the plot.

"""
import matplotlib.pyplot as plt
import numpy as np

# Fix random state for reproducibility
np.random.seed(19680801)

# Create N random (x, y) points
N = 100
x = 0.9 * np.random.rand(N)
y = 0.9 * np.random.rand(N)

# Create masked size array based on calculation of x and y values
size = np.full(N, 36)
radius = 0.6
masked_size = np.ma.masked_where(radius > np.sqrt(x ** 2 + y ** 2), size)

# Plot data points using masked array
subplot_kw = {
    'aspect': 'equal',
    'xlim': (0, max(x)),
    'ylim': (0, max(y))
}
fig, ax = plt.subplots(subplot_kw=subplot_kw)
ax.scatter(x, y, s=masked_size, marker='^', c="mediumseagreen")

# Show the boundary between the regions
theta = np.arange(-0, np.pi * 2, 0.01)
circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)
ax.plot(circle_x, circle_y, c="black")

plt.show()

###############################################################################
# This technique can also be used to plot a decision boundary, rather than
# masking certain data points so that they don't appear at all. This example
# uses the same data points and boundary as the example above, this time in the
# style of a decision boundary.

# Create a masked array for values within the radius
masked_size_2 = np.ma.masked_where(radius <= np.sqrt(x ** 2 + y ** 2), size)

# Plot solution regions
fig, ax = plt.subplots(subplot_kw=subplot_kw)
ax.patch.set_facecolor('#D8EFE2')  # equivalent of 'mediumseagreen', alpha=0.2
ax.fill(circle_x, circle_y, color='#FFF7CC')  # equivalent of 'gold', alpha=0.2

# Plot data points using two different masked arrays
ax.scatter(x, y, s=masked_size, marker='^', c='mediumseagreen')
ax.scatter(x, y, s=masked_size_2, marker='o', c='gold')

# Plot boundary
ax.plot(circle_x, circle_y, c='black')

plt.show()

###############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.scatter` / `matplotlib.pyplot.scatter`
#    - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
#    - `matplotlib.axes.Axes.fill` / `matplotlib.pyplot.fill`
