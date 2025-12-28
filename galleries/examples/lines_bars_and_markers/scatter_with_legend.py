"""
==========================
Scatter plot with a legend
==========================

This example demonstrates how to create scatter plots with legends in Matplotlib.

Legend labels are created by setting the 'label' parameter in the scatter() function.
Each scatter plot call with a unique label will appear as a separate entry in the legend.

The example also shows how to adjust marker transparency using the 'alpha' parameter.
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

# Create figure and axes for the plot
fig, ax = plt.subplots()

# Create scatter plots for each color, using 'label' parameter for legend entries
for color in ['tab:blue', 'tab:orange', 'tab:green']:
    n = 750
    x, y = np.random.rand(2, n)
    scale = 200.0 * np.random.rand(n)
    ax.scatter(x, y, c=color, s=scale, label=f"Points in {color}",
               alpha=0.3, edgecolors='none')

# Add legend using labels from scatter calls
ax.legend()

# Enable grid to improve visual readability
ax.grid(True)

# Add descriptive title and axis labels
ax.set_title('Scatter Plot with Automatically Generated Legend')
ax.set_xlabel('X Values')
ax.set_ylabel('Y Values')

plt.show()


# %%
# .. _automatedlegendcreation:
#
# Automated legend creation
# -------------------------
#
# Another option for creating a legend for a scatter is to use the
# `.PathCollection.legend_elements` method.  It will automatically try to
# determine a useful number of legend entries to be shown and return a tuple of
# handles and labels. Those can be passed to the call to `~.axes.Axes.legend`.


# Generate random data for the scatter plot
N = 45
x, y = np.random.rand(2, N)
c = np.random.randint(1, 5, size=N)
s = np.random.randint(10, 220, size=N)

# Create figure and axes
fig, ax = plt.subplots()

# Create scatter plot with color and size variations
scatter = ax.scatter(x, y, c=c, s=s)

# Create legend for colors (classes) using legend_elements method
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)

# Create legend for sizes using legend_elements with prop="sizes"
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")

# Add title and axis labels
ax.set_title('Automated Legend Creation')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()


# %%
# Further arguments to the `.PathCollection.legend_elements` method
# can be used to steer how many legend entries are to be created and how they
# should be labeled. The following shows how to use some of them.

# Generate random data for volume, amount, ranking, and price
volume = np.random.rayleigh(27, size=40)
amount = np.random.poisson(10, size=40)
ranking = np.random.normal(size=40)
price = np.random.uniform(1, 10, size=40)

# Create figure and axes
fig, ax = plt.subplots()

# Create scatter plot with ranking as color and normalized price as size
# Because the price is much too small when being provided as size for ``s``,
# we normalize it to some useful point sizes, s=0.3*(price*3)**2
scatter = ax.scatter(volume, amount, c=ranking, s=0.3*(price*3)**2,
                     vmin=-3, vmax=3, cmap="Spectral")

# Create legend for ranking (colors) with limited entries
legend1 = ax.legend(*scatter.legend_elements(num=5),
                    loc="upper left", title="Ranking")
ax.add_artist(legend1)

# Create legend for price (sizes) with custom formatting
kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7), fmt="$ {x:.2f}",
          func=lambda s: np.sqrt(s/.3)/3)
legend2 = ax.legend(*scatter.legend_elements(**kw),
                    loc="lower right", title="Price")

# Add title and axis labels
ax.set_title('Legend with Custom Elements')
ax.set_xlabel('Volume')
ax.set_ylabel('Amount')

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.scatter` / `matplotlib.pyplot.scatter`
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`
#    - `matplotlib.collections.PathCollection.legend_elements`
#
# .. tags::
#
#    component: legend
#    plot-type: scatter
#    level: intermediate
