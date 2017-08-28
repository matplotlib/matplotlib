"""
==============
Bivariate Demo
==============

Plotting bivariate data.

imshow, pcolor, pcolormesh, pcolorfast allows you to plot bivariate data
using a bivaraite colormap.

In this example we use imshow to plot air temperature with surface pressure
alongwith a color square.
"""
import matplotlib.colors as colors
from matplotlib.cbook import get_sample_data
import matplotlib.pyplot as plt
import numpy as np


###############################################################################
# Bivariate plotting demo
# -----------------------

air_temp = np.load(get_sample_data('air_temperature.npy'))
surf_pres = np.load(get_sample_data('surface_pressure.npy'))

fig, ax = plt.subplots()

bivariate = [air_temp, surf_pres]

###############################################################################
# To distinguish bivariate data either BivariateNorm or BivariateColormap must
# be passed in as argument

cax = ax.imshow(bivariate, norm=colors.BivariateNorm(),
                cmap=colors.BivariateColormap())

###############################################################################
# If input data is bivariate then colorbar automatically draws colorsquare
# instead of colorbar

cbar = fig.colorbar(cax, xlabel='air_temp', ylabel='surf_pres')

plt.show()
