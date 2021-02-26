"""
===========================
Stackplots and streamgraphs
===========================
"""

##############################################################################
# Stackplots
# ----------
#
# Stackplots draw multiple datasets as vertically stacked areas. This is
# useful when the individual data values and additionally their cumulative
# value are of interest.


import numpy as np
import matplotlib.pyplot as plt

# data from United Nations World Population Prospects (Revision 2019)
# https://population.un.org/wpp/, license: CC BY 3.0 IGO
year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018]
population_by_continent = {
    'africa': [228, 284, 365, 477, 631, 814, 1044, 1275],
    'americas': [340, 425, 519, 619, 727, 840, 943, 1006],
    'asia': [1394, 1686, 2120, 2625, 3202, 3714, 4169, 4560],
    'europe': [220, 253, 276, 295, 310, 303, 294, 293],
    'oceania': [12, 15, 19, 22, 26, 31, 36, 39],
}

fig, ax = plt.subplots()
ax.stackplot(year, population_by_continent.values(),
             labels=population_by_continent.keys(), alpha=0.8)
ax.legend(loc='upper left')
ax.set_title('World population')
ax.set_xlabel('Year')
ax.set_ylabel('Number of people (millions)')

plt.show()

##############################################################################
# Streamgraphs
# ------------
#
# Using the *baseline* parameter, you can turn an ordinary stacked area plot
# with baseline 0 into a stream graph.


# Fixing random state for reproducibility
np.random.seed(19680801)


def gaussian_mixture(x, n=5):
    """Return a random mixture of *n* Gaussians, evaluated at positions *x*."""
    def add_random_gaussian(a):
        amplitude = 1 / (.1 + np.random.random())
        dx = x[-1] - x[0]
        x0 = (2 * np.random.random() - .5) * dx
        z = 10 / (.1 + np.random.random()) / dx
        a += amplitude * np.exp(-(z * (x - x0))**2)
    a = np.zeros_like(x)
    for j in range(n):
        add_random_gaussian(a)
    return a


x = np.linspace(0, 100, 101)
ys = [gaussian_mixture(x) for _ in range(3)]

fig, ax = plt.subplots()
ax.stackplot(x, ys, baseline='wiggle')
plt.show()

##############################################################################
# stackplot top_to_bottom option
# ------------------------------
#
# When cycling through the color palette more than once, it can become
# difficult to match legend entries to the correct plot area.
# In this example a color palette with 8 different colors is chosen for
# a stackplot with 16 components.
# "entry 1" and "entry 9" will use the same color, yet a viewer of the
# plot may wonder which of the two is the large band, and which the smaller.
# With the ``top_to_bottom`` option, entries in the legend are stacked in the
# same way as in the plotting area - i.e. "entry 9" appears higher up in the
# legend than and higher up in the stack than "entry 1".

x = np.linspace(0, 10, 11)
many_ys = {"entry 1": np.random.normal(3., 1., 11),
           "entry 2": np.random.normal(3., .1, 11),
           "entry 3": np.random.normal(1., .1, 11),
           "entry 4": np.random.normal(1., .1, 11),
           "entry 5": np.random.normal(1., .1, 11),
           "entry 6": np.random.normal(3., .3, 11),
           "entry 7": np.random.normal(10., 0., 11),
           "entry 8": np.random.normal(8., 0., 11),

           "entry 9": np.random.normal(30., 1., 11),
           "entry 10": np.random.normal(30., 5., 11),
           "entry 11": np.random.normal(1., .1, 11),
           "entry 12": np.random.normal(1., .1, 11),
           "entry 13": np.random.normal(5., .1, 11),
           "entry 14": np.random.normal(5., 1., 11),
           "entry 15": np.random.normal(5., .5, 11),
           "entry 16": np.random.normal(5., .2, 11)}

fig, ax = plt.subplots()
ax.set_prop_cycle(color=plt.cm.Dark2.colors)
ax.stackplot(x, many_ys.values(), labels=many_ys.keys(), top_to_bottom=True)
ax.legend(loc='upper left')
plt.show()
