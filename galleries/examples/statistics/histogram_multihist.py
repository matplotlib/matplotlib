"""
=====================================================
The histogram (hist) function with multiple data sets
=====================================================

Plot histogram with multiple sample sets and demonstrate:

* Use of legend with multiple sample sets
* Stacked bars
* Step curve with no fill
* Data sets of different sample sizes

Selecting different bin counts and sizes can significantly affect the
shape of a histogram. The Astropy docs have a great section on how to
select these parameters:
http://docs.astropy.org/en/stable/visualization/histogram.html

.. redirect-from:: /gallery/lines_bars_and_markers/filled_step

"""
# %%
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

n_bins = 10
x = np.random.randn(1000, 3)

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

colors = ['red', 'tan', 'lime']
ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('bars with legend')

ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)
ax1.set_title('stacked bar')

ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
ax2.set_title('stack step (unfilled)')

# Make a multiple-histogram of data-sets with different length.
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
ax3.hist(x_multi, n_bins, histtype='bar')
ax3.set_title('different sample sizes')

fig.tight_layout()
plt.show()

# %%
# -----------------------------------
# Setting properties for each dataset
# -----------------------------------
#
# You can style the histograms individually by passing a list of values to the
# following parameters:
#
# * edgecolor
# * facecolor
# * hatch
# * linewidth
# * linestyle
#
#
# edgecolor
# .........

fig, ax = plt.subplots()

edgecolors = ['green', 'red', 'blue']

ax.hist(x, n_bins, fill=False, histtype="step", stacked=True,
        edgecolor=edgecolors, label=edgecolors)
ax.legend()
ax.set_title('Stacked Steps with Edgecolors')

plt.show()

# %%
# facecolor
# .........

fig, ax = plt.subplots()

facecolors = ['green', 'red', 'blue']

ax.hist(x, n_bins, histtype="barstacked", facecolor=facecolors, label=facecolors)
ax.legend()
ax.set_title("Bars with different Facecolors")

plt.show()

# %%
# hatch
# .....

fig, ax = plt.subplots()

hatches = [".", "o", "x"]

ax.hist(x, n_bins, histtype="barstacked", hatch=hatches, label=hatches)
ax.legend()
ax.set_title("Hatches on Stacked Bars")

plt.show()

# %%
# linewidth
# .........

fig, ax = plt.subplots()

linewidths = [1, 2, 3]
edgecolors = ["green", "red", "blue"]

ax.hist(x, n_bins, fill=False, histtype="bar", linewidth=linewidths,
        edgecolor=edgecolors, label=linewidths)
ax.legend()
ax.set_title("Bars with Linewidths")

plt.show()

# %%
# linestyle
# .........

fig, ax = plt.subplots()

linestyles = ['-', ':', '--']

ax.hist(x, n_bins, fill=False, histtype='bar', linestyle=linestyles,
        edgecolor=edgecolors, label=linestyles)
ax.legend()
ax.set_title('Bars with Linestyles')

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`
