"""
.. redirect-from:: gallery/pie_and_polar_charts/pie_demo2

==========
Pie charts
==========

Demo of plotting a pie chart.

This example illustrates various parameters of `~matplotlib.axes.Axes.pie`.
"""

# %%
# Label slices
# ------------
#
# Plot a pie chart of animals and label the slices. To add
# labels, pass a list of labels to the *wedge_labels* parameter.

import matplotlib.pyplot as plt

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [12, 24, 36, 8]

fig, ax = plt.subplots()
ax.pie(sizes, wedge_labels=labels)

# %%
# Each slice of the pie chart is a `.patches.Wedge` object; therefore in
# addition to the customizations shown here, each wedge can be customized using
# the *wedgeprops* argument, as demonstrated in
# :doc:`/gallery/pie_and_polar_charts/nested_pie`.
#
# Controlling label positions
# ---------------------------
# If you want the labels outside the pie, set a *wedge_label_distance* greater than 1.
# This is the distance from the center of the pie as a fraction of its radius.

fig, ax = plt.subplots()
ax.pie(sizes, wedge_labels=labels, wedge_label_distance=1.1)

# %%
#
# Auto-label slices
# -----------------
#
# Pass a format string to *wedge_labels* to label slices with their values...

fig, ax = plt.subplots()
ax.pie(sizes, wedge_labels='{absval:.1f}')

# %%
#
# ...or with their percentages...

fig, ax = plt.subplots()
ax.pie(sizes, wedge_labels='{frac:.1%}')

# %%
#
# ...or both.

fig, ax = plt.subplots()
ax.pie(sizes, wedge_labels='{absval:d}\n{frac:.1%}')

# %%
#
# For more control over labels, or to add multiple sets, see
# :doc:`/gallery/pie_and_polar_charts/pie_label`.

# %%
#
# Color slices
# ------------
#
# Pass a list of colors to *colors* to set the color of each slice.

fig, ax = plt.subplots()
ax.pie(sizes, colors=['olivedrab', 'rosybrown', 'gray', 'saddlebrown'])

# %%
# Hatch slices
# ------------
#
# Pass a list of hatch patterns to *hatch* to set the pattern of each slice.

fig, ax = plt.subplots()
ax.pie(sizes, hatch=['**O', 'oO', 'O.O', '.||.'])

# %%
#
# Explode, shade, and rotate slices
# ---------------------------------
#
# In addition to the basic pie chart, this demo shows a few optional features:
#
# * offsetting a slice using *explode*
# * add a drop-shadow using *shadow*
# * custom start angle using *startangle*
#
# This example orders the slices, separates (explodes) them, and rotates them.

explode = (0, 0.2, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, wedge_labels='{frac:.1%}', shadow=True, startangle=90)
plt.show()

# %%
# The default *startangle* is 0, which would start the first slice ("Frogs") on
# the positive x-axis. This example sets ``startangle = 90`` such that all the
# slices are rotated counter-clockwise by 90 degrees, and the frog slice starts
# on the positive y-axis.
#
# Controlling the size
# --------------------
#
# By changing the *radius* parameter, and often the text size for better visual
# appearance, the pie chart can be scaled.

fig, ax = plt.subplots()

ax.pie(sizes, wedge_labels='{frac:.1%}', textprops={'size': 'small'}, radius=0.5)
plt.show()

# %%
# Modifying the shadow
# --------------------
#
# The *shadow* parameter may optionally take a dictionary with arguments to
# the `.Shadow` patch. This can be used to modify the default shadow.

fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9},
       startangle=90)
plt.show()

# %%
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.pie` / `matplotlib.pyplot.pie`
#
# .. tags::
#
#    plot-type: pie
#    level: beginner
