"""
.. redirect-from:: /tutorials/intermediate/autoscale

.. _autoscale:

Axis autoscaling
================

Basic concept
-------------

Autoscaling ensures that data is visible within the Axes by automatically adjusting
the axis limits. When you plot data, Matplotlib's autoscaling mechanism updates the
axis limits accordingly.
"""

import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-6, 6, 201)
y = np.sinc(x)

fig, ax = plt.subplots()
ax.plot(x, y)

# %%
#
# .. _autoscale_margins:
#
# Margins
# -------
# To ensure that the data is not at the very edge of the plot, Matplotlib adds a
# margin around the data limits. Note that the *x* data range in the above plot is
# [-6, 6], but the x-axis limits are slightly wider due to the margin.
#
# The default margin is 5%, defined via
#
# - :rc:`axes.xmargin`
# - :rc:`axes.ymargin`
# - :rc:`axes.zmargin`

print(ax.get_xmargin(), ax.get_ymargin())

# %%
# The margin size can be overridden to make them smaller or larger using
# `~matplotlib.axes.Axes.margins`:

fig, ax = plt.subplots()
ax.plot(x, y)
ax.margins(0.2, 0.2)

# %%
# In general, margins can be in the range (-0.5, ∞), where negative margins set
# the axes limits to a subrange of the data range, i.e. they clip data.
# Using a single number for margins affects both axes, a single margin can be
# customized using keyword arguments ``x`` or ``y``, but positional and keyword
# interface cannot be combined.

fig, ax = plt.subplots()
ax.plot(x, y)
ax.margins(y=-0.2)

# %%
# Sticky edges
# ------------
# There are plot elements (`.Artist`\s) that are usually used without margins.
# For example false-color images (e.g. created with `.Axes.imshow`) are not
# considered in the margins calculation.
#

xx, yy = np.meshgrid(x, x)
zz = np.sinc(np.sqrt((xx - 1)**2 + (yy - 1)**2))

fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
ax[0].imshow(zz)
ax[0].set_title("default margins")
ax[1].imshow(zz)
ax[1].margins(0.2)
ax[1].set_title("margins(0.2)")

# %%
# This override of margins is determined by "sticky edges", a
# property of `.Artist` class that can suppress adding margins to axis
# limits. The effect of sticky edges can be disabled on an Axes by changing
# `~matplotlib.axes.Axes.use_sticky_edges`.
# Artists have a property `.Artist.sticky_edges`, and the values of
# sticky edges can be changed by writing to ``Artist.sticky_edges.x`` or
# ``Artist.sticky_edges.y``.
#
# The following example shows how overriding works and when it is needed.

fig, ax = plt.subplots(ncols=3, figsize=(16, 10))
ax[0].imshow(zz)
ax[0].margins(0.2)
ax[0].set_title("default use_sticky_edges\nmargins(0.2)")
ax[1].imshow(zz)
ax[1].margins(0.2)
ax[1].use_sticky_edges = False
ax[1].set_title("use_sticky_edges=False\nmargins(0.2)")
ax[2].imshow(zz)
ax[2].margins(-0.2)
ax[2].set_title("default use_sticky_edges\nmargins(-0.2)")

# %%
# We can see that setting ``use_sticky_edges`` to *False* renders the image
# with requested margins.
#
# While sticky edges don't increase the axis limits through extra margins,
# negative margins are still taken into account. This can be seen in
# the reduced limits of the third image.
#
# Controlling autoscale
# ---------------------
#
# By default, the limits are
# recalculated every time you add a new curve to the plot:

fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
ax[0].plot(x, y)
ax[0].set_title("Single curve")
ax[1].plot(x, y)
ax[1].plot(x * 2.0, y)
ax[1].set_title("Two curves")

# %%
# If you don't want automatic updates of the axis limits, either deactivate
# autoscaling with `~.axes.Axes.autoscale` or set the limits
# manually with `~.axes.Axes.set_xlim` / `~.axes.Axes.set_ylim`.
#
# Let's say that we want to see only a part of the data in
# greater detail. Setting the ``xlim`` persists even if we add more curves to
# the data. Calling `.Axes.autoscale` will again re-enable the autoscaling and
# recalculate the limits to fit all the data.

fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
ax[0].plot(x, y)
ax[0].set_xlim(left=-1, right=1)
ax[0].plot(x + np.pi * 0.5, y)
ax[0].set_title("set_xlim(left=-1, right=1)\n")
ax[1].plot(x, y)
ax[1].set_xlim(left=-1, right=1)
ax[1].plot(x + np.pi * 0.5, y)
ax[1].autoscale()
ax[1].set_title("set_xlim(left=-1, right=1)\nautoscale()")

# %%
# We can check that the first plot has autoscale disabled and that the second
# plot has it enabled again by using `.Axes.get_autoscale_on()`:

print(ax[0].get_autoscale_on())  # False means disabled
print(ax[1].get_autoscale_on())  # True means enabled -> recalculated

# %%
# Arguments of the autoscale function give us precise control over the process
# of autoscaling. A combination of arguments ``enable``, and ``axis`` sets the
# autoscaling feature for the selected axis (or both). The argument ``tight``
# sets the margin of the selected axis to zero. To preserve settings of either
# ``enable`` or ``tight`` you can set the opposite one to *None*, that way
# it should not be modified. However, setting ``enable`` to *None* and tight
# to *True* affects both axes regardless of the ``axis`` argument.

fig, ax = plt.subplots()
ax.plot(x, y)
ax.margins(0.2, 0.2)
ax.autoscale(enable=None, axis="x", tight=True)

print(ax.margins())
