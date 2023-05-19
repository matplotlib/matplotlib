"""
================
Hover event demo
================

.. note::
    Data tooltips are currently only supported for the TkAgg backend.

You can enable hovering by setting the "hover" property of an artist.
Hovering adds a tooltip to the bottom right corner
of the figure canvas, which is displayed when the mouse pointer hovers over the
artist.

The hover behavior depends on the type of the argument passed to the
``set_hover`` method:

* *None* - hovering is disabled for this artist (default)

* list of string literals - hovering is enabled, and hovering over a point
  displays the corresponding string literal.

* dictionary - hovering is enabled, and hovering over a point
  displays the string literal corresponding to the coordinate tuple.

* function - if hover is callable, it is a user supplied function which
  takes a ``mouseevent`` object (see below), and returns a tuple of transformed
  coordinates

After you have enabled an artist for picking by setting the "hover"
property, you need to connect to the figure canvas hover_event to get
hover callbacks on mouse over events.  For example, ::

  def hover_handler(event):
      mouseevent = event.mouseevent
      artist = event.artist
      # now do something with this...


The hover event (matplotlib.backend_bases.HoverEvent) which is passed to
your callback is always fired with two attributes:

mouseevent
  the mouse event that generate the hover event.

  The mouse event in turn has attributes like x and y (the coordinates in
  display space, such as pixels from left, bottom) and xdata, ydata (the
  coords in data space).  Additionally, you can get information about
  which buttons were pressed, which keys were pressed, which Axes
  the mouse is over, etc.  See matplotlib.backend_bases.MouseEvent
  for details.

artist
  the matplotlib.artist that generated the hover event.

You can set the ``hover`` property of an artist by supplying a ``hover``
argument to ``Axes.plot()``

The examples below illustrate the different ways to use the ``hover`` property.

.. note::
    These examples exercises the interactive capabilities of Matplotlib, and
    this will not appear in the static documentation. Please run this code on
    your machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""
# %%
# Hover with string literal labels
# --------------------------------
import matplotlib.pyplot as plt
from numpy.random import rand

fig, ax = plt.subplots()

ax.plot(rand(3), 'o', hover=['London', 'Paris', 'Barcelona'])
plt.show()

# %%
# Hover with dictionary data
# --------------------------------
fig, ax = plt.subplots()
x = rand(3)
y = rand(3)
ax.plot(x, y, 'o', hover={
    (x[0], y[0]): "London",
    (x[1], y[1]): "Paris",
    (x[2], y[2]): "Barcelona"})
plt.show()

# %%
# Hover with a callable transformation function
# ---------------------------------------------
fig, ax = plt.subplots()


def user_defined_function(event):
    return round(event.xdata * 10, 1), round(event.ydata + 3, 3)

ax.plot(rand(100), 'o', hover=user_defined_function)
plt.show()
