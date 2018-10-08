"""
===========
Cursor Data
===========

This example demonstrates how to monkeypatch `.Artist.get_cursor_data` and
`.Artist.format_cursor_data` to show the elements under the cursor in the
status bar of the plot window.

.. image:: ../../_static/cursor_data.png

"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def lines_under_cursor(self, event):
    """
    Return a list of lines under the cursor for the given MouseMove event.
    """
    if not event.inaxes:
        return []
    return [line for line in event.inaxes.lines if line.contains(event)[0]]


def format_lines(self, data):
    """Convert a list of lines to a comma-separted string."""
    return ', '.join("Line('%s')" % line.get_label() for line in data)


Line2D.mouseover = True
Line2D.get_cursor_data = lines_under_cursor
Line2D.format_cursor_data = format_lines

fig, ax = plt.subplots()
ax.plot([1, 3, 2], label="Label 1", lw=3)
ax.plot([2, 1, 3], label="Label 2", lw=3)

plt.show()


#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

import matplotlib
matplotlib.artist.Artist.get_cursor_data
matplotlib.artist.Artist.format_cursor_data
