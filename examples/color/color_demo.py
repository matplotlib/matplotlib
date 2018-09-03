"""
==========
Color Demo
==========

Matplotlib gives you 7 ways to specify colors,

1) an RGB or RGBA tuple of float values in ``[0, 1]`` (e.g. ``(0.1, 0.2, 0.5)``
   or  ``(0.1, 0.2, 0.5, 0.3)``).  RGBA is short for Red, Green, Blue, Alpha;
2) a hex RGB or RGBA string (e.g., ``'#0F0F0F'`` or ``'#0F0F0F0F'``);
3) a string representation of a float value in ``[0, 1]`` inclusive for gray
   level (e.g., ``'0.5'``);
4) a single letter string, i.e. one of
   ``{'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}``;
5) a X11/CSS4 ("html") color name, e.g. ``"blue"``;
6) a color name from a palette, prefixed with the palette's name:

   a. a name from the `xkcd color survey <https://xkcd.com/color/rgb/>`__;
      prefixed with ``'xkcd:'`` (e.g., ``'xkcd:sky blue'``);
   b. one of ``{'tab:blue', 'tab:orange', 'tab:green',
      'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
      'tab:gray', 'tab:olive', 'tab:cyan'}`` which are the Tableau Colors from
      the 'T10' categorical palette (which is the default color cycle);
   c. one of ``{'tabx:blue', 'tabx:orange', 'tabx:red', 'tabx:cyan',
      'tabx:green', 'tabx:yellow', 'tabx:purple', 'tabx:pink', 'tabx:brown',
      'tabx:gray' }``  which are the colors from the
      `new Tableau10 categorical palette
      <https://www.tableau.com/about/blog/2016/7/colors-upgrade-tableau-10-\
56782>`__;

7) a "Cn" color spec, i.e. `'C'` followed by a single digit, which is an index
   into the default property cycle
   (``matplotlib.rcParams['axes.prop_cycle']``); the indexing occurs at artist
   creation time and defaults to black if the cycle does not include color.

For more information on colors in matplotlib see

* the :doc:`/tutorials/colors/colors` tutorial;
* the `matplotlib.colors` API;
* the :doc:`/gallery/color/named_colors` example.
"""

import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0.0, 2.0, 201)
s = np.sin(2 * np.pi * t)

# 1) RGB tuple:
fig, ax = plt.subplots(facecolor=(.18, .31, .31))
# 2) hex string:
ax.set_facecolor('#eafff5')
# 3) gray level string:
ax.set_title('Voltage vs. time chart', color='0.7')
# 4) single letter color string
ax.set_xlabel('time (s)', color='c')
# 5) a named color:
ax.set_ylabel('voltage (mV)', color='peachpuff')
# 6a) a named xkcd color:
ax.plot(t, s, 'xkcd:crimson')
# 6b) tab notation:
ax.tick_params(axis="x", labelcolor='tab:orange')
# 6c) tabx notation:
ax.tick_params(axis="y", labelcolor='tabx:yellow')
# 7) Cn notation:
ax.plot(t, .7*s, color='C4', linestyle='--')


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
matplotlib.colors
matplotlib.axes.Axes.plot
matplotlib.axes.Axes.set_facecolor
matplotlib.axes.Axes.set_title
matplotlib.axes.Axes.set_xlabel
matplotlib.axes.Axes.set_ylabel
matplotlib.axes.Axes.tick_params
