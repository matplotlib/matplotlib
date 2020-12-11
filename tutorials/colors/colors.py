"""
*****************
Specifying Colors
*****************

Matplotlib recognizes the following formats to specify a color:

+--------------------------------------+--------------------------------------+
| Format                               | Example                              |
+======================================+======================================+
| RGB or RGBA (red, green, blue, alpha)| - ``(0.1, 0.2, 0.5)``                |
| tuple of float values in a closed    | - ``(0.1, 0.2, 0.5, 0.3)``           |
| interval [0, 1]                      |                                      |
+--------------------------------------+--------------------------------------+
| Case-insensitive hex RGB or RGBA     | - ``'#0f0f0f'``                      |
| string                               | - ``'#0f0f0f80'``                    |
+--------------------------------------+--------------------------------------+
| Case-insensitive shorthand equivalent| - ``'#abc'`` as ``'#aabbcc'``        |
| string of RGB orRGBA from duplicated | - ``'#abcd'`` as ``'#aabbccdd'``     |
| characters                           |                                      |
+--------------------------------------+--------------------------------------+
| String representation of float value | - ``'0.8'`` as light gray            |
| in closed interval ``[0, 1]``,       |                                      |
| inclusive for gray level             |                                      |
+--------------------------------------+--------------------------------------+
| Single character shorthand notation  | - ``'b'`` as blue                    |
| for shades of colors                 | - ``'g'`` as green                   |
|                                      | - ``'r'`` as red                     |
| .. note:: The colors green, cyan,    | - ``'c'`` as cyan                    |
|           magenta, and yellow do not | - ``'m'`` as magenta                 |
|           coincide with X11/CSS4     | - ``'y'`` as yellow                  |
|           colors.                    | - ``'k'`` as black                   |
|                                      | - ``'w'`` as white                   |
+--------------------------------------+--------------------------------------+
| Case-insensitive color name from     | - ``'xkcd:sky blue'``                |
| `xkcd color survey`_ with ``'xkcd:'``| - ``'xkcd:eggshell'``                |
| prefix                               |                                      |
+--------------------------------------+--------------------------------------+
| Case-insensitive Tableau Colors from | - 'tab:blue'                         |
| 'T10' categorical palette            | - 'tab:orange'                       |
|                                      | - 'tab:green'                        |
|                                      | - 'tab:red'                          |
|                                      | - 'tab:purple'                       |
| .. note:: This is the default color  | - 'tab:brown'                        |
|           cycle.                     | - 'tab:pink'                         |
|                                      | - 'tab:gray'                         |
|                                      | - 'tab:olive'                        |
|                                      | - 'tab:cyan'                         |
+--------------------------------------+--------------------------------------+
| "CN" color spec where ``'C'``        | - ``'C0'``                           |
| precedes a number acting as an index | - ``'C1'``                           |
| into the default property cycle      +--------------------------------------+
|                                      | :rc:`axes.prop_cycle`                |
| .. note:: Indexing occurs at         |                                      |
|           rendering time and defaults|                                      |
|           to black if cycle does not |                                      |
|           inlcude color.             |                                      |
+--------------------------------------+--------------------------------------+

.. _xkcd color survey: https://xkcd.com/color/rgb/

"Red", "Green", and "Blue" are the intensities of those colors. In combination,
they represent the colorspace.

"Alpha" depends on the ``zorder`` of the Artist.  Matplotlib draws higher
``zorder`` Artists on top of lower Artists. "Alpha" determines
whether the lower Artist is covered by the higher.

If the previous RGB of a pixel is ``RGBold`` and the RGB of the pixel of the
added Artist is ``RGBnew`` with Alpha ``alpha``, then the RGB of the pixel
updates to: ``RGB = RGBOld * (1 - Alpha) + RGBnew * Alpha``.

Alpha of 1 indicates the new Artist completely covering the previous color.
Alpha of 0 indicates that pixel of the Artist is transparent.

.. seealso::

    The following links provide more information on colors in Matplotlib.
        * :doc:`/gallery/color/color_demo` Example
        * `matplotlib.colors` API
        * :doc:`/gallery/color/named_colors` Example

"CN" color selection
--------------------

"CN" colors convert to RGBA when creating Artists.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

th = np.linspace(0, 2*np.pi, 128)


def demo(sty):
    mpl.style.use(sty)
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.set_title('style: {!r}'.format(sty), color='C0')

    ax.plot(th, np.cos(th), 'C1', label='C1')
    ax.plot(th, np.sin(th), 'C2', label='C2')
    ax.legend()


demo('default')
demo('seaborn')

###############################################################################
# The first color ``'C0'`` is the title. Each plot uses the second and third
# colors of each style's :rc:`axes.prop_cycle`. They are ``'C1'`` and ``'C2'``,
# respectively.
#
# .. _xkcd-colors:
#
# xkcd v X11/CSS4
# ---------------
#
# The xkcd colors come from a user survey conducted by the webcomic xkcd.
# Details of the survey are available on the `xkcd blog
# <https://blog.xkcd.com/2010/05/03/color-survey-results/>`__.
#
# There are 95 out of 148 colors with name collisions between the X11/CSS4
# names and the xkcd names. Only three of these colors have the same hex
# values.
#
# For example, ``'blue'`` maps to ``'#0000FF'`` whereas ``'xkcd:blue'`` maps to
# ``'#0343DF'``.  Due to these name collisions, all xkcd colors have the
# ``'xkcd:'`` prefix.
#
# The visual below shows name collisions. Color names where hex values agree
# are bold.

import matplotlib._color_data as mcd
import matplotlib.patches as mpatch

overlap = {name for name in mcd.CSS4_COLORS
           if "xkcd:" + name in mcd.XKCD_COLORS}

fig = plt.figure(figsize=[4.8, 16])
ax = fig.add_axes([0, 0, 1, 1])

for j, n in enumerate(sorted(overlap, reverse=True)):
    weight = None
    cn = mcd.CSS4_COLORS[n]
    xkcd = mcd.XKCD_COLORS["xkcd:" + n].upper()
    if cn == xkcd:
        weight = 'bold'

    r1 = mpatch.Rectangle((0, j), 1, 1, color=cn)
    r2 = mpatch.Rectangle((1, j), 1, 1, color=xkcd)
    txt = ax.text(2, j+.5, '  ' + n, va='center', fontsize=10,
                  weight=weight)
    ax.add_patch(r1)
    ax.add_patch(r2)
    ax.axhline(j, color='k')

ax.text(.5, j + 1.5, 'X11', ha='center', va='center')
ax.text(1.5, j + 1.5, 'xkcd', ha='center', va='center')
ax.set_xlim(0, 3)
ax.set_ylim(0, j + 2)
ax.axis('off')

plt.show()
