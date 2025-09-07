"""

.. redirect-from:: /tutorials/text/text_props

.. _text_props:

============================
 Text properties and layout
============================

Controlling properties of text and its layout with Matplotlib.

`matplotlib.text.Text` instances have a variety of properties which can be
configured via keyword arguments to `~.Axes.set_title`, `~.Axes.set_xlabel`,
`~.Axes.text`, etc.

==========================  ======================================================================================================================
Property                    Value Type
==========================  ======================================================================================================================
alpha                       `float`
backgroundcolor             any matplotlib :ref:`color <colors_def>`
bbox                        `~matplotlib.patches.Rectangle` prop dict plus key ``'pad'`` which is a pad in points
clip_box                    a matplotlib.transform.Bbox instance
clip_on                     bool
clip_path                   a `~matplotlib.path.Path` instance and a `~matplotlib.transforms.Transform` instance, a `~matplotlib.patches.Patch`
color                       any matplotlib :ref:`color <colors_def>`
family                      [ ``'serif'`` | ``'sans-serif'`` | ``'cursive'`` | ``'fantasy'`` | ``'monospace'`` ]
fontproperties              `~matplotlib.font_manager.FontProperties`
horizontalalignment or ha   [ ``'center'`` | ``'right'`` | ``'left'`` ]
label                       any string
linespacing                 `float`
multialignment              [``'left'`` | ``'right'`` | ``'center'`` ]
name or fontname            string e.g., [``'Sans'`` | ``'Courier'`` | ``'Helvetica'`` ...]
picker                      [None|float|bool|callable]
position                    (x, y)
rotation                    [ angle in degrees | ``'vertical'`` | ``'horizontal'`` ]
        size or fontsize            [ size in points | relative size, e.g., ``'small'``, ``'x-large'`` ]
style or fontstyle          [ ``'normal'`` | ``'italic'`` | ``'oblique'`` ]
text                        string or anything printable with '%s' conversion
transform                   `~matplotlib.transforms.Transform` subclass
variant                     [ ``'normal'`` | ``'small-caps'`` ]
verticalalignment or va     [ ``'center'`` | ``'top'`` | ``'bottom'`` | ``'baseline'`` ]
visible                     bool
weight or fontweight        [ ``'normal'`` | ``'bold'`` | ``'heavy'`` | ``'light'`` | ``'ultrabold'`` | ``'ultralight'``]
x                           `float`
y                           `float`
zorder                      any number
==========================  ======================================================================================================================


Text alignment
==============

You can lay out text with the alignment arguments
``horizontalalignment``, ``verticalalignment``, and
``multialignment``.  ``horizontalalignment`` controls whether the x
positional argument for the text indicates the left, center or right
side of the text bounding box. ``verticalalignment`` controls whether
the y positional argument for the text indicates the bottom, center or
top side of the text bounding box.  ``multialignment``, for newline
separated strings only, controls whether the different lines are left,
center or right justified.  Here is an example which uses the
:func:`~matplotlib.pyplot.text` command to show the various alignment
possibilities.  The use of ``transform=ax.transAxes`` throughout the
code indicates that the coordinates are given relative to the Axes
bounding box, with (0, 0) being the lower left of the Axes and (1, 1) the
upper right.
"""

import matplotlib.pyplot as plt

import matplotlib.patches as patches

# build a rectangle in axes coords
left, width = 0.25, 0.5
bottom, height = 0.25, 0.5
right = left + width
top = bottom + height

fig = plt.figure()
ax = fig.add_axes((0, 0, 1, 1))

# axes coordinates: (0, 0) is bottom left and (1, 1) is upper right
p = patches.Rectangle(
    (left, bottom), width, height,
    fill=False, transform=ax.transAxes, clip_on=False
    )

ax.add_patch(p)

ax.text(left, bottom, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, bottom, 'left bottom',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right bottom',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right top',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(right, bottom, 'center top',
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, 0.5*(bottom+top), 'right center',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, 0.5*(bottom+top), 'left center',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(0.5*(left+right), 0.5*(bottom+top), 'middle',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=20, color='red',
        transform=ax.transAxes)

ax.text(right, 0.5*(bottom+top), 'centered',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, top, 'rotated\nwith newlines',
        horizontalalignment='center',
        verticalalignment='center',
        rotation=45,
        transform=ax.transAxes)

ax.set_axis_off()
plt.show()

# %%
# Relative font sizes
# ===================
#
# Font sizes can be specified in points, or as a string that indicates the size
# relative to the default font size. The following are valid values for relative font sizes:
#
# ====================    ================================
# Relative font size      Scaling of default font size
# ====================    ================================
# xx-small                0.579
# x-small                 0.694
# small                   0.833
# medium                  1.000
# large                   1.200
# x-large                 1.440
# xx-large                1.728
# ====================    ================================
#
#
# ==============
#  Default Font
# ==============
#
# The base default font is controlled by a set of rcParams. To set the font
# for mathematical expressions, use the rcParams beginning with ``mathtext``
# (see :ref:`mathtext <mathtext-fonts>`).
#
# +---------------------+----------------------------------------------------+
# | rcParam             | usage                                              |
# +=====================+====================================================+
# | ``'font.family'``   | List of font families (installed on user's machine)|
# |                     | and/or ``{'cursive', 'fantasy', 'monospace',       |
# |                     | 'sans', 'sans serif', 'sans-serif', 'serif'}``.    |
# |                     |                                                    |
# +---------------------+----------------------------------------------------+
# |  ``'font.style'``   | The default style, ex ``'normal'``,                |
# |                     | ``'italic'``.                                      |
# |                     |                                                    |
# +---------------------+----------------------------------------------------+
# | ``'font.variant'``  | Default variant, ex ``'normal'``, ``'small-caps'`` |
# |                     | (untested)                                         |
# +---------------------+----------------------------------------------------+
# | ``'font.stretch'``  | Default stretch, ex ``'normal'``, ``'condensed'``  |
# |                     | (incomplete)                                       |
# |                     |                                                    |
# +---------------------+----------------------------------------------------+
# |  ``'font.weight'``  | Default weight.  Either string or integer          |
# |                     |                                                    |
# |                     |                                                    |
# +---------------------+----------------------------------------------------+
# |   ``'font.size'``   | Default font size in points.  Relative font sizes  |
# |                     | (``'large'``, ``'x-small'``) are computed against  |
# |                     | this size.                                         |
# +---------------------+----------------------------------------------------+
#
# Matplotlib can use font families installed on the user's computer, i.e.
# Helvetica, Times, etc. Font families can also be specified with
# generic-family aliases like (``{'cursive', 'fantasy', 'monospace',
# 'sans', 'sans serif', 'sans-serif', 'serif'}``).
#
# .. note::
#    To access the full list of available fonts: ::
#
#       matplotlib.font_manager.get_font_names()
#
# The mapping between the generic family aliases and actual font families
# (mentioned at :ref:`default rcParams <customizing>`)
# is controlled by the following rcParams:
#
#
# +------------------------------------------+--------------------------------+
# | CSS-based generic-family alias           | rcParam with mappings          |
# +==========================================+================================+
# | ``'serif'``                              | ``'font.serif'``               |
# +------------------------------------------+--------------------------------+
# | ``'monospace'``                          | ``'font.monospace'``           |
# +------------------------------------------+--------------------------------+
# | ``'fantasy'``                            | ``'font.fantasy'``             |
# +------------------------------------------+--------------------------------+
# | ``'cursive'``                            | ``'font.cursive'``             |
# +------------------------------------------+--------------------------------+
# | ``{'sans', 'sans serif', 'sans-serif'}`` | ``'font.sans-serif'``          |
# +------------------------------------------+--------------------------------+
#
#
# If any of generic family names appear in ``'font.family'``, we replace that entry
# by all the entries in the corresponding rcParam mapping.
# For example: ::
#
#    matplotlib.rcParams['font.family'] = ['Family1', 'serif', 'Family2']
#    matplotlib.rcParams['font.serif'] = ['SerifFamily1', 'SerifFamily2']
#
#    # This is effectively translated to:
#    matplotlib.rcParams['font.family'] = ['Family1', 'SerifFamily1', 'SerifFamily2', 'Family2']
#
#
# .. _font-nonlatin:
#
# Text with non-latin glyphs
# ==========================
#
# As of v2.0 the :ref:`default font <default_changes_font>`, DejaVu, contains
# glyphs for many western alphabets, but not other scripts, such as Chinese,
# Korean, or Japanese.
#
# To set the default font to be one that supports the code points you
# need, prepend the font name to ``'font.family'`` (recommended), or to the
# desired alias lists. ::
#
#    # first method
#    matplotlib.rcParams['font.family'] = ['Source Han Sans TW', 'sans-serif']
#
#    # second method
#    matplotlib.rcParams['font.family'] = ['sans-serif']
#    matplotlib.rcParams['sans-serif'] = ['Source Han Sans TW', ...]
#
# The generic family alias lists contain fonts that are either shipped
# alongside Matplotlib (so they have 100% chance of being found), or fonts
# which have a very high probability of being present in most systems.
#
# A good practice when setting custom font families is to append
# a generic-family to the font-family list as a last resort.
#
# You can also set it in your :file:`.matplotlibrc` file::
#
#    font.family: Source Han Sans TW, Arial, sans-serif
#
# To control the font used on per-artist basis use the *name*, *fontname* or
# *fontproperties* keyword arguments documented in :ref:`text_props`.
#
#
# On linux, `fc-list <https://linux.die.net/man/1/fc-list>`__ can be a
# useful tool to discover the font name; for example ::
#
#    $ fc-list :lang=zh family
#    Noto to Sans Mono CJK TC,Noto Sans Mono CJK TC Bold
#    Noto Sans CJK TC,Noto Sans CJK TC Medium
#    Noto Sans CJK TC,Noto Sans CJK TC DemiLight
#    Noto Sans CJK KR,Noto Sans CJK KR Black
#    Noto Sans CJK TC,Noto Sans CJK TC Black
#    Noto Sans Mono CJK TC,Noto Sans Mono CJK TC Regular
#    Noto Sans CJK SC,Noto Sans CJK SC Light
#
# lists all of the fonts that support Chinese.
#
