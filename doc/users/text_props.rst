.. _text-properties:

============================
 Text properties and layout
============================

The :class:`matplotlib.text.Text` instances have a variety of
properties which can be configured via keyword arguments to the text
commands (e.g., :func:`~matplotlib.pyplot.title`,
:func:`~matplotlib.pyplot.xlabel` and :func:`~matplotlib.pyplot.text`).

==========================  ======================================================================================================================
Property                    Value Type
==========================  ======================================================================================================================
alpha                       `float`
backgroundcolor             any matplotlib :ref:`color <colors>`
bbox                        `~matplotlib.patches.Rectangle` prop dict plus key ``'pad'`` which is a pad in points
clip_box                    a matplotlib.transform.Bbox instance
clip_on                     [True | False]
clip_path                   a `~matplotlib.path.Path` instance and a `~matplotlib.transforms.Transform` instance, a `~matplotlib.patches.Patch`
color                       any matplotlib :ref:`color <colors>`
family                      [ ``'serif'`` | ``'sans-serif'`` | ``'cursive'`` | ``'fantasy'`` | ``'monospace'`` ]
fontproperties              a `~matplotlib.font_manager.FontProperties` instance
horizontalalignment or ha   [ ``'center'`` | ``'right'`` | ``'left'`` ]
label                       any string
linespacing                 `float`
multialignment              [``'left'`` | ``'right'`` | ``'center'`` ]
name or fontname            string e.g., [``'Sans'`` | ``'Courier'`` | ``'Helvetica'`` ...]
picker                      [None|float|boolean|callable]
position                    (x, y)
rotation                    [ angle in degrees | ``'vertical'`` | ``'horizontal'`` ]
size or fontsize            [ size in points | relative size, e.g., ``'smaller'``, ``'x-large'`` ]
style or fontstyle          [ ``'normal'`` | ``'italic'`` | ``'oblique'`` ]
text                        string or anything printable with '%s' conversion
transform                   a `~matplotlib.transforms.Transform` instance
variant                     [ ``'normal'`` | ``'small-caps'`` ]
verticalalignment or va     [ ``'center'`` | ``'top'`` | ``'bottom'`` | ``'baseline'`` ]
visible                     [True | False]
weight or fontweight        [ ``'normal'`` | ``'bold'`` | ``'heavy'`` | ``'light'`` | ``'ultrabold'`` | ``'ultralight'``]
x                           `float`
y                           `float`
zorder                      any number
==========================  ======================================================================================================================


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
code indicates that the coordinates are given relative to the axes
bounding box, with 0,0 being the lower left of the axes and 1,1 the
upper right.

.. plot:: mpl_examples/pyplots/text_layout.py
   :include-source:


==============
 Default Font
==============

The base default font is controlled by a set of rcParams:

+---------------------+----------------------------------------------------+
| rcParam             | usage                                              |
+=====================+====================================================+
| ``'font.family'``   | List of either names of font or ``{'cursive',      |
|                     | 'fantasy', 'monospace', 'sans', 'sans serif',      |
|                     | 'sans-serif', 'serif'}``.                          |
|                     |                                                    |
+---------------------+----------------------------------------------------+
|  ``'font.style'``   | The default style, ex ``'normal'``,                |
|                     | ``'italic'``.                                      |
|                     |                                                    |
+---------------------+----------------------------------------------------+
| ``'font.variant'``  | Default variant, ex ``'normal'``, ``'small-caps'`` |
|                     | (untested)                                         |
+---------------------+----------------------------------------------------+
| ``'font.stretch'``  | Default stretch, ex ``'normal'``, ``'condensed'``  |
|                     | (incomplete)                                       |
|                     |                                                    |
+---------------------+----------------------------------------------------+
|  ``'font.weight'``  | Default weight.  Either string or integer          |
|                     |                                                    |
|                     |                                                    |
+---------------------+----------------------------------------------------+
|   ``'font.size'``   | Default font size in points.  Relative font sizes  |
|                     | (``'large'``, ``'x-small'``) are computed against  |
|                     | this size.                                         |
+---------------------+----------------------------------------------------+

The mapping between the family aliases (``{'cursive', 'fantasy',
'monospace', 'sans', 'sans serif', 'sans-serif', 'serif'}``) and actual font names
is controlled by the following rcParams:


+------------------------------------------+--------------------------------+
| family alias                             | rcParam with mappings          |
+==========================================+================================+
| ``'serif'``                              | ``'font.serif'``               |
+------------------------------------------+--------------------------------+
| ``'monospace'``                          | ``'font.monospace'``           |
+------------------------------------------+--------------------------------+
| ``'fantasy'``                            | ``'font.fantasy'``             |
+------------------------------------------+--------------------------------+
| ``'cursive'``                            | ``'font.cursive'``             |
+------------------------------------------+--------------------------------+
| ``{'sans', 'sans serif', 'sans-serif'}`` | ``'font.sans-serif'``          |
+------------------------------------------+--------------------------------+


which are lists of font names.

Text with non-latin glyphs
==========================

As of v2.0 the :ref:`default font <default_changes_font>` contains
glyphs for many western alphabets, but still does not cover all of the
glyphs that may be required by mpl users.  For example, DejaVu has no
coverage of Chinese, Korean, or Japanese.


To set the default font to be one that supports the code points you
need, prepend the font name to ``'font.family'`` or the desired alias
lists ::

   matplotlib.rcParams['font.sans-serif'] = ['Source Han Sans TW', 'sans-serif']

or set it in your :file:`.matplotlibrc` file::

   font.sans-serif: Source Han Sans TW, Ariel, sans-serif

To control the font used on per-artist basis use the ``'name'``,
``'fontname'`` or ``'fontproperties'`` kwargs documented :ref:`above
<text-properties>`.


On linux, `fc-list <http://linux.die.net/man/1/fc-list>`__ can be a
useful tool to discover the font name; for example ::

   $ fc-list :lang=zh family
   Source Han Sans TW,思源黑體 TW,思源黑體 TW ExtraLight,Source Han Sans TW ExtraLight
   Source Han Sans TW,思源黑體 TW,思源黑體 TW Regular,Source Han Sans TW Regular
   Droid Sans Fallback
   Source Han Sans TW,思源黑體 TW,思源黑體 TW Bold,Source Han Sans TW Bold
   Source Han Sans TW,思源黑體 TW,思源黑體 TW Medium,Source Han Sans TW Medium
   Source Han Sans TW,思源黑體 TW,思源黑體 TW Normal,Source Han Sans TW Normal
   Fixed
   Source Han Sans TW,思源黑體 TW,思源黑體 TW Heavy,Source Han Sans TW Heavy
   Source Han Sans TW,思源黑體 TW,思源黑體 TW Light,Source Han Sans TW Light

lists all of the fonts that support Chinese.
