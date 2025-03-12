Deprecations
------------


Positional parameters in plotting functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many plotting functions will restrict positional arguments to the first few parameters
in the future. All further configuration parameters will have to be passed as keyword
arguments. This is to enforce better code and and allow for future changes with reduced
risk of breaking existing code.

Changing ``Figure.number``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Changing ``Figure.number`` is deprecated. This value is used by `.pyplot`
to identify figures. It must stay in sync with the pyplot internal state
and is not intended to be modified by the user.

``PdfFile.hatchPatterns``
~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated.

(Sub)Figure.set_figure
~~~~~~~~~~~~~~~~~~~~~~

...is deprecated and in future will always raise an exception.  The parent and
root figures of a (Sub)Figure are set at instantiation and cannot be changed.

``Poly3DCollection.get_vector``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated with no replacement.

Deprecated ``register`` on ``matplotlib.patches._Styles`` and subclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class method is never used internally.  Due to the internal check in the
method it only accepts subclasses of a private baseclass embedded in the host
class which makes it unlikely that it has been used externally.

matplotlib.validate_backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~

...is deprecated. Please use `matplotlib.rcsetup.validate_backend` instead.


matplotlib.sanitize_sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

...is deprecated. Please use `matplotlib.cbook.sanitize_sequence` instead.

ft2font module-level constants replaced by enums
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.ft2font`-level constants have been converted to `enum` classes, and all API using
them now take/return the new types.

The following constants are now part of `.ft2font.Kerning` (without the ``KERNING_``
prefix):

- ``KERNING_DEFAULT``
- ``KERNING_UNFITTED``
- ``KERNING_UNSCALED``

The following constants are now part of `.ft2font.LoadFlags` (without the ``LOAD_``
prefix):

- ``LOAD_DEFAULT``
- ``LOAD_NO_SCALE``
- ``LOAD_NO_HINTING``
- ``LOAD_RENDER``
- ``LOAD_NO_BITMAP``
- ``LOAD_VERTICAL_LAYOUT``
- ``LOAD_FORCE_AUTOHINT``
- ``LOAD_CROP_BITMAP``
- ``LOAD_PEDANTIC``
- ``LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH``
- ``LOAD_NO_RECURSE``
- ``LOAD_IGNORE_TRANSFORM``
- ``LOAD_MONOCHROME``
- ``LOAD_LINEAR_DESIGN``
- ``LOAD_NO_AUTOHINT``
- ``LOAD_TARGET_NORMAL``
- ``LOAD_TARGET_LIGHT``
- ``LOAD_TARGET_MONO``
- ``LOAD_TARGET_LCD``
- ``LOAD_TARGET_LCD_V``

The following constants are now part of `.ft2font.FaceFlags`:

- ``EXTERNAL_STREAM``
- ``FAST_GLYPHS``
- ``FIXED_SIZES``
- ``FIXED_WIDTH``
- ``GLYPH_NAMES``
- ``HORIZONTAL``
- ``KERNING``
- ``MULTIPLE_MASTERS``
- ``SCALABLE``
- ``SFNT``
- ``VERTICAL``

The following constants are now part of `.ft2font.StyleFlags`:

- ``ITALIC``
- ``BOLD``

FontProperties initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.FontProperties` initialization is limited to the two call patterns:

- single positional parameter, interpreted as fontconfig pattern
- only keyword parameters for setting individual properties

All other previously supported call patterns are deprecated.

``AxLine`` ``xy1`` and ``xy2`` setters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These setters now each take a single argument, ``xy1`` or ``xy2`` as a tuple.
The old form, where ``x`` and ``y`` were passed as separate arguments, is
deprecated.

Calling ``pyplot.polar()`` with an existing non-polar Axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This currently plots the data into the non-polar Axes, ignoring
the "polar" intention. This usage scenario is deprecated and
will raise an error in the future.

Passing floating-point values to ``RendererAgg.draw_text_image``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any floating-point values passed to the *x* and *y* parameters were truncated to integers
silently. This behaviour is now deprecated, and only `int` values should be used.

Passing floating-point values to ``FT2Image``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any floating-point values passed to the `.FT2Image` constructor, or the *x0*, *y0*, *x1*,
and *y1* parameters of `.FT2Image.draw_rect_filled` were truncated to integers silently.
This behaviour is now deprecated, and only `int` values should be used.

``boxplot`` and ``bxp`` *vert* parameter, and ``rcParams["boxplot.vertical"]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter *vert: bool* has been deprecated on `~.Axes.boxplot` and
`~.Axes.bxp`. It is replaced by *orientation: {"vertical", "horizontal"}*
for API consistency.

``rcParams["boxplot.vertical"]``, which controlled the orientation of ``boxplot``,
is deprecated without replacement.

This deprecation is currently marked as pending and will be fully deprecated in Matplotlib 3.11.

``violinplot`` and ``violin`` *vert* parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter *vert: bool* has been deprecated on `~.Axes.violinplot` and
`~.Axes.violin`.
It will be replaced by *orientation: {"vertical", "horizontal"}* for API
consistency.

This deprecation is currently marked as pending and will be fully deprecated in Matplotlib 3.11.

``proj3d.proj_transform_clip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated with no replacement.
