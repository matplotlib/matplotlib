Changes to the default style
----------------------------

The most important changes in matplotlib 2.0 are the changes to the
default style.

While it is impossible to select the best default for all cases, these
are designed to work well in the most common cases.

These changes include:

Colors
``````

- The default figure background color has changed from grey to white.
  Use the rcParam ``figure.facecolor`` to control this.

- The default cycle of colors to draw lines, markers and other content
  has been changed.  It is based on the `Vega category10 palette
  <https://github.com/vega/vega/wiki/Scales#scale-range-literals>`__.

- The default color map used for images and pcolor meshes, etc., has
  changed from ``jet`` to ``viridis``.

- For markers, scatter plots, bar charts and pie charts, there is no
  longer a black outline around filled markers by default.

- Grid lines are light grey solid 1pt lines.  They are no longer dashed by
  default.

Plots
`````

- The default size of the elements in a scatter plot is now based on
  the rcParam ``lines.markersize`` so it is consistent with ``plot(X,
  Y, 'o')``.  The old value was 20, and the new value is 36 (6^2).

Hatching
````````

- The width of the lines in a hatch pattern is now configurable by the
  rcParam `hatch.linewidth`, with a default of 1 point.  The old
  behavior was different depending on backend:

    - PDF: 0.1 pt
    - SVG: 1.0 pt
    - PS:  1 px
    - Agg: 1 px

Plot layout
```````````

- The default dpi used for on-screen is now 100, which is the same as
  the old default for saving files.  Due to this, the on-screen
  display is now more what-you-see-is-what-you-get.

- The number of ticks on an axis is now automatically determined based
  on the length of the axis.

- The limits are scaled to exactly the dimensions of the data, plus 5%
  padding.  The old behavior was to scale to the nearest "round"
  numbers.  To use the old behavior, set the ``rcParam``
  ``axes.autolimit_mode`` to ``round_numbers``.  To control the
  margins on particular side individually, pass any of the following
  to any artist or plotting function:

  - ``top_margin=False``
  - ``bottom_margin=False``
  - ``left_margin=False``
  - ``right_margin=False``

- Ticks now point outward by default.  To have ticks pointing inward,
  use the ``rcParams`` ``xtick.direction`` and ``ytick.direction``.

- Ticks and grids are now plotted above solid elements such as
  filled contours, but below lines.  To return to the previous
  behavior of plotting ticks and grids above lines, set
  ``rcParams['axes.axisbelow'] = False``.

- By default, caps on the ends of errorbars are not present.  Use the
  rcParam ``errorbar.capsize`` to control this.

Images
``````

- The default mode for image interpolation, in the rcParam
  ``image.interpolation``, is now ``nearest``.

- The default shading mode for light source shading, in
  ``matplotlib.colors.LightSource.shade``, is now ``overlay``.
  Formerly, it was ``hsv``.

- The default value for the rcParam ``image.resample`` is now
  ``True``.  This will apply interpolation for both upsampling and
  downsampling of an image.

Fonts
`````

- The default font has changed from "Bitstream Vera Sans" to "DejaVu
  Sans".  "DejaVu Sans" is an improvement on "Bistream Vera Sans" that
  adds more international and math characters, but otherwise has the
  same appearance.

- The default math font when using the built-in math rendering engine
  (mathtext) has changed from "Computer Modern" (i.e. LaTeX-like) to
  "DejaVu Sans".  To revert to the old behavior, set the ``rcParam``
  ``mathtext.fontset`` to ``cm``.  This change has no effect if the
  TeX backend is used (i.e. ``text.usetex`` is ``True``).

Dates
`````

- The default date formats are now all based on ISO format, i.e., with
  the slowest-moving value first.  The date formatters are still
  changeable through the ``date.autoformatter.*`` rcParams.  Python's
  ``%x`` and ``%X`` date formats may be of particular interest to
  format dates based on the current locale.

Legends
```````

- By default, the number of points displayed in a legend is now 1.

- The default legend location is ``best``, so the legend will be
  automatically placed in a location to obscure the least amount of
  data possible.

- The legend now has rounded corners by default.
