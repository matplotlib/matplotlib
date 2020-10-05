"""
==========
Linestyles
==========

The Matplotlib `~mpl._enums.LineStyle` specifies the dash pattern used to draw
a given line. The simplest line styles can be accessed by name using the
strings "solid", "dotted", "dashed" and "dashdot" (or their short names, "-",
":", "--", and "-.", respectively).

The exact spacing of the dashes used can be controlled using the
'lines.*_pattern' family of rc parameters. For example,
:rc:`lines.dashdot_pattern` controls the exact spacing of dashed used whenever
the '-.' `~mpl._enums.LineStyle` is specified.

For more information about how to create custom `~mpl._enums.LineStyle`
specifications, see `the LineStyle docs <mpl._enums.LineStyle>`.

*Note*: For historical reasons, one can also specify the dash pattern for a
particular line using `.Line2D.set_dashes` as shown in
:doc:`/gallery/lines_bars_and_markers/line_demo_dash_control` (or by passing a
list of dash sequences using the keyword *dashes* to the cycler in
:doc:`property_cycle </tutorials/intermediate/color_cycle>`). This interface is
strictly less expressive, and we recommend using LineStyle (or the keyword
*linestyle* to the :doc:`property cycler
</tutorials/intermediate/color_cycle>`).
"""
from matplotlib._enums import LineStyle

LineStyle.demo()
