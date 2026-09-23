Axisartist axis labels follow the font size set through Axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Axis labels in `~mpl_toolkits.axisartist` now use the font size set through
`~matplotlib.axes.Axes.set_xlabel` and `~matplotlib.axes.Axes.set_ylabel`.
Previously, these labels kept their initial font size.

A font size set directly on an axisartist label, for example with
``ax.axis["left"].label.set_fontsize(20)``, continues to take precedence.
