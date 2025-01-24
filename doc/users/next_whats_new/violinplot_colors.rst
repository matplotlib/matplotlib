``violinplot`` now accepts color arguments
-------------------------------------------

The ``~.Axes.violinplot`` constructor now accepts ``facecolor``, ``edgecolor``
and ``alpha`` as input arguments. This means that users can set the color of
violinplots as they make them, rather than setting the color of individual
objects afterwards. It is possible to pass a single color to be used for all
violins, or pass a sequence of colors.

The ``alpha`` argument is used to set the transparency of the violins. By
default, ``alpha`` is set to 0.3. However, if ``alpha`` is set to ``None``,
the violin alpha(s) will be set to the alpha value of the facecolor(s).
