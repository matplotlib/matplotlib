Streamplot now draws streamlines as one piece if no width or no color variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since there is no need to draw streamlines piece by piece if there is no color
change or width change, now streamplot will draw each streamline in one piece.

The behavior for varying width or varying color is not changed, same logic is
used for these kinds of streamplots.
