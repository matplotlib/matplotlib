Contourf now accepts edgecolors and linewidths arguments
--------------------------------------------------------

`matplotlib.axes.contourf` creates adjoining polygons for each contour level
that by default do not have lines drawn to delineate their boundaries.
If a PDF viewer applies anti-aliasing to these polygons their edges are made
a bit brighter as they are anti-aliased with the background figure color.
This creates a light border between contour levels that appears to be a light
stroke, depending on how the anti-aliasing happens in teh viewer.

Some viewers can turn off anti-aliasing for images (i.e. Adobe Acrobat) and
this effect dissapears.  But others do not. A work around is to apply a
stroke to the edge of the polygons, making them overlap slightly, and hence
removing the anti-aliasing artifact.  `matplotlib.axes.contourf` now accepts
the new kwarguments ``edgecolors`` with arguments either ``'none'`` or
``'face'``.  If ``'face'`` then the stroke is applied.

We don't do this by default, because unfortunately, PNG rendering has the
opposite error where a faint line appears between contour polygons if
``edgecolors='face'``.

The kwarg ``linewidths`` is also supplied, defaulting to ``linewidths=0.2``
to allow the user to control the linewidth if they want to tune to the ammount
of the anti-aliasing in their favourite viewer.

 
