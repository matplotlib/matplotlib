Added "legend.framealpha" key to rcParams
`````````````````````````````````````````
Added a key and the corresponding logic to control the default transparency of
legend frames. This feature was written into the docstring of axes.legend(),
but not yet implemented.

Added "figure.titlesize" and "figure.titleweight" keys to rcParams
``````````````````````````````````````````````````````````````````
Two new keys were added to rcParams to control the default font size and weight
used by the figure title (as emitted by ``pyplot.suptitle()``).

``image.composite_image`` added to rcParams
```````````````````````````````````````````
Controls whether vector graphics backends (i.e. PDF, PS, and SVG) combine
multiple images on a set of axes into a single composite image.  Saving each
image individually can be useful if you generate vector graphics files in
matplotlib and then edit the files further in Inkscape or other programs.
