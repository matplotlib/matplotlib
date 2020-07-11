Add ``_repr_png_`` for figures with no configured IPython backend
-----------------------------------------------------------------

Previously, IPython would not show figures as images unless using the
``matplotlib.pyplot`` interface or with an IPython magic statement like
``%matplotlib backend``. Now, no magic is required to view PNG figure
representations.
