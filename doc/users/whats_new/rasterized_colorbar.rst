Dense colorbars are rasterized
``````````````````````````````
Vector file formats (pdf, ps, svg) are efficient for
many types of plot element, but for some they can yield
excessive file size and even rendering artifacts, depending
on the renderer used for screen display.  This is a problem
for colorbars that show a large number of shades, as is
most commonly the case.  Now, if a colorbar is showing
50 or more colors, it will be rasterized in vector
backends.
