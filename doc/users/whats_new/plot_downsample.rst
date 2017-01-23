Downsample line plots
---------------------

A ``downsample`` parameter now exists for the method :func:`plot`.  This
allows line plots to be intelligently downsampled so that rendering and
interaction is faster. The downsampling algorithm will render an image
very similar to a non-downsampled image.
