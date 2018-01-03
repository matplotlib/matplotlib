Plot Directive `outname` and `plot_preserve_dir`
----------------------------------------------------

The Sphinx plot directive can be used to automagically generate figures for
documentation like so:

    .. plot::

       import matplotlib.pyplot as plt
       import matplotlib.image as mpimg
       import numpy as np
       img = mpimg.imread('_static/stinkbug.png')
       imgplot = plt.imshow(img)

But, if you reorder the figures in the documentation then all the figures may
need to be rebuilt. This takes time. The names given to the figures are also
fairly meaningless, making them more difficult to index by search engines or to
find on a filesystem.

Alternatively, if you are compiling on a limited-resource service like
ReadTheDocs, you may wish to build imagery locally to avoid hitting resource
limits on the server. Using the new changes allows extensive dynamically
generated imagery to be used on services like ReadTheDocs.

The `:outname:` property
~~~~~~~~~~~~~~~~~~~~~~~~

These problems are address through two new features in the plot directive. The
first is the introduction of the `:outname:` property. It is used like so:

    .. plot::
       :outname: stinkbug_plot

       import matplotlib.pyplot as plt
       import matplotlib.image as mpimg
       import numpy as np
       img = mpimg.imread('_static/stinkbug.png')
       imgplot = plt.imshow(img)

Without `:outname:`, the figure generated above would normally be called, e.g.
`docfile3-4-01.png` or something equally mysterious. With `:outname:` the figure
generated will instead be named `stinkbug_plot-01.png` or even
`stinkbug_plot.png`. This makes it easy to understand which output image is
which and, more importantly, uniquely keys output images to code snippets.

The `plot_preserve_dir` configuration value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting the `plot_preserve_dir` configuration value to the name of a directory
will cause all images with `:outname:` set to be copied to this directory upon
generation.

If an image is already in `plot_preserve_dir` when documentation is being
generated, this image is copied to the build directory thereby pre-empting
generation and reducing computation time in low-resource environments.

