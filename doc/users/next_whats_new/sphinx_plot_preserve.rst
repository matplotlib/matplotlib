Caching sphinx directive figure outputs
---------------------------------------

The new ``:outname:`` property for the Sphinx plot directive can
be used to cache generated images. It is used like:

.. code-block:: rst

    .. plot::
       :outname: stinkbug_plot

       import matplotlib.pyplot as plt
       import matplotlib.image as mpimg
       import numpy as np
       img = mpimg.imread('_static/stinkbug.png')
       imgplot = plt.imshow(img)

Without ``:outname:``, the figure generated above would normally be called,
e.g. :file:`docfile3-4-01.png` or something equally mysterious. With
``:outname:`` the figure generated will instead be named
:file:`stinkbug_plot-01.png` or even :file:`stinkbug_plot.png`. This makes it
easy to understand which output image is which and, more importantly, uniquely
keys output images to the code snippets that generated them.

Configuring the cache directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The directory that images are cached to can be configured using the
``plot_cache_dir`` configuration value in the Sphinx configuration file.

If an image is already in ``plot_cache_dir`` when documentation is being
generated, this image is copied to the build directory thereby pre-empting
generation and reducing computation time in low-resource environments.
