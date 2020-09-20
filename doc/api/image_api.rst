********************
``matplotlib.image``
********************

.. currentmodule:: matplotlib.image

.. automodule:: matplotlib.image
   :no-members:
   :no-inherited-members:


Image Artists
-------------

.. inheritance-diagram::    matplotlib.image._ImageBase matplotlib.image.BboxImage matplotlib.image.FigureImage matplotlib.image.PcolorImage matplotlib.image.AxesImage matplotlib.image.NonUniformImage
   :parts: 1
   :private-bases:


.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   _ImageBase
   AxesImage
   NonUniformImage
   PcolorImage
   FigureImage
   BboxImage


Resampling
~~~~~~~~~~

When Matplotlib rasterizes an image when saving / displaying a Figure
we need to, in general, resample the data (either up or down) in
addition to normalizing and color mapping it.  This is because the
exact size of the input, in "data" pixels, will not match the size, in
"screen" pixels, of the output.  The details of how we do the
resampling is controlled by the *interpolation* specified.  This
resampling process can introduce a variety of artifacts and the
default interpolation is chosen to avoid aliasis in common cases (see
:doc:`/gallery/images_contours_and_fields/image_antialiasing`).

Floating point and you
----------------------

The processing steps for rendering a pseudo color image are:

1. rasample to user input to the required dimensions
2. normalize the user data via a `~.colors.Normalize` instance
3. color map from the normalized data to RGBA via a `~.colors.ColorMap` instance

Prior to Matplotlib 2.0 we re



Helper functions
~~~~~~~~~~~~~~~~



.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst


   composite_images
   pil_to_array



Image I/O functions
-------------------

This functions can be used to read, save, and generate thumbnails of
files on disk.  These are here for historical reasons, and while it is
unlikely we will remove them, please prefer to use a dedicated image
I/O library (such as `imageio <https://imageio.github.io/>`__, `pillow
<https://pillow.readthedocs.io/en/stable/>`__, or `tifffile
<https://pypi.org/project/tifffile/>`__) instead.


.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   imread
   imsave
   thumbnail
