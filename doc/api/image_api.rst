********************
``matplotlib.image``
********************

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

When Matplotlib rasterizes an image to save / display a Figure,
we need to, in general, resample the data (either up or down) in
addition to normalizing and color mapping it.  This is because the
exact size of the input, in "data" pixels, will not match the size, in
"screen" pixels, of the output.  The details of how we do the
resampling is controlled by the *interpolation* specified.  This
resampling process can introduce a variety of artifacts and the
default interpolation is chosen to avoid aliasing in common cases (see
:doc:`/gallery/images_contours_and_fields/image_antialiasing`).

Colormapping
~~~~~~~~~~~~

The processing steps for rendering a pseudo color image are:

1. resample the user input to the required dimensions
2. normalize the user data via a `~.colors.Normalize` instance
3. colormap from the normalized data to RGBA via a `~.colors.Colormap` instance

Prior to Matplotlib 2.0 we did the normalization and colormapping
first and then resampled to fit the screen.  However this can produce
artifacts in the visualization when the data is changing close to the
full range on the scale of a few screen pixels.  Because most
colormaps are not straight lines in RGB space the interpolated values
"cut the corner" and produce colors in the output image that are not
present in the colormap.  To fix this problem we re-ordered the
processing, however this has lead to a number of subtle issues with
floating point discussed below.


What you need to know about Floating Point Arithmetic for Colormapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Floating point numbers, despite being ubiquitous, are not fully
understood by most practitioners.  For a concise discussion of how
floating point numbers work see `https://floating-point-gui.de/`__,
for a through review see `Goldberg, ACM Computing Surveys (1991)
10.1145/103162.103163 <https://doi.org/10.1145/103162.103163>`__
(paywall), or to for all of the details see `IEEE Standard for
Floating Point Arithmetic (IEEE std 754) 10.1109/IEEESTD.2008.4610935
<https://doi.org/10.1109/IEEESTD.2008.4610935>`__ (paywall).  For the
purposes of this discussion we need to know:

1. There are only a finite number of "floating point numbers" (that is,
   values that can be represented by a IEEE float in the computer) and
   hence they can not exactly represent all Real Numbers.  Between
   two Real Numbers there is an infinite number of Real numbers, hence
   the floating point numbers and computation expressed in a computer
   are an approximation of the Real Numbers.
2. The absolute distance between adjacent floating point numbers
   scales with the magnitude, while the relative distance remains
   the same.  This is a consequence of the implementation of IEEE
   floats.
3. During computation results are rounded to the nearest
   representable value.  Working with numbers that are either almost
   identical or vastly different orders of magnitude exaggerates the
   errors due to this rounding.

This is relevant to images because, as an implementation detail, we
make use of the Agg library to do the resampling from the data space
to screen space and that code clips all input values to the range
:math:`[0, 1]`.  In addition to mapping the colors "in range" we also
map over, under, and bad values (see :ref:`norms_and_colormaps`) which need to be
preserved through the resampling process.  Thus, we:

1. scale the data to :math:`[.1, .9]`
2. pass the data to Agg to resample the pixels
3. scale back to the original data range

and then resume going through the user supplied normalization and colormap.

Naively, this could be expressed as ::

  data_min, data_max = data.min, data.max
  # scale to [.1, .9]
  rescaled = .1 + .8 * (data - data_min) / (data_max - data_min)
  # get the correct number of pixels
  resampled = resample(scaled)
  # scale back to original data range
  scaled = (resampled - .1) * (data_max - data_min)  + data_min

For "most" user data is OK, but can fail in interesting ways.

If the range of the input data is large, but the range the user actually
cares about is small this will effectively map all of the interesting
data to the same value!  To counteract this, we have a check if min /
max of the data are drastically different than the vmin / vmax of the
norm we use a data range expanded from vmin/vmax in the rescaling.
This was addressed in :ghissue:`10072`, :ghpull:`10133`, and
:ghpull:`11047`.

Due floating point math being an approximation of the exact infinite
precision computation not all values "round trip" identically.  This
cause the rescaling to move values in the input data that are very
close to the values of vmin or vmax to the other side.  In the default
case, when the over and under colors are equal to the top and bottom
colors of the colormap respectively this is not visually apparent,
however if the user sets a different color for over/under this is
extremely apparent.  The solution is to also rescale the vmin and vmax
values.  Despite accumulating errors, the float operations will
preserve the relative ordering of values under :math:`\geq` and
:math:`\leq`.  This was reported in :ghissue:`16910` and fixed in
:ghpull:`17636`.

Due to rescaling the vmin and vmax, under certain conditions the sign
of the vmin may change.  In the case of a linear `~.colors.Normalize`
this is not a problem, but in the case of a `~.colors.LogNorm` we
check that both vmin and vmax are greater than 0.  This was reported
in :ghissue:`18415` and fixed in :ghpull:`18458` by special casing
`~.colors.LogNorm` and clipping vmin to be greater than 0.




Helper functions
----------------



.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst


   composite_images
   pil_to_array



Image I/O functions
-------------------

These functions can be used to read, save, and generate thumbnails of
files on disk.  These are here for historical reasons, and while it is
unlikely we will remove them, please use a dedicated image I/O library
(such as `imageio <https://imageio.github.io/>`__, `pillow
<https://pillow.readthedocs.io/en/stable/>`__, or `tifffile
<https://pypi.org/project/tifffile/>`__) instead.


.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   imread
   imsave
   thumbnail
