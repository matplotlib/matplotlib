Changes for 0.81
================

.. code-block:: text

  - pylab and artist "set" functions renamed to setp to avoid clash
    with python2.4 built-in set.  Current version will issue a
    deprecation warning which will be removed in future versions

  - imshow interpolation arguments changes for advanced interpolation
    schemes.  See help imshow, particularly the interpolation,
    filternorm and filterrad kwargs

  - Support for masked arrays has been added to the plot command and
    to the Line2D object.  Only the valid points are plotted.  A
    "valid_only" kwarg was added to the get_xdata() and get_ydata()
    methods of Line2D; by default it is False, so that the original
    data arrays are returned. Setting it to True returns the plottable
    points.

  - contour changes:

    Masked arrays: contour and contourf now accept masked arrays as
      the variable to be contoured.  Masking works correctly for
      contour, but a bug remains to be fixed before it will work for
      contourf.  The "badmask" kwarg has been removed from both
      functions.

     Level argument changes:

       Old version: a list of levels as one of the positional
       arguments specified the lower bound of each filled region; the
       upper bound of the last region was taken as a very large
       number.  Hence, it was not possible to specify that z values
       between 0 and 1, for example, be filled, and that values
       outside that range remain unfilled.

       New version: a list of N levels is taken as specifying the
       boundaries of N-1 z ranges.  Now the user has more control over
       what is colored and what is not.  Repeated calls to contourf
       (with different colormaps or color specifications, for example)
       can be used to color different ranges of z.  Values of z
       outside an expected range are left uncolored.

       Example:
         Old: contourf(z, [0, 1, 2]) would yield 3 regions: 0-1, 1-2, and >2.
         New: it would yield 2 regions: 0-1, 1-2.  If the same 3 regions were
         desired, the equivalent list of levels would be [0, 1, 2,
         1e38].
