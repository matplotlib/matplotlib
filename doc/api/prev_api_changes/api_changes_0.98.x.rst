
Changes for 0.98.x
==================
* ``psd()``, ``csd()``, and ``cohere()`` will now automatically wrap negative
  frequency components to the beginning of the returned arrays.
  This is much more sensible behavior and makes them consistent
  with ``specgram()``.  The previous behavior was more of an oversight
  than a design decision.

* Added new keyword parameters *nonposx*, *nonposy* to
  :class:`matplotlib.axes.Axes` methods that set log scale
  parameters.  The default is still to mask out non-positive
  values, but the kwargs accept 'clip', which causes non-positive
  values to be replaced with a very small positive value.

* Added new :func:`matplotlib.pyplot.fignum_exists` and
  :func:`matplotlib.pyplot.get_fignums`; they merely expose
  information that had been hidden in ``matplotlib._pylab_helpers``.

* Deprecated numerix package.

* Added new :func:`matplotlib.image.imsave` and exposed it to the
  :mod:`matplotlib.pyplot` interface.

* Remove support for pyExcelerator in exceltools -- use xlwt
  instead

* Changed the defaults of acorr and xcorr to use usevlines=True,
  maxlags=10 and normed=True since these are the best defaults

* Following keyword parameters for :class:`matplotlib.legend.Legend` are now
  deprecated and new set of parameters are introduced. The new parameters
  are given as a fraction of the font-size. Also, *scatteryoffsets*,
  *fancybox* and *columnspacing* are added as keyword parameters.

        ================   ================
        Deprecated         New
        ================   ================
        pad                borderpad
        labelsep           labelspacing
        handlelen          handlelength
        handlestextsep     handletextpad
        axespad            borderaxespad
        ================   ================


* Removed the configobj and experimental traits rc support

* Modified :func:`matplotlib.mlab.psd`, :func:`matplotlib.mlab.csd`,
  :func:`matplotlib.mlab.cohere`, and :func:`matplotlib.mlab.specgram`
  to scale one-sided densities by a factor of 2.  Also, optionally
  scale the densities by the sampling frequency, which gives true values
  of densities that can be integrated by the returned frequency values.
  This also gives better MATLAB compatibility.  The corresponding
  :class:`matplotlib.axes.Axes` methods and :mod:`matplotlib.pyplot`
  functions were updated as well.

* Font lookup now uses a nearest-neighbor approach rather than an
  exact match.  Some fonts may be different in plots, but should be
  closer to what was requested.

* :meth:`matplotlib.axes.Axes.set_xlim`,
  :meth:`matplotlib.axes.Axes.set_ylim` now return a copy of the
  ``viewlim`` array to avoid modify-in-place surprises.

* ``matplotlib.afm.AFM.get_fullname`` and
  ``matplotlib.afm.AFM.get_familyname`` no longer raise an
  exception if the AFM file does not specify these optional
  attributes, but returns a guess based on the required FontName
  attribute.

* Changed precision kwarg in :func:`matplotlib.pyplot.spy`; default is
  0, and the string value 'present' is used for sparse arrays only to
  show filled locations.

* :class:`matplotlib.collections.EllipseCollection` added.

* Added ``angles`` kwarg to :func:`matplotlib.pyplot.quiver` for more
  flexible specification of the arrow angles.

* Deprecated (raise NotImplementedError) all the mlab2 functions from
  :mod:`matplotlib.mlab` out of concern that some of them were not
  clean room implementations.

* Methods :meth:`matplotlib.collections.Collection.get_offsets` and
  :meth:`matplotlib.collections.Collection.set_offsets` added to
  :class:`~matplotlib.collections.Collection` base class.

* ``matplotlib.figure.Figure.figurePatch`` renamed
  ``matplotlib.figure.Figure.patch``;
  ``matplotlib.axes.Axes.axesPatch`` renamed
  ``matplotlib.axes.Axes.patch``;
  ``matplotlib.axes.Axes.axesFrame`` renamed
  ``matplotlib.axes.Axes.frame``.
  ``matplotlib.axes.Axes.get_frame``, which returns
  ``matplotlib.axes.Axes.patch``, is deprecated.

* Changes in the :class:`matplotlib.contour.ContourLabeler` attributes
  (:func:`matplotlib.pyplot.clabel` function) so that they all have a
  form like ``.labelAttribute``.  The three attributes that are most
  likely to be used by end users, ``.cl``, ``.cl_xy`` and
  ``.cl_cvalues`` have been maintained for the moment (in addition to
  their renamed versions), but they are deprecated and will eventually
  be removed.

* Moved several functions in :mod:`matplotlib.mlab` and
  :mod:`matplotlib.cbook` into a separate module
  ``matplotlib.numerical_methods`` because they were unrelated to
  the initial purpose of mlab or cbook and appeared more coherent
  elsewhere.
