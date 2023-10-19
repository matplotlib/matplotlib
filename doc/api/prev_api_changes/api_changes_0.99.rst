Changes in 0.99
===============

* pylab no longer provides a load and save function.  These are
  available in matplotlib.mlab, or you can use numpy.loadtxt and
  numpy.savetxt for text files, or np.save and np.load for binary
  NumPy arrays.

* User-generated colormaps can now be added to the set recognized
  by :func:`matplotlib.cm.get_cmap`.  Colormaps can be made the
  default and applied to the current image using
  :func:`matplotlib.pyplot.set_cmap`.

* changed use_mrecords default to False in mlab.csv2rec since this is
  partially broken

* Axes instances no longer have a "frame" attribute. Instead, use the
  new "spines" attribute. Spines is a dictionary where the keys are
  the names of the spines (e.g., 'left','right' and so on) and the
  values are the artists that draw the spines. For normal
  (rectilinear) axes, these artists are Line2D instances. For other
  axes (such as polar axes), these artists may be Patch instances.

* Polar plots no longer accept a resolution kwarg.  Instead, each Path
  must specify its own number of interpolation steps.  This is
  unlikely to be a user-visible change -- if interpolation of data is
  required, that should be done before passing it to Matplotlib.
