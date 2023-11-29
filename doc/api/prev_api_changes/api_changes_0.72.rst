Changes for 0.72
================

.. code-block:: text

  - Line2D, Text, and Patch copy_properties renamed update_from and
    moved into artist base class

  - LineCollections.color renamed to LineCollections.set_color for
    consistency with set/get introspection mechanism,

  - pylab figure now defaults to num=None, which creates a new figure
    with a guaranteed unique number

  - contour method syntax changed - now it is MATLAB compatible

      unchanged: contour(Z)
      old: contour(Z, x=Y, y=Y)
      new: contour(X, Y, Z)

    see http://matplotlib.sf.net/matplotlib.pylab.html#-contour


   - Increased the default resolution for save command.

   - Renamed the base attribute of the ticker classes to _base to avoid conflict
     with the base method.  Sitt for subs

   - subs=none now does autosubbing in the tick locator.

   - New subplots that overlap old will delete the old axes.  If you
     do not want this behavior, use fig.add_subplot or the axes
     command
