*****************
``matplotlib.cm``
*****************

.. automodule:: matplotlib.cm
   :members:
   :undoc-members:
   :show-inheritance:


.. class:: ScalarMappable(colorizer, **kwargs)
   :canonical: matplotlib.colorizer._ScalarMappable

   .. attribute:: colorbar
   .. method::  changed

      Call this whenever the mappable is changed to notify all the
      callbackSM listeners to the 'changed' signal.

   .. method::  set_array(A)

      Set the value array from array-like *A*.


      :Parameters:

          **A** : array-like or None
              The values that are mapped to colors.

              The base class `.ScalarMappable` does not make any assumptions on
              the dimensionality and shape of the value array *A*.



   .. method::  set_cmap(A)

      Set the colormap for luminance data.


      :Parameters:

          **cmap** : `.Colormap` or str or None
              ..


   .. method::  set_clim(vmin=None, vmax=None)

      Set the norm limits for image scaling.


      :Parameters:

          **vmin, vmax** : float
              The limits.

              For scalar data, the limits may also be passed as a
              tuple (*vmin*, *vmax*) as a single positional argument.

              .. ACCEPTS: (vmin: float, vmax: float)
