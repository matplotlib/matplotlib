Enhancements to polar plot
--------------------------

The polar axes transforms have been greatly re-factored to allow for more
customization of view limits and tick labelling. Additional options for view
limits allow for creating an annulus, a sector, or some combination of the two.

The :meth:`~matplotlib.axes.projections.polar.PolarAxes.set_rorigin` method may
be used to provide an offset to the minimum plotting radius, producing an
annulus.

The :meth:`~matplotlib.projections.polar.PolarAxes.set_theta_zero_location` now
has an optional :code:`offset` argument. This argument may be used to further
specify the zero location based on the given anchor point.

.. figure:: ../../gallery/pie_and_polar_charts/images/sphx_glr_polar_scatter_001.png
   :target: ../../gallery/pie_and_polar_charts/polar_scatter.html
   :align: center
   :scale: 50

   Polar Offset Demo

The :meth:`~matplotlib.axes.projections.polar.PolarAxes.set_thetamin` and
:meth:`~matplotlib.axes.projections.polar.PolarAxes.set_thetamax` methods may
be used to limit the range of angles plotted, producing sectors of a circle.

.. figure:: ../../gallery/pie_and_polar_charts/images/sphx_glr_polar_scatter_002.png
   :target: ../../gallery/pie_and_polar_charts/polar_scatter.html
   :align: center
   :scale: 50

   Polar Sector Demo

Previous releases allowed plots containing negative radii for which the
negative values are simply used as labels, and the real radius is shifted by
the configured minimum. This release also allows negative radii to be used for
grids and ticks, which were previously silently ignored.
