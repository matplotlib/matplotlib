Rotating 3d plots with the mouse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rotating three-dimensional plots with the mouse has been made more intuitive.
The plot now reacts the same way to mouse movement, independent of the
particular orientation at hand; and it is possible to control all 3 rotational
degrees of freedom (azimuth, elevation, and roll). By default,
it uses a variation on Ken Shoemake's ARCBALL [1]_.
The particular style of mouse rotation can be set via
``rcParams.axes3d.mouserotationstyle``.
See also :doc:`/api/toolkits/mplot3d/view_angles`.

.. [1] Ken Shoemake, "ARCBALL: A user interface for specifying
  three-dimensional rotation using a mouse", in Proceedings of Graphics
  Interface '92, 1992, pp. 151-156, https://doi.org/10.20380/GI1992.18
