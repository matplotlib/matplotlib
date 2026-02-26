Snapping 3D rotation angles with Control key
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3D axes rotation now supports snapping to fixed angular increments
when holding the ``Control`` key during mouse rotation.

The snap step size is controlled by the new
``axes3d.snap_rotation`` rcParam (default: 5.0 degrees).
Setting it to 0 disables snapping.

For example::

    mpl.rcParams["axes3d.snap_rotation"] = 10

will snap elevation, azimuth, and roll angles to multiples
of 10 degrees while rotating with the mouse.
