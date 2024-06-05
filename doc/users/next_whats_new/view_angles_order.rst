Order of view angles
~~~~~~~~~~~~~~~~~~~~

The order of the 'view angles' for three-dimensional plots (with ``mplot3d``)
has been changed to ``azim=, elev=, roll=``. This is consistent with the order
in which the rotations take place [1]. It is suggested to use keyword arguments
(in this order) for the angles, so there will be no confusion as to which angle
is which (even if the order would be different).


For backward compatibility, the old ``elev``, ``azim`` etc. positional arguments
will still be accepted (but it is not recommended).


[1]: See :doc: `/api/toolkits/mplot3d/view_angles` for details. Also, this particular
order appears to be most common; and it is consistent with the ordering in
matplotlib's colors.py - see also :ghissue:`28353`.
