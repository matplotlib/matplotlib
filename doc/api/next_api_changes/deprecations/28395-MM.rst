``Axes3D.view_init``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... preferably should be used with keyword arguments, azim=, elev=, roll=,
in this order (i.e., the order in which the rotations are applied).

For backwards compatibility, positional arguments in the old sequence
(first elev, then azim) will still be accepted
(but a deprecation warning will be given).
