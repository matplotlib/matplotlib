Axes3D now allows manual control of draw order
----------------------------------------------

The :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` class now has
``computed_zorder`` parameter. When set to False, Artists are drawn using their
``zorder`` attribute.
