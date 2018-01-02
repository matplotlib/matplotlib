mplot3d Line3D now allows {set,get}_data_3d
-------------------------------------------

Lines created with the 3d projection in mplot3d can now access the data using
``mplot3d.art3d.Line3D.get_data_3d()`` which returns a tuple of array_likes containing
the (x, y, z) data. The equivalent ``mplot3d.art3d.Line3D.set_data_3d(x, y, z)``
can be used to modify the data of an existing Line3D.
