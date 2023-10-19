API Changes for 3.5.2
=====================

.. contents::
   :local:
   :depth: 1

QuadMesh mouseover defaults to False
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New in 3.5, `.QuadMesh.get_cursor_data` allows display of data values
under the cursor.  However, this can be very slow for large meshes, so
by ``.QuadMesh.set_mouseover`` defaults to *False*.
