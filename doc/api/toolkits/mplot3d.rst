.. _toolkit_mplot3d-api:

***********
mplot3d API
***********

.. note::
   `.pyplot` cannot be used to add content to 3D plots, because its function
   signatures are strictly 2D and cannot handle the additional information
   needed for 3D. Instead, use the explicit API by calling the respective
   methods on the `.Axes3D` object.

.. automodule:: mpl_toolkits.mplot3d
   :no-members:
   :no-undoc-members:

.. module:: mpl_toolkits.mplot3d.axes3d
.. currentmodule:: mpl_toolkits.mplot3d

:mod:`~mpl_toolkits.mplot3d.axes3d`
===================================

.. note::
   3D plotting in Matplotlib is still not as mature as the 2D case.
   Please report any functions that do not behave as expected as a bug.
   In addition, help and patches would be greatly appreciated!

.. autosummary::
   :toctree: ../_as_gen
   :template: autosummary.rst

   axes3d.Axes3D


.. module:: mpl_toolkits.mplot3d.axis3d
.. currentmodule:: mpl_toolkits.mplot3d

:mod:`~mpl_toolkits.mplot3d.axis3d`
===================================

.. note::
   See :attr:`mpl_toolkits.mplot3d.axis3d._axinfo` for a dictionary containing
   constants that may be modified for controlling the look and feel
   of mplot3d axes (e.g., label spacing, font colors and panel colors).
   Historically, axis3d has suffered from having hard-coded constants
   that precluded user adjustments, and this dictionary was implemented
   in version 1.1 as a stop-gap measure.


.. autosummary::
   :toctree: ../_as_gen
   :template: autosummary.rst

   axis3d.Axis


.. module:: mpl_toolkits.mplot3d.art3d
.. currentmodule:: mpl_toolkits.mplot3d

:mod:`~mpl_toolkits.mplot3d.art3d`
==================================

.. autosummary::
   :toctree: ../_as_gen
   :template: autosummary.rst

   art3d.Line3D
   art3d.Line3DCollection
   art3d.Patch3D
   art3d.Patch3DCollection
   art3d.Path3DCollection
   art3d.PathPatch3D
   art3d.Poly3DCollection
   art3d.Text3D
   art3d.get_dir_vector
   art3d.juggle_axes
   art3d.line_2d_to_3d
   art3d.line_collection_2d_to_3d
   art3d.patch_2d_to_3d
   art3d.patch_collection_2d_to_3d
   art3d.pathpatch_2d_to_3d
   art3d.poly_collection_2d_to_3d
   art3d.rotate_axes
   art3d.text_2d_to_3d

.. module:: mpl_toolkits.mplot3d.proj3d
.. currentmodule:: mpl_toolkits.mplot3d

:mod:`~mpl_toolkits.mplot3d.proj3d`
===================================

.. autosummary::
   :toctree: ../_as_gen
   :template: autosummary.rst

   proj3d.inv_transform
   proj3d.persp_transformation
   proj3d.proj_points
   proj3d.proj_trans_points
   proj3d.proj_transform
   proj3d.proj_transform_clip
   proj3d.rot_x
   proj3d.transform
   proj3d.view_transformation
   proj3d.world_transformation
