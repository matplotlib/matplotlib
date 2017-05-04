.. _toolkit_mplot3d-api:

***********
mplot3d API
***********

.. contents::
      :backlinks: none

.. currentmodule:: mpl_toolkits.mplot3d

.. _toolkit_mplot3d-axesapi:

:mod:`~mpl_toolkits.mplot3d.axes3d`
===================================

.. note::
   Significant effort went into bringing axes3d to feature-parity with
   regular axes objects for version 1.1.0. However, more work remains.
   Please report any functions that do not behave as expected as a bug.
   In addition, help and patches would be greatly appreciated!

.. autosummary::
   :toctree: ../_as_gen
   :template: autosummary.rst

   axes3d.Axes
   axes3d.Axes3D
   axes3d.Bbox
   axes3d.LightSource
   axes3d.Normalize
   axes3d.Triangulation


.. _toolkit_mplot3d-axisapi:

:mod:`~mpl_toolkits.mplot3d.axis3d`
===================================

.. note::
   Historically, axis3d has suffered from having hard-coded constants
   controlling the look and feel of the 3D plot. This precluded user
   level adjustments such as label spacing, font colors and panel colors.
   For version 1.1.0, these constants have been consolidated into a single
   private member dictionary, `self._axinfo`, for the axis object. This is
   intended only as a stop-gap measure to allow user-level customization,
   but it is not intended to be permanent.

.. autosummary::
   :toctree: ../_as_gen
   :template: autosummary.rst

   axis3d.Axis
   axis3d.XAxis
   axis3d.YAxis
   axis3d.ZAxis
   axis3d.get_flip_min_max
   axis3d.move_from_center
   axis3d.tick_update_position


.. _toolkit_mplot3d-artapi:

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
   art3d.get_colors
   art3d.get_dir_vector
   art3d.get_patch_verts
   art3d.iscolor
   art3d.juggle_axes
   art3d.line_2d_to_3d
   art3d.line_collection_2d_to_3d
   art3d.norm_angle
   art3d.norm_text_angle
   art3d.patch_2d_to_3d
   art3d.patch_collection_2d_to_3d
   art3d.path_to_3d_segment
   art3d.path_to_3d_segment_with_codes
   art3d.pathpatch_2d_to_3d
   art3d.paths_to_3d_segments
   art3d.paths_to_3d_segments_with_codes
   art3d.poly_collection_2d_to_3d
   art3d.rotate_axes
   art3d.text_2d_to_3d
   art3d.zalpha

.. _toolkit_mplot3d-projapi:

:mod:`~mpl_toolkits.mplot3d.proj3d`
===================================

.. autosummary::
   :toctree: ../_as_gen
   :template: autosummary.rst

   proj3d.inv_transform
   proj3d.line2d
   proj3d.line2d_dist
   proj3d.line2d_seg_dist
   proj3d.mod
   proj3d.persp_transformation
   proj3d.proj_points
   proj3d.proj_trans_clip_points
   proj3d.proj_trans_points
   proj3d.proj_transform
   proj3d.proj_transform_clip
   proj3d.proj_transform_vec
   proj3d.proj_transform_vec_clip
   proj3d.rot_x
   proj3d.transform
   proj3d.vec_pad_ones
   proj3d.view_transformation
   proj3d.world_transformation
