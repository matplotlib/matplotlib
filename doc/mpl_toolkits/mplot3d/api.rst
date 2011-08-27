.. _toolkit_mplot3d-api:

***********
mplot3d API
***********
.. contents::
      :backlinks: none

.. _toolkit_mplot3d-axesapi:

:mod:`~mpl_toolkits.mplot3d.axes3d`
===================================

.. note::
   Significant effort went into bringing axes3d to feature-parity with
   regular axes objects for version 1.1.0. However, more work remains.
   Please report any functions that do not behave as expected as a bug.
   In addition, help and patches would be greatly appreciated!

.. automodule:: mpl_toolkits.mplot3d.axes3d
    :members:
    :undoc-members:
    :show-inheritance:


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

.. automodule:: mpl_toolkits.mplot3d.axis3d
    :members:
    :undoc-members:
    :show-inheritance:

.. _toolkit_mplot3d-artapi:

:mod:`~mpl_toolkits.mplot3d.art3d`
==================================

.. automodule:: mpl_toolkits.mplot3d.art3d
    :members:
    :undoc-members:
    :show-inheritance:

.. _toolkit_mplot3d-projapi:

:mod:`~mpl_toolkits.mplot3d.proj3d`
===================================

.. automodule:: mpl_toolkits.mplot3d.proj3d
    :members:
    :undoc-members:
    :show-inheritance:

