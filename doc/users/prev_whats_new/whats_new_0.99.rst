.. _whats-new-0-99:

New in matplotlib 0.99
======================

.. contents:: Table of Contents
   :depth: 2



New documentation
-----------------

Jae-Joon Lee has written two new guides :doc:`/tutorials/intermediate/legend_guide`
and :ref:`plotting-guide-annotation`.  Michael Sarahan has written
:doc:`/tutorials/introductory/images`.  John Hunter has written two new tutorials on
working with paths and transformations: :doc:`/tutorials/advanced/path_tutorial` and
:doc:`/tutorials/advanced/transforms_tutorial`.

.. _whats-new-mplot3d:

mplot3d
--------


Reinier Heeres has ported John Porter's mplot3d over to the new
matplotlib transformations framework, and it is now available as a
toolkit mpl_toolkits.mplot3d (which now comes standard with all mpl
installs).  See :ref:`mplot3d-examples-index` and
:ref:`toolkit_mplot3d-tutorial`

.. figure:: ../../gallery/pyplots/images/sphx_glr_whats_new_99_mplot3d_001.png
   :target: ../../gallery/pyplots/whats_new_99_mplot3d.html
   :align: center
   :scale: 50

   Whats New 99 Mplot3d

.. _whats-new-axes-grid:

axes grid toolkit
-----------------

Jae-Joon Lee has added a new toolkit to ease displaying multiple images in
matplotlib, as well as some support for curvilinear grids to support
the world coordinate system. The toolkit is included standard with all
new mpl installs.   See :ref:`axes_grid1-examples-index`,
:ref:`axisartist-examples-index`, :ref:`axes_grid1_users-guide-index` and
:ref:`axisartist_users-guide-index`

.. figure:: ../../gallery/pyplots/images/sphx_glr_whats_new_99_axes_grid_001.png
   :target: ../../gallery/pyplots/whats_new_99_axes_grid.html
   :align: center
   :scale: 50

   Whats New 99 Axes Grid

.. _whats-new-spine:

Axis spine placement
--------------------

Andrew Straw has added the ability to place "axis spines" -- the lines
that denote the data limits -- in various arbitrary locations.  No
longer are your axis lines constrained to be a simple rectangle around
the figure -- you can turn on or off left, bottom, right and top, as
well as "detach" the spine to offset it away from the data.  See
:doc:`/gallery/ticks_and_spines/spine_placement_demo` and
:class:`matplotlib.spines.Spine`.

.. figure:: ../../gallery/pyplots/images/sphx_glr_whats_new_99_spines_001.png
   :target: ../../gallery/pyplots/whats_new_99_spines.html
   :align: center
   :scale: 50

   Whats New 99 Spines
