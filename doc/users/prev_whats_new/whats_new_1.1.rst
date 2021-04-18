.. _whats-new-1-1:

New in matplotlib 1.1
=====================

.. contents:: Table of Contents
   :depth: 2


.. note::

   matplotlib 1.1 supports Python 2.4 to 2.7


Sankey Diagrams
---------------

Kevin Davies has extended Yannick Copin's original Sankey example into a module
(:mod:`~matplotlib.sankey`) and provided new examples
(:doc:`/gallery/specialty_plots/sankey_basics`,
:doc:`/gallery/specialty_plots/sankey_links`,
:doc:`/gallery/specialty_plots/sankey_rankine`).

.. figure:: ../../gallery/specialty_plots/images/sphx_glr_sankey_rankine_001.png
   :target: ../../gallery/specialty_plots/sankey_rankine.html
   :align: center
   :scale: 50

   Sankey Rankine


Animation
---------

Ryan May has written a backend-independent framework for creating
animated figures. The :mod:`~matplotlib.animation` module is intended
to replace the backend-specific examples formerly in the
:ref:`examples-index` listings.  Examples using the new framework are
in :ref:`animation-examples-index`; see the entrancing :file:`double
pendulum <gallery/animation/double_pendulum_sgskip.py>` which uses
:meth:`matplotlib.animation.Animation.save` to create the movie below.

.. raw:: html

    <iframe width="420" height="315" src="http://www.youtube.com/embed/32cjc6V0OZY" frameborder="0" allowfullscreen></iframe>

This should be considered as a beta release of the framework;
please try it and provide feedback.


Tight Layout
------------

A frequent issue raised by users of matplotlib is the lack of a layout
engine to nicely space out elements of the plots. While matplotlib still
adheres to the philosophy of giving users complete control over the placement
of plot elements, Jae-Joon Lee created the :mod:`~matplotlib.tight_layout`
module and introduced a new
command :func:`~matplotlib.pyplot.tight_layout`
to address the most common layout issues.

.. plot::

    plt.rcParams['savefig.facecolor'] = "0.8"
    plt.rcParams['figure.figsize'] = 4, 3

    fig, axes_list = plt.subplots(2, 1)
    for ax in axes_list.flat:
        ax.set(xlabel="x-label", ylabel="y-label", title="before tight_layout")
    ax.locator_params(nbins=3)

    plt.show()

    plt.rcParams['savefig.facecolor'] = "0.8"
    plt.rcParams['figure.figsize'] = 4, 3

    fig, axes_list = plt.subplots(2, 1)
    for ax in axes_list.flat:
        ax.set(xlabel="x-label", ylabel="y-label", title="after tight_layout")
    ax.locator_params(nbins=3)

    plt.tight_layout()
    plt.show()

The usage of this functionality can be as simple as ::

    plt.tight_layout()

and it will adjust the spacing between subplots
so that the axis labels do not overlap with neighboring subplots. A
:doc:`/tutorials/intermediate/tight_layout_guide` has been created to show how to use
this new tool.

PyQT4, PySide, and IPython
--------------------------

Gerald Storer made the Qt4 backend compatible with PySide as
well as PyQT4.  At present, however, PySide does not support
the PyOS_InputHook mechanism for handling gui events while
waiting for text input, so it cannot be used with the new
version 0.11 of `IPython <http://ipython.org>`__. Until this
feature appears in PySide, IPython users should use
the PyQT4 wrapper for QT4, which remains the matplotlib default.

An rcParam entry, "backend.qt4", has been added to allow users
to select PyQt4, PyQt4v2, or PySide.  The latter two use the
Version 2 Qt API.  In most cases, users can ignore this rcParam
variable; it is available to aid in testing, and to provide control
for users who are embedding matplotlib in a PyQt4 or PySide app.


Legend
------

Jae-Joon Lee has improved plot legends. First,
legends for complex plots such as :meth:`~matplotlib.pyplot.stem` plots
will now display correctly. Second, the 'best' placement of a legend has
been improved in the presence of NANs.

See the :doc:`/tutorials/intermediate/legend_guide` for more detailed explanation and
examples.

.. figure:: ../../gallery/text_labels_and_annotations/images/sphx_glr_legend_demo_004.png
   :target: ../../gallery/text_labels_and_annotations/legend_demo.html
   :align: center
   :scale: 50

   Legend Demo4

mplot3d
-------

In continuing the efforts to make 3D plotting in matplotlib just as easy
as 2D plotting, Ben Root has made several improvements to the
:mod:`~mpl_toolkits.mplot3d` module.

* :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` has been
  improved to bring the class towards feature-parity with regular
  Axes objects

* Documentation for :ref:`toolkit_mplot3d-tutorial` was significantly expanded

* Axis labels and orientation improved

* Most 3D plotting functions now support empty inputs

* Ticker offset display added:

.. figure:: ../../gallery/mplot3d/images/sphx_glr_offset_001.png
   :target: ../../gallery/mplot3d/offset.html
   :align: center
   :scale: 50

   Offset

* :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.contourf`
  gains *zdir* and *offset* kwargs. You can now do this:

.. figure:: ../../gallery/mplot3d/images/sphx_glr_contourf3d_2_001.png
   :target: ../../gallery/mplot3d/contourf3d_2.html
   :align: center
   :scale: 50

   Contourf3d 2

Numerix support removed
-----------------------

After more than two years of deprecation warnings, Numerix support has
now been completely removed from matplotlib.

Markers
-------

The list of available markers for :meth:`~matplotlib.pyplot.plot` and
:meth:`~matplotlib.pyplot.scatter` has now been merged. While they
were mostly similar, some markers existed for one function, but not
the other. This merge did result in a conflict for the 'd' diamond
marker. Now, 'd' will be interpreted to always mean "thin" diamond
while 'D' will mean "regular" diamond.

Thanks to Michael Droettboom for this effort.

Other improvements
------------------

* Unit support for polar axes and :func:`~matplotlib.axes.Axes.arrow`

* :class:`~matplotlib.projections.polar.PolarAxes` gains getters and setters for
  "theta_direction", and "theta_offset" to allow for theta to go in
  either the clock-wise or counter-clockwise direction and to specify where zero
  degrees should be placed.
  :meth:`~matplotlib.projections.polar.PolarAxes.set_theta_zero_location` is an
  added convenience function.

* Fixed error in argument handling for tri-functions such as
  :meth:`~matplotlib.pyplot.tripcolor`

* ``axes.labelweight`` parameter added to rcParams.

* For :meth:`~matplotlib.pyplot.imshow`, *interpolation='nearest'* will
  now always perform an interpolation. A "none" option has been added to
  indicate no interpolation at all.

* An error in the Hammer projection has been fixed.

* *clabel* for :meth:`~matplotlib.pyplot.contour` now accepts a callable.
  Thanks to Daniel Hyams for the original patch.

* Jae-Joon Lee added the `~mpl_toolkits.axes_grid1.axes_divider.HBoxDivider`
  and `~mpl_toolkits.axes_grid1.axes_divider.VBoxDivider` classes.

* Christoph Gohlke reduced memory usage in :meth:`~matplotlib.pyplot.imshow`.

* :meth:`~matplotlib.pyplot.scatter` now accepts empty inputs.

* The behavior for 'symlog' scale has been fixed, but this may result
  in some minor changes to existing plots.  This work was refined by
  ssyr.

* Peter Butterworth added named figure support to
  :func:`~matplotlib.pyplot.figure`.

* Michiel de Hoon has modified the MacOSX backend to make
  its interactive behavior consistent with the other backends.

* Pim Schellart added a new colormap called "cubehelix".
  Sameer Grover also added a colormap called "coolwarm". See it and all
  other colormaps :ref:`here <color-colormaps_reference>`.

* Many bug fixes and documentation improvements.
