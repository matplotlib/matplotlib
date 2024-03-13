*********************
``matplotlib.figure``
*********************

.. currentmodule:: matplotlib.figure

.. automodule:: matplotlib.figure
   :no-members:
   :no-undoc-members:

Figure
======

Figure class
------------
.. autosummary::
   :toctree: _as_gen
   :template: autosummary_class_only.rst
   :nosignatures:

   Figure


Adding Axes and SubFigures
--------------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Figure.add_axes
   Figure.add_subplot
   Figure.subplots
   Figure.subplot_mosaic
   Figure.add_gridspec
   Figure.get_axes
   Figure.axes
   Figure.delaxes
   Figure.subfigures
   Figure.add_subfigure

Saving
------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Figure.savefig


Annotating
----------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Figure.colorbar
   Figure.legend
   Figure.text
   Figure.suptitle
   Figure.get_suptitle
   Figure.supxlabel
   Figure.get_supxlabel
   Figure.supylabel
   Figure.get_supylabel
   Figure.align_labels
   Figure.align_xlabels
   Figure.align_ylabels
   Figure.autofmt_xdate


Figure geometry
---------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Figure.set_size_inches
   Figure.get_size_inches
   Figure.set_figheight
   Figure.get_figheight
   Figure.set_figwidth
   Figure.get_figwidth
   Figure.dpi
   Figure.set_dpi
   Figure.set_dpi

Subplot layout
--------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Figure.subplots_adjust
   Figure.set_layout_engine
   Figure.get_layout_engine

Discouraged or deprecated
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Figure.tight_layout
   Figure.set_tight_layout
   Figure.get_tight_layout
   Figure.set_constrained_layout
   Figure.get_constrained_layout
   Figure.set_constrained_layout_pads
   Figure.get_constrained_layout_pads

Interactive
-----------

.. seealso::

   - :ref:`event-handling`

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Figure.ginput
   Figure.add_axobserver
   Figure.waitforbuttonpress
   Figure.pick

Modifying appearance
--------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Figure.set_frameon
   Figure.get_frameon
   Figure.set_linewidth
   Figure.get_linewidth
   Figure.set_facecolor
   Figure.get_facecolor
   Figure.set_edgecolor
   Figure.get_edgecolor

Adding and getting Artists
--------------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Figure.add_artist
   Figure.get_children
   Figure.figimage

Getting and modifying state
---------------------------

.. seealso::

   - :ref:`interactive_figures`

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Figure.clear
   Figure.gca
   Figure.sca
   Figure.get_tightbbox
   Figure.get_window_extent
   Figure.show
   Figure.set_canvas
   Figure.draw
   Figure.draw_without_rendering
   Figure.draw_artist

.. _figure-api-subfigure:

SubFigure
=========

Matplotlib has the concept of a `~.SubFigure`, which is a logical figure inside
a parent `~.Figure`.  It has many of the same methods as the parent.  See
:ref:`nested_axes_layouts`.

.. plot::

   fig = plt.figure(layout='constrained', figsize=(4, 2.5), facecolor='lightgoldenrodyellow')

   # Make two subfigures, left ones more narrow than right ones:
   sfigs = fig.subfigures(1, 2, width_ratios=[0.8, 1])
   sfigs[0].set_facecolor('khaki')
   sfigs[1].set_facecolor('lightsalmon')

   # Add subplots to left subfigure:
   lax = sfigs[0].subplots(2, 1)
   sfigs[0].suptitle('Left subfigure')

   # Add subplots to right subfigure:
   rax = sfigs[1].subplots(1, 2)
   sfigs[1].suptitle('Right subfigure')

   # suptitle for the main figure:
   fig.suptitle('Figure')

SubFigure class
---------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary_class_only.rst
   :nosignatures:

   SubFigure

Adding Axes and SubFigures
--------------------------
.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   SubFigure.add_axes
   SubFigure.add_subplot
   SubFigure.subplots
   SubFigure.subplot_mosaic
   SubFigure.add_gridspec
   SubFigure.delaxes
   SubFigure.add_subfigure
   SubFigure.subfigures

Annotating
----------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   SubFigure.colorbar
   SubFigure.legend
   SubFigure.text
   SubFigure.suptitle
   SubFigure.get_suptitle
   SubFigure.supxlabel
   SubFigure.get_supxlabel
   SubFigure.supylabel
   SubFigure.get_supylabel
   SubFigure.align_labels
   SubFigure.align_xlabels
   SubFigure.align_ylabels

Adding and getting Artists
--------------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   SubFigure.add_artist
   SubFigure.get_children

Modifying appearance
--------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   SubFigure.set_frameon
   SubFigure.get_frameon
   SubFigure.set_linewidth
   SubFigure.get_linewidth
   SubFigure.set_facecolor
   SubFigure.get_facecolor
   SubFigure.set_edgecolor
   SubFigure.get_edgecolor

Passthroughs
------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   SubFigure.set_dpi
   SubFigure.get_dpi


FigureBase parent class
=======================

.. autoclass:: FigureBase

Helper functions
================

.. autofunction:: figaspect
