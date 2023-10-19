*******************
``matplotlib.axes``
*******************

The `~.axes.Axes` class represents one (sub-)plot in a figure. It contains the
plotted data, axis ticks, labels, title, legend, etc. Its methods are the main
interface for manipulating the plot.

.. currentmodule:: matplotlib.axes

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: entry
   :class: multicol-toc

.. automodule:: matplotlib.axes
   :no-members:
   :no-undoc-members:

The Axes class
==============

.. autosummary::
   :toctree: _as_gen
   :template: autosummary_class_only.rst
   :nosignatures:

   Axes

Plotting
========

Basic
-----

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.plot
   Axes.errorbar
   Axes.scatter

   Axes.plot_date
   Axes.step

   Axes.loglog
   Axes.semilogx
   Axes.semilogy

   Axes.fill_between
   Axes.fill_betweenx

   Axes.bar
   Axes.barh
   Axes.bar_label

   Axes.stem
   Axes.eventplot

   Axes.pie

   Axes.stackplot


   Axes.broken_barh
   Axes.vlines
   Axes.hlines
   Axes.fill

Spans
-----

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.axhline
   Axes.axhspan
   Axes.axvline
   Axes.axvspan
   Axes.axline

Spectral
--------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.acorr
   Axes.angle_spectrum
   Axes.cohere
   Axes.csd
   Axes.magnitude_spectrum
   Axes.phase_spectrum
   Axes.psd
   Axes.specgram
   Axes.xcorr

Statistics
----------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.ecdf
   Axes.boxplot
   Axes.violinplot

   Axes.bxp
   Axes.violin

Binned
------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.hexbin
   Axes.hist
   Axes.hist2d
   Axes.stairs

Contours
--------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.clabel
   Axes.contour
   Axes.contourf

2D arrays
---------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.imshow
   Axes.matshow
   Axes.pcolor
   Axes.pcolorfast
   Axes.pcolormesh
   Axes.spy

Unstructured triangles
----------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.tripcolor
   Axes.triplot
   Axes.tricontour
   Axes.tricontourf


Text and annotations
--------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.annotate
   Axes.text
   Axes.table
   Axes.arrow
   Axes.inset_axes
   Axes.indicate_inset
   Axes.indicate_inset_zoom
   Axes.secondary_xaxis
   Axes.secondary_yaxis


Vector fields
-------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.barbs
   Axes.quiver
   Axes.quiverkey
   Axes.streamplot


Clearing
========

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.cla
   Axes.clear


Appearance
==========

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:


   Axes.axis

   Axes.set_axis_off
   Axes.set_axis_on
   Axes.set_frame_on
   Axes.get_frame_on

   Axes.set_axisbelow
   Axes.get_axisbelow

   Axes.grid

   Axes.get_facecolor
   Axes.set_facecolor


Property cycle
==============

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.set_prop_cycle


Axis / limits
=============

.. For families of methods of the form {get,set}_{x,y}foo, try to list them in
   the order set_xfoo, get_xfoo, set_yfoo, get_yfoo

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.get_xaxis
   Axes.get_yaxis

Axis limits and direction
-------------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.invert_xaxis
   Axes.xaxis_inverted
   Axes.invert_yaxis
   Axes.yaxis_inverted

   Axes.set_xlim
   Axes.get_xlim
   Axes.set_ylim
   Axes.get_ylim

   Axes.update_datalim

   Axes.set_xbound
   Axes.get_xbound
   Axes.set_ybound
   Axes.get_ybound

Axis labels, title, and legend
------------------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.set_xlabel
   Axes.get_xlabel
   Axes.set_ylabel
   Axes.get_ylabel
   Axes.label_outer

   Axes.set_title
   Axes.get_title
   Axes.legend
   Axes.get_legend
   Axes.get_legend_handles_labels

Axis scales
-----------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.set_xscale
   Axes.get_xscale
   Axes.set_yscale
   Axes.get_yscale

Autoscaling and margins
-----------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.use_sticky_edges

   Axes.margins
   Axes.get_xmargin
   Axes.get_ymargin
   Axes.set_xmargin
   Axes.set_ymargin

   Axes.relim

   Axes.autoscale
   Axes.autoscale_view

   Axes.set_autoscale_on
   Axes.get_autoscale_on

   Axes.set_autoscalex_on
   Axes.get_autoscalex_on

   Axes.set_autoscaley_on
   Axes.get_autoscaley_on

Aspect ratio
------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.apply_aspect
   Axes.set_aspect
   Axes.get_aspect

   Axes.set_box_aspect
   Axes.get_box_aspect

   Axes.set_adjustable
   Axes.get_adjustable

Ticks and tick labels
---------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.set_xticks
   Axes.get_xticks

   Axes.set_xticklabels
   Axes.get_xticklabels
   Axes.get_xmajorticklabels
   Axes.get_xminorticklabels

   Axes.get_xgridlines
   Axes.get_xticklines

   Axes.xaxis_date

   Axes.set_yticks
   Axes.get_yticks

   Axes.set_yticklabels
   Axes.get_yticklabels
   Axes.get_ymajorticklabels
   Axes.get_yminorticklabels

   Axes.get_ygridlines
   Axes.get_yticklines

   Axes.yaxis_date

   Axes.minorticks_off
   Axes.minorticks_on

   Axes.ticklabel_format
   Axes.tick_params

   Axes.locator_params


Units
=====

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.convert_xunits
   Axes.convert_yunits
   Axes.have_units


Adding artists
==============

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.add_artist
   Axes.add_child_axes
   Axes.add_collection
   Axes.add_container
   Axes.add_image
   Axes.add_line
   Axes.add_patch
   Axes.add_table


Twinning and sharing
====================

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.twinx
   Axes.twiny

   Axes.sharex
   Axes.sharey

   Axes.get_shared_x_axes
   Axes.get_shared_y_axes


Axes position
=============
.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.get_anchor
   Axes.set_anchor

   Axes.get_axes_locator
   Axes.set_axes_locator

   Axes.get_subplotspec
   Axes.set_subplotspec

   Axes.reset_position

   Axes.get_position
   Axes.set_position


Async/event based
=================

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.stale
   Axes.pchanged
   Axes.add_callback
   Axes.remove_callback


Interactive
===========

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:


   Axes.can_pan
   Axes.can_zoom

   Axes.get_navigate
   Axes.set_navigate
   Axes.get_navigate_mode
   Axes.set_navigate_mode

   Axes.get_forward_navigation_events
   Axes.set_forward_navigation_events

   Axes.start_pan
   Axes.drag_pan
   Axes.end_pan

   Axes.format_coord
   Axes.format_cursor_data
   Axes.format_xdata
   Axes.format_ydata

   Axes.mouseover
   Axes.in_axes

   Axes.contains
   Axes.contains_point

   Axes.get_cursor_data

Children
========

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.get_children
   Axes.get_images
   Axes.get_lines
   Axes.findobj


Drawing
=======

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.draw
   Axes.draw_artist
   Axes.redraw_in_frame

   Axes.get_rasterization_zorder
   Axes.set_rasterization_zorder

   Axes.get_window_extent
   Axes.get_tightbbox


Projection
==========

Methods used by `~matplotlib.axis.Axis` that must be overridden for
non-rectilinear Axes.

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.name
   Axes.get_xaxis_transform
   Axes.get_yaxis_transform
   Axes.get_data_ratio

   Axes.get_xaxis_text1_transform
   Axes.get_xaxis_text2_transform
   Axes.get_yaxis_text1_transform
   Axes.get_yaxis_text2_transform


Other
=====

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.zorder
   Axes.get_default_bbox_extra_artists
   Axes.get_transformed_clip_path_and_affine
   Axes.has_data
   Axes.set

.. autoclass:: matplotlib.axes.Axes.ArtistList
