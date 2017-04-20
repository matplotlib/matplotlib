================
 ``Axes`` class
================
.. currentmodule:: matplotlib.axes

.. autoclass:: Axes

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: entry


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

   Axes.boxplot
   Axes.violinplot

   Axes.violin
   Axes.bxp

Binned
------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.hexbin
   Axes.hist
   Axes.hist2d

Contours
--------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.clabel
   Axes.contour
   Axes.contourf

Array
-----

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

Unstructured Triangles
----------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.tripcolor
   Axes.triplot
   Axes.tricontour
   Axes.tricontourf


Text and Annotations
--------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.annotate
   Axes.text
   Axes.table
   Axes.arrow


Fields
------

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

   Axes.get_axis_bgcolor
   Axes.get_facecolor
   Axes.get_fc

   Axes.set_facecolor
   Axes.set_fc
   Axes.set_axis_bgcolor



Property cycle
==============

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.set_prop_cycle
   Axes.set_color_cycle


Axis / limits
=============

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:


   Axes.get_yaxis
   Axes.get_xaxis



Axis Limits and direction
-------------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.invert_xaxis
   Axes.invert_yaxis
   Axes.xaxis_inverted
   Axes.yaxis_inverted

   Axes.set_xlim
   Axes.set_ylim
   Axes.get_ylim
   Axes.get_xlim

   Axes.update_datalim
   Axes.update_datalim_bounds
   Axes.update_datalim_numerix

   Axes.set_ybound
   Axes.set_xbound
   Axes.get_ybound
   Axes.get_xbound

Axis Labels, title, and legend
------------------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.get_xlabel
   Axes.get_ylabel

   Axes.set_xlabel
   Axes.set_ylabel

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

   Axes.get_yscale
   Axes.set_yscale



Autoscaling and margins
-----------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.use_sticky_edges

   Axes.margins
   Axes.set_xmargin
   Axes.set_ymargin

   Axes.relim

   Axes.autoscale
   Axes.autoscale_view

   Axes.get_autoscale_on
   Axes.set_autoscale_on

   Axes.get_autoscalex_on
   Axes.set_autoscalex_on

   Axes.get_autoscaley_on
   Axes.set_autoscaley_on


Aspect ratio
------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.apply_aspect
   Axes.get_aspect
   Axes.set_aspect

   Axes.get_adjustable
   Axes.set_adjustable


Ticks and tick labels
---------------------


.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.xaxis_date
   Axes.yaxis_date

   Axes.get_xmajorticklabels
   Axes.get_xminorticklabels
   Axes.get_xticklabels
   Axes.get_xticklines
   Axes.get_xticks

   Axes.get_ymajorticklabels
   Axes.get_yminorticklabels
   Axes.get_yticklabels
   Axes.get_yticklines
   Axes.get_yticks

   Axes.minorticks_off
   Axes.minorticks_on

   Axes.set_xticklabels
   Axes.set_xticks
   Axes.set_yticklabels
   Axes.set_yticks

   Axes.get_xgridlines
   Axes.get_ygridlines


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


Adding Artists
==============

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.add_artist
   Axes.add_collection
   Axes.add_container
   Axes.add_image
   Axes.add_line
   Axes.add_patch
   Axes.add_table


Twinning
========

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.twinx
   Axes.twiny

   Axes.get_shared_x_axes
   Axes.get_shared_y_axes


Axes Position
=============
.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.get_anchor
   Axes.set_anchor

   Axes.get_axes_locator
   Axes.set_axes_locator

   Axes.reset_position

   Axes.get_position
   Axes.set_position


Async/Event based
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

   Axes.start_pan
   Axes.drag_pan
   Axes.end_pan

   Axes.format_coord
   Axes.format_cursor_data
   Axes.format_xdata
   Axes.format_ydata

   Axes.hitlist
   Axes.mouseover
   Axes.in_axes

   Axes.pick
   Axes.pickable
   Axes.get_picker
   Axes.set_picker

   Axes.set_contains
   Axes.get_contains

   Axes.contains
   Axes.contains_point

   Axes.get_cursor_data
   Axes.get_cursor_props
   Axes.set_cursor_props

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
   Axes.get_renderer_cache

   Axes.get_rasterization_zorder
   Axes.set_rasterization_zorder

   Axes.get_window_extent
   Axes.get_tightbbox


Bulk property manipulation
==========================

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.set
   Axes.update
   Axes.properties
   Axes.update_from


General Artist Properties
=========================

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.set_alpha
   Axes.set_animated
   Axes.set_clip_box
   Axes.set_clip_on
   Axes.set_clip_path
   Axes.set_gid
   Axes.set_label
   Axes.set_url
   Axes.set_visible
   Axes.set_zorder
   Axes.set_rasterized
   Axes.set_sketch_params
   Axes.set_agg_filter
   Axes.set_snap
   Axes.set_transform
   Axes.set_path_effects

   Axes.get_agg_filter
   Axes.get_sketch_params
   Axes.get_alpha
   Axes.get_animated
   Axes.get_clip_box
   Axes.get_clip_on
   Axes.get_clip_path
   Axes.get_gid
   Axes.get_label
   Axes.get_url
   Axes.get_visible
   Axes.get_zorder
   Axes.get_rasterized
   Axes.get_transform
   Axes.get_snap
   Axes.get_path_effects


   Axes.axes
   Axes.get_axes
   Axes.set_axes
   Axes.set_figure
   Axes.get_figure

Artist Methods
==============

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axes.is_figure_set
   Axes.remove
   Axes.is_transform_set


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
   Axes.get_data_ratio_log

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
   Axes.aname
   Axes.get_default_bbox_extra_artists
   Axes.get_transformed_clip_path_and_affine
   Axes.has_data
   Axes.hold
   Axes.ishold


Inheritance
===========
.. inheritance-diagram:: matplotlib.axes.Axes
   :private-bases:
