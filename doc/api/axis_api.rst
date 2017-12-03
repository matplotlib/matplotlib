===================
 axis and tick API
===================

.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: entry

.. automodule:: matplotlib.axis
   :no-members:
   :no-undoc-members:

Inheritance
===========

.. inheritance-diagram:: Tick Ticker XAxis YAxis XTick YTick
   :private-bases:


``Axis`` objects
================

.. autoclass:: Axis
   :no-members:
   :no-undoc-members:
.. autoclass:: XAxis
   :no-members:
   :no-undoc-members:
.. autoclass:: YAxis
   :no-members:
   :no-undoc-members:
.. autoclass:: Ticker
   :no-members:
   :no-undoc-members:


.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.cla
   Axis.get_scale


Formatters and Locators
-----------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.get_major_formatter
   Axis.get_major_locator
   Axis.get_minor_formatter
   Axis.get_minor_locator
   Axis.set_major_formatter
   Axis.set_major_locator
   Axis.set_minor_formatter
   Axis.set_minor_locator


Axis Label
----------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.set_label_coords
   Axis.set_label_position
   Axis.set_label_text
   Axis.get_label_position
   Axis.get_label_text

Ticks, tick labels and Offset text
----------------------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.get_major_ticks
   Axis.get_majorticklabels
   Axis.get_majorticklines
   Axis.get_majorticklocs
   Axis.get_minor_ticks
   Axis.get_minorticklabels
   Axis.get_minorticklines
   Axis.get_minorticklocs

   Axis.get_offset_text

   Axis.get_tick_padding
   Axis.get_ticklabels
   Axis.get_ticklines
   Axis.get_ticklocs

   Axis.get_gridlines
   Axis.grid

   Axis.iter_ticks
   Axis.set_tick_params

   Axis.axis_date


Data and view intervals
-----------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.get_data_interval
   Axis.get_view_interval
   Axis.set_data_interval
   Axis.set_view_interval

Rendering helpers
-----------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.get_minpos
   Axis.get_tick_space
   Axis.get_ticklabel_extents
   Axis.get_tightbbox


Interactive
-----------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:


   Axis.get_pickradius
   Axis.set_pickradius


Units
-----

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.convert_units
   Axis.set_units
   Axis.get_units
   Axis.update_units


Incremental navigation
----------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.pan
   Axis.zoom


YAxis Specific
--------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   YAxis.axis_name
   YAxis.get_text_widths
   YAxis.get_ticks_position
   YAxis.set_offset_position
   YAxis.set_ticks_position
   YAxis.tick_left
   YAxis.tick_right


XAxis Specific
--------------
.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   XAxis.axis_name
   XAxis.get_text_heights
   XAxis.get_ticks_position
   XAxis.set_ticks_position
   XAxis.tick_bottom
   XAxis.tick_top


Other
-----

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:


   Axis.OFFSETTEXTPAD

   Axis.limit_range_for_scale
   Axis.reset_ticks
   Axis.set_default_intervals
   Axis.get_smart_bounds
   Axis.set_smart_bounds

Discouraged
-----------

These methods implicitly use `~matplotlib.ticker.FixedLocator` and
`~matplotlib.ticker.FixedFormatter`.  They can be convenient, but if
not used together may de-couple your tick labels from your data.

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.set_ticklabels
   Axis.set_ticks



``Tick`` objects
================

.. autoclass:: Tick
   :no-members:
   :no-undoc-members:
.. autoclass:: XTick
   :no-members:
   :no-undoc-members:
.. autoclass:: YTick
   :no-members:
   :no-undoc-members:


.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:


   Tick.apply_tickdir
   Tick.get_loc
   Tick.get_pad
   Tick.get_pad_pixels
   Tick.get_tick_padding
   Tick.get_tickdir
   Tick.get_view_interval
   Tick.set_label1
   Tick.set_label2
   Tick.set_pad
   Tick.update_position


Common and inherited methods
============================

``XTick``
---------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   XTick.apply_tickdir
   XTick.get_loc
   XTick.get_pad
   XTick.get_pad_pixels
   XTick.get_tick_padding
   XTick.get_tickdir
   XTick.get_view_interval
   XTick.set_label1
   XTick.set_label2
   XTick.set_pad
   XTick.update_position



YTick
-----


.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   YTick.apply_tickdir
   YTick.get_loc
   YTick.get_pad
   YTick.get_pad_pixels
   YTick.get_tick_padding
   YTick.get_tickdir
   YTick.get_view_interval
   YTick.set_label1
   YTick.set_label2
   YTick.set_pad
   YTick.update_position

YAxis
-----

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:



   YAxis.OFFSETTEXTPAD
   YAxis.axis_date
   YAxis.cla
   YAxis.convert_units
   YAxis.get_data_interval
   YAxis.get_gridlines
   YAxis.get_label_position
   YAxis.get_label_text
   YAxis.get_major_formatter
   YAxis.get_major_locator
   YAxis.get_major_ticks
   YAxis.get_majorticklabels
   YAxis.get_majorticklines
   YAxis.get_majorticklocs
   YAxis.get_minor_formatter
   YAxis.get_minor_locator
   YAxis.get_minor_ticks
   YAxis.get_minorticklabels
   YAxis.get_minorticklines
   YAxis.get_minorticklocs
   YAxis.get_minpos
   YAxis.get_offset_text
   YAxis.get_pickradius
   YAxis.get_scale
   YAxis.get_smart_bounds
   YAxis.get_tick_padding
   YAxis.get_tick_space
   YAxis.get_ticklabel_extents
   YAxis.get_ticklabels
   YAxis.get_ticklines
   YAxis.get_ticklocs
   YAxis.get_tightbbox
   YAxis.get_units
   YAxis.get_view_interval
   YAxis.grid
   YAxis.iter_ticks
   YAxis.limit_range_for_scale
   YAxis.pan
   YAxis.reset_ticks
   YAxis.set_data_interval
   YAxis.set_default_intervals
   YAxis.set_label_coords
   YAxis.set_label_position
   YAxis.set_label_text
   YAxis.set_major_formatter
   YAxis.set_major_locator
   YAxis.set_minor_formatter
   YAxis.set_minor_locator
   YAxis.set_pickradius
   YAxis.set_smart_bounds
   YAxis.set_tick_params
   YAxis.set_ticklabels
   YAxis.set_ticks
   YAxis.set_units
   YAxis.set_view_interval
   YAxis.update_units
   YAxis.zoom

XAxis
---------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:


   XAxis.OFFSETTEXTPAD
   XAxis.axis_date
   XAxis.cla
   XAxis.convert_units
   XAxis.get_data_interval
   XAxis.get_gridlines
   XAxis.get_label_position
   XAxis.get_label_text
   XAxis.get_major_formatter
   XAxis.get_major_locator
   XAxis.get_major_ticks
   XAxis.get_majorticklabels
   XAxis.get_majorticklines
   XAxis.get_majorticklocs
   XAxis.get_minor_formatter
   XAxis.get_minor_locator
   XAxis.get_minor_ticks
   XAxis.get_minorticklabels
   XAxis.get_minorticklines
   XAxis.get_minorticklocs
   XAxis.get_minpos
   XAxis.get_offset_text
   XAxis.get_pickradius
   XAxis.get_scale
   XAxis.get_smart_bounds
   XAxis.get_tick_padding
   XAxis.get_tick_space
   XAxis.get_ticklabel_extents
   XAxis.get_ticklabels
   XAxis.get_ticklines
   XAxis.get_ticklocs
   XAxis.get_tightbbox
   XAxis.get_units
   XAxis.get_view_interval
   XAxis.grid
   XAxis.iter_ticks
   XAxis.limit_range_for_scale
   XAxis.pan
   XAxis.reset_ticks
   XAxis.set_data_interval
   XAxis.set_default_intervals
   XAxis.set_label_coords
   XAxis.set_label_position
   XAxis.set_label_text
   XAxis.set_major_formatter
   XAxis.set_major_locator
   XAxis.set_minor_formatter
   XAxis.set_minor_locator
   XAxis.set_pickradius
   XAxis.set_smart_bounds
   XAxis.set_tick_params
   XAxis.set_ticklabels
   XAxis.set_ticks
   XAxis.set_units
   XAxis.set_view_interval
   XAxis.update_units
   XAxis.zoom



Inherited from artist
---------------------

Ticks
~~~~~

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:


   Tick.add_callback
   Tick.aname
   Tick.axes
   Tick.contains
   Tick.convert_xunits
   Tick.convert_yunits
   Tick.draw
   Tick.findobj
   Tick.format_cursor_data
   Tick.get_agg_filter
   Tick.get_alpha
   Tick.get_animated
   Tick.get_children
   Tick.get_clip_box
   Tick.get_clip_on
   Tick.get_clip_path
   Tick.get_contains
   Tick.get_cursor_data
   Tick.get_figure
   Tick.get_gid
   Tick.get_label
   Tick.get_path_effects
   Tick.get_picker
   Tick.get_rasterized
   Tick.get_sketch_params
   Tick.get_snap
   Tick.get_transform
   Tick.get_transformed_clip_path_and_affine
   Tick.get_url
   Tick.get_visible
   Tick.get_window_extent
   Tick.get_zorder
   Tick.have_units
   Tick.is_transform_set
   Tick.mouseover
   Tick.pchanged
   Tick.pick
   Tick.pickable
   Tick.properties
   Tick.remove
   Tick.remove_callback
   Tick.set
   Tick.set_agg_filter
   Tick.set_alpha
   Tick.set_animated
   Tick.set_clip_box
   Tick.set_clip_on
   Tick.set_clip_path
   Tick.set_contains
   Tick.set_figure
   Tick.set_gid
   Tick.set_label
   Tick.set_path_effects
   Tick.set_picker
   Tick.set_rasterized
   Tick.set_sketch_params
   Tick.set_snap
   Tick.set_transform
   Tick.set_url
   Tick.set_visible
   Tick.set_zorder
   Tick.stale
   Tick.update
   Tick.update_from
   Tick.zorder

   XTick.add_callback
   XTick.aname
   XTick.axes
   XTick.contains
   XTick.convert_xunits
   XTick.convert_yunits
   XTick.draw
   XTick.findobj
   XTick.format_cursor_data
   XTick.get_agg_filter
   XTick.get_alpha
   XTick.get_animated
   XTick.get_children
   XTick.get_clip_box
   XTick.get_clip_on
   XTick.get_clip_path
   XTick.get_contains
   XTick.get_cursor_data
   XTick.get_figure
   XTick.get_gid
   XTick.get_label
   XTick.get_path_effects
   XTick.get_picker
   XTick.get_rasterized
   XTick.get_sketch_params
   XTick.get_snap
   XTick.get_transform
   XTick.get_transformed_clip_path_and_affine
   XTick.get_url
   XTick.get_visible
   XTick.get_window_extent
   XTick.get_zorder
   XTick.have_units
   XTick.is_transform_set
   XTick.mouseover
   XTick.pchanged
   XTick.pick
   XTick.pickable
   XTick.properties
   XTick.remove
   XTick.remove_callback
   XTick.set
   XTick.set_agg_filter
   XTick.set_alpha
   XTick.set_animated
   XTick.set_clip_box
   XTick.set_clip_on
   XTick.set_clip_path
   XTick.set_contains
   XTick.set_figure
   XTick.set_gid
   XTick.set_label
   XTick.set_path_effects
   XTick.set_picker
   XTick.set_rasterized
   XTick.set_sketch_params
   XTick.set_snap
   XTick.set_transform
   XTick.set_url
   XTick.set_visible
   XTick.set_zorder
   XTick.stale
   XTick.update
   XTick.update_from
   XTick.zorder

   YTick.add_callback
   YTick.aname
   YTick.axes
   YTick.contains
   YTick.convert_xunits
   YTick.convert_yunits
   YTick.draw
   YTick.findobj
   YTick.format_cursor_data
   YTick.get_agg_filter
   YTick.get_alpha
   YTick.get_animated
   YTick.get_children
   YTick.get_clip_box
   YTick.get_clip_on
   YTick.get_clip_path
   YTick.get_contains
   YTick.get_cursor_data
   YTick.get_figure
   YTick.get_gid
   YTick.get_label
   YTick.get_path_effects
   YTick.get_picker
   YTick.get_rasterized
   YTick.get_sketch_params
   YTick.get_snap
   YTick.get_transform
   YTick.get_transformed_clip_path_and_affine
   YTick.get_url
   YTick.get_visible
   YTick.get_window_extent
   YTick.get_zorder
   YTick.have_units
   YTick.is_transform_set
   YTick.mouseover
   YTick.pchanged
   YTick.pick
   YTick.pickable
   YTick.properties
   YTick.remove
   YTick.remove_callback
   YTick.set
   YTick.set_agg_filter
   YTick.set_alpha
   YTick.set_animated
   YTick.set_clip_box
   YTick.set_clip_on
   YTick.set_clip_path
   YTick.set_contains
   YTick.set_figure
   YTick.set_gid
   YTick.set_label
   YTick.set_path_effects
   YTick.set_picker
   YTick.set_rasterized
   YTick.set_sketch_params
   YTick.set_snap
   YTick.set_transform
   YTick.set_url
   YTick.set_visible
   YTick.set_zorder
   YTick.stale
   YTick.update
   YTick.update_from
   YTick.zorder


Axis
~~~~

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:


   Axis.add_callback
   Axis.aname
   Axis.axes
   Axis.contains
   Axis.convert_xunits
   Axis.convert_yunits
   Axis.draw
   Axis.findobj
   Axis.format_cursor_data
   Axis.get_agg_filter
   Axis.get_alpha
   Axis.get_animated
   Axis.get_children
   Axis.get_clip_box
   Axis.get_clip_on
   Axis.get_clip_path
   Axis.get_contains
   Axis.get_cursor_data
   Axis.get_figure
   Axis.get_gid
   Axis.get_label
   Axis.get_path_effects
   Axis.get_picker
   Axis.get_rasterized
   Axis.get_sketch_params
   Axis.get_snap
   Axis.get_transform
   Axis.get_transformed_clip_path_and_affine
   Axis.get_url
   Axis.get_visible
   Axis.get_window_extent
   Axis.get_zorder
   Axis.have_units
   Axis.is_transform_set
   Axis.mouseover
   Axis.pchanged
   Axis.pick
   Axis.pickable
   Axis.properties
   Axis.remove
   Axis.remove_callback
   Axis.set
   Axis.set_agg_filter
   Axis.set_alpha
   Axis.set_animated
   Axis.set_clip_box
   Axis.set_clip_on
   Axis.set_clip_path
   Axis.set_contains
   Axis.set_figure
   Axis.set_gid
   Axis.set_label
   Axis.set_path_effects
   Axis.set_picker
   Axis.set_rasterized
   Axis.set_sketch_params
   Axis.set_snap
   Axis.set_transform
   Axis.set_url
   Axis.set_visible
   Axis.set_zorder
   Axis.stale
   Axis.update
   Axis.update_from
   Axis.zorder

   XAxis.add_callback
   XAxis.aname
   XAxis.axes
   XAxis.contains
   XAxis.convert_xunits
   XAxis.convert_yunits
   XAxis.draw
   XAxis.findobj
   XAxis.format_cursor_data
   XAxis.get_agg_filter
   XAxis.get_alpha
   XAxis.get_animated
   XAxis.get_children
   XAxis.get_clip_box
   XAxis.get_clip_on
   XAxis.get_clip_path
   XAxis.get_contains
   XAxis.get_cursor_data
   XAxis.get_figure
   XAxis.get_gid
   XAxis.get_label
   XAxis.get_path_effects
   XAxis.get_picker
   XAxis.get_rasterized
   XAxis.get_sketch_params
   XAxis.get_snap
   XAxis.get_transform
   XAxis.get_transformed_clip_path_and_affine
   XAxis.get_url
   XAxis.get_visible
   XAxis.get_window_extent
   XAxis.get_zorder
   XAxis.have_units
   XAxis.is_transform_set
   XAxis.mouseover
   XAxis.pchanged
   XAxis.pick
   XAxis.pickable
   XAxis.properties
   XAxis.remove
   XAxis.remove_callback
   XAxis.set
   XAxis.set_agg_filter
   XAxis.set_alpha
   XAxis.set_animated
   XAxis.set_clip_box
   XAxis.set_clip_on
   XAxis.set_clip_path
   XAxis.set_contains
   XAxis.set_figure
   XAxis.set_gid
   XAxis.set_label
   XAxis.set_path_effects
   XAxis.set_picker
   XAxis.set_rasterized
   XAxis.set_sketch_params
   XAxis.set_snap
   XAxis.set_transform
   XAxis.set_url
   XAxis.set_visible
   XAxis.set_zorder
   XAxis.stale
   XAxis.update
   XAxis.update_from
   XAxis.zorder

   YAxis.add_callback
   YAxis.aname
   YAxis.axes
   YAxis.contains
   YAxis.convert_xunits
   YAxis.convert_yunits
   YAxis.draw
   YAxis.findobj
   YAxis.format_cursor_data
   YAxis.get_agg_filter
   YAxis.get_alpha
   YAxis.get_animated
   YAxis.get_children
   YAxis.get_clip_box
   YAxis.get_clip_on
   YAxis.get_clip_path
   YAxis.get_contains
   YAxis.get_cursor_data
   YAxis.get_figure
   YAxis.get_gid
   YAxis.get_label
   YAxis.get_path_effects
   YAxis.get_picker
   YAxis.get_rasterized
   YAxis.get_sketch_params
   YAxis.get_snap
   YAxis.get_transform
   YAxis.get_transformed_clip_path_and_affine
   YAxis.get_url
   YAxis.get_visible
   YAxis.get_window_extent
   YAxis.get_zorder
   YAxis.have_units
   YAxis.is_transform_set
   YAxis.mouseover
   YAxis.pchanged
   YAxis.pick
   YAxis.pickable
   YAxis.properties
   YAxis.remove
   YAxis.remove_callback
   YAxis.set
   YAxis.set_agg_filter
   YAxis.set_alpha
   YAxis.set_animated
   YAxis.set_clip_box
   YAxis.set_clip_on
   YAxis.set_clip_path
   YAxis.set_contains
   YAxis.set_figure
   YAxis.set_gid
   YAxis.set_label
   YAxis.set_path_effects
   YAxis.set_picker
   YAxis.set_rasterized
   YAxis.set_sketch_params
   YAxis.set_snap
   YAxis.set_transform
   YAxis.set_url
   YAxis.set_visible
   YAxis.set_zorder
   YAxis.stale
   YAxis.update
   YAxis.update_from
   YAxis.zorder
