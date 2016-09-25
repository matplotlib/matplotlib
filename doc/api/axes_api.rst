********
``Axes``
********

.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: entry


:mod:`matplotlib.axes`
======================

.. automodule:: matplotlib.axes

Plotting
--------

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Axes.acorr
   Axes.angle_spectrum
   Axes.annotate
   Axes.arrow
   Axes.axhline
   Axes.axhspan
   Axes.axvline
   Axes.axvspan
   Axes.bar
   Axes.barbs
   Axes.barh
   Axes.boxplot
   Axes.broken_barh
   Axes.bxp
   Axes.clabel
   Axes.cohere
   Axes.contour
   Axes.contourf
   Axes.csd
   Axes.errorbar
   Axes.eventplot
   Axes.fill
   Axes.fill_between
   Axes.fill_betweenx
   Axes.hexbin
   Axes.hist
   Axes.hist2d
   Axes.hlines
   Axes.imshow
   Axes.loglog
   Axes.magnitude_spectrum
   Axes.matshow
   Axes.pcolor
   Axes.pcolorfast
   Axes.pcolormesh
   Axes.phase_spectrum
   Axes.pie
   Axes.plot
   Axes.plot_date
   Axes.psd
   Axes.quiver
   Axes.quiverkey
   Axes.scatter
   Axes.semilogx
   Axes.semilogy
   Axes.specgram
   Axes.spy
   Axes.stackplot
   Axes.stem
   Axes.step
   Axes.streamplot
   Axes.table
   Axes.text
   Axes.tricontour
   Axes.tricontourf
   Axes.tripcolor
   Axes.triplot
   Axes.violin
   Axes.violinplot
   Axes.vlines
   Axes.xcorr


Axis / limits
-------------


.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Axes.axis
   Axes.set_axis_off
   Axes.set_axis_on
   Axes.set_axisbelow
   Axes.invert_xaxis
   Axes.invert_yaxis
   Axes.xaxis_inverted
   Axes.yaxis_inverted


Axis scales
~~~~~~~~~~~

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Axes.set_xscale
   Axes.get_xscale
   Axes.get_yscale
   Axes.set_yscale


Autoscaling
~~~~~~~~~~~

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Axes.autoscale
   Axes.autoscale_view

   Axes.get_autoscale_on
   Axes.set_autoscale_on

   Axes.get_autoscalex_on
   Axes.set_autoscalex_on

   Axes.get_autoscaley_on
   Axes.set_autoscaley_on


Aspect ratio
~~~~~~~~~~~~

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Axes.apply_aspect
   Axes.get_aspect
   Axes.set_aspect

   Axes.get_adjustable
   Axes.set_adjustable


Ticks and tick labels
~~~~~~~~~~~~~~~~~~~~~


.. autosummary::
   :toctree: _as_gen
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
   Axes.set_ybound
   Axes.set_ylabel
   Axes.set_ylim
   Axes.set_ymargin
   Axes.set_yticklabels
   Axes.set_yticks


Units
-----

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Axes.convert_xunits
   Axes.convert_yunits


Adding Artists
--------------

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Axes.add_artist
   Axes.add_callback
   Axes.add_collection
   Axes.add_container
   Axes.add_image
   Axes.add_line
   Axes.add_patch
   Axes.add_table


Async/Event based
-----------------

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Axes.stale
   Axes.pchanged


Twinning
--------

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Axes.twinx
   Axes.twiny

   Axes.get_shared_x_axes
   Axes.get_shared_y_axes

Interactive helpers
-------------------

.. autosummary::
   :toctree: _as_gen
   :nosignatures:


   Axes.can_pan
   Axes.can_zoom

   Axes.get_navigate
   Axes.set_navigate
   Axes.get_navigate_mode
   Axes.set_navigate_mode

   Axes.drag_pan
   Axes.end_pan

   Axes.format_coord
   Axes.format_cursor_data
   Axes.format_xdata
   Axes.format_ydata

Other
-----

.. autosummary::
   :toctree: _as_gen
   :nosignatures:

   Axes.aname
   Axes.axes
   Axes.bottom_margin
   Axes.cla
   Axes.clear
   Axes.contains
   Axes.contains_point
   Axes.draw
   Axes.draw_artist
   Axes.findobj
   Axes.get_agg_filter
   Axes.get_alpha
   Axes.get_anchor
   Axes.get_animated
   Axes.get_axes
   Axes.get_axes_locator
   Axes.get_axis_bgcolor
   Axes.get_axisbelow
   Axes.get_bottom_margin
   Axes.get_children
   Axes.get_clip_box
   Axes.get_clip_on
   Axes.get_clip_path
   Axes.get_contains
   Axes.get_cursor_data
   Axes.get_cursor_props
   Axes.get_data_ratio
   Axes.get_data_ratio_log
   Axes.get_default_bbox_extra_artists
   Axes.get_facecolor
   Axes.get_fc
   Axes.get_figure
   Axes.get_frame_on
   Axes.get_gid
   Axes.get_images
   Axes.get_label
   Axes.get_left_margin
   Axes.get_legend
   Axes.get_legend_handles_labels
   Axes.get_lines
   Axes.get_margins
   Axes.get_path_effects
   Axes.get_picker
   Axes.get_position
   Axes.get_rasterization_zorder
   Axes.get_rasterized
   Axes.get_renderer_cache
   Axes.get_right_margin
   Axes.get_sketch_params
   Axes.get_snap
   Axes.get_tightbbox
   Axes.get_title
   Axes.get_top_margin
   Axes.get_transform
   Axes.get_transformed_clip_path_and_affine
   Axes.get_url
   Axes.get_visible
   Axes.get_window_extent
   Axes.get_xaxis
   Axes.get_xaxis_text1_transform
   Axes.get_xaxis_text2_transform
   Axes.get_xaxis_transform
   Axes.get_xbound
   Axes.get_xgridlines
   Axes.get_xlabel
   Axes.get_xlim
   Axes.get_yaxis
   Axes.get_yaxis_text1_transform
   Axes.get_yaxis_text2_transform
   Axes.get_yaxis_transform
   Axes.get_ybound
   Axes.get_ygridlines
   Axes.get_ylabel
   Axes.get_ylim
   Axes.get_zorder
   Axes.grid
   Axes.has_data
   Axes.have_units
   Axes.hitlist
   Axes.hold
   Axes.in_axes
   Axes.is_figure_set
   Axes.is_transform_set
   Axes.ishold
   Axes.left_margin
   Axes.legend
   Axes.locator_params
   Axes.margins
   Axes.mouseover
   Axes.name
   Axes.pick
   Axes.pickable
   Axes.properties
   Axes.redraw_in_frame
   Axes.relim
   Axes.remove
   Axes.remove_callback
   Axes.reset_position
   Axes.right_margin
   Axes.set
   Axes.set_agg_filter
   Axes.set_alpha
   Axes.set_anchor
   Axes.set_animated
   Axes.set_axes
   Axes.set_axes_locator
   Axes.set_axis_bgcolor
   Axes.set_bottom_margin
   Axes.set_clip_box
   Axes.set_clip_on
   Axes.set_clip_path
   Axes.set_color_cycle
   Axes.set_contains
   Axes.set_cursor_props
   Axes.set_facecolor
   Axes.set_fc
   Axes.set_figure
   Axes.set_frame_on
   Axes.set_gid
   Axes.set_label
   Axes.set_left_margin
   Axes.set_margins
   Axes.set_path_effects
   Axes.set_picker
   Axes.set_position
   Axes.set_prop_cycle
   Axes.set_rasterization_zorder
   Axes.set_rasterized
   Axes.set_right_margin
   Axes.set_sketch_params
   Axes.set_snap
   Axes.set_title
   Axes.set_top_margin
   Axes.set_transform
   Axes.set_url
   Axes.set_visible
   Axes.set_xbound
   Axes.set_xlabel
   Axes.set_xlim
   Axes.set_xmargin
   Axes.set_zorder
   Axes.start_pan
   Axes.tick_params
   Axes.ticklabel_format
   Axes.top_margin
   Axes.update
   Axes.update_datalim
   Axes.update_datalim_bounds
   Axes.update_datalim_numerix
   Axes.update_from
   Axes.zorder
