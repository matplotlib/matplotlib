*******************
``matplotlib.axis``
*******************

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

   Axis.clear
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

   Axis.remove_overlapping_locs
   Axis.get_remove_overlapping_locs
   Axis.set_remove_overlapping_locs

Axis Label
----------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.set_label_coords
   Axis.set_label_position
   Axis.set_label_text
   Axis.get_label
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
   Axis.get_tick_params
   Axis.get_ticklabels
   Axis.get_ticklines
   Axis.get_ticklocs

   Axis.get_gridlines
   Axis.grid

   Axis.set_tick_params

   Axis.axis_date

   Axis.minorticks_off
   Axis.minorticks_on


Data and view intervals
-----------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.get_data_interval
   Axis.get_view_interval
   Axis.get_inverted
   Axis.set_data_interval
   Axis.set_view_interval
   Axis.set_inverted

Rendering helpers
-----------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.get_minpos
   Axis.get_tick_space
   Axis.get_tightbbox


Interactive
-----------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.contains
   Axis.pickradius
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


XAxis Specific
--------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   XAxis.axis_name
   XAxis.get_ticks_position
   XAxis.set_ticks_position
   XAxis.set_label_position
   XAxis.tick_bottom
   XAxis.tick_top

YAxis Specific
--------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   YAxis.axis_name
   YAxis.get_ticks_position
   YAxis.set_offset_position
   YAxis.set_ticks_position
   YAxis.set_label_position
   YAxis.tick_left
   YAxis.tick_right

Other
-----

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:


   Axis.OFFSETTEXTPAD

   Axis.axes
   Axis.limit_range_for_scale
   Axis.reset_ticks
   Axis.set_default_intervals

Discouraged
-----------

These methods should be used together with care, calling ``set_ticks``
to specify the desired tick locations **before** calling ``set_ticklabels`` to
specify a matching series of labels.  Calling ``set_ticks`` makes a
`~matplotlib.ticker.FixedLocator`; it's list of locations is then used by
``set_ticklabels`` to make an appropriate
`~matplotlib.ticker.FuncFormatter`.

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   Axis.set_ticks
   Axis.set_ticklabels



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

   Tick.get_loc
   Tick.get_pad
   Tick.get_tick_padding
   Tick.get_tickdir
   Tick.get_view_interval
   Tick.set_label1
   Tick.set_label2
   Tick.set_pad
   Tick.set_url
   Tick.update_position
