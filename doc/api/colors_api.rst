*********************
``matplotlib.colors``
*********************

.. note::

   The Color :ref:`tutorials <tutorials-colors>` and :ref:`examples
   <color_examples>` demonstrate how to set colors and colormaps. You may want
   to read those instead.

.. currentmodule:: matplotlib.colors

.. automodule:: matplotlib.colors
   :no-members:
   :no-inherited-members:

Color norms
-----------

.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   Normalize
   NoNorm
   AsinhNorm
   BoundaryNorm
   CenteredNorm
   FuncNorm
   LogNorm
   PowerNorm
   SymLogNorm
   TwoSlopeNorm

Colormaps
---------

.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   Colormap
   LinearSegmentedColormap
   ListedColormap

Other classes
-------------

.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   ColorSequenceRegistry
   LightSource

Functions
---------

.. autosummary::
   :toctree: _as_gen/
   :template: autosummary.rst

   from_levels_and_colors
   hsv_to_rgb
   rgb_to_hsv
   to_hex
   to_rgb
   to_rgba
   to_rgba_array
   is_color_like
   same_color
   get_named_colors_mapping
   make_norm_from_scale

Exported colors
---------------

The data used to populate the :doc:`/gallery/color/named_colors` are exposed
as dictionaries that map color names to hex strings.

.. py:data:: BASE_COLORS

.. py:data:: TABLEAU_COLORS

.. py:data:: CSS4_COLORS

.. py:data:: XKCD_COLORS
