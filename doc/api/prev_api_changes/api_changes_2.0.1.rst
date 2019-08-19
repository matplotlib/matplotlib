
API Changes in 2.0.1
====================

Extensions to `matplotlib.backend_bases.GraphicsContextBase`
------------------------------------------------------------

To better support controlling the color of hatches, the method
`matplotlib.backend_bases.GraphicsContextBase.set_hatch_color` was
added to the expected API of ``GraphicsContext`` classes.  Calls to
this method are currently wrapped with a ``try:...except Attribute:``
block to preserve back-compatibility with any third-party backends
which do not extend `~matplotlib.backend_bases.GraphicsContextBase`.

This value can be accessed in the backends via
`matplotlib.backend_bases.GraphicsContextBase.get_hatch_color` (which
was added in 2.0 see :ref:`gc_get_hatch_color_wn`) and should be used
to color the hatches.

In the future there may also be ``hatch_linewidth`` and
``hatch_density`` related methods added.  It is encouraged, but not
required that third-party backends extend
`~matplotlib.backend_bases.GraphicsContextBase` to make adapting to
these changes easier.


``afm.get_fontconfig_fonts`` returns a list of paths and does not check for existence
-------------------------------------------------------------------------------------

``afm.get_fontconfig_fonts`` used to return a set of paths encoded as a
``{key: 1, ...}`` dict, and checked for the existence of the paths.  It now
returns a list and dropped the existence check, as the same check is performed
by the caller (``afm.findSystemFonts``) as well.


``bar`` now returns rectangles of negative height or width if the corresponding input is negative
-------------------------------------------------------------------------------------------------

`.pyplot.bar` used to normalize the coordinates of the rectangles that it
created, to keep their height and width positives, even if the corresponding
input was negative.  This normalization has been removed to permit a simpler
computation of the correct `.Artist.sticky_edges` to use.


Do not clip line width when scaling dashes
------------------------------------------

The algorithm to scale dashes was changed to no longer clip the
scaling factor: the dash patterns now continue to shrink at thin line widths.
If the line width is smaller than the effective pixel size, this may result in
dashed lines turning into solid gray-ish lines.  This also required slightly
tweaking the default patterns for '--', ':', and '.-' so that with the default
line width the final patterns would not change.

There is no way to restore the old behavior.


Deprecate 'Vega' color maps
---------------------------

The "Vega" colormaps are deprecated in Matplotlib 2.0.1 and will be
removed in Matplotlib 2.2. Use the "tab" colormaps instead: "tab10",
"tab20", "tab20b", "tab20c".
