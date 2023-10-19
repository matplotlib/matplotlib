Changes for 0.42
================

.. code-block:: text

  * Refactoring AxisText to be backend independent.  Text drawing and
    get_window_extent functionality will be moved to the Renderer.

  * backend_bases.AxisTextBase is now text.Text module

  * All the erase and reset functionality removed from AxisText - not
    needed with double buffered drawing.  Ditto with state change.
    Text instances have a get_prop_tup method that returns a hashable
    tuple of text properties which you can use to see if text props
    have changed, e.g., by caching a font or layout instance in a dict
    with the prop tup as a key -- see RendererGTK.get_pango_layout in
    backend_gtk for an example.

  * Text._get_xy_display renamed Text.get_xy_display

  * Artist set_renderer and wash_brushes methods removed

  * Moved Legend class from matplotlib.axes into matplotlib.legend

  * Moved Tick, XTick, YTick, Axis, XAxis, YAxis from matplotlib.axes
    to matplotlib.axis

  * moved process_text_args to matplotlib.text

  * After getting Text handled in a backend independent fashion, the
    import process is much cleaner since there are no longer cyclic
    dependencies

  * matplotlib.matlab._get_current_fig_manager renamed to
    matplotlib.matlab.get_current_fig_manager to allow user access to
    the GUI window attribute, e.g., figManager.window for GTK and
    figManager.frame for wx
