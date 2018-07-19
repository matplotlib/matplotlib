
Changes for 0.40
================

.. code-block:: text

  - Artist
      * __init__ takes a DPI instance and a Bound2D instance which is
        the bounding box of the artist in display coords
      * get_window_extent returns a Bound2D instance
      * set_size is removed; replaced by bbox and dpi
      * the clip_gc method is removed.  Artists now clip themselves with
        their box
      * added _clipOn boolean attribute.  If True, gc clip to bbox.

  - AxisTextBase
      * Initialized with a transx, transy which are Transform instances
      * set_drawing_area removed
      * get_left_right and get_top_bottom are replaced by get_window_extent

  - Line2D Patches now take transx, transy
      * Initialized with a transx, transy which are Transform instances

  - Patches
     * Initialized with a transx, transy which are Transform instances

  - FigureBase attributes dpi is a DPI instance rather than scalar and
    new attribute bbox is a Bound2D in display coords, and I got rid
    of the left, width, height, etc... attributes.  These are now
    accessible as, for example, bbox.x.min is left, bbox.x.interval()
    is width, bbox.y.max is top, etc...

  - GcfBase attribute pagesize renamed to figsize

  - Axes
      * removed figbg attribute
      * added fig instance to __init__
      * resizing is handled by figure call to resize.

  - Subplot
      * added fig instance to __init__

  - Renderer methods for patches now take gcEdge and gcFace instances.
    gcFace=None takes the place of filled=False

  - True and False symbols provided by cbook in a python2.3 compatible
    way

  - new module transforms supplies Bound1D, Bound2D and Transform
    instances and more

  - Changes to the MATLAB helpers API

    * _matlab_helpers.GcfBase is renamed by Gcf.  Backends no longer
      need to derive from this class.  Instead, they provide a factory
      function new_figure_manager(num, figsize, dpi).  The destroy
      method of the GcfDerived from the backends is moved to the derived
      FigureManager.

    * FigureManagerBase moved to backend_bases

    * Gcf.get_all_figwins renamed to Gcf.get_all_fig_managers

  Jeremy:

    Make sure to self._reset = False in AxisTextWX._set_font.  This was
    something missing in my backend code.
