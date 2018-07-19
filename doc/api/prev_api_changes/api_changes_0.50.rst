
Changes for 0.50
================

.. code-block:: text

  * refactored Figure class so it is no longer backend dependent.
    FigureCanvasBackend takes over the backend specific duties of the
    Figure.  matplotlib.backend_bases.FigureBase moved to
    matplotlib.figure.Figure.

  * backends must implement FigureCanvasBackend (the thing that
    controls the figure and handles the events if any) and
    FigureManagerBackend (wraps the canvas and the window for MATLAB
    interface).  FigureCanvasBase implements a backend switching
    mechanism

  * Figure is now an Artist (like everything else in the figure) and
    is totally backend independent

  * GDFONTPATH renamed to TTFPATH

  * backend faceColor argument changed to rgbFace

  * colormap stuff moved to colors.py

  * arg_to_rgb in backend_bases moved to class ColorConverter in
    colors.py

  * GD users must upgrade to gd-2.0.22 and gdmodule-0.52 since new gd
    features (clipping, antialiased lines) are now used.

  * Renderer must implement points_to_pixels

  Migrating code:

  MATLAB interface:

    The only API change for those using the MATLAB interface is in how
    you call figure redraws for dynamically updating figures.  In the
    old API, you did

      fig.draw()

    In the new API, you do

      manager = get_current_fig_manager()
      manager.canvas.draw()

    See the examples system_monitor.py, dynamic_demo.py, and anim.py

  API

    There is one important API change for application developers.
    Figure instances used subclass GUI widgets that enabled them to be
    placed directly into figures.  e.g., FigureGTK subclassed
    gtk.DrawingArea.  Now the Figure class is independent of the
    backend, and FigureCanvas takes over the functionality formerly
    handled by Figure.  In order to include figures into your apps,
    you now need to do, for example

      # gtk example
      fig = Figure(figsize=(5,4), dpi=100)
      canvas = FigureCanvasGTK(fig)  # a gtk.DrawingArea
      canvas.show()
      vbox.pack_start(canvas)

    If you use the NavigationToolbar, this in now initialized with a
    FigureCanvas, not a Figure.  The examples embedding_in_gtk.py,
    embedding_in_gtk2.py, and mpl_with_glade.py all reflect the new
    API so use these as a guide.

    All prior calls to

     figure.draw()  and
     figure.print_figure(args)

    should now be

     canvas.draw()  and
     canvas.print_figure(args)

    Apologies for the inconvenience.  This refactorization brings
    significant more freedom in developing matplotlib and should bring
    better plotting capabilities, so I hope the inconvenience is worth
    it.
