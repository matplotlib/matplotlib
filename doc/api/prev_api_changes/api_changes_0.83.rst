Changes for 0.83
================

.. code-block:: text

  - Made HOME/.matplotlib the new config dir where the matplotlibrc
    file, the ttf.cache, and the tex.cache live.  The new default
    filenames in .matplotlib have no leading dot and are not hidden.
    e.g., the new names are matplotlibrc, tex.cache, and ttffont.cache.
    This is how ipython does it so it must be right.

    If old files are found, a warning is issued and they are moved to
    the new location.

  - backends/__init__.py no longer imports new_figure_manager,
    draw_if_interactive and show from the default backend, but puts
    these imports into a call to pylab_setup.  Also, the Toolbar is no
    longer imported from WX/WXAgg.  New usage:

      from backends import pylab_setup
      new_figure_manager, draw_if_interactive, show = pylab_setup()

  - Moved Figure.get_width_height() to FigureCanvasBase. It now
    returns int instead of float.
