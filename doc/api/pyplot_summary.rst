*********************
``matplotlib.pyplot``
*********************

.. currentmodule:: matplotlib.pyplot

.. automodule:: matplotlib.pyplot
   :no-members:
   :no-undoc-members:


Managing Figure and Axes
------------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   axes
   cla
   clf
   close
   delaxes
   fignum_exists
   figure
   gca
   gcf
   get_figlabels
   get_fignums
   sca
   subplot
   subplot2grid
   subfigure_mosaic
   subplot_mosaic
   subplots
   twinx
   twiny


Adding data to the plot
-----------------------

Basic
^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   plot
   errorbar
   scatter
   plot_date
   step
   loglog
   semilogx
   semilogy
   fill_between
   fill_betweenx
   bar
   barh
   bar_label
   stem
   eventplot
   pie
   stackplot
   broken_barh
   vlines
   hlines
   fill
   polar


Spans
^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   axhline
   axhspan
   axvline
   axvspan
   axline


Spectral
^^^^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   acorr
   angle_spectrum
   cohere
   csd
   magnitude_spectrum
   phase_spectrum
   psd
   specgram
   xcorr


Statistics
^^^^^^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   ecdf
   boxplot
   violinplot


Binned
^^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   hexbin
   hist
   hist2d
   stairs


Contours
^^^^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   clabel
   contour
   contourf


2D arrays
^^^^^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   imshow
   matshow
   pcolor
   pcolormesh
   spy
   figimage


Unstructured triangles
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   triplot
   tripcolor
   tricontour
   tricontourf


Text and annotations
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   annotate
   text
   figtext
   table
   arrow
   figlegend
   legend


Vector fields
^^^^^^^^^^^^^

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   barbs
   quiver
   quiverkey
   streamplot


Axis configuration
------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   autoscale
   axis
   box
   grid
   locator_params
   minorticks_off
   minorticks_on
   rgrids
   thetagrids
   tick_params
   ticklabel_format
   xlabel
   xlim
   xscale
   xticks
   ylabel
   ylim
   yscale
   yticks
   suptitle
   title


Layout
------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   margins
   subplots_adjust
   subplot_tool
   tight_layout


Colormapping
------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   clim
   colorbar
   gci
   sci
   get_cmap
   set_cmap
   imread
   imsave

Colormaps are available via the colormap registry `matplotlib.colormaps`. For
convenience this registry is available in ``pyplot`` as

.. autodata:: colormaps
   :no-value:

Additionally, there are shortcut functions to set builtin colormaps; e.g.
``plt.viridis()`` is equivalent to ``plt.set_cmap('viridis')``.


.. autodata:: color_sequences
   :no-value:


Configuration
-------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   rc
   rc_context
   rcdefaults


Output
------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   draw
   draw_if_interactive
   ioff
   ion
   install_repl_displayhook
   isinteractive
   pause
   savefig
   show
   switch_backend
   uninstall_repl_displayhook


Other
-----

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   connect
   disconnect
   findobj
   get
   getp
   get_current_fig_manager
   ginput
   new_figure_manager
   set_loglevel
   setp
   waitforbuttonpress
   xkcd
