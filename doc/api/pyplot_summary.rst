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
   subplot_mosaic
   subplots
   twinx
   twiny


Adding data to the plot
-----------------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   acorr
   angle_spectrum
   annotate
   arrow
   axhline
   axhspan
   axline
   axvline
   axvspan
   bar
   bar_label
   barbs
   barh
   boxplot
   broken_barh
   clabel
   cohere
   contour
   contourf
   csd
   errorbar
   eventplot
   figimage
   figlegend
   figtext
   fill
   fill_between
   fill_betweenx
   hexbin
   hist
   hist2d
   hlines
   imshow
   legend
   loglog
   magnitude_spectrum
   matshow
   pcolor
   pcolormesh
   phase_spectrum
   pie
   plot
   plot_date
   polar
   psd
   quiver
   quiverkey
   scatter
   semilogx
   semilogy
   specgram
   spy
   stackplot
   stairs
   stem
   step
   streamplot
   suptitle
   table
   text
   title
   tricontour
   tricontourf
   tripcolor
   triplot
   violinplot
   vlines
   xcorr


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
