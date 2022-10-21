*********************
``matplotlib.pyplot``
*********************

.. currentmodule:: matplotlib.pyplot

.. automodule:: matplotlib.pyplot
   :no-members:
   :no-undoc-members:


Plotting commands
-----------------

.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   acorr
   angle_spectrum
   annotate
   arrow
   autoscale
   axes
   axhline
   axhspan
   axis
   axline
   axvline
   axvspan
   bar
   bar_label
   barbs
   barh
   box
   boxplot
   broken_barh
   cla
   clabel
   clf
   clim
   close
   cohere
   colorbar
   contour
   contourf
   csd
   delaxes
   draw
   draw_if_interactive
   errorbar
   eventplot
   figimage
   figlegend
   fignum_exists
   figtext
   figure
   fill
   fill_between
   fill_betweenx
   findobj
   gca
   gcf
   gci
   get
   get_cmap
   get_figlabels
   get_fignums
   getp
   grid
   hexbin
   hist
   hist2d
   hlines
   imread
   imsave
   imshow
   install_repl_displayhook
   ioff
   ion
   isinteractive
   legend
   locator_params
   loglog
   magnitude_spectrum
   margins
   matshow
   minorticks_off
   minorticks_on
   pause
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
   rc
   rc_context
   rcdefaults
   rgrids
   savefig
   sca
   scatter
   sci
   semilogx
   semilogy
   set_cmap
   set_loglevel
   setp
   show
   specgram
   spy
   stackplot
   stairs
   stem
   step
   streamplot
   subplot
   subplot2grid
   subplot_mosaic
   subplot_tool
   subplots
   subplots_adjust
   suptitle
   switch_backend
   table
   text
   thetagrids
   tick_params
   ticklabel_format
   tight_layout
   title
   tricontour
   tricontourf
   tripcolor
   triplot
   twinx
   twiny
   uninstall_repl_displayhook
   violinplot
   vlines
   xcorr
   xkcd
   xlabel
   xlim
   xscale
   xticks
   ylabel
   ylim
   yscale
   yticks


Other commands
--------------
.. autosummary::
   :toctree: _as_gen
   :template: autosummary.rst
   :nosignatures:

   connect
   disconnect
   get_current_fig_manager
   ginput
   new_figure_manager
   waitforbuttonpress


Colormaps
---------
Colormaps are available via the colormap registry `matplotlib.colormaps`. For
convenience this registry is available in ``pyplot`` as

.. autodata:: colormaps
   :no-value:

Additionally, there are shortcut functions to set builtin colormaps; e.g.
``plt.viridis()`` is equivalent to ``plt.set_cmap('viridis')``.

.. autodata:: color_sequences
   :no-value:
