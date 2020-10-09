New subfigure functionality
---------------------------
New `.figure.Figure.add_subfigure` and `.figure.Figure.subfigures`
functionalities allow creating virtual figures within figures.  Similar
nesting was previously done with nested gridspecs
( see :doc:`/gallery/subplots_axes_and_figures/gridspec_nested`).  However, this
did not allow localized figure artists (i.e. a colorbar or suptitle) that
only pertained to each subgridspec.

The new methods `.figure.Figure.add_subfigure` and `.figure.Figure.subfigures`
are meant to rhyme with `.figure.Figure.add_subplot` and
`.figure.Figure.subplots` and have most of the same arguments.

See :doc:`/gallery/subplots_axes_and_figures/subfigures`.

.. note::

  The subfigure functionality is experimental API as of v3.4.

.. plot::

  def example_plot(ax, fontsize=12, hide_labels=False):
      pc = ax.pcolormesh(np.random.randn(30, 30))
      if not hide_labels:
          ax.set_xlabel('x-label', fontsize=fontsize)
          ax.set_ylabel('y-label', fontsize=fontsize)
          ax.set_title('Title', fontsize=fontsize)
      return pc

  np.random.seed(19680808)
  fig = plt.figure(constrained_layout=True, figsize=(10, 4))
  subfigs = fig.subfigures(1, 2, wspace=0.07)

  axsLeft = subfigs[0].subplots(1, 2, sharey=True)
  subfigs[0].set_facecolor('0.75')
  for ax in axsLeft:
      pc = example_plot(ax)
  subfigs[0].suptitle('Left plots', fontsize='x-large')
  subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

  axsRight = subfigs[1].subplots(3, 1, sharex=True)
  for nn, ax in enumerate(axsRight):
      pc = example_plot(ax, hide_labels=True)
      if nn == 2:
          ax.set_xlabel('xlabel')
      if nn == 1:
          ax.set_ylabel('ylabel')
  subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight)
  subfigs[1].suptitle('Right plots', fontsize='x-large')

  fig.suptitle('Figure suptitle', fontsize='xx-large')

  plt.show()
