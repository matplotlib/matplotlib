Interactive OO usage
--------------------

All `Artists` now keep track of if their internal state has been
changed but not reflected in the display ('stale') by a call to
``draw``.  It is thus possible to pragmatically determine if a given
`Figure` needs to be re-drawn in an interactive session.

To facilitate interactive usage a ``draw_all`` method has been added
to ``pyplot`` which will redraw all of the figures which are 'stale'.

To make this convenient for interactive use matplotlib now registers
a function either with IPython's 'post_execute' event or with the
displayhook in the standard python REPL to automatically call
``plt.draw_all`` just before control is returned to the REPL.  This ensures
that the draw command is deferred and only called once.

The upshot of this is that for interactive backends (including
``%matplotlib notebook``) in interactive mode (with ``plt.ion()``)

.. ipython :: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots()

   ln, = ax.plot([0, 1, 4, 9, 16])

   plt.show()

   ln.set_color('g')


will automatically update the plot to be green.  Any subsequent
modifications to the ``Artist`` objects will do likewise.

This is the first step of a larger consolidation and simplification of
the pyplot internals.
