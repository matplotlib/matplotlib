Subplot and subplot2grid can now work with constrained layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``constrained_layout`` depends on a single ``GridSpec``
for each logical layout on a figure. Previously, ``plt.subplot`` and
``plt.subplot2grid`` added a new ``GridSpec`` each time they were called and
were therefore incompatible with ``constrained_layout``.

Now ``plt.subplot`` attempts to reuse the ``GridSpec`` if the number of rows
and columns is the same as the top level gridspec already in the figure.
i.e. ``plt.subplot(2, 1, 2)`` will use the same gridspec as
``plt.subplot(2, 1, 1)`` and the ``constrained_layout=True`` option to
`~.figure.Figure` will work.

In contrast, mixing ``nrows`` and ``ncols`` will *not* work with
``constrained_lyaout``: ``plt.subplot(2, 2, 1)`` followed by
``plt.subplots(2, 1, 2)`` will still produce two gridspecs, and
``constrained_layout=True`` will give bad results.  In order to get the
desired effect, the second call can specify the cells the second axes is meant
to cover:  ``plt.subplots(2, 2, (2, 4))``, or the more pythonic
``plt.subplot2grid((2, 2), (0, 1), rowspan=2)`` can be used.
