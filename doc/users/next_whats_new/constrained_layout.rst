Constrained Layout Manager
---------------------------

.. warning::

    Constrained Layout is **experimental**.  The
    behaviour and API are subject to change, or the whole functionality
    may be removed without a deprecation period.


A new method to automatically decide spacing between subplots and their
organizing ``GridSpec`` instances has been added.  It is meant to
replace the venerable ``tight_layout`` method.  It is invoked via
a new ``constrained_layout=True`` kwarg to
`~.figure.Figure` or `~.figure.subplots`.

There are new ``rcParams`` for this package, and spacing can be
more finely tuned with the new `~.set_constrained_layout_pads`.

Features include:

  - Automatic spacing for subplots with a fixed-size padding in inches around
    subplots and all their decorators, and space between as a fraction
    of subplot size between subplots.
  - Spacing for `~.figure.suptitle`, and colorbars that are attached to
    more than one axes.
  - Nested `~.GridSpec` layouts using `~.GridSpecFromSubplotSpec`.

  For more details and capabilities please see the new tutorial:
  :doc:`/tutorials/intermediate/constrainedlayout_guide`

Note the new API to access this:

New ``plt.figure`` and ``plt.subplots`` kwarg: ``constrained_layout``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~matplotlib.pyplot.figure` and :meth:`~matplotlib.pyplot.subplots`
can now be called with ``constrained_layout=True`` kwarg to enable
constrained_layout.

New ``ax.set_position`` behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~matplotlib.axes.set_position` now makes the specified axis no
longer responsive to ``constrained_layout``, consistent with the idea that the
user wants to place an axis manually.

Internally, this means that old ``ax.set_position`` calls *inside* the library
are changed to private ``ax._set_position`` calls so that
``constrained_layout`` will still work with these axes.

New ``figure`` kwarg for ``GridSpec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to facilitate ``constrained_layout``, ``GridSpec`` now accepts a
``figure`` keyword.  This is backwards compatible, in that not supplying this
will simply cause ``constrained_layout`` to not operate on the subplots
orgainzed by this ``GridSpec`` instance.  Routines that use ``GridSpec`` (e.g.
``fig.subplots``) have been modified to pass the figure to ``GridSpec``.
