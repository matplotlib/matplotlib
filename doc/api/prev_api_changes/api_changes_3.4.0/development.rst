Development changes
-------------------

Increase to minimum supported versions of Python and dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Maptlotlib 3.4, the :ref:`minimum supported versions <dependencies>` are
being bumped:

+------------+-----------------+---------------+
| Dependency |  min in mpl3.3  | min in mpl3.4 |
+============+=================+===============+
|   Python   |       3.6       |      3.7      |
+------------+-----------------+---------------+
|  dateutil  |       2.1       |      2.7      |
+------------+-----------------+---------------+
|   numpy    |      1.15       |     1.16      |
+------------+-----------------+---------------+
| pyparsing  |      2.0.3      |     2.2.1     |
+------------+-----------------+---------------+

This is consistent with our :ref:`min_deps_policy` and `NEP29
<https://numpy.org/neps/nep-0029-deprecation_policy.html>`__

Qhull downloaded at build-or-sdist time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Much like FreeType, Qhull is now downloaded at build time, or upon creation of
the sdist. To link against system Qhull, set the ``system_qhull`` option to
`True` in the :file:`setup.cfg` file. Note that Matplotlib now requires the
re-entrant version of Qhull (``qhull_r``).

``FigureBase`` class added, and ``Figure`` class made a child
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new subfigure feature motivated some re-organization of the
`.figure.Figure` class, so that the new `.figure.SubFigure` class could have
all the capabilities of a figure.

The `.figure.Figure` class is now a subclass of `.figure.FigureBase`, where
`.figure.FigureBase` contains figure-level artist addition routines, and the
`.figure.Figure` subclass just contains features that are unique to the outer
figure.

Note that there is a new *transSubfigure* transform associated with the
subfigure.  This transform also exists for a `.Figure` instance, and is equal
to *transFigure* in that case, so code that uses the transform stack that wants
to place objects on either the parent figure or one of the subfigures should
use *transSubfigure*.
