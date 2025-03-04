Development changes
-------------------

Documentation-specific custom Sphinx roles are now semi-public
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For third-party packages that derive types from Matplotlib, our use of custom roles may
prevent Sphinx from building their docs. These custom Sphinx roles are now public solely
for the purposes of use within projects that derive from Matplotlib types. See
:mod:`matplotlib.sphinxext.roles` for details.

Increase to minimum supported versions of dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Matplotlib 3.10, the :ref:`minimum supported versions <dependencies>` are
being bumped:

+------------+-----------------+----------------+
| Dependency |  min in mpl3.9  | min in mpl3.10 |
+============+=================+================+
|   Python   |       3.9       |      3.10      |
+------------+-----------------+----------------+

This is consistent with our :ref:`min_deps_policy` and `SPEC0
<https://scientific-python.org/specs/spec-0000/>`__
