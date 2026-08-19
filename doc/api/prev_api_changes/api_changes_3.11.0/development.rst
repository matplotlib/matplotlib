Development changes
-------------------

Increase to minimum supported versions of dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Matplotlib 3.11, the :ref:`minimum supported versions <dependencies>` are being
bumped:

+------------+-----------------+----------------+
| Dependency |  min in mpl3.10 | min in mpl3.11 |
+============+=================+================+
|   Python   |       3.10      |      3.11      |
+------------+-----------------+----------------+
|   NumPy    |       1.23      |      1.25      |
+------------+-----------------+----------------+
| pyparsing  |      2.3.1      |     3.0.0      |
+------------+-----------------+----------------+

This is consistent with our :ref:`min_deps_policy` and `SPEC0
<https://scientific-python.org/specs/spec-0000/>`__

pip 25.1 suggested for development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dependencies for development (build and testing) are now specified as `Dependency Groups
<https://packaging.python.org/en/latest/specifications/dependency-groups/#dependency-groups>`_
instead of `individual requirements files
<https://pip.pypa.io/en/stable/reference/requirements-file-format/>`_.

Consequently, a version of pip that supports Dependency Groups is suggested, namely
version 25.1 or higher. Note that if you install build/testing dependencies manually (by
copying the list from ``pyproject.toml``), then an older version of pip is sufficient.

Glyph indices now typed distinctly from character codes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, character codes and glyph indices were both typed as `int`, which means you
could mix and match them erroneously. While the character code can't be made a distinct
type (because it's used for `chr`/`ord`), typing glyph indices as a distinct type means
these can't be fully swapped.
