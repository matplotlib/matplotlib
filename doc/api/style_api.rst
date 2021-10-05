********************
``matplotlib.style``
********************

Styles are predefined sets of `.rcParams` that define the visual appearance of
a plot.

:doc:`/tutorials/introductory/customizing` describes the mechanism and usage
of styles.

The :doc:`/gallery/style_sheets/style_sheets_reference` gives an overview of
the builtin styles.

.. automodule:: matplotlib.style
   :members:
   :undoc-members:
   :show-inheritance:
   :imported-members:

.. imported variables have to be specified explicitly due to
   https://github.com/sphinx-doc/sphinx/issues/6607

.. data:: matplotlib.style.library

   A dict mapping from style name to `.RcParams` defining that style.

   This is meant to be read-only. Use `.reload_library` to update.

.. data:: matplotlib.style.available

   List of the names of the available styles.

   This is meant to be read-only. Use `.reload_library` to update.
