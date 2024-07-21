API Changes for 3.9.1
=====================

Development
-----------

Documentation-specific custom Sphinx roles are now semi-public
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For third-party packages that derive types from Matplotlib, our use of custom roles may
prevent Sphinx from building their docs. These custom Sphinx roles are now public solely
for the purposes of use within projects that derive from Matplotlib types. See
:mod:`matplotlib.sphinxext.roles` for details.
