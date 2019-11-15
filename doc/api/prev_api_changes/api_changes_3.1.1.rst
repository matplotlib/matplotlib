API Changes for 3.1.1
=====================

.. contents::
   :local:
   :depth: 1

Behavior changes
----------------

Locator.nonsingular return order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.Locator.nonsingular` (introduced in mpl 3.1) now returns a range ``v0, v1``
with ``v0 <= v1``.  This behavior is consistent with the implementation of
``nonsingular`` by the `.LogLocator` and `.LogitLocator` subclasses.
