******************************************************************************
``matplotlib.backends.backend_qtagg``, ``matplotlib.backends.backend_qtcairo``
******************************************************************************

**NOTE** These :ref:`backends` are not (auto) documented here, to avoid adding
a dependency to building the docs.

.. redirect-from:: /api/backend_qt4agg_api
.. redirect-from:: /api/backend_qt4cairo_api
.. redirect-from:: /api/backend_qt5agg_api
.. redirect-from:: /api/backend_qt5cairo_api

.. module:: matplotlib.backends.qt_compat
.. module:: matplotlib.backends.backend_qt
.. module:: matplotlib.backends.backend_qtagg
.. module:: matplotlib.backends.backend_qtcairo
.. module:: matplotlib.backends.backend_qt5agg
.. module:: matplotlib.backends.backend_qt5cairo

.. _QT_bindings:

Qt Bindings
-----------

There are currently 2 actively supported Qt versions, Qt5 and Qt6, and two
supported Python bindings per version -- `PyQt5
<https://www.riverbankcomputing.com/static/Docs/PyQt5/>`_ and `PySide2
<https://doc.qt.io/qtforpython-5/contents.html>`_ for Qt5 and `PyQt6
<https://www.riverbankcomputing.com/static/Docs/PyQt6/>`_ and `PySide6
<https://doc.qt.io/qtforpython/contents.html>`_ for Qt6 [#]_.  Matplotlib's
qtagg and qtcairo backends (``matplotlib.backends.backend_qtagg`` and
``matplotlib.backend.backend_qtcairo``) support all these bindings, with common
parts factored out in the ``matplotlib.backends.backend_qt`` module.

At runtime, these backends select the actual binding used as follows:

1. If a binding's ``QtCore`` subpackage is already imported, that binding is
   selected (the order for the check is ``PyQt6``, ``PySide6``, ``PyQt5``,
   ``PySide2``).
2. If the :envvar:`QT_API` environment variable is set to one of "PyQt6",
   "PySide6", "PyQt5", "PySide2" (case-insensitive), that binding is selected.
   (See also the documentation on :ref:`environment-variables`.)
3. Otherwise, the first available backend in the order ``PyQt6``, ``PySide6``,
   ``PyQt5``, ``PySide2`` is selected.

In the past, Matplotlib used to have separate backends for each version of Qt
(e.g. qt4agg/``matplotlib.backends.backend_qt4agg`` and
qt5agg/``matplotlib.backends.backend_qt5agg``).  This scheme was dropped when
support for Qt6 was added.  For back-compatibility, qt5agg/``backend_qt5agg``
and qt5cairo/``backend_qt5cairo`` remain available; selecting one of these
backends forces the use of a Qt5 binding.  Their use is discouraged and
``backend_qtagg`` or ``backend_qtcairo`` should be preferred instead.  However,
these modules will not be deprecated until we drop support for Qt5.

While both PyQt
and Qt for Python (aka PySide) closely mirror the underlying C++ API they are
wrapping, they are not drop-in replacements for each other [#]_.  To account
for this, Matplotlib has an internal API compatibility layer in
`matplotlib.backends.qt_compat` which covers our needs.  Despite being a public
module, we do not consider this to be a stable user-facing API and it may
change without warning [#]_.

.. [#] There is also `PyQt4
       <https://www.riverbankcomputing.com/static/Docs/PyQt4/>`_ and `PySide
       <https://srinikom.github.io/pyside-docs/>`_ for Qt4 but these are no
       longer supported by Matplotlib and upstream support for Qt4 ended
       in 2015.
.. [#] Despite the slight API differences, the more important distinction
       between the PyQt and Qt for Python series of bindings is licensing.
.. [#] If you are looking for a general purpose compatibility library please
       see `qtpy <https://github.com/spyder-ide/qtpy>`_.
