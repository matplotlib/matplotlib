:mod:`.backend_qtagg`, :mod:`.backend_qtcairo`
==============================================

**NOTE** These backends are not (auto) documented here, to avoid adding a
dependency to building the docs.

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
<https://doc.qt.io/qtforpython/contents.html>`_ for Qt6 [#]_.  While both PyQt
and Qt for Python (aka PySide) closely mirror the underlying C++ API they are
wrapping, they are not drop-in replacements for each other [#]_.  To account
for this, Matplotlib has an internal API compatibility layer in
`matplotlib.backends.qt_compat` which covers our needs.  Despite being a public
module, we do not consider this to be a stable user-facing API and it may
change without warning [#]_.

Previously Matplotlib's Qt backends had the Qt version number in the name, both
in the module and the :rc:`backend` value
(e.g. ``matplotlib.backends.backend_qt4agg`` and
``matplotlib.backends.backend_qt5agg``). However as part of adding support for
Qt6 we were able to support both Qt5 and Qt6 with a single implementation with
all of the Qt version and binding support handled in
`~matplotlib.backends.qt_compat`.  A majority of the renderer agnostic Qt code
is now in `matplotlib.backends.backend_qt` with specialization for AGG in
``backend_qtagg`` and cairo in ``backend_qtcairo``.

The binding is selected at run time based on what bindings are already imported
(by checking for the ``QtCore`` sub-package), then by the :envvar:`QT_API`
environment variable, and finally by the :rc:`backend`.  In all cases when we
need to search, the order is ``PyQt6``, ``PySide6``, ``PyQt5``, ``PySide2``.
See :ref:`QT_API-usage` for usage instructions.

The ``backend_qt5``, ``backend_qt5agg``, and ``backend_qt5cairo`` are provided
and force the use of a Qt5 binding for backwards compatibility.  Their use is
discouraged (but not deprecated) and ``backend_qt``, ``backend_qtagg``, or
``backend_qtcairo`` should be preferred instead.  However, these modules will
not be deprecated until we drop support for Qt5.




.. [#] There is also `PyQt4
       <https://www.riverbankcomputing.com/static/Docs/PyQt4/>`_ and `PySide
       <https://srinikom.github.io/pyside-docs/>`_ for Qt4 but these are no
       longer supported by Matplotlib and upstream support for Qt4 ended
       in 2015.
.. [#] Despite the slight API differences, the more important distinction
       between the PyQt and Qt for Python series of bindings is licensing.
.. [#] If you are looking for a general purpose compatibility library please
       see `qtpy <https://github.com/spyder-ide/qtpy>`_.
