""" A Qt API selector that can be used to switch between PyQt and PySide.
"""

import os
import warnings

# Available APIs.
QT_API_PYQT = 'pyqt'
QT_API_PYSIDE = 'pyside'

def prepare_pyqt4():
    # For PySide compatibility, use the new-style string API that automatically
    # converts QStrings to Unicode Python strings. Also, automatically unpack
    # QVariants to their underlying objects.
    import sip
    sip.setapi('QString', 2)
    sip.setapi('QVariant', 2)

# Select Qt binding, using the QT_API environment variable if available.
QT_API = os.environ.get('QT_API')
if QT_API is None:
    try:
        import PySide
        if PySide.__version_info__ < (1,0,3):
            warnings.warn("PySide found with version < 1.0.3; trying PyQt4")
            raise ImportError
        QT_API = QT_API_PYSIDE
    except ImportError:
        try:
            prepare_pyqt4()
            import PyQt4
            QT_API = QT_API_PYQT
        except ImportError:
            raise ImportError('Cannot import PySide or PyQt4')

elif QT_API == QT_API_PYQT:
    # Note: This must be called *before* PyQt4 is imported.
    prepare_pyqt4()

# Now perform the imports.
if QT_API == QT_API_PYQT:
    from PyQt4 import QtCore, QtGui, QtSvg

    # Alias PyQt-specific functions for PySide compatibility.
    QtCore.Signal = QtCore.pyqtSignal
    try:
        QtCore.Slot = QtCore.pyqtSlot
    except AttributeError:
        QtCore.Slot = pyqtSignature # Not a perfect match but
                                    # works in simple cases
    QtCore.Property = QtCore.pyqtProperty
    __version__ = QtCore.PYQT_VERSION_STR

    # Use new getSaveFileNameAndFilter()
    _getSaveFileName = lambda self, msg, start, filters, \
                              selectedFilter : \
                        QtGui.QFileDialog.getSaveFileNameAndFilter( \
                        self, msg, start, filters, selectedFilter)[0]



elif QT_API == QT_API_PYSIDE:
    from PySide import QtCore, QtGui, __version__, __version_info__
    if __version_info__ < (1,0,3):
        raise ImportError(
            "Matplotlib backend_qt4 and backend_qt4agg require PySide >=1.0.3")

else:
    raise RuntimeError('Invalid Qt API %r, valid values are: %r or %r' %
                       (QT_API, QT_API_PYQT, QT_API_PYSIDE))
