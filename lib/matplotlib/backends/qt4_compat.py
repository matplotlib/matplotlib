""" A Qt API selector that can be used to switch between PyQt and PySide.
"""

import os

# Available APIs.
QT_API_PYQT = 'pyqt'
QT_API_PYSIDE = 'pyside'

# Select PyQt4 or PySide using an environment variable, in the same way IPython does.
# IPython is using PyQt as default (for now) so we will too.
QT_API = os.environ.get('QT_API', QT_API_PYQT)

if QT_API == QT_API_PYQT:
    try:
        from PyQt4 import QtCore, QtGui
    except ImportError:
        raise ImportError("Qt4 backend requires that PyQt4 is installed.")
    # Alias PyQt-specific functions for PySide compatibility.
    try:
        QtCore.Slot = QtCore.pyqtSlot
    except AttributeError:
        QtCore.Slot = pyqtSignature # Not a perfect match but 
                                    # works in simple cases
    QtCore.Property = QtCore.pyqtProperty
    __version__ = QtCore.PYQT_VERSION_STR
    import sip
    try :
        if sip.getapi("QString") > 1 :
            # Use new getSaveFileNameAndFilter()
            _getSaveFileName = lambda self, msg, start, filters, \
                                      selectedFilter : \
                                QtGui.QFileDialog.getSaveFileNameAndFilter( \
                                self, msg, start, filters, selectedFilter)[0]
        else :
            # Use old getSaveFileName()
            _getSaveFileName = QtGui.QFileDialog.getSaveFileName
    except (AttributeError, KeyError) :
        # call to getapi() can fail in older versions of sip
        # Use the old getSaveFileName()
        _getSaveFileName = QtGui.QFileDialog.getSaveFileName

elif QT_API == QT_API_PYSIDE:
    try:
        from PySide import QtCore, QtGui, __version__, __version_info__
    except ImportError:
        raise ImportError("Qt4 backend requires that PySide is installed.")
    if __version_info__ < (1,0,3):
        raise ImportError("Matplotlib backend_qt4 and backend_qt4agg require PySide >=1.0.3")

    # Alias PySide-specific function for PyQt compatibilty
    QtCore.pyqtProperty = QtCore.Property
    QtCore.pyqtSignature = QtCore.Slot # Not a perfect match but 
                                       # works in simple cases

    _getSaveFileName = lambda self, msg, start, filters, selectedFilter : \
                        QtGui.QFileDialog.getSaveFileName(self,  \
                        msg, start, filters, selectedFilter)[0]
else:
    raise RuntimeError('Invalid Qt API %r, valid values are: %r or %r' %
                       (QT_API, QT_API_PYQT, QT_API_PYSIDE))