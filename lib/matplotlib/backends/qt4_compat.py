""" A Qt API selector that can be used to switch between PyQt and PySide.
"""

import os
import warnings
from matplotlib import rcParams

# Available APIs.
QT_API_PYQT = 'PyQt4'
QT_API_PYSIDE = 'PySide'

# Select Qt binding, using the rcParams variable if available.
QT_API = rcParams.setdefault('backend.qt4', QT_API_PYQT)

# We will define an appropriate wrapper for the differing versions
# of file dialog.
_getSaveFileName = None

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

    import sip
    try :
        if sip.getapi("QString") > 1 :
            # Use new getSaveFileNameAndFilter()
            _get_save = QtGui.QFileDialog.getSaveFileNameAndFilter
        else :
            # Use old getSaveFileName()
            _getSaveFileName = QtGui.QFileDialog.getSaveFileName
    except (AttributeError, KeyError) :
        # call to getapi() can fail in older versions of sip
        _getSaveFileName = QtGui.QFileDialog.getSaveFileName

elif QT_API == QT_API_PYSIDE:
    from PySide import QtCore, QtGui, __version__, __version_info__
    if __version_info__ < (1,0,3):
        raise ImportError(
            "Matplotlib backend_qt4 and backend_qt4agg require PySide >=1.0.3")

    _get_save = QtGui.QFileDialog.getSaveFileName

else:
    raise RuntimeError('Invalid Qt API %r, valid values are: %r or %r' %
                       (QT_API, QT_API_PYQT, QT_API_PYSIDE))

if _getSaveFileName is None:

    def _getSaveFileName(self, msg, start, filters, selectedFilter):
        return _get_save(self, msg, start, filters, selectedFilter)[0]

