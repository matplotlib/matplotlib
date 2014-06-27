""" A Qt API selector that can be used to switch between PyQt and PySide.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
from matplotlib import rcParams, verbose

# Available APIs.
QT_API_PYQT = 'PyQt4'       # API is not set here; Python 2.x default is V 1
QT_API_PYQTv2 = 'PyQt4v2'   # forced to Version 2 API
QT_API_PYSIDE = 'PySide'    # only supports Version 2 API
QT_API_PYQT5 = 'PyQt5'       # use PyQt5 API; Version 2 with module shim

ETS = dict(pyqt=QT_API_PYQTv2, pyside=QT_API_PYSIDE, pyqt5=QT_API_PYQT5)

# If the ETS QT_API environment variable is set, use it.  Note that
# ETS requires the version 2 of PyQt4, which is not the platform
# default for Python 2.x.

QT_API_ENV = os.environ.get('QT_API')
if QT_API_ENV is not None:
    try:
        QT_API = ETS[QT_API_ENV]
    except KeyError:
        raise RuntimeError(
          'Unrecognized environment variable %r, valid values are: %r or %r' %
          (QT_API_ENV, 'pyqt', 'pyside', 'pyqt5'))
else:
    # No ETS environment, so use rcParams.
    if rcParams['backend'] == 'Qt5Agg':
        QT_API = rcParams['backend.qt5']
    else:
        QT_API = rcParams['backend.qt4']

# We will define an appropriate wrapper for the differing versions
# of file dialog.
_getSaveFileName = None

# Flag to check if sip could be imported
_sip_imported = False

# Now perform the imports.
if QT_API in (QT_API_PYQT, QT_API_PYQTv2, QT_API_PYQT5):
    try:
        import sip
        _sip_imported = True
    except ImportError:
        # Try using PySide
        QT_API = QT_API_PYSIDE

if _sip_imported:
    if QT_API == QT_API_PYQTv2:
        if QT_API_ENV == 'pyqt':
            cond = ("Found 'QT_API=pyqt' environment variable. "
                    "Setting PyQt4 API accordingly.\n")
        else:
            cond = "PyQt API v2 specified."
        try:
            sip.setapi('QString', 2)
        except:
            res = 'QString API v2 specification failed. Defaulting to v1.'
            verbose.report(cond+res, 'helpful')
            # condition has now been reported, no need to repeat it:
            cond = ""
        try:
            sip.setapi('QVariant', 2)
        except:
            res = 'QVariant API v2 specification failed. Defaulting to v1.'
            verbose.report(cond+res, 'helpful')

    if QT_API in [QT_API_PYQT, QT_API_PYQTv2]:  # PyQt4 API

        from PyQt4 import QtCore, QtGui

        try:
            if sip.getapi("QString") > 1:
                # Use new getSaveFileNameAndFilter()
                _get_save = QtGui.QFileDialog.getSaveFileNameAndFilter
            else:
                # Use old getSaveFileName()
                _getSaveFileName = QtGui.QFileDialog.getSaveFileName
        except (AttributeError, KeyError):
            # call to getapi() can fail in older versions of sip
            _getSaveFileName = QtGui.QFileDialog.getSaveFileName

    else:  # PyQt5 API

        from PyQt5 import QtCore, QtGui, QtWidgets

        # Additional PyQt5 shimming to make it appear as for PyQt4

        _get_save = QtWidgets.QFileDialog.getSaveFileName
        _getSaveFileName = QtWidgets.QFileDialog.getSaveFileName

    # Alias PyQt-specific functions for PySide compatibility.
    QtCore.Signal = QtCore.pyqtSignal
    try:
        QtCore.Slot = QtCore.pyqtSlot
    except AttributeError:
        # Not a perfect match but works in simple cases
        QtCore.Slot = QtCore.pyqtSignature

    QtCore.Property = QtCore.pyqtProperty
    __version__ = QtCore.PYQT_VERSION_STR

else:  # try importing pyside
    from PySide import QtCore, QtGui, __version__, __version_info__
    if __version_info__ < (1, 0, 3):
        raise ImportError(
            "Matplotlib backend_qt4 and backend_qt4agg require PySide >=1.0.3")

    _get_save = QtGui.QFileDialog.getSaveFileName

if _getSaveFileName is None:
    def _getSaveFileName(self, msg, start, filters, selectedFilter):
        return _get_save(self, msg, start, filters, selectedFilter)[0]

# Apply shim to Qt4 APIs to make them look like Qt5
if QT_API in (QT_API_PYQT, QT_API_PYQTv2, QT_API_PYSIDE):
    '''Import all used QtGui objects into QtWidgets

    Here I've opted to simple copy QtGui into QtWidgets as that
    achieves the same result as copying over the objects, and will
    continue to work if other objects are used.

    '''
    QtWidgets = QtGui
