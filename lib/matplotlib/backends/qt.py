""" A Qt API selector that can be used to switch between PyQt and PySide.
"""

import os

# Available APIs.
QT_API_PYQT = 'pyqt'
QT_API_PYSIDE = 'pyside'

# Use PyQt by default until PySide is stable.
QT_API = os.environ.get('QT_API', QT_API_PYQT)

if QT_API == QT_API_PYQT:
    from PyQt4 import QtCore, QtGui

    # Alias PyQt-specific functions for PySide compatibility.
    try:
        QtCore.Slot = QtCore.pyqtSlot
    except AttributeError:
        QtCore.Slot = pyqtSignature # Not a perfect match but 
                                    # works in simple cases
    QtCore.Property = QtCore.pyqtProperty
    
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
    from PySide import QtCore, QtGui
    
    # Alias PySide-specific function for PyQt compatibilty
    QtCore.pyqtProperty = QtCore.Property
    QtCore.pyqtSignature = QtCore.Slot # Not a perfect match but 
                                       # works in simple cases
    
    _getSaveFileName = lambda self, msg, start, filters, selectedFilter : \
                        QtGui.QFileDialog.getSaveFileName(self,  \
                        msg, start, filters, #selectedFilter
                        )[0]              #Commmented out due to PySide bug 819
                        
    # Fix for PySide bug 489 - Remove when fixed
    class QImage(QtGui.QImage):
        def __init__(self,data,width,height,format,*args,**kwargs):
            super(QImage,self).__init__(buffer(data),width,height,
                                        format,*args,**kwargs)
    QtGui.QImage = QImage
    
else:
    raise RuntimeError('Invalid Qt API %r, valid values are: %r or %r' %
                       (QT_API, QT_API_PYQT, QT_API_PYSIDE))