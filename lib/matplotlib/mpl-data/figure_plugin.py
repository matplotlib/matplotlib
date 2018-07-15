"""
Plugin for drag and drop matplotlib.Figure in QtDesigner
"""
import os.path

from matplotlib.backends.backend_qt5 import FigureCanvasQT
from matplotlib.figure import Figure

# Pyside and Pyside2 do not support the QtDesigner functionality that is
# contained in both PyQt4 and PyQt5. This feature will not be supported with
# those backends until this feature set is available in those libraries
from matplotlib.backends.qt_compat import QtDesigner, QtGui


class FigureDesignerPlugin(QtDesigner.QPyDesignerCustomWidgetPlugin):
    """
    QtDesigner Plugin for a matplotlib FigureCanvas

    Notes
    -----
    In order to load this plugin, set the ``PYQTDESIGNERPATH`` environment
    variable to the directory that contains this file.
    """
    def __init__(self):
        QtDesigner.QPyDesignerCustomWidgetPlugin.__init__(self)
        self.initialized = False

    def initialize(self, core):
        """Mark the QtDesigner plugin as initialized"""
        if self.initialized:
            return
        self.initialized = True

    def isInitialized(self):
        """Whether the widget has been initialized"""
        return self.initialized

    def createWidget(self, parent):
        """Create a FigureCanvasQT instance"""
        # Create the Canvas with a new Figure
        fig = FigureCanvasQT(Figure())
        # Set the parent of the newly created widget
        fig.setParent(parent)
        return fig

    def name(self):
        """Name of plugin displayed in QtDesigner"""
        return "FigureCanvasQT"

    def group(self):
        """Name of plugin group header in QtDesigner"""
        return "Matplotlib Widgets"

    def isContainer(self):
        """Whether to allow  widgets to be dragged in inside QtCanvas"""
        # Someday we may want to set this to True if we can drop in curve
        # objects, but this first draft will not include that functionality
        return False

    def toolTip(self):
        """Short description of Widget"""
        return "A matplotlib FigureCanvas"

    def whatsThis(self):
        """Long explanation of Widget"""
        return self.__doc__

    def icon(self):
        """Icon displayed alongside Widget selection"""
        mpl_data = os.path.dirname(__file__)
        mpl_icon = os.path.join(mpl_data, 'images/matplotlib_large.png')
        return QtGui.QIcon(mpl_icon)

    def domXml(self):
        """XML Description of the widget's properties"""
        return (
                "<widget class=\"{0}\" name=\"{0}\">\n"
                " <property name=\"toolTip\" >\n"
                "  <string>{1}</string>\n"
                " </property>\n"
                " <property name=\"whatsThis\" >\n"
                "  <string>{2}</string>\n"
                " </property>\n"
                "</widget>\n"
               ).format(self.name(), self.toolTip(), self.whatsThis())

    def includeFile(self):
        """Include a link to this file for reference"""
        return FigureCanvasQT.__module__
