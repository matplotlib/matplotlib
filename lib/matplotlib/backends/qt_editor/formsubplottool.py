# -*- coding: utf-8 -*-
"""
formsubplottool.py

backend.qt4 (PyQt4|PySide) independent form of the subplot tool.

"""

__author__ = 'rudolf.hoefler@gmail.com'

from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets


class UiSubplotTool(QtWidgets.QDialog):

    def __init__(self, *args, **kwargs):
        super(UiSubplotTool, self).__init__(*args, **kwargs)
        self.setObjectName('SubplotTool')
        self.resize(450, 265)

        gbox = QtWidgets.QGridLayout(self)
        self.setLayout(gbox)

        # groupbox borders
        groupbox = QtWidgets.QGroupBox('Borders', self)
        gbox.addWidget(groupbox, 6, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout(groupbox)
        self.verticalLayout.setSpacing(0)

        # slider top
        self.hboxtop = QtWidgets.QHBoxLayout()
        self.labeltop = QtWidgets.QLabel('top', self)
        self.labeltop.setMinimumSize(QtCore.QSize(50, 0))
        self.labeltop.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.slidertop = QtWidgets.QSlider(self)
        self.slidertop.setMouseTracking(False)
        self.slidertop.setProperty("value", 0)
        self.slidertop.setOrientation(QtCore.Qt.Horizontal)
        self.slidertop.setInvertedAppearance(False)
        self.slidertop.setInvertedControls(False)
        self.slidertop.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slidertop.setTickInterval(100)

        self.topvalue = QtWidgets.QLabel('0', self)
        self.topvalue.setMinimumSize(QtCore.QSize(30, 0))
        self.topvalue.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.verticalLayout.addLayout(self.hboxtop)
        self.hboxtop.addWidget(self.labeltop)
        self.hboxtop.addWidget(self.slidertop)
        self.hboxtop.addWidget(self.topvalue)

        # slider bottom
        hboxbottom = QtWidgets.QHBoxLayout()
        labelbottom = QtWidgets.QLabel('bottom', self)
        labelbottom.setMinimumSize(QtCore.QSize(50, 0))
        labelbottom.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.sliderbottom = QtWidgets.QSlider(self)
        self.sliderbottom.setMouseTracking(False)
        self.sliderbottom.setProperty("value", 0)
        self.sliderbottom.setOrientation(QtCore.Qt.Horizontal)
        self.sliderbottom.setInvertedAppearance(False)
        self.sliderbottom.setInvertedControls(False)
        self.sliderbottom.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.sliderbottom.setTickInterval(100)

        self.bottomvalue = QtWidgets.QLabel('0', self)
        self.bottomvalue.setMinimumSize(QtCore.QSize(30, 0))
        self.bottomvalue.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.verticalLayout.addLayout(hboxbottom)
        hboxbottom.addWidget(labelbottom)
        hboxbottom.addWidget(self.sliderbottom)
        hboxbottom.addWidget(self.bottomvalue)

        # slider left
        hboxleft = QtWidgets.QHBoxLayout()
        labelleft = QtWidgets.QLabel('left', self)
        labelleft.setMinimumSize(QtCore.QSize(50, 0))
        labelleft.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.sliderleft = QtWidgets.QSlider(self)
        self.sliderleft.setMouseTracking(False)
        self.sliderleft.setProperty("value", 0)
        self.sliderleft.setOrientation(QtCore.Qt.Horizontal)
        self.sliderleft.setInvertedAppearance(False)
        self.sliderleft.setInvertedControls(False)
        self.sliderleft.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.sliderleft.setTickInterval(100)

        self.leftvalue = QtWidgets.QLabel('0', self)
        self.leftvalue.setMinimumSize(QtCore.QSize(30, 0))
        self.leftvalue.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.verticalLayout.addLayout(hboxleft)
        hboxleft.addWidget(labelleft)
        hboxleft.addWidget(self.sliderleft)
        hboxleft.addWidget(self.leftvalue)

        # slider right
        hboxright = QtWidgets.QHBoxLayout()
        self.labelright = QtWidgets.QLabel('right', self)
        self.labelright.setMinimumSize(QtCore.QSize(50, 0))
        self.labelright.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.sliderright = QtWidgets.QSlider(self)
        self.sliderright.setMouseTracking(False)
        self.sliderright.setProperty("value", 0)
        self.sliderright.setOrientation(QtCore.Qt.Horizontal)
        self.sliderright.setInvertedAppearance(False)
        self.sliderright.setInvertedControls(False)
        self.sliderright.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.sliderright.setTickInterval(100)

        self.rightvalue = QtWidgets.QLabel('0', self)
        self.rightvalue.setMinimumSize(QtCore.QSize(30, 0))
        self.rightvalue.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.verticalLayout.addLayout(hboxright)
        hboxright.addWidget(self.labelright)
        hboxright.addWidget(self.sliderright)
        hboxright.addWidget(self.rightvalue)

        # groupbox spacings
        groupbox = QtWidgets.QGroupBox('Spacings', self)
        gbox.addWidget(groupbox, 7, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout(groupbox)
        self.verticalLayout.setSpacing(0)

        # slider hspace
        hboxhspace = QtWidgets.QHBoxLayout()
        self.labelhspace = QtWidgets.QLabel('hspace', self)
        self.labelhspace.setMinimumSize(QtCore.QSize(50, 0))
        self.labelhspace.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.sliderhspace = QtWidgets.QSlider(self)
        self.sliderhspace.setMouseTracking(False)
        self.sliderhspace.setProperty("value", 0)
        self.sliderhspace.setOrientation(QtCore.Qt.Horizontal)
        self.sliderhspace.setInvertedAppearance(False)
        self.sliderhspace.setInvertedControls(False)
        self.sliderhspace.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.sliderhspace.setTickInterval(100)

        self.hspacevalue = QtWidgets.QLabel('0', self)
        self.hspacevalue.setMinimumSize(QtCore.QSize(30, 0))
        self.hspacevalue.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.verticalLayout.addLayout(hboxhspace)
        hboxhspace.addWidget(self.labelhspace)
        hboxhspace.addWidget(self.sliderhspace)
        hboxhspace.addWidget(self.hspacevalue)  # slider hspace

        # slider wspace
        hboxwspace = QtWidgets.QHBoxLayout()
        self.labelwspace = QtWidgets.QLabel('wspace', self)
        self.labelwspace.setMinimumSize(QtCore.QSize(50, 0))
        self.labelwspace.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.sliderwspace = QtWidgets.QSlider(self)
        self.sliderwspace.setMouseTracking(False)
        self.sliderwspace.setProperty("value", 0)
        self.sliderwspace.setOrientation(QtCore.Qt.Horizontal)
        self.sliderwspace.setInvertedAppearance(False)
        self.sliderwspace.setInvertedControls(False)
        self.sliderwspace.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.sliderwspace.setTickInterval(100)

        self.wspacevalue = QtWidgets.QLabel('0', self)
        self.wspacevalue.setMinimumSize(QtCore.QSize(30, 0))
        self.wspacevalue.setAlignment(
            QtCore.Qt.AlignRight |
            QtCore.Qt.AlignTrailing |
            QtCore.Qt.AlignVCenter)

        self.verticalLayout.addLayout(hboxwspace)
        hboxwspace.addWidget(self.labelwspace)
        hboxwspace.addWidget(self.sliderwspace)
        hboxwspace.addWidget(self.wspacevalue)

        # button bar
        hbox2 = QtWidgets.QHBoxLayout()
        gbox.addLayout(hbox2, 8, 0, 1, 1)
        self.tightlayout = QtWidgets.QPushButton('Tight Layout', self)
        spacer = QtWidgets.QSpacerItem(
            5, 20, QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Minimum)
        self.resetbutton = QtWidgets.QPushButton('Reset', self)
        self.donebutton = QtWidgets.QPushButton('Close', self)
        self.donebutton.setFocus()
        hbox2.addWidget(self.tightlayout)
        hbox2.addItem(spacer)
        hbox2.addWidget(self.resetbutton)
        hbox2.addWidget(self.donebutton)

        self.donebutton.clicked.connect(self.accept)
