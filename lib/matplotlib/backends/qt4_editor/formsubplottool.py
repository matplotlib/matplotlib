# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lib/matplotlib/backends/qt4_editor/subplot_conftool.ui'
#
# Created: Mon Jun  3 01:47:41 2013
#      by: PyQt4 UI code generator 4.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_SubplotTool(object):
    def setupUi(self, SubplotTool):
        SubplotTool.setObjectName(_fromUtf8("SubplotTool"))
        SubplotTool.resize(447, 265)
        self.horizontalLayout = QtGui.QHBoxLayout(SubplotTool)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.tightLayout = QtGui.QPushButton(SubplotTool)
        self.tightLayout.setObjectName(_fromUtf8("tightLayout"))
        self.horizontalLayout_2.addWidget(self.tightLayout)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.resetButton = QtGui.QPushButton(SubplotTool)
        self.resetButton.setObjectName(_fromUtf8("resetButton"))
        self.horizontalLayout_2.addWidget(self.resetButton)
        spacerItem1 = QtGui.QSpacerItem(5, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.doneButton = QtGui.QPushButton(SubplotTool)
        self.doneButton.setEnabled(True)
        self.doneButton.setFlat(False)
        self.doneButton.setObjectName(_fromUtf8("doneButton"))
        self.horizontalLayout_2.addWidget(self.doneButton)
        self.gridLayout.addLayout(self.horizontalLayout_2, 8, 0, 1, 1)
        self.groupBox = QtGui.QGroupBox(SubplotTool)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.verticalLayout = QtGui.QVBoxLayout(self.groupBox)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.hboxtop = QtGui.QHBoxLayout()
        self.hboxtop.setObjectName(_fromUtf8("hboxtop"))
        self.labeltop = QtGui.QLabel(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labeltop.sizePolicy().hasHeightForWidth())
        self.labeltop.setSizePolicy(sizePolicy)
        self.labeltop.setMinimumSize(QtCore.QSize(50, 0))
        self.labeltop.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labeltop.setObjectName(_fromUtf8("labeltop"))
        self.hboxtop.addWidget(self.labeltop)
        self.slidertop = QtGui.QSlider(self.groupBox)
        self.slidertop.setMouseTracking(False)
        self.slidertop.setProperty("value", 0)
        self.slidertop.setOrientation(QtCore.Qt.Horizontal)
        self.slidertop.setInvertedAppearance(False)
        self.slidertop.setInvertedControls(False)
        self.slidertop.setTickPosition(QtGui.QSlider.TicksAbove)
        self.slidertop.setTickInterval(100)
        self.slidertop.setObjectName(_fromUtf8("slidertop"))
        self.hboxtop.addWidget(self.slidertop)
        self.topvalue = QtGui.QLabel(self.groupBox)
        self.topvalue.setMinimumSize(QtCore.QSize(30, 0))
        self.topvalue.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.topvalue.setObjectName(_fromUtf8("topvalue"))
        self.hboxtop.addWidget(self.topvalue)
        self.verticalLayout.addLayout(self.hboxtop)
        self.hboxbottom = QtGui.QHBoxLayout()
        self.hboxbottom.setObjectName(_fromUtf8("hboxbottom"))
        self.labelbottom = QtGui.QLabel(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelbottom.sizePolicy().hasHeightForWidth())
        self.labelbottom.setSizePolicy(sizePolicy)
        self.labelbottom.setMinimumSize(QtCore.QSize(50, 0))
        self.labelbottom.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelbottom.setObjectName(_fromUtf8("labelbottom"))
        self.hboxbottom.addWidget(self.labelbottom)
        self.sliderbottom = QtGui.QSlider(self.groupBox)
        self.sliderbottom.setOrientation(QtCore.Qt.Horizontal)
        self.sliderbottom.setInvertedAppearance(False)
        self.sliderbottom.setTickPosition(QtGui.QSlider.TicksAbove)
        self.sliderbottom.setTickInterval(100)
        self.sliderbottom.setObjectName(_fromUtf8("sliderbottom"))
        self.hboxbottom.addWidget(self.sliderbottom)
        self.bottomvalue = QtGui.QLabel(self.groupBox)
        self.bottomvalue.setMinimumSize(QtCore.QSize(30, 0))
        self.bottomvalue.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.bottomvalue.setObjectName(_fromUtf8("bottomvalue"))
        self.hboxbottom.addWidget(self.bottomvalue)
        self.verticalLayout.addLayout(self.hboxbottom)
        self.hboxleft = QtGui.QHBoxLayout()
        self.hboxleft.setObjectName(_fromUtf8("hboxleft"))
        self.labelleft = QtGui.QLabel(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelleft.sizePolicy().hasHeightForWidth())
        self.labelleft.setSizePolicy(sizePolicy)
        self.labelleft.setMinimumSize(QtCore.QSize(50, 0))
        self.labelleft.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelleft.setObjectName(_fromUtf8("labelleft"))
        self.hboxleft.addWidget(self.labelleft)
        self.sliderleft = QtGui.QSlider(self.groupBox)
        self.sliderleft.setOrientation(QtCore.Qt.Horizontal)
        self.sliderleft.setInvertedAppearance(False)
        self.sliderleft.setTickPosition(QtGui.QSlider.TicksAbove)
        self.sliderleft.setTickInterval(100)
        self.sliderleft.setObjectName(_fromUtf8("sliderleft"))
        self.hboxleft.addWidget(self.sliderleft)
        self.leftvalue = QtGui.QLabel(self.groupBox)
        self.leftvalue.setMinimumSize(QtCore.QSize(30, 0))
        self.leftvalue.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.leftvalue.setObjectName(_fromUtf8("leftvalue"))
        self.hboxleft.addWidget(self.leftvalue)
        self.verticalLayout.addLayout(self.hboxleft)
        self.hboxright = QtGui.QHBoxLayout()
        self.hboxright.setObjectName(_fromUtf8("hboxright"))
        self.labelright = QtGui.QLabel(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelright.sizePolicy().hasHeightForWidth())
        self.labelright.setSizePolicy(sizePolicy)
        self.labelright.setMinimumSize(QtCore.QSize(50, 0))
        self.labelright.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelright.setObjectName(_fromUtf8("labelright"))
        self.hboxright.addWidget(self.labelright)
        self.sliderright = QtGui.QSlider(self.groupBox)
        self.sliderright.setOrientation(QtCore.Qt.Horizontal)
        self.sliderright.setInvertedAppearance(False)
        self.sliderright.setTickPosition(QtGui.QSlider.TicksAbove)
        self.sliderright.setTickInterval(100)
        self.sliderright.setObjectName(_fromUtf8("sliderright"))
        self.hboxright.addWidget(self.sliderright)
        self.rightvalue = QtGui.QLabel(self.groupBox)
        self.rightvalue.setMinimumSize(QtCore.QSize(30, 0))
        self.rightvalue.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.rightvalue.setObjectName(_fromUtf8("rightvalue"))
        self.hboxright.addWidget(self.rightvalue)
        self.verticalLayout.addLayout(self.hboxright)
        self.gridLayout.addWidget(self.groupBox, 5, 0, 1, 1)
        self.groupBox_2 = QtGui.QGroupBox(SubplotTool)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setMargin(0)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.hboxhspace = QtGui.QHBoxLayout()
        self.hboxhspace.setObjectName(_fromUtf8("hboxhspace"))
        self.labelhspace = QtGui.QLabel(self.groupBox_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelhspace.sizePolicy().hasHeightForWidth())
        self.labelhspace.setSizePolicy(sizePolicy)
        self.labelhspace.setMinimumSize(QtCore.QSize(50, 0))
        self.labelhspace.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelhspace.setObjectName(_fromUtf8("labelhspace"))
        self.hboxhspace.addWidget(self.labelhspace)
        self.sliderhspace = QtGui.QSlider(self.groupBox_2)
        self.sliderhspace.setOrientation(QtCore.Qt.Horizontal)
        self.sliderhspace.setInvertedAppearance(False)
        self.sliderhspace.setTickPosition(QtGui.QSlider.TicksAbove)
        self.sliderhspace.setTickInterval(100)
        self.sliderhspace.setObjectName(_fromUtf8("sliderhspace"))
        self.hboxhspace.addWidget(self.sliderhspace)
        self.hspacevalue = QtGui.QLabel(self.groupBox_2)
        self.hspacevalue.setMinimumSize(QtCore.QSize(30, 0))
        self.hspacevalue.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.hspacevalue.setObjectName(_fromUtf8("hspacevalue"))
        self.hboxhspace.addWidget(self.hspacevalue)
        self.verticalLayout_2.addLayout(self.hboxhspace)
        self.hboxwspace = QtGui.QHBoxLayout()
        self.hboxwspace.setObjectName(_fromUtf8("hboxwspace"))
        self.labelwspace = QtGui.QLabel(self.groupBox_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelwspace.sizePolicy().hasHeightForWidth())
        self.labelwspace.setSizePolicy(sizePolicy)
        self.labelwspace.setMinimumSize(QtCore.QSize(50, 0))
        self.labelwspace.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelwspace.setObjectName(_fromUtf8("labelwspace"))
        self.hboxwspace.addWidget(self.labelwspace)
        self.sliderwspace = QtGui.QSlider(self.groupBox_2)
        self.sliderwspace.setTracking(True)
        self.sliderwspace.setOrientation(QtCore.Qt.Horizontal)
        self.sliderwspace.setInvertedAppearance(False)
        self.sliderwspace.setTickPosition(QtGui.QSlider.TicksAbove)
        self.sliderwspace.setTickInterval(100)
        self.sliderwspace.setObjectName(_fromUtf8("sliderwspace"))
        self.hboxwspace.addWidget(self.sliderwspace)
        self.wspacevalue = QtGui.QLabel(self.groupBox_2)
        self.wspacevalue.setMinimumSize(QtCore.QSize(30, 0))
        self.wspacevalue.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.wspacevalue.setObjectName(_fromUtf8("wspacevalue"))
        self.hboxwspace.addWidget(self.wspacevalue)
        self.verticalLayout_2.addLayout(self.hboxwspace)
        self.gridLayout.addWidget(self.groupBox_2, 6, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)

        self.retranslateUi(SubplotTool)
        QtCore.QObject.connect(self.doneButton, QtCore.SIGNAL(_fromUtf8("clicked()")), SubplotTool.accept)
        QtCore.QMetaObject.connectSlotsByName(SubplotTool)

    def retranslateUi(self, SubplotTool):
        SubplotTool.setWindowTitle(_translate("SubplotTool", "Dialog", None))
        self.tightLayout.setText(_translate("SubplotTool", "tight layout", None))
        self.resetButton.setText(_translate("SubplotTool", "reset", None))
        self.doneButton.setText(_translate("SubplotTool", "close", None))
        self.groupBox.setTitle(_translate("SubplotTool", "Borders", None))
        self.labeltop.setText(_translate("SubplotTool", "top", None))
        self.topvalue.setText(_translate("SubplotTool", "0", None))
        self.labelbottom.setText(_translate("SubplotTool", "bottom", None))
        self.bottomvalue.setText(_translate("SubplotTool", "0", None))
        self.labelleft.setText(_translate("SubplotTool", "left", None))
        self.leftvalue.setText(_translate("SubplotTool", "0", None))
        self.labelright.setText(_translate("SubplotTool", "right", None))
        self.rightvalue.setText(_translate("SubplotTool", "0", None))
        self.groupBox_2.setTitle(_translate("SubplotTool", "Spaces", None))
        self.labelhspace.setText(_translate("SubplotTool", "hspace", None))
        self.hspacevalue.setText(_translate("SubplotTool", "0", None))
        self.labelwspace.setText(_translate("SubplotTool", "wspace", None))
        self.wspacevalue.setText(_translate("SubplotTool", "0", None))
