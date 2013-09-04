# -*- coding: utf-8 -*-
"""
formlayout
==========

Module creating Qt form dialogs/layouts to edit various type of parameters


formlayout License Agreement (MIT License)
------------------------------------------

Copyright (c) 2009 Pierre Raybaut

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function

# History:
# 1.0.10: added float validator (disable "Ok" and "Apply" button when not valid)
# 1.0.7: added support for "Apply" button
# 1.0.6: code cleaning

__version__ = '1.0.10'
__license__ = __doc__

DEBUG = False

import sys
STDERR = sys.stderr

from matplotlib.backends.qt4_compat import QtGui,QtCore
from matplotlib.colors import rgb2hex

if not hasattr(QtGui,'QFormLayout'):
    raise ImportError, "Warning: formlayout requires PyQt4 >v4.3 or PySide"

(QWidget, QLineEdit, QComboBox, QLabel, QSpinBox, QIcon,QStyle,
 QDialogButtonBox, QHBoxLayout, QVBoxLayout, QDialog, QColor, QPushButton,
 QCheckBox, QColorDialog, QPixmap, QTabWidget, QApplication, QStackedWidget,
 QDateEdit, QDateTimeEdit, QFont, QFontComboBox, QFontDatabase, QGridLayout,
 QFormLayout, QDoubleValidator) =\
 (QtGui.QWidget, QtGui.QLineEdit, QtGui.QComboBox, QtGui.QLabel,
 QtGui.QSpinBox, QtGui.QIcon, QtGui.QStyle, QtGui.QDialogButtonBox,
 QtGui.QHBoxLayout, QtGui.QVBoxLayout, QtGui.QDialog, QtGui.QColor,
 QtGui.QPushButton, QtGui.QCheckBox, QtGui.QColorDialog, QtGui.QPixmap,
 QtGui.QTabWidget, QtGui.QApplication, QtGui.QStackedWidget, QtGui.QDateEdit,
 QtGui.QDateTimeEdit, QtGui.QFont, QtGui.QFontComboBox, QtGui.QFontDatabase,
 QtGui.QGridLayout, QtGui.QFormLayout, QtGui.QDoubleValidator)

(Qt, SIGNAL, SLOT, QObject, QSize,pyqtSignature, pyqtProperty) =\
(QtCore.Qt, QtCore.SIGNAL, QtCore.SLOT, QtCore.QObject, QtCore.QSize,
 QtCore.Slot, QtCore.Property)

import datetime

class ColorButton(QPushButton):
    """
    Color choosing push button
    """
    __pyqtSignals__ = ("colorChanged(QColor)",)

    def __init__(self, parent=None):
        QPushButton.__init__(self, parent)
        self.setFixedSize(20, 20)
        self.setIconSize(QSize(12, 12))
        self.connect(self, SIGNAL("clicked()"), self.choose_color)
        self._color = QColor()

    def choose_color(self):
        color = QColorDialog.getColor(self._color,self.parentWidget(),'')
        if color.isValid():
            self.set_color(color)

    def get_color(self):
        return self._color

    @QtCore.Slot("QColor")
    def set_color(self, color):
        if color != self._color:
            self._color = color
            self.emit(SIGNAL("colorChanged(QColor)"), self._color)
            pixmap = QPixmap(self.iconSize())
            pixmap.fill(color)
            self.setIcon(QtGui.QIcon(pixmap))

    color = QtCore.Property("QColor", get_color, set_color)

def col2hex(color):
    """Convert matplotlib color to hex before passing to Qt"""
    return rgb2hex(colorConverter.to_rgb(color))

def to_qcolor(color):
    """Create a QColor from a matplotlib color"""
    qcolor = QtGui.QColor()
    color = str(color)
    try:
        color = col2hex(color)
    except ValueError:
        #print('WARNING: ignoring invalid color %r' % color)
        return qcolor # return invalid QColor
    qcolor.setNamedColor(color) # set using hex color
    return qcolor # return valid QColor

def text_to_qcolor(text):
    """
    Create a QColor from specified string
    Avoid warning from Qt when an invalid QColor is instantiated
    """
    color = QColor()
    if isinstance(text, QObject):
        # actually a QString, which is not provided by the new PyQt4 API:
        text = str(text)
    if not isinstance(text, (unicode, str)):
        return color
    if text.startswith('#') and len(text)==7:
        correct = '#0123456789abcdef'
        for char in text:
            if char.lower() not in correct:
                return color
    elif text not in list(QColor.colorNames()):
        return color
    color.setNamedColor(text)
    return color

def is_matplotlib_color(value):
    """
    Check if value is a color passed to us from matplotlib.
    It could either be a valid color string or a 3-tuple of floats between 0. and 1.
    """
    if text_to_qcolor(value).isValid():
        return True
    if isinstance(value,tuple) and len(value)==3 and all(map(lambda v: isinstance(v,float),value)):
        for c in value:
            if c < 0. or c > 1.:
                return False
        return True
    return False

class ColorLayout(QHBoxLayout):
    """Color-specialized QLineEdit layout"""
    def __init__(self, color, parent=None):
        QHBoxLayout.__init__(self)
        assert isinstance(color, QColor)
        self.lineedit = QLineEdit(color.name(), parent)
        self.connect(self.lineedit, SIGNAL("textChanged(QString)"),
                     self.update_color)
        self.addWidget(self.lineedit)
        self.colorbtn = ColorButton(parent)
        self.colorbtn.color = color
        self.connect(self.colorbtn, SIGNAL("colorChanged(QColor)"),
                     self.update_text)
        self.addWidget(self.colorbtn)

    def update_color(self, text):
        color = text_to_qcolor(text)
        if color.isValid():
            self.colorbtn.color = color

    def update_text(self, color):
        self.lineedit.setText(color.name())

    def text(self):
        return self.lineedit.text()


def font_is_installed(font):
    """Check if font is installed"""
    return [fam for fam in QFontDatabase().families() if unicode(fam)==font]

def tuple_to_qfont(tup):
    """
    Create a QFont from tuple:
        (family [string], size [int], italic [bool], bold [bool])
    """
    if not isinstance(tup, tuple) or len(tup) != 4 \
       or not font_is_installed(tup[0]) \
       or not isinstance(tup[1], int) \
       or not isinstance(tup[2], bool) \
       or not isinstance(tup[3], bool):
        return None
    font = QFont()
    family, size, italic, bold = tup
    font.setFamily(family)
    font.setPointSize(size)
    font.setItalic(italic)
    font.setBold(bold)
    return font

def qfont_to_tuple(font):
    return (unicode(font.family()), int(font.pointSize()),
            font.italic(), font.bold())

class FontLayout(QGridLayout):
    """Font selection"""
    def __init__(self, value, parent=None):
        QGridLayout.__init__(self)
        font = tuple_to_qfont(value)
        assert font is not None

        # Font family
        self.family = QFontComboBox(parent)
        self.family.setCurrentFont(font)
        self.addWidget(self.family, 0, 0, 1, -1)

        # Font size
        self.size = QComboBox(parent)
        self.size.setEditable(True)
        sizelist = range(6, 12) + range(12, 30, 2) + [36, 48, 72]
        size = font.pointSize()
        if size not in sizelist:
            sizelist.append(size)
            sizelist.sort()
        self.size.addItems([str(s) for s in sizelist])
        self.size.setCurrentIndex(sizelist.index(size))
        self.addWidget(self.size, 1, 0)

        # Italic or not
        self.italic = QCheckBox(self.tr("Italic"), parent)
        self.italic.setChecked(font.italic())
        self.addWidget(self.italic, 1, 1)

        # Bold or not
        self.bold = QCheckBox(self.tr("Bold"), parent)
        self.bold.setChecked(font.bold())
        self.addWidget(self.bold, 1, 2)

    def get_font(self):
        font = self.family.currentFont()
        font.setItalic(self.italic.isChecked())
        font.setBold(self.bold.isChecked())
        font.setPointSize(int(self.size.currentText()))
        return qfont_to_tuple(font)


def is_edit_valid(edit):
    text = edit.text()
    state = edit.validator().validate(text, 0)[0]

    return state == QDoubleValidator.Acceptable

class FormWidget(QWidget):
    def __init__(self, data, comment="", parent=None):
        QWidget.__init__(self, parent)
        from copy import deepcopy
        self.data = deepcopy(data)
        self.widgets = []
        self.formlayout = QFormLayout(self)
        if comment:
            self.formlayout.addRow(QLabel(comment))
            self.formlayout.addRow(QLabel(" "))
        if DEBUG:
            print("\n"+("*"*80))
            print("DATA:", self.data)
            print("*"*80)
            print("COMMENT:", comment)
            print("*"*80)

    def get_dialog(self):
        """Return FormDialog instance"""
        dialog = self.parent()
        while not isinstance(dialog, QDialog):
            dialog = dialog.parent()
        return dialog

    def setup(self):
        for label, value in self.data:
            if DEBUG:
                print("value:", value)
            if label is None and value is None:
                # Separator: (None, None)
                self.formlayout.addRow(QLabel(" "), QLabel(" "))
                self.widgets.append(None)
                continue
            elif label is None:
                # Comment
                self.formlayout.addRow(QLabel(value))
                self.widgets.append(None)
                continue
            elif tuple_to_qfont(value) is not None:
                field = FontLayout(value, self)
            elif is_matplotlib_color(value):
                field = ColorLayout(QColor(value), self)
            elif isinstance(value, (str, unicode)):
                field = QLineEdit(value, self)
            elif isinstance(value, (list, tuple)):
                if isinstance(value, tuple):
                    value = list(value)
                selindex = value.pop(0)
                field = QComboBox(self)
                if isinstance(value[0], (list, tuple)):
                    keys = [ key for key, _val in value ]
                    value = [ val for _key, val in value ]
                else:
                    keys = value
                field.addItems(value)
                if selindex in value:
                    selindex = value.index(selindex)
                elif selindex in keys:
                    selindex = keys.index(selindex)
                elif not isinstance(selindex, int):
                    print("Warning: '%s' index is invalid (label: " \
                                    "%s, value: %s)" % (selindex, label, value), file=STDERR)
                    selindex = 0
                field.setCurrentIndex(selindex)
            elif isinstance(value, bool):
                field = QCheckBox(self)
                if value:
                    field.setCheckState(Qt.Checked)
                else :
                    field.setCheckState(Qt.Unchecked)
            elif isinstance(value, float):
                field = QLineEdit(repr(value), self)
                field.setValidator(QDoubleValidator(field))
                dialog = self.get_dialog()
                dialog.register_float_field(field)
                self.connect(field, SIGNAL('textChanged(QString)'),
                             lambda text: dialog.update_buttons())
            elif isinstance(value, int):
                field = QSpinBox(self)
                field.setRange(-1e9, 1e9)
                field.setValue(value)
            elif isinstance(value, datetime.datetime):
                field = QDateTimeEdit(self)
                field.setDateTime(value)
            elif isinstance(value, datetime.date):
                field = QDateEdit(self)
                field.setDate(value)
            else:
                field = QLineEdit(repr(value), self)
            self.formlayout.addRow(label, field)
            self.widgets.append(field)

    def get(self):
        valuelist = []
        for index, (label, value) in enumerate(self.data):
            field = self.widgets[index]
            if label is None:
                # Separator / Comment
                continue
            elif tuple_to_qfont(value) is not None:
                value = field.get_font()
            elif isinstance(value, (str, unicode)) or is_matplotlib_color(value):
                value = unicode(field.text())
            elif isinstance(value, (list, tuple)):
                index = int(field.currentIndex())
                if isinstance(value[0], (list, tuple)):
                    value = value[index][0]
                else:
                    value = value[index]
            elif isinstance(value, bool):
                value = field.checkState() == Qt.Checked
            elif isinstance(value, float):
                value = float(str(field.text()))
            elif isinstance(value, int):
                value = int(field.value())
            elif isinstance(value, datetime.datetime):
                value = field.dateTime().toPyDateTime()
            elif isinstance(value, datetime.date):
                value = field.date().toPyDate()
            else:
                value = eval(str(field.text()))
            valuelist.append(value)
        return valuelist


class FormComboWidget(QWidget):
    def __init__(self, datalist, comment="", parent=None):
        QWidget.__init__(self, parent)
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.combobox = QComboBox()
        layout.addWidget(self.combobox)

        self.stackwidget = QStackedWidget(self)
        layout.addWidget(self.stackwidget)
        self.connect(self.combobox, SIGNAL("currentIndexChanged(int)"),
                     self.stackwidget, SLOT("setCurrentIndex(int)"))

        self.widgetlist = []
        for data, title, comment in datalist:
            self.combobox.addItem(title)
            widget = FormWidget(data, comment=comment, parent=self)
            self.stackwidget.addWidget(widget)
            self.widgetlist.append(widget)

    def setup(self):
        for widget in self.widgetlist:
            widget.setup()

    def get(self):
        return [ widget.get() for widget in self.widgetlist]


class FormTabWidget(QWidget):
    def __init__(self, datalist, comment="", parent=None):
        QWidget.__init__(self, parent)
        layout = QVBoxLayout()
        self.tabwidget = QTabWidget()
        layout.addWidget(self.tabwidget)
        self.setLayout(layout)
        self.widgetlist = []
        for data, title, comment in datalist:
            if len(data[0])==3:
                widget = FormComboWidget(data, comment=comment, parent=self)
            else:
                widget = FormWidget(data, comment=comment, parent=self)
            index = self.tabwidget.addTab(widget, title)
            self.tabwidget.setTabToolTip(index, comment)
            self.widgetlist.append(widget)

    def setup(self):
        for widget in self.widgetlist:
            widget.setup()

    def get(self):
        return [ widget.get() for widget in self.widgetlist]


class FormDialog(QDialog):
    """Form Dialog"""
    def __init__(self, data, title="", comment="",
                 icon=None, parent=None, apply=None):
        QDialog.__init__(self, parent)

        self.apply_callback = apply

        # Form
        if isinstance(data[0][0], (list, tuple)):
            self.formwidget = FormTabWidget(data, comment=comment,
                                            parent=self)
        elif len(data[0])==3:
            self.formwidget = FormComboWidget(data, comment=comment,
                                              parent=self)
        else:
            self.formwidget = FormWidget(data, comment=comment,
                                         parent=self)
        layout = QVBoxLayout()
        layout.addWidget(self.formwidget)

        self.float_fields = []
        self.formwidget.setup()

        # Button box
        self.bbox = bbox = QDialogButtonBox(QDialogButtonBox.Ok
                                            |QDialogButtonBox.Cancel)
        self.connect(self.formwidget, SIGNAL('update_buttons()'),
                     self.update_buttons)
        if self.apply_callback is not None:
            apply_btn = bbox.addButton(QDialogButtonBox.Apply)
            self.connect(apply_btn, SIGNAL("clicked()"), self.apply)
        self.connect(bbox, SIGNAL("accepted()"), SLOT("accept()"))
        self.connect(bbox, SIGNAL("rejected()"), SLOT("reject()"))
        layout.addWidget(bbox)

        self.setLayout(layout)

        self.setWindowTitle(title)
        if not isinstance(icon, QIcon):
            icon = QWidget().style().standardIcon(QStyle.SP_MessageBoxQuestion)
        self.setWindowIcon(icon)

    def register_float_field(self, field):
        self.float_fields.append(field)

    def update_buttons(self):
        valid = True
        for field in self.float_fields:
            if not is_edit_valid(field):
                valid = False
        for btn_type in (QDialogButtonBox.Ok, QDialogButtonBox.Apply):
            btn = self.bbox.button(btn_type)
            if btn is not None:
                btn.setEnabled(valid)

    def accept(self):
        self.data = self.formwidget.get()
        QDialog.accept(self)

    def reject(self):
        self.data = None
        QDialog.reject(self)

    def apply(self):
        self.apply_callback(self.formwidget.get())

    def get(self):
        """Return form result"""
        return self.data


def fedit(data, title="", comment="", icon=None, parent=None, apply=None):
    """
    Create form dialog and return result
    (if Cancel button is pressed, return None)

    data: datalist, datagroup
    title: string
    comment: string
    icon: QIcon instance
    parent: parent QWidget
    apply: apply callback (function)

    datalist: list/tuple of (field_name, field_value)
    datagroup: list/tuple of (datalist *or* datagroup, title, comment)

    -> one field for each member of a datalist
    -> one tab for each member of a top-level datagroup
    -> one page (of a multipage widget, each page can be selected with a combo
       box) for each member of a datagroup inside a datagroup

    Supported types for field_value:
      - int, float, str, unicode, bool
      - colors: in Qt-compatible text form, i.e. in hex format or name (red,...)
                (automatically detected from a string)
      - list/tuple:
          * the first element will be the selected index (or value)
          * the other elements can be couples (key, value) or only values
    """

    # Create a QApplication instance if no instance currently exists
    # (e.g., if the module is used directly from the interpreter)
    if QApplication.startingUp():
        _app = QApplication([])
    dialog = FormDialog(data, title, comment, icon, parent, apply)
    if dialog.exec_():
        return dialog.get()



if __name__ == "__main__":

    def create_datalist_example():
        return [('str', 'this is a string'),
                ('list', [0, '1', '3', '4']),
                ('list2', ['--', ('none', 'None'), ('--', 'Dashed'),
                           ('-.', 'DashDot'), ('-', 'Solid'),
                           ('steps', 'Steps'), (':', 'Dotted')]),
                ('float', 1.2),
                (None, 'Other:'),
                ('int', 12),
                ('font', ('Arial', 10, False, True)),
                ('color', '#123409'),
                ('bool', True),
                ('date', datetime.date(2010, 10, 10)),
                ('datetime', datetime.datetime(2010, 10, 10)),
                ]

    def create_datagroup_example():
        datalist = create_datalist_example()
        return ((datalist, "Category 1", "Category 1 comment"),
                (datalist, "Category 2", "Category 2 comment"),
                (datalist, "Category 3", "Category 3 comment"))

    #--------- datalist example
    datalist = create_datalist_example()
    def apply_test(data):
        print("data:", data)
    print("result:", fedit(datalist, title="Example",
                           comment="This is just an <b>example</b>.",
                           apply=apply_test))

    #--------- datagroup example
    datagroup = create_datagroup_example()
    print("result:", fedit(datagroup, "Global title"))

    #--------- datagroup inside a datagroup example
    datalist = create_datalist_example()
    datagroup = create_datagroup_example()
    print("result:", fedit(((datagroup, "Title 1", "Tab 1 comment"),
                            (datalist, "Title 2", "Tab 2 comment"),
                            (datalist, "Title 3", "Tab 3 comment")),
                            "Global title"))
