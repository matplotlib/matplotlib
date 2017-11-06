import sys

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock


class MyCairoCffi(MagicMock):
    version_info = (1, 4, 0)


class MyPyQt4(MagicMock):
    class QtGui(object):
        # PyQt4.QtGui public classes.
        # Generated with
        # textwrap.fill([name for name in dir(PyQt4.QtGui)
        #                if isinstance(getattr(PyQt4.QtGui, name), type)])
        _QtGui_public_classes = """\
        Display QAbstractButton QAbstractGraphicsShapeItem
        QAbstractItemDelegate QAbstractItemView QAbstractPrintDialog
        QAbstractProxyModel QAbstractScrollArea QAbstractSlider
        QAbstractSpinBox QAbstractTextDocumentLayout QAction QActionEvent
        QActionGroup QApplication QBitmap QBoxLayout QBrush QButtonGroup
        QCalendarWidget QCheckBox QClipboard QCloseEvent QColor QColorDialog
        QColumnView QComboBox QCommandLinkButton QCommonStyle QCompleter
        QConicalGradient QContextMenuEvent QCursor QDataWidgetMapper QDateEdit
        QDateTimeEdit QDesktopServices QDesktopWidget QDial QDialog
        QDialogButtonBox QDirModel QDockWidget QDoubleSpinBox QDoubleValidator
        QDrag QDragEnterEvent QDragLeaveEvent QDragMoveEvent QDropEvent
        QErrorMessage QFileDialog QFileIconProvider QFileOpenEvent
        QFileSystemModel QFocusEvent QFocusFrame QFont QFontComboBox
        QFontDatabase QFontDialog QFontInfo QFontMetrics QFontMetricsF
        QFormLayout QFrame QGesture QGestureEvent QGestureRecognizer QGlyphRun
        QGradient QGraphicsAnchor QGraphicsAnchorLayout QGraphicsBlurEffect
        QGraphicsColorizeEffect QGraphicsDropShadowEffect QGraphicsEffect
        QGraphicsEllipseItem QGraphicsGridLayout QGraphicsItem
        QGraphicsItemAnimation QGraphicsItemGroup QGraphicsLayout
        QGraphicsLayoutItem QGraphicsLineItem QGraphicsLinearLayout
        QGraphicsObject QGraphicsOpacityEffect QGraphicsPathItem
        QGraphicsPixmapItem QGraphicsPolygonItem QGraphicsProxyWidget
        QGraphicsRectItem QGraphicsRotation QGraphicsScale QGraphicsScene
        QGraphicsSceneContextMenuEvent QGraphicsSceneDragDropEvent
        QGraphicsSceneEvent QGraphicsSceneHelpEvent QGraphicsSceneHoverEvent
        QGraphicsSceneMouseEvent QGraphicsSceneMoveEvent
        QGraphicsSceneResizeEvent QGraphicsSceneWheelEvent
        QGraphicsSimpleTextItem QGraphicsTextItem QGraphicsTransform
        QGraphicsView QGraphicsWidget QGridLayout QGroupBox QHBoxLayout
        QHeaderView QHelpEvent QHideEvent QHoverEvent QIcon QIconDragEvent
        QIconEngine QIconEngineV2 QIdentityProxyModel QImage QImageIOHandler
        QImageReader QImageWriter QInputContext QInputContextFactory
        QInputDialog QInputEvent QInputMethodEvent QIntValidator QItemDelegate
        QItemEditorCreatorBase QItemEditorFactory QItemSelection
        QItemSelectionModel QItemSelectionRange QKeyEvent QKeyEventTransition
        QKeySequence QLCDNumber QLabel QLayout QLayoutItem QLineEdit
        QLinearGradient QListView QListWidget QListWidgetItem QMainWindow
        QMatrix QMatrix2x2 QMatrix2x3 QMatrix2x4 QMatrix3x2 QMatrix3x3
        QMatrix3x4 QMatrix4x2 QMatrix4x3 QMatrix4x4 QMdiArea QMdiSubWindow
        QMenu QMenuBar QMessageBox QMimeSource QMouseEvent
        QMouseEventTransition QMoveEvent QMovie QPageSetupDialog QPaintDevice
        QPaintEngine QPaintEngineState QPaintEvent QPainter QPainterPath
        QPainterPathStroker QPalette QPanGesture QPen QPicture QPictureIO
        QPinchGesture QPixmap QPixmapCache QPlainTextDocumentLayout
        QPlainTextEdit QPolygon QPolygonF QPrintDialog QPrintEngine
        QPrintPreviewDialog QPrintPreviewWidget QPrinter QPrinterInfo
        QProgressBar QProgressDialog QProxyModel QPushButton QPyTextObject
        QQuaternion QRadialGradient QRadioButton QRawFont QRegExpValidator
        QRegion QResizeEvent QRubberBand QScrollArea QScrollBar
        QSessionManager QShortcut QShortcutEvent QShowEvent QSizeGrip
        QSizePolicy QSlider QSortFilterProxyModel QSound QSpacerItem QSpinBox
        QSplashScreen QSplitter QSplitterHandle QStackedLayout QStackedWidget
        QStandardItem QStandardItemModel QStaticText QStatusBar
        QStatusTipEvent QStringListModel QStyle QStyleFactory QStyleHintReturn
        QStyleHintReturnMask QStyleHintReturnVariant QStyleOption
        QStyleOptionButton QStyleOptionComboBox QStyleOptionComplex
        QStyleOptionDockWidget QStyleOptionDockWidgetV2 QStyleOptionFocusRect
        QStyleOptionFrame QStyleOptionFrameV2 QStyleOptionFrameV3
        QStyleOptionGraphicsItem QStyleOptionGroupBox QStyleOptionHeader
        QStyleOptionMenuItem QStyleOptionProgressBar QStyleOptionProgressBarV2
        QStyleOptionRubberBand QStyleOptionSizeGrip QStyleOptionSlider
        QStyleOptionSpinBox QStyleOptionTab QStyleOptionTabBarBase
        QStyleOptionTabBarBaseV2 QStyleOptionTabV2 QStyleOptionTabV3
        QStyleOptionTabWidgetFrame QStyleOptionTabWidgetFrameV2
        QStyleOptionTitleBar QStyleOptionToolBar QStyleOptionToolBox
        QStyleOptionToolBoxV2 QStyleOptionToolButton QStyleOptionViewItem
        QStyleOptionViewItemV2 QStyleOptionViewItemV3 QStyleOptionViewItemV4
        QStylePainter QStyledItemDelegate QSwipeGesture QSyntaxHighlighter
        QSystemTrayIcon QTabBar QTabWidget QTableView QTableWidget
        QTableWidgetItem QTableWidgetSelectionRange QTabletEvent
        QTapAndHoldGesture QTapGesture QTextBlock QTextBlockFormat
        QTextBlockGroup QTextBlockUserData QTextBrowser QTextCharFormat
        QTextCursor QTextDocument QTextDocumentFragment QTextDocumentWriter
        QTextEdit QTextFormat QTextFragment QTextFrame QTextFrameFormat
        QTextImageFormat QTextInlineObject QTextItem QTextLayout QTextLength
        QTextLine QTextList QTextListFormat QTextObject QTextObjectInterface
        QTextOption QTextTable QTextTableCell QTextTableCellFormat
        QTextTableFormat QTimeEdit QToolBar QToolBox QToolButton QToolTip
        QTouchEvent QTransform QTreeView QTreeWidget QTreeWidgetItem
        QTreeWidgetItemIterator QUndoCommand QUndoGroup QUndoStack QUndoView
        QVBoxLayout QValidator QVector2D QVector3D QVector4D QWhatsThis
        QWhatsThisClickedEvent QWheelEvent QWidget QWidgetAction QWidgetItem
        QWindowStateChangeEvent QWizard QWizardPage QWorkspace
        QX11EmbedContainer QX11EmbedWidget QX11Info
        """
        for _name in _QtGui_public_classes.split():
            locals()[_name] = type(_name, (), {})
        del _name


class MySip(MagicMock):
    def getapi(*args):
        return 1


class MyWX(MagicMock):
    class Panel(object):
        pass

    class ToolBar(object):
        pass

    class Frame(object):
        pass

    VERSION_STRING = '2.9'


def setup(app):
    sys.modules['cairocffi'] = MyCairoCffi()
    sys.modules['PyQt4'] = MyPyQt4()
    sys.modules['sip'] = MySip()
    sys.modules['wx'] = MyWX()
    sys.modules['wxversion'] = MagicMock()

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
