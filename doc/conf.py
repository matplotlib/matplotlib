# -*- coding: utf-8 -*-
#
# Matplotlib documentation build configuration file, created by
# sphinx-quickstart on Fri May  2 12:33:25 2008.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

import os
import sys
import sphinx

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.append(os.path.abspath('.'))

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['matplotlib.sphinxext.mathmpl', 'sphinxext.math_symbol_table',
              'sphinx.ext.autodoc', 'matplotlib.sphinxext.only_directives',
              'sphinx.ext.doctest', 'sphinx.ext.autosummary',
              'matplotlib.sphinxext.plot_directive',
              'sphinx.ext.inheritance_diagram',
              'sphinxext.gen_gallery', 'sphinxext.gen_rst',
              'sphinxext.github',
              'numpydoc']

exclude_patterns = ['api/api_changes/*', 'users/whats_new/*']

# Use IPython's console highlighting by default
try:
    from IPython.sphinxext import ipython_console_highlighting
except ImportError:
    raise ImportError(
        "IPython must be installed to build the Matplotlib docs")
else:
    extensions.append('IPython.sphinxext.ipython_console_highlighting')
    extensions.append('IPython.sphinxext.ipython_directive')

try:
    import numpydoc
except ImportError:
    raise ImportError("No module named numpydoc - you need to install "
                      "numpydoc to build the documentation.")

try:
    import colorspacious
except ImportError:
    raise ImportError("No module named colorspacious - you need to install "
                      "colorspacious to build the documentation")

try:
    from unittest.mock import MagicMock
except ImportError:
    try:
        from mock import MagicMock
    except ImportError:
        raise ImportError("No module named mock - you need to install "
                          "mock to build the documentation")

try:
    from PIL import Image
except ImportError:
    raise ImportError("No module named Image - you need to install "
                      "pillow to build the documentation")


try:
    import matplotlib
except ImportError:
    msg = "Error: Matplotlib must be installed before building the documentation"
    sys.exit(msg)


autosummary_generate = True

autodoc_docstring_signature = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# This is the default encoding, but it doesn't hurt to be explicit
source_encoding = "utf-8"

# The master toctree document.
master_doc = 'contents'

# General substitutions.
project = 'Matplotlib'
copyright = ('2002 - 2012 John Hunter, Darren Dale, Eric Firing, '
             'Michael Droettboom and the Matplotlib development '
             'team; 2012 - 2016 The Matplotlib development team')

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.

version = matplotlib.__version__
# The full version, including alpha/beta/rc tags.
release = version

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
unused_docs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

default_role = 'obj'

# Plot directive configuration
# ----------------------------

plot_formats = [('png', 100), ('pdf', 100)]

# Subdirectories in 'examples/' directory of package and titles for gallery
mpl_example_sections = [
    ('lines_bars_and_markers', 'Lines, bars, and markers'),
    ('shapes_and_collections', 'Shapes and collections'),
    ('statistics', 'Statistical plots'),
    ('images_contours_and_fields', 'Images, contours, and fields'),
    ('pie_and_polar_charts', 'Pie and polar charts'),
    ('color', 'Color'),
    ('text_labels_and_annotations', 'Text, labels, and annotations'),
    ('ticks_and_spines', 'Ticks and spines'),
    ('scales', 'Axis scales'),
    ('subplots_axes_and_figures', 'Subplots, axes, and figures'),
    ('style_sheets', 'Style sheets'),
    ('specialty_plots', 'Specialty plots'),
    ('showcase', 'Showcase'),
    ('api', 'API'),
    ('pylab_examples', 'pylab examples'),
    ('mplot3d', 'mplot3d toolkit'),
    ('axes_grid1', 'axes_grid1 toolkit'),
    ('axisartist', 'axisartist toolkit'),
    ('units', 'units'),
    ('widgets', 'widgets'),
    ('misc', 'Miscellaneous examples'),
    ]


# Github extension

github_project_url = "https://github.com/matplotlib/matplotlib/"

# Options for HTML output
# -----------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
#html_style = 'matplotlib.css'
html_style = 'mpl.css'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# The name of an image file (within the static path) to place at the top of
# the sidebar.
#html_logo = 'logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If nonempty, this is the file name suffix for generated HTML files.  The
# default is ``".html"``.
html_file_suffix = '.html'

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Content template for the index page.
html_index = 'index.html'

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Custom sidebar templates, maps page names to templates.
html_sidebars = {
    'index': ['badgesidebar.html','donate_sidebar.html',
              'indexsidebar.html', 'searchbox.html'],
    '**': ['badgesidebar.html', 'localtoc.html',
           'relations.html', 'sourcelink.html', 'searchbox.html']
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {'index': 'index.html',
                         'gallery':'gallery.html',
                         'citing': 'citing.html'}

# If false, no module index is generated.
#html_use_modindex = True
html_domain_indices = ["py-modindex"]

# If true, the reST sources are included in the HTML build as _sources/<name>.
#html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.
html_use_opensearch = 'False'

# Output file base name for HTML help builder.
htmlhelp_basename = 'Matplotlibdoc'

# Path to favicon
html_favicon = '_static/favicon.ico'

# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
latex_paper_size = 'letter'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).

latex_documents = [
    ('contents', 'Matplotlib.tex', 'Matplotlib',
     'John Hunter, Darren Dale, Eric Firing, Michael Droettboom and the '
     'matplotlib development team', 'manual'),
]


# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = None

latex_elements = {}
# Additional stuff for the LaTeX preamble.
latex_elements['preamble'] = r"""
   % In the parameters section, place a newline after the Parameters
   % header.  (This is stolen directly from Numpy's conf.py, since it
   % affects Numpy-style docstrings).
   \usepackage{expdlist}
   \let\latexdescription=\description
   \def\description{\latexdescription{}{} \breaklabel}

   \usepackage{amsmath}
   \usepackage{amsfonts}
   \usepackage{amssymb}
   \usepackage{txfonts}

   % The enumitem package provides unlimited nesting of lists and
   % enums.  Sphinx may use this in the future, in which case this can
   % be removed.  See
   % https://bitbucket.org/birkenfeld/sphinx/issue/777/latex-output-too-deeply-nested
   \usepackage{enumitem}
   \setlistdepth{2048}
"""
latex_elements['pointsize'] = '11pt'

# Documents to append as an appendix to all manuals.
latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = True

if hasattr(sphinx, 'version_info') and sphinx.version_info[:2] >= (1, 4):
    latex_toplevel_sectioning = 'part'
else:
    latex_use_parts = True

# Show both class-level docstring and __init__ docstring in class
# documentation
autoclass_content = 'both'

rst_epilog = """
.. |minimum_numpy_version| replace:: %s
""" % matplotlib.__version__numpy__

texinfo_documents = [
    ("contents", 'matplotlib', 'Matplotlib Documentation',
     'John Hunter@*Darren Dale@*Eric Firing@*Michael Droettboom@*'
     'The matplotlib development team',
     'Matplotlib', "Python plotting package", 'Programming',
     1),
]


class MyWX(MagicMock):
    class Panel(object):
        pass

    class ToolBar(object):
        pass

    class Frame(object):
        pass

    VERSION_STRING = '2.8.12'


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


mockwxversion = MagicMock()
mockwx = MyWX()
mocksip = MySip()
mockpyqt4 = MyPyQt4()
sys.modules['wxversion'] = mockwxversion
sys.modules['wx'] = mockwx
sys.modules['sip'] = mocksip
sys.modules['PyQt4'] = mockpyqt4

# numpydoc config

numpydoc_show_class_members = False

# Skip deprecated members

def skip_deprecated(app, what, name, obj, skip, options):
    if skip:
        return skip
    skipped = {"matplotlib.colors": ["ColorConverter", "hex2color", "rgb2hex"]}
    skip_list = skipped.get(getattr(obj, "__module__", None))
    if skip_list is not None:
        return getattr(obj, "__name__", None) in skip_list

def setup(app):
    app.connect('autodoc-skip-member', skip_deprecated)
