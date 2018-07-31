from .backend_qt5 import (
    backend_version, SPECIAL_KEYS, SUPER, ALT, CTRL, SHIFT, MODIFIER_KEYS,
    cursord, _create_qApp, _BackendQT5, TimerQT, MainWindow, FigureManagerQT,
    NavigationToolbar2QT, SubplotToolQt, error_msg_qt, exception_handler)
from .backend_qt5 import FigureCanvasQT as FigureCanvasQT5


@_BackendQT5.export
class _BackendQT4(_BackendQT5):
    required_interactive_framework = "qt4"
