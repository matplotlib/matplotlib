"""
Render to qt from agg
"""

from .backend_qt5agg import (
    _BackendQT5Agg, FigureCanvasQTAgg, FigureManagerQT, NavigationToolbar2QT)


@_BackendQT5Agg.export
class _BackendQT4Agg(_BackendQT5Agg):
    pass
