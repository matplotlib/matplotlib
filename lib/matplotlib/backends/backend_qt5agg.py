"""
Render to qt from agg
"""

from .backend_qtagg import (
    _BackendQTAgg, FigureCanvasQTAgg, FigureManagerQT, NavigationToolbar2QT,
    backend_version)


@_BackendQTAgg.export
class _BackendQT5Agg(_BackendQTAgg):
    pass
