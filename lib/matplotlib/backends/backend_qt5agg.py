"""
Render to qt from agg
"""

from .backend_qtagg import _BackendQTAgg
from .backend_qtagg import (  # noqa: F401 # pylint: disable=W0611
    FigureCanvasQTAgg, FigureManagerQT, NavigationToolbar2QT,
    backend_version,  FigureCanvasAgg,  FigureCanvasQT
)


@_BackendQTAgg.export
class _BackendQT5Agg(_BackendQTAgg):
    pass
