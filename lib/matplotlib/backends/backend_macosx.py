import sys
from . import backend_legacymac as _legacymac
from matplotlib.backend_bases import _Backend

_RENAMES = {
    'FigureCanvasMac': 'FigureCanvasLegacyMac',
    'FigureManagerMac': 'FigureManagerLegacyMac',
    'NavigationToolbar2': 'NavigationToolbar2LegacyMac'
}

_mod = sys.modules[__name__]
for _new, _old in _RENAMES.items():
    setattr(_mod, _new, getattr(_legacymac, _old))
del _mod, _new, _old


@_Backend.export
class _BackendMac(_Backend):
    FigureCanvas = _legacymac.FigureCanvasLegacyMac
    FigureManager = _legacymac.FigureManagerLegacyMac
    mainloop = _legacymac.FigureManagerLegacyMac.start_main_loop