from . import backend_legacymac as _legacymac
from matplotlib.backend_bases import _Backend


class TimerMac(_legacymac.TimerLegacyMac):
    pass


class FigureCanvasMac(_legacymac.FigureCanvasLegacyMac):
    pass


class FigureManagerMac(_legacymac.FigureManagerLegacyMac):
    pass


class NavigationToolbar2Mac(_legacymac.NavigationToolbar2LegacyMac):
    pass


@_Backend.export
class _BackendMac(_Backend):
    FigureCanvas = FigureCanvasMac
    FigureManager = FigureManagerMac
    mainloop = _legacymac.FigureManagerLegacyMac.start_main_loop
