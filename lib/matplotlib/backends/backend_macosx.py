from matplotlib.backend_bases import _Backend

from ._backend_legacymac import (
    FigureCanvasLegacyMac as FigureCanvasMac,
    FigureManagerLegacyMac as FigureManagerMac,
    NavigationToolbar2LegacyMac as NavigationToolbar2Mac,
    TimerLegacyMac as TimerMac)

__all__ = [FigureCanvasMac, FigureManagerMac, NavigationToolbar2Mac, TimerMac]


@_Backend.export
class _BackendMac(_Backend):
    FigureCanvas = FigureCanvasMac
    FigureManager = FigureManagerMac
    mainloop = FigureManagerMac.start_main_loop
