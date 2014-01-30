from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import signal
import matplotlib

from matplotlib._pylab_helpers import Gcf

# Qt compat imports, can probably trim these
# TODO expose less compatibility
from .qt4_compat import QtCore, QtGui, _getSaveFileName, __version__

# pull in QT specific part
from .base_backend_qt4 import (new_figure_manager,
                            new_figure_manager_given_figure,
                            TimerQT,
                            FigureCanvasQT,
                            FigureManagerQT,
                            NavigationToolbar2QT,
                            SubplotToolQt, backend_version)
# pull in Gcf contaminate parts
from matplotlib.backend_bases import (ShowBase,
                                      key_press_handler)


# unclear why this is per-backend, should probably be pushed up the
# hierarchy

def draw_if_interactive():
    """
    Is called after every pylab drawing command
    """
    if matplotlib.is_interactive():
        figManager = Gcf.get_active()
        if figManager is not None:
            figManager.canvas.draw_idle()


class Show(ShowBase):
    def mainloop(self):
        # allow KeyboardInterrupt exceptions to close the plot window.
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        QtGui.qApp.exec_()
show = Show()

FigureCanvas = FigureCanvasQT
FigureManager = FigureManagerQT
# register default key_handlers
FigureManager._key_press_handler = staticmethod(key_press_handler)
FigureManager._destroy_callback = staticmethod(Gcf.destroy)
