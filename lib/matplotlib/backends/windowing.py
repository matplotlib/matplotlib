"""
MS Windows-specific helper for the TkAgg backend.

With rcParams['tk.window_focus'] default of False, it is
effectively disabled.

It uses a tiny C++ extension module to access MS Win functions.

This module is deprecated and will be removed in version 3.2
"""

from matplotlib import rcParams, cbook

cbook.warn_deprecated('3.0', obj_type='module', name='backends.windowing')

try:
    if not rcParams['tk.window_focus']:
        raise ImportError
    from matplotlib._windowing import GetForegroundWindow, SetForegroundWindow
except ImportError:
    def GetForegroundWindow():
        return 0
    def SetForegroundWindow(hwnd):
        pass

class FocusManager(object):
    def __init__(self):
        self._shellWindow = GetForegroundWindow()

    def __del__(self):
        SetForegroundWindow(self._shellWindow)
