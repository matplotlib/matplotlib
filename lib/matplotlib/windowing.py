from matplotlib import rcParams

try:
    if not rcParams['tk.window_focus']:
        raise ImportError
    from _windowing import GetForegroundWindow, SetForegroundWindow
except ImportError:
    def GetForegroundWindow():
        return 0
    def SetForegroundWindow(hwnd):
        pass

class FocusManager:
    def __init__(self):
        self._shellWindow = GetForegroundWindow()

    def __del__(self):
        SetForegroundWindow(self._shellWindow)
