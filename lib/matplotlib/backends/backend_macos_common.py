import sys

if sys.platform != "darwin":
    def set_mac_icon(path):
        pass
else:
    from . import _macosx
    def set_mac_icon(path):
        _macosx.set_icon(path)
