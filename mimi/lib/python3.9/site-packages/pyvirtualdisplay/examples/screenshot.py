"Create screenshot of xmessage in background using 'smartdisplay' submodule"
from easyprocess import EasyProcess

from pyvirtualdisplay.smartdisplay import SmartDisplay

# 'SmartDisplay' instead of 'Display'
# It has 'waitgrab()' method.
# It has more dependencies than Display.
with SmartDisplay() as disp:
    with EasyProcess(["xmessage", "hello"]):
        # wait until something is displayed on the virtual display (polling method)
        # and then take a fullscreen screenshot
        # and then crop it. Background is black.
        img = disp.waitgrab()
img.save("xmessage.png")
