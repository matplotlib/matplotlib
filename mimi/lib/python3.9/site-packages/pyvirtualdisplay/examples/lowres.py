"Testing gnumeric on low resolution."
from easyprocess import EasyProcess

from pyvirtualdisplay import Display

# start Xephyr
with Display(visible=True, size=(320, 240)) as disp:
    # start Gnumeric
    with EasyProcess(["gnumeric"]) as proc:
        proc.wait()
