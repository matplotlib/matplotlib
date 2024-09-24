"Start Xvfb server. Open xmessage window."

from easyprocess import EasyProcess

from pyvirtualdisplay import Display

with Display(visible=False, size=(100, 60)) as disp:
    with EasyProcess(["xmessage", "hello"]) as proc:
        proc.wait()
