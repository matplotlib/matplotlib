import os
import subprocess
import sys
import tkinter

import numpy as np
import pytest

from matplotlib import pyplot as plt


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
def test_blit():
    from matplotlib.backends import _tkagg
    def evil_blit(photoimage, aggimage, offsets, bboxptr):
        data = np.asarray(aggimage)
        height, width = data.shape[:2]
        dataptr = (height, width, data.ctypes.data)
        _tkagg.blit(
            photoimage.tk.interpaddr(), str(photoimage), dataptr, offsets,
            bboxptr)

    fig, ax = plt.subplots()
    for bad_boxes in ((-1, 2, 0, 2),
                      (2, 0, 0, 2),
                      (1, 6, 0, 2),
                      (0, 2, -1, 2),
                      (0, 2, 2, 0),
                      (0, 2, 1, 6)):
        with pytest.raises(ValueError):
            evil_blit(fig.canvas._tkphoto,
                      np.ones((4, 4, 4)),
                      (0, 1, 2, 3),
                      bad_boxes)


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
def test_figuremanager_preserves_host_mainloop():
    success = False

    def do_plot():
        plt.figure()
        plt.plot([1, 2], [3, 5])
        plt.close()
        root.after(0, legitimate_quit)

    def legitimate_quit():
        root.quit()
        nonlocal success
        success = True

    root = tkinter.Tk()
    root.after(0, do_plot)
    root.mainloop()

    assert success


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
@pytest.mark.flaky(reruns=3)
def test_figuremanager_cleans_own_mainloop():
    script = '''
import tkinter
import time
import matplotlib.pyplot as plt
import threading
from matplotlib.cbook import _get_running_interactive_framework

root = tkinter.Tk()
plt.plot([1, 2, 3], [1, 2, 5])

def target():
    while not 'tk' == _get_running_interactive_framework():
        time.sleep(.01)
    plt.close()
    if show_finished_event.wait():
        print('success')

show_finished_event = threading.Event()
thread = threading.Thread(target=target, daemon=True)
thread.start()
plt.show(block=True)  # testing if this function hangs
show_finished_event.set()
thread.join()

'''
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ,
                 "MPLBACKEND": "TkAgg",
                 "SOURCE_DATE_EPOCH": "0"},
            timeout=10,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Most likely plot.show(block=True) hung")
    except subprocess.CalledProcessError:
        pytest.fail("Subprocess failed to test intended behavior")
    assert proc.stdout.count("success") == 1


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
def test_missing_back_button():
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
    class Toolbar(NavigationToolbar2Tk):
        # only display the buttons we need
        toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                     t[0] in ('Home', 'Pan', 'Zoom')]

    fig = plt.figure()
    # this should not raise
    Toolbar(fig.canvas, fig.canvas.manager.window)
