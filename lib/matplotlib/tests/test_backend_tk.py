import threading
import time
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
        root.after(0, legitmate_quit)

    def legitmate_quit():
        root.quit()
        nonlocal success
        success = True

    root = tkinter.Tk()
    root.after(0, do_plot)
    root.mainloop()

    assert success


@pytest.mark.backend('TkAgg', skip_on_importerror=True)
def test_figuremanager_cleans_own_mainloop():
    root = tkinter.Tk()
    plt.plot([1, 2, 3], [1, 2, 5])
    can_detect_mainloop = False
    thread_died_before_quit = True

    def target():
        nonlocal can_detect_mainloop
        nonlocal thread_died_before_quit
        from matplotlib.cbook import _get_running_interactive_framework

        time.sleep(0.1)  # should poll for mainloop being up
        can_detect_mainloop = 'tk' == _get_running_interactive_framework()
        plt.close()
        time.sleep(0.1)  # should poll for mainloop going down
        root.quit()
        thread_died_before_quit = False

    threading.Thread(target=target, daemon=True).start()
    plt.show(block=True)
    assert can_detect_mainloop
    assert thread_died_before_quit
