import pytest
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import sys
import time
import threading


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
def test_draw_after_destroy():
    """
    Idle callbacks should not trigger exceptions after canvas been destroyed.
    """
    sys.last_type = None  # reset last exception

    def worker():
        fig, ax = plt.subplots()
        graph, = ax.plot([], [])

        def update(t):
            graph.axes.plot([1, 2], [1, t])
            if t == 1:
                plt.close()
            return graph,

        ani = animation.FuncAnimation(fig, update, 3, blit=False, repeat=False)
        plt.show()

        input()  # will trigger idle callback, but no exception shall be thrown

    th = threading.Thread(target=worker)
    th.daemon = True  # in order to pass through input() blocking
    th.start()
    time.sleep(3)
    assert sys.last_type is None
