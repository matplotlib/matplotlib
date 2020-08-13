import pytest
import numpy as np
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
def test_missing_back_button():
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
    class Toolbar(NavigationToolbar2Tk):
        # only display the buttons we need
        toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                     t[0] in ('Home', 'Pan', 'Zoom')]

    fig = plt.figure()
    # this should not raise
    Toolbar(fig.canvas, fig.canvas.manager.window)
