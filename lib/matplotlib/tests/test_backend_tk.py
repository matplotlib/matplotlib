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
def test_blit_error_on_closed_figure():
    # render figure with TkAgg backend and then close
    fig, axes = plt.subplots()
    fig.canvas.draw()
    plt.close(fig.number)
    # assert runtime error is thrown after figure is closed
    with pytest.raises(RuntimeError) as seg_info:
        fig.canvas.blit(axes.bbox)
    assert f"{fig.canvas._tkphoto} has been deleted" in str(seg_info.value)
