"""
Display an Agg figure in a notebook using jupyter_rfb.

Install the widget extra, then run this in a notebook::

    import matplotlib
    matplotlib.use("module://matplotlib.backends.backend_anywidget")
    import matplotlib.pyplot as plt

    plt.plot([0, 1, 2], [0, 1, 0])
    plt.show()
"""

import numpy as np
from jupyter_rfb import RemoteFrameBuffer

from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_agg import FigureCanvasAgg


class _FigureWidget(RemoteFrameBuffer):
    def __init__(self, canvas):
        self.canvas = canvas
        width, height = canvas.get_width_height()
        super().__init__(css_width=f"{width}px", css_height=f"{height}px",
                         resizable=False, quality=100)

    def get_frame(self):
        self.canvas.draw()
        # RFB retains the frame, while Agg reuses its pixel buffer.
        return np.asarray(self.canvas.buffer_rgba()).copy()


class FigureManager(FigureManagerBase):
    def __init__(self, canvas, num):
        super().__init__(canvas, num)
        self._widget = None

    def show(self):
        if self._widget is None:
            self._widget = _FigureWidget(self.canvas)
            self._widget.display()
        self._widget.request_draw()

    def destroy(self):
        if self._widget is not None:
            self._widget.close()
        super().destroy()


class FigureCanvas(FigureCanvasAgg):
    manager_class = FigureManager
    supports_blit = False
