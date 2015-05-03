"""iTerm2 exterimental backend.

Based on iTerm2 nightly build feature - displaying images in terminal.
http://iterm2.com/images.html#/section/home

Example:

    import matplotlib
    matplotlib.use('xxx')
    from pylab import *
    plot([1,2,3])
    show()
"""

__author__ = 'Oleg Selivanov <oleg.a.selivanov@gmail.com>'

import os
import subprocess
import tempfile

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.figure import Figure
from PIL import Image

# TODO(oleg): Show better message if PIL/Pillow is not installed.


def show():
    for manager in Gcf.get_all_fig_managers():
        manager.show()
        # TODO(oleg): Check if it's okay to destroy manager here.
        Gcf.destroy(manager.num)


def new_figure_manager(num, *args, **kwargs):
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    canvas = FigureCanvasAgg(thisFig)
    manager = FigureManagerTemplate(canvas, num)
    return manager


class FigureManagerTemplate(FigureManagerBase):
    def show(self):
        canvas = self.canvas
        canvas.draw()
        buf = canvas.buffer_rgba(0, 0)
        render = canvas.get_renderer()
        w = int(render.width)
        h = int(render.height)
        im = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'RGBA', 0, 1)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            im.save(f.name)
            subprocess.call(['imgcat', f.name])
            os.unlink(f.name)


FigureManager = FigureManagerBase
