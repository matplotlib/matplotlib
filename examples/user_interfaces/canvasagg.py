"""
==============
CanvasAgg demo
==============

This example shows how to use the agg backend directly to create images, which
may be of use to web application developers who want full control over their
code without using the pyplot interface to manage figures, figure closing etc.

.. note::

    It is not necessary to avoid using the pyplot interface in order to
    create figures without a graphical front-end - simply setting
    the backend to "Agg" would be sufficient.

In this example, we show how to save the contents of the agg canvas to a file,
and how to extract them to a string, which can in turn be passed off to PIL or
put in a numpy array.  The latter functionality allows e.g. to use Matplotlib
inside a cgi-script *without* needing to write a figure to disk.
"""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

fig = Figure(figsize=(5, 4), dpi=100)
# A canvas must be manually attached to the figure (pyplot would automatically
# do it).  This is done by instantiating the canvas with the figure as
# argument.
canvas = FigureCanvasAgg(fig)

# Do some plotting.
ax = fig.add_subplot(111)
ax.plot([1, 2, 3])

# Option 1: Save the figure to a file; can also be a file-like object (BytesIO,
# etc.).
fig.savefig("test.png")

# Option 2: Retrieve a view on the renderer buffer...
canvas.draw()
buf = canvas.buffer_rgba()
# ... convert to a NumPy array ...
X = np.asarray(buf)
# ... and pass it to PIL.
from PIL import Image
im = Image.fromarray(X)

# Uncomment this line to display the image using ImageMagick's `display` tool.
# im.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

import matplotlib
matplotlib.backends.backend_agg.FigureCanvasAgg
matplotlib.figure.Figure
matplotlib.figure.Figure.add_subplot
matplotlib.figure.Figure.savefig
matplotlib.axes.Axes.plot
