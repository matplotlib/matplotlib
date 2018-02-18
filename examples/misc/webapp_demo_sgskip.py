"""
===========
Webapp Demo
===========

This example shows how to use the agg backend directly to create
images, which may be of use to web application developers who want
full control over their code without using the pyplot interface to
manage figures, figure closing etc.

.. note::

    It is not necessary to avoid using the pyplot interface in order to
    create figures without a graphical front-end - simply setting
    the backend to "Agg" would be sufficient.

It is also worth noting that, because matplotlib can save figures to file-like
object, matplotlib can also be used inside a cgi-script *without* needing to
write a figure to disk.

"""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def make_fig():
    """
    Make a figure and save it to "webagg.png".

    """
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([1, 2, 3], 'ro--', markersize=12, markerfacecolor='g')

    # make a translucent scatter collection
    x = np.random.rand(100)
    y = np.random.rand(100)
    area = np.pi * (10 * np.random.rand(100)) ** 2  # 0 to 10 point radii
    c = ax.scatter(x, y, area)
    c.set_alpha(0.5)

    # add some text decoration
    ax.set_title('My first image')
    ax.set_ylabel('Some numbers')
    ax.set_xticks((.2, .4, .6, .8))
    labels = ax.set_xticklabels(('Bill', 'Fred', 'Ted', 'Ed'))

    # To set object properties, you can either iterate over the
    # objects manually, or define you own set command, as in setapi
    # above.
    for label in labels:
        label.set_rotation(45)
        label.set_fontsize(12)

    FigureCanvasAgg(fig).print_png('webapp.png', dpi=150)


make_fig()
