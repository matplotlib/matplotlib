"""
===================
Custom Figure Class
===================

You can pass a custom Figure constructor to figure if you want to derive from
the default Figure.  This simple example creates a figure with a figure title.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class MyFigure(Figure):
    def __init__(self, *args, **kwargs):
        """
        custom kwarg figtitle is a figure title
        """
        figtitle = kwargs.pop('figtitle', 'hi mom')
        Figure.__init__(self, *args, **kwargs)
        self.text(0.5, 0.95, figtitle, ha='center')


fig = plt.figure(FigureClass=MyFigure, figtitle='my title')
ax = fig.subplots()
ax.plot([1, 2, 3])

plt.show()
