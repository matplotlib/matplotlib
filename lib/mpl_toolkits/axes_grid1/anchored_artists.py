
from matplotlib.patches import Rectangle, Ellipse

import numpy as np

from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker,\
     TextArea, AnchoredText, DrawingArea, AnnotationBbox


class AnchoredDrawingArea(AnchoredOffsetbox):
    """
    AnchoredOffsetbox with DrawingArea
    """

    def __init__(self, width, height, xdescent, ydescent,
                 loc, pad=0.4, borderpad=0.5, prop=None, frameon=True,
                 **kwargs):
        """
        *width*, *height*, *xdescent*, *ydescent* : the dimensions of the DrawingArea.
        *prop* : font property. This is only used for scaling the paddings.
        """

        self.da = DrawingArea(width, height, xdescent, ydescent, clip=True)
        self.drawing_area = self.da

        super(AnchoredDrawingArea, self).__init__(loc, pad=pad, borderpad=borderpad,
                                                  child=self.da,
                                                  prop=None,
                                                  frameon=frameon,
                                                  **kwargs)


class AnchoredAuxTransformBox(AnchoredOffsetbox):
    def __init__(self, transform, loc,
                 pad=0.4, borderpad=0.5, prop=None, frameon=True, **kwargs):

        self.drawing_area = AuxTransformBox(transform)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self.drawing_area,
                                   prop=prop,
                                   frameon=frameon,
                                   **kwargs)



class AnchoredEllipse(AnchoredOffsetbox):
    def __init__(self, transform, width, height, angle, loc,
                 pad=0.1, borderpad=0.1, prop=None, frameon=True, **kwargs):
        """
        Draw an ellipse the size in data coordinate of the give axes.

        pad, borderpad in fraction of the legend font size (or prop)
        """
        self._box = AuxTransformBox(transform)
        self.ellipse = Ellipse((0,0), width, height, angle)
        self._box.add_artist(self.ellipse)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon, **kwargs)



class AnchoredSizeBar(AnchoredOffsetbox):
    def __init__(self, transform, size, label, loc,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, frameon=True,
                 **kwargs):
        """
        Draw a horizontal bar with the size in data coordinate of the give axes.
        A label will be drawn underneath (center-aligned).

        pad, borderpad in fraction of the legend font size (or prop)
        sep in points.
        """
        self.size_bar = AuxTransformBox(transform)
        self.size_bar.add_artist(Rectangle((0,0), size, 0, fc="none"))

        self.txt_label = TextArea(label, minimumdescent=False)

        self._box = VPacker(children=[self.size_bar, self.txt_label],
                            align="center",
                            pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=prop,
                                   frameon=frameon, **kwargs)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    fig = plt.gcf()
    fig.clf()
    ax = plt.subplot(111)

    offsetbox = AnchoredText("Test", loc=6, pad=0.3,
                             borderpad=0.3, prop=None)
    xy = (0.5, 0.5)
    ax.plot([0.5], [0.5], "xk")
    ab = AnnotationBbox(offsetbox, xy,
                        xybox=(1., .5),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        arrowprops=dict(arrowstyle="->"))
                        #arrowprops=None)

    ax.add_artist(ab)


    from matplotlib.patches import Circle
    ada = AnchoredDrawingArea(20, 20, 0, 0,
                              loc=6, pad=0.1, borderpad=0.3, frameon=True)
    p = Circle((10, 10), 10)
    ada.da.add_artist(p)

    ab = AnnotationBbox(ada, (0.3, 0.4),
                        xybox=(1., 0.4),
                        xycoords='data',
                        boxcoords=("axes fraction", "data"),
                        arrowprops=dict(arrowstyle="->"))
                        #arrowprops=None)

    ax.add_artist(ab)


    arr = np.arange(100).reshape((10,10))
    im = AnchoredImage(arr,
                       loc=4,
                       pad=0.5, borderpad=0.2, prop=None, frameon=True,
                       zoom=1,
                       cmap = None,
                       norm = None,
                       interpolation=None,
                       origin=None,
                       extent=None,
                       filternorm=1,
                       filterrad=4.0,
                       resample = False,
                       )

    ab = AnnotationBbox(im, (0.5, 0.5),
                        xybox=(-10., 10.),
                        xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
                        #arrowprops=None)

    ax.add_artist(ab)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


    plt.draw()
    plt.show()

