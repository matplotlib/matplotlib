from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

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

        self.da = DrawingArea(width, height, xdescent, ydescent)
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
                 pad=0.1, borderpad=0.1, sep=2,
                 frameon=True, size_vertical=0, color='black',
                 label_top=False, fontproperties=None,
                 **kwargs):
        """
        Draw a horizontal bar with the size in data coordinate of the given axes.
        A label will be drawn underneath (center-aligned).

        Parameters:
        -----------
        transform : matplotlib transformation object
        size : int or float
          horizontal length of the size bar, given in data coordinates
        label : str
        loc : int
        pad : int or float, optional
          in fraction of the legend font size (or prop)
        borderpad : int or float, optional
          in fraction of the legend font size (or prop)
        sep : int or float, optional
          in points
        frameon : bool, optional
          if True, will draw a box around the horizontal bar and label
        size_vertical : int or float, optional
          vertical length of the size bar, given in data coordinates
        color : str, optional
          color for the size bar and label
        label_top : bool, optional
          if True, the label will be over the rectangle
        fontproperties: a matplotlib.font_manager.FontProperties instance, optional
          sets the font properties for the label text

        Returns:
        --------
        AnchoredSizeBar object

        Example:
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        >>> fig, ax = plt.subplots()
        >>> ax.imshow(np.random.random((10,10)))
        >>> bar = AnchoredSizeBar(ax.transData, 3, '3 units', 4)
        >>> ax.add_artist(bar)
        >>> fig.show()

        Using all the optional parameters

        >>> import matplotlib.font_manager as fm
        >>> fontprops = fm.FontProperties(size=14, family='monospace')
        >>> bar = AnchoredSizeBar(ax.transData, 3, '3 units', 4, pad=0.5, sep=5, borderpad=0.5, frameon=False, size_vertical=0.5, color='white', fontproperties=fontprops)

        """

        self.size_bar = AuxTransformBox(transform)
        self.size_bar.add_artist(Rectangle((0, 0), size, size_vertical,
                                           fill=True, facecolor=color,
                                           edgecolor=color))

        # if fontproperties is None, but `prop` is not, assume that
        # prop should be used to set the font properties. This is
        # questionable behavior
        if fontproperties is None and 'prop' in kwargs:
            fontproperties = kwargs.pop('prop')

        if fontproperties is None:
            textprops = {'color': color}
        else:
            textprops = {'color': color, 'fontproperties': fontproperties}

        self.txt_label = TextArea(
            label,
            minimumdescent=False,
            textprops=textprops)

        if label_top:
            _box_children = [self.txt_label, self.size_bar]
        else:
            _box_children = [self.size_bar, self.txt_label]

        self._box = VPacker(children=_box_children,
                            align="center",
                            pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box,
                                   prop=fontproperties,
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
