

from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib.patches import Rectangle, Ellipse

from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker,\
     TextArea, DrawingArea


class AnchoredText(AnchoredOffsetbox):
    def __init__(self, s, loc, pad=0.4, borderpad=0.5, prop=None, **kwargs):

        self.txt = TextArea(s, textprops=prop,
                            minimumdescent=False)
        fp = self.txt._text.get_fontproperties()

        super(AnchoredText, self).__init__(loc, pad=pad, borderpad=borderpad,
                                           child=self.txt,
                                           prop=fp,
                                           **kwargs)



class AnchoredDrawingArea(AnchoredOffsetbox):
    def __init__(self, width, height, xdescent, ydescent,
                 loc, pad=0.4, borderpad=0.5, prop=None, frameon=True,
                 **kwargs):

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
        A label will be drawn underneath (center-alinged).

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

