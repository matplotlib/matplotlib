"""
Place a text (or any offsetbox artist) at the corner of the axes, like a lenged.
"""

from matplotlib.offsetbox import TextArea, OffsetBox, DrawingArea
from matplotlib.transforms import Bbox
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Circle


class AnchoredOffsetbox(OffsetBox):
    def __init__(self, loc, pad=0.4, borderpad=0.5,
                 child=None, fontsize=None, frameon=True):

        super(AnchoredOffsetbox, self).__init__()

        self.set_child(child)

        self.loc = loc
        self.borderpad=borderpad
        self.pad = pad

        if fontsize is None:
            prop=FontProperties(size=rcParams["legend.fontsize"])
            self._fontsize = prop.get_size_in_points()
        else:
            self._fontsize = fontsize



        self.patch = FancyBboxPatch(
            xy=(0.0, 0.0), width=1., height=1.,
            facecolor='w', edgecolor='k',
            mutation_scale=self._fontsize,
            snap=True
            )
        self.patch.set_boxstyle("square",pad=0)
        self._drawFrame =  frameon

    def set_child(self, child):
        self._child = child

    def get_children(self):
        return [self._child]

    def get_child(self):
        return self._child

    def get_extent(self, renderer):
        w, h, xd, yd =  self.get_child().get_extent(renderer)
        fontsize = renderer.points_to_pixels(self._fontsize)
        pad = self.pad * fontsize

        return w+2*pad, h+2*pad, xd+pad, yd+pad

    def get_window_extent(self, renderer):
        '''
        get the bounding box in display space.
        '''
        w, h, xd, yd = self.get_extent(renderer)
        ox, oy = self.get_offset(w, h, xd, yd)
        return Bbox.from_bounds(ox-xd, oy-yd, w, h)

    def draw(self, renderer):

        if not self.get_visible(): return

        fontsize = renderer.points_to_pixels(self._fontsize)

        def _offset(w, h, xd, yd, fontsize=fontsize, self=self):
            bbox = Bbox.from_bounds(0, 0, w, h)
            borderpad = self.borderpad*fontsize
            x0, y0 = self._get_anchored_bbox(self.loc,
                                             bbox,
                                             self.axes.bbox,
                                             borderpad)
            return x0+xd, y0+yd

        self.set_offset(_offset)

        if self._drawFrame:
            # update the location and size of the legend
            bbox = self.get_window_extent(renderer)
            self.patch.set_bounds(bbox.x0, bbox.y0,
                                  bbox.width, bbox.height)

            self.patch.set_mutation_scale(fontsize)

            self.patch.draw(renderer)


        width, height, xdescent, ydescent = self.get_extent(renderer)

        px, py = self.get_offset(width, height, xdescent, ydescent)

        self.get_child().set_offset((px, py))
        self.get_child().draw(renderer)



    def _get_anchored_bbox(self, loc, bbox, parentbbox, borderpad):
        assert loc in range(1,11) # called only internally

        BEST, UR, UL, LL, LR, R, CL, CR, LC, UC, C = range(11)

        anchor_coefs={UR:"NE",
                      UL:"NW",
                      LL:"SW",
                      LR:"SE",
                      R:"E",
                      CL:"W",
                      CR:"E",
                      LC:"S",
                      UC:"N",
                      C:"C"}

        c = anchor_coefs[loc]

        container = parentbbox.padded(-borderpad)
        anchored_box = bbox.anchored(c, container=container)
        return anchored_box.x0, anchored_box.y0


class AnchoredText(AnchoredOffsetbox):
    def __init__(self, s, loc, pad=0.4, borderpad=0.5, prop=None, frameon=True):

        self.txt = TextArea(s,
                            minimumdescent=False)


        if prop is None:
            self.prop=FontProperties(size=rcParams["legend.fontsize"])
        else:
            self.prop=prop


        super(AnchoredText, self).__init__(loc, pad=pad, borderpad=borderpad,
                                           child=self.txt,
                                           fontsize=self.prop.get_size_in_points(),
                                           frameon=frameon)


class AnchoredDrawingArea(AnchoredOffsetbox):
    def __init__(self, width, height, xdescent, ydescent,
                 loc, pad=0.4, borderpad=0.5, fontsize=None, frameon=True):

        self.da = DrawingArea(width, height, xdescent, ydescent, clip=True)

        super(AnchoredDrawingArea, self).__init__(loc, pad=pad, borderpad=borderpad,
                                                  child=self.da,
                                                  fontsize=fontsize,
                                                  frameon=frameon)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #ax = plt.subplot(1,1,1)
    plt.clf()
    plt.cla()
    plt.draw()
    ax = plt.gca()
    #ax.set_aspect(1.)

    at = AnchoredText("Figure 1(a)", loc=2, frameon=False)
    ax.add_artist(at)

    ada = AnchoredDrawingArea(20, 20, 0, 0, loc=3, pad=0., frameon=False)

    p = Circle((10, 10), 10)
    ada.da.add_artist(p)
    ax.add_artist(ada)

    ax.plot([0,1])
    plt.draw()

    plt.show()



