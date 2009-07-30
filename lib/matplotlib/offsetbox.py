"""
The OffsetBox is a simple container artist. The child artist are meant
to be drawn at a relative position to its parent.  The [VH]Packer,
DrawingArea and TextArea are derived from the OffsetBox.

The [VH]Packer automatically adjust the relative postisions of their
children, which should be instances of the OffsetBox. This is used to
align similar artists together, e.g., in legend.

The DrawingArea can contain any Artist as a child. The
DrawingArea has a fixed width and height. The position of children
relative to the parent is fixed.  The TextArea is contains a single
Text instance. The width and height of the TextArea instance is the
width and height of the its child text.
"""


import matplotlib.transforms as mtransforms
import matplotlib.artist as martist
import matplotlib.text as mtext
import numpy as np
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox, BboxTransformTo

from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

from matplotlib.patches import bbox_artist as mbbox_artist
DEBUG=False
# for debuging use
def bbox_artist(*args, **kwargs):
    if DEBUG:
        mbbox_artist(*args, **kwargs)


# _get_packed_offsets() and _get_aligned_offsets() are coded assuming
# that we are packing boxes horizontally. But same function will be
# used with vertical packing.

def _get_packed_offsets(wd_list, total, sep, mode="fixed"):
    """
    Geiven a list of (width, xdescent) of each boxes, calculate the
    total width and the x-offset positions of each items according to
    *mode*. xdescent is analagous to the usual descent, but along the
    x-direction. xdescent values are currently ignored.

    *wd_list* : list of (width, xdescent) of boxes to be packed.
    *sep* : spacing between boxes
    *total* : Intended total length. None if not used.
    *mode* : packing mode. 'fixed', 'expand', or 'equal'.
    """

    w_list, d_list = zip(*wd_list)
    # d_list is currently not used.

    if mode == "fixed":
        offsets_ = np.add.accumulate([0]+[w + sep for w in w_list])
        offsets = offsets_[:-1]

        if total is None:
            total = offsets_[-1] - sep

        return total, offsets

    elif mode == "expand":
        sep = (total - sum(w_list))/(len(w_list)-1.)
        offsets_ = np.add.accumulate([0]+[w + sep for w in w_list])
        offsets = offsets_[:-1]

        return total, offsets

    elif mode == "equal":
        maxh = max(w_list)
        if total is None:
            total = (maxh+sep)*len(w_list)
        else:
            sep = float(total)/(len(w_list)) - maxh

        offsets = np.array([(maxh+sep)*i for i in range(len(w_list))])

        return total, offsets

    else:
        raise ValueError("Unknown mode : %s" % (mode,))


def _get_aligned_offsets(hd_list, height, align="baseline"):
    """
    Geiven a list of (height, descent) of each boxes, align the boxes
    with *align* and calculate the y-offsets of each boxes.
    total width and the offset positions of each items according to
    *mode*. xdescent is analagous to the usual descent, but along the
    x-direction. xdescent values are currently ignored.

    *hd_list* : list of (width, xdescent) of boxes to be aligned.
    *sep* : spacing between boxes
    *height* : Intended total length. None if not used.
    *align* : align mode. 'baseline', 'top', 'bottom', or 'center'.
    """

    if height is None:
        height = max([h for h, d in hd_list])

    if align == "baseline":
        height_descent = max([h-d for h, d in hd_list])
        descent = max([d for h, d in hd_list])
        height = height_descent + descent
        offsets = [0. for h, d in hd_list]
    elif align in ["left","top"]:
        descent=0.
        offsets = [d for h, d in hd_list]
    elif align in ["right","bottom"]:
        descent=0.
        offsets = [height-h+d for h, d in hd_list]
    elif align == "center":
        descent=0.
        offsets = [(height-h)*.5+d for h, d in hd_list]
    else:
        raise ValueError("Unknown Align mode : %s" % (align,))

    return height, descent, offsets



class OffsetBox(martist.Artist):
    """
    The OffsetBox is a simple container artist. The child artist are meant
    to be drawn at a relative position to its parent.
    """
    def __init__(self, *args, **kwargs):

        super(OffsetBox, self).__init__(*args, **kwargs)

        self._children = []
        self._offset = (0, 0)

    def set_figure(self, fig):
        """
        Set the figure

        accepts a class:`~matplotlib.figure.Figure` instance
        """
        martist.Artist.set_figure(self, fig)
        for c in self.get_children():
            c.set_figure(fig)

    def set_offset(self, xy):
        """
        Set the offset

        accepts x, y, tuple, or a callable object.
        """
        self._offset = xy

    def get_offset(self, width, height, xdescent, ydescent):
        """
        Get the offset

        accepts extent of the box
        """
        if callable(self._offset):
            return self._offset(width, height, xdescent, ydescent)
        else:
            return self._offset

    def set_width(self, width):
        """
        Set the width

        accepts float
        """
        self.width = width

    def set_height(self, height):
        """
        Set the height

        accepts float
        """
        self.height = height

    def get_visible_children(self):
        """
        Return a list of visible artists it contains.
        """
        return [c for c in self._children if c.get_visible()]

    def get_children(self):
        """
        Return a list of artists it contains.
        """
        return self._children

    def get_extent_offsets(self, renderer):
        raise Exception("")

    def get_extent(self, renderer):
        """
        Return with, height, xdescent, ydescent of box
        """
        w, h, xd, yd, offsets = self.get_extent_offsets(renderer)
        return w, h, xd, yd

    def get_window_extent(self, renderer):
        '''
        get the bounding box in display space.
        '''
        w, h, xd, yd, offsets = self.get_extent_offsets(renderer)
        px, py = self.get_offset(w, h, xd, yd)
        return mtransforms.Bbox.from_bounds(px-xd, py-yd, w, h)

    def draw(self, renderer):
        """
        Update the location of children if necessary and draw them
        to the given *renderer*.
        """

        width, height, xdescent, ydescent, offsets = self.get_extent_offsets(renderer)

        px, py = self.get_offset(width, height, xdescent, ydescent)

        for c, (ox, oy) in zip(self.get_visible_children(), offsets):
            c.set_offset((px+ox, py+oy))
            c.draw(renderer)

        bbox_artist(self, renderer, fill=False, props=dict(pad=0.))


class PackerBase(OffsetBox):
    def __init__(self, pad=None, sep=None, width=None, height=None,
                 align=None, mode=None,
                 children=None):
        """
        *pad* : boundary pad
        *sep* : spacing between items
        *width*, *height* : width and height of the container box.
           calculated if None.
        *align* : alignment of boxes. Can be one of 'top', 'bottom',
           'left', 'right', 'center' and 'baseline'
        *mode* : packing mode

        .. note::
          *pad* and *sep* need to given in points and will be
          scale with the renderer dpi, while *width* and *hight*
          need to be in pixels.
        """
        super(PackerBase, self).__init__()

        self.height = height
        self.width = width
        self.sep = sep
        self.pad = pad
        self.mode = mode
        self.align = align

        self._children = children


class VPacker(PackerBase):
    """
    The VPacker has its children packed vertically. It automatically
    adjust the relative postisions of children in the drawing time.
    """
    def __init__(self, pad=None, sep=None, width=None, height=None,
                 align="baseline", mode="fixed",
                 children=None):
        """
        *pad* : boundary pad
        *sep* : spacing between items
        *width*, *height* : width and height of the container box.
           calculated if None.
        *align* : alignment of boxes
        *mode* : packing mode

        .. note::
          *pad* and *sep* need to given in points and will be
          scale with the renderer dpi, while *width* and *hight*
          need to be in pixels.
        """
        super(VPacker, self).__init__(pad, sep, width, height,
                                      align, mode,
                                      children)



    def get_extent_offsets(self, renderer):
        """
        update offset of childrens and return the extents of the box
        """

        dpicor = renderer.points_to_pixels(1.)
        pad = self.pad * dpicor
        sep = self.sep * dpicor

        if self.width is not None:
            for c in self.get_visible_children():
                if isinstance(c, PackerBase) and c.mode == "expand":
                    c.set_width(self.width)

        whd_list = [c.get_extent(renderer) for c in self.get_visible_children()]
        whd_list = [(w, h, xd, (h-yd)) for w, h, xd, yd in whd_list]


        wd_list = [(w, xd) for w, h, xd, yd in whd_list]
        width, xdescent, xoffsets = _get_aligned_offsets(wd_list,
                                                         self.width,
                                                         self.align)

        pack_list = [(h, yd) for w,h,xd,yd in whd_list]
        height, yoffsets_ = _get_packed_offsets(pack_list, self.height,
                                                sep, self.mode)

        yoffsets = yoffsets_  + [yd for w,h,xd,yd in whd_list]
        ydescent = height - yoffsets[0]
        yoffsets = height - yoffsets

        #w, h, xd, h_yd = whd_list[-1]
        yoffsets = yoffsets - ydescent


        return width + 2*pad, height + 2*pad, \
               xdescent+pad, ydescent+pad, \
               zip(xoffsets, yoffsets)


class HPacker(PackerBase):
    """
    The HPacker has its children packed horizontally. It automatically
    adjust the relative postisions of children in the drawing time.
    """
    def __init__(self, pad=None, sep=None, width=None, height=None,
                 align="baseline", mode="fixed",
                 children=None):
        """
        *pad* : boundary pad
        *sep* : spacing between items
        *width*, *height* : width and height of the container box.
           calculated if None.
        *align* : alignment of boxes
        *mode* : packing mode

        .. note::
          *pad* and *sep* need to given in points and will be
          scale with the renderer dpi, while *width* and *hight*
          need to be in pixels.
        """
        super(HPacker, self).__init__(pad, sep, width, height,
                                      align, mode, children)


    def get_extent_offsets(self, renderer):
        """
        update offset of childrens and return the extents of the box
        """

        dpicor = renderer.points_to_pixels(1.)
        pad = self.pad * dpicor
        sep = self.sep * dpicor

        whd_list = [c.get_extent(renderer) for c in self.get_visible_children()]

        if self.height is None:
            height_descent = max([h-yd for w,h,xd,yd in whd_list])
            ydescent = max([yd for w,h,xd,yd in whd_list])
            height = height_descent + ydescent
        else:
            height = self.height - 2*pad # width w/o pad

        hd_list = [(h, yd) for w, h, xd, yd in whd_list]
        height, ydescent, yoffsets = _get_aligned_offsets(hd_list,
                                                          self.height,
                                                          self.align)


        pack_list = [(w, xd) for w,h,xd,yd in whd_list]
        width, xoffsets_ = _get_packed_offsets(pack_list, self.width,
                                               sep, self.mode)

        xoffsets = xoffsets_  + [xd for w,h,xd,yd in whd_list]

        xdescent=whd_list[0][2]
        xoffsets = xoffsets - xdescent

        return width + 2*pad, height + 2*pad, \
               xdescent + pad, ydescent + pad, \
               zip(xoffsets, yoffsets)



class DrawingArea(OffsetBox):
    """
    The DrawingArea can contain any Artist as a child. The DrawingArea
    has a fixed width and height. The position of children relative to
    the parent is fixed.
    """

    def __init__(self, width, height, xdescent=0.,
                 ydescent=0., clip=True):
        """
        *width*, *height* : width and height of the container box.
        *xdescent*, *ydescent* : descent of the box in x- and y-direction.
        """

        super(DrawingArea, self).__init__()

        self.width = width
        self.height = height
        self.xdescent = xdescent
        self.ydescent = ydescent

        self.offset_transform = mtransforms.Affine2D()
        self.offset_transform.clear()
        self.offset_transform.translate(0, 0)

        self.dpi_transform = mtransforms.Affine2D()



    def get_transform(self):
        """
        Return the :class:`~matplotlib.transforms.Transform` applied
        to the children
        """
        return self.dpi_transform + self.offset_transform

    def set_transform(self, t):
        """
        set_transform is ignored.
        """
        pass


    def set_offset(self, xy):
        """
        set offset of the container.

        Accept : tuple of x,y cooridnate in disokay units.
        """
        self._offset = xy

        self.offset_transform.clear()
        self.offset_transform.translate(xy[0], xy[1])


    def get_offset(self):
        """
        return offset of the container.
        """
        return self._offset


    def get_window_extent(self, renderer):
        '''
        get the bounding box in display space.
        '''
        w, h, xd, yd = self.get_extent(renderer)
        ox, oy = self.get_offset() #w, h, xd, yd)
        return mtransforms.Bbox.from_bounds(ox-xd, oy-yd, w, h)


    def get_extent(self, renderer):
        """
        Return with, height, xdescent, ydescent of box
        """

        dpi_cor = renderer.points_to_pixels(1.)

        return self.width*dpi_cor, self.height*dpi_cor, \
               self.xdescent*dpi_cor, self.ydescent*dpi_cor


    def add_artist(self, a):
        'Add any :class:`~matplotlib.artist.Artist` to the container box'
        self._children.append(a)
        a.set_transform(self.get_transform())


    def draw(self, renderer):
        """
        Draw the children
        """

        dpi_cor = renderer.points_to_pixels(1.)
        self.dpi_transform.clear()
        self.dpi_transform.scale(dpi_cor, dpi_cor)

        for c in self._children:
            c.draw(renderer)

        bbox_artist(self, renderer, fill=False, props=dict(pad=0.))


class TextArea(OffsetBox):
    """
    The TextArea is contains a single Text instance. The text is
    placed at (0,0) with baseline+left alignment. The width and height
    of the TextArea instance is the width and height of the its child
    text.
    """



    def __init__(self, s,
                 textprops=None,
                 multilinebaseline=None,
                 minimumdescent=True,
                 ):
        """
        *s* : a string to be displayed.
        *textprops* : property dictionary for the text
        *multilinebaseline* : If True, baseline for multiline text is
                              adjusted so that it is (approximatedly)
                              center-aligned with singleline text.
        *minimumdescent*  : If True, the box has a minimum descent of "p".
        """
        if textprops is None:
            textprops = {}

        if not textprops.has_key("va"):
            textprops["va"]="baseline"

        self._text = mtext.Text(0, 0, s, **textprops)

        OffsetBox.__init__(self)

        self._children = [self._text]


        self.offset_transform = mtransforms.Affine2D()
        self.offset_transform.clear()
        self.offset_transform.translate(0, 0)
        self._baseline_transform = mtransforms.Affine2D()
        self._text.set_transform(self.offset_transform+self._baseline_transform)

        self._multilinebaseline = multilinebaseline
        self._minimumdescent = minimumdescent


    def set_text(self, s):
        "set text"
        self._text.set_text(s)

    def get_text(self):
        "get text"
        return self._text.get_text()

    def set_multilinebaseline(self, t):
        """
        Set multilinebaseline .

        If True, baseline for multiline text is
        adjusted so that it is (approximatedly) center-aligned with
        singleline text.
        """
        self._multilinebaseline = t


    def get_multilinebaseline(self):
        """
        get multilinebaseline .
        """
        return self._multilinebaseline


    def set_minimumdescent(self, t):
        """
        Set minimumdescent .

        If True, extent of the single line text is adjusted so that
        it has minimum descent of "p"
        """
        self._minimumdescent = t


    def get_minimumdescent(self):
        """
        get minimumdescent.
        """
        return self._minimumdescent


    def set_transform(self, t):
        """
        set_transform is ignored.
        """
        pass


    def set_offset(self, xy):
        """
        set offset of the container.

        Accept : tuple of x,y cooridnate in disokay units.
        """
        self._offset = xy

        self.offset_transform.clear()
        self.offset_transform.translate(xy[0], xy[1])


    def get_offset(self):
        """
        return offset of the container.
        """
        return self._offset


    def get_window_extent(self, renderer):
        '''
        get the bounding box in display space.
        '''
        w, h, xd, yd = self.get_extent(renderer)
        ox, oy = self.get_offset() #w, h, xd, yd)
        return mtransforms.Bbox.from_bounds(ox-xd, oy-yd, w, h)


    def get_extent(self, renderer):
        clean_line, ismath = self._text.is_math_text(self._text._text)
        _, h_, d_ = renderer.get_text_width_height_descent(
            "lp", self._text._fontproperties, ismath=False)

        bbox, info = self._text._get_layout(renderer)
        w, h = bbox.width, bbox.height

        line = info[0][0] # first line

        _, hh, dd = renderer.get_text_width_height_descent(
            line, self._text._fontproperties, ismath=ismath)


        self._baseline_transform.clear()
        d = h-(hh-dd)  # the baseline of the first line
        if len(info) > 1 and self._multilinebaseline:
            d_new = 0.5 * h  - 0.5 * (h_ - d_)
            self._baseline_transform.translate(0, d - d_new)
            d = d_new

        else: # single line

            h_d = max(h_ - d_, h-d)

            if self.get_minimumdescent():
                ## to have a minimum descent, #i.e., "l" and "p" have same
                ## descents.
                d = max(d, d_)
            #else:
            #    d = d

            h = h_d + d

        return w, h, 0., d


    def draw(self, renderer):
        """
        Draw the children
        """

        self._text.draw(renderer)

        bbox_artist(self, renderer, fill=False, props=dict(pad=0.))



class AuxTransformBox(OffsetBox):
    """
    Offset Box with the aux_transform . Its children will be
    transformed with the aux_transform first then will be
    offseted. The absolute coordinate of the aux_transform is meaning
    as it will be automaticcaly adjust so that the left-lower corner
    of the bounding box of children will be set to (0,0) before the
    offset trnasform.

    It is similar to drawing area, except that the extent of the box
    is not predetemined but calculated from the window extent of its
    children. Furthermore, the extent of the children will be
    calculated in the transformed coordinate.
    """

    def __init__(self, aux_transform):
        self.aux_transform = aux_transform
        OffsetBox.__init__(self)

        self.offset_transform = mtransforms.Affine2D()
        self.offset_transform.clear()
        self.offset_transform.translate(0, 0)

        # ref_offset_transform is used to make the offset_transform is
        # always reference to the lower-left corner of the bbox of its
        # children.
        self.ref_offset_transform = mtransforms.Affine2D()
        self.ref_offset_transform.clear()

    def add_artist(self, a):
        'Add any :class:`~matplotlib.artist.Artist` to the container box'
        self._children.append(a)
        a.set_transform(self.get_transform())

    def get_transform(self):
        """
        Return the :class:`~matplotlib.transforms.Transform` applied
        to the children
        """

        return self.aux_transform + \
               self.ref_offset_transform + \
               self.offset_transform

    def set_transform(self, t):
        """
        set_transform is ignored.
        """
        pass


    def set_offset(self, xy):
        """
        set offset of the container.

        Accept : tuple of x,y cooridnate in disokay units.
        """
        self._offset = xy

        self.offset_transform.clear()
        self.offset_transform.translate(xy[0], xy[1])


    def get_offset(self):
        """
        return offset of the container.
        """
        return self._offset


    def get_window_extent(self, renderer):
        '''
        get the bounding box in display space.
        '''
        w, h, xd, yd = self.get_extent(renderer)
        ox, oy = self.get_offset() #w, h, xd, yd)
        return mtransforms.Bbox.from_bounds(ox-xd, oy-yd, w, h)


    def get_extent(self, renderer):

        # clear the offset transforms
        _off = self.ref_offset_transform.to_values() # to be restored later
        self.ref_offset_transform.clear()
        self.offset_transform.clear()

        # calculate the extent
        bboxes = [c.get_window_extent(renderer) for c in self._children]
        ub = mtransforms.Bbox.union(bboxes)


        # adjust ref_offset_tansform
        self.ref_offset_transform.translate(-ub.x0, -ub.y0)
        # restor offset transform
        self.offset_transform.matrix_from_values(*_off)

        return ub.width, ub.height, 0., 0.


    def draw(self, renderer):
        """
        Draw the children
        """

        for c in self._children:
            c.draw(renderer)

        bbox_artist(self, renderer, fill=False, props=dict(pad=0.))



class AnchoredOffsetbox(OffsetBox):
    """
    An offset box placed according to the legend location
    loc. AnchoredOffsetbox has a single child. When multiple children
    is needed, use other OffsetBox class to enlose them.  By default,
    the offset box is anchored against its parent axes. You may
    explicitly specify the bbox_to_anchor.
    """

    zorder = 5 # zorder of the legend

    def __init__(self, loc,
                 pad=0.4, borderpad=0.5,
                 child=None, prop=None, frameon=True,
                 bbox_to_anchor=None,
                 bbox_transform=None,
                 **kwargs):
        """
        loc is a string or an integer specifying the legend location.
        The valid  location codes are::

        'upper right'  : 1,
        'upper left'   : 2,
        'lower left'   : 3,
        'lower right'  : 4,
        'right'        : 5,
        'center left'  : 6,
        'center right' : 7,
        'lower center' : 8,
        'upper center' : 9,
        'center'       : 10,

        pad : pad around the child for drawing a frame. given in
          fraction of fontsize.

        borderpad : pad between offsetbox frame and the bbox_to_anchor,

        child : OffsetBox instance that will be anchored.

        prop : font property. This is only used as a reference for paddings.

        frameon : draw a frame box if True.

        bbox_to_anchor : bbox to anchor. Use self.axes.bbox if None.

        bbox_transform : with which the bbox_to_anchor will be transformed.

        """

        super(AnchoredOffsetbox, self).__init__(**kwargs)

        self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)
        self.set_child(child)

        self.loc = loc
        self.borderpad=borderpad
        self.pad = pad

        if prop is None:
            self.prop=FontProperties(size=rcParams["legend.fontsize"])
        elif isinstance(prop, dict):
            self.prop=FontProperties(**prop)
            if "size" not in prop:
                self.prop.set_size(rcParams["legend.fontsize"])
        else:
            self.prop = prop

        self.patch = FancyBboxPatch(
            xy=(0.0, 0.0), width=1., height=1.,
            facecolor='w', edgecolor='k',
            mutation_scale=self.prop.get_size_in_points(),
            snap=True
            )
        self.patch.set_boxstyle("square",pad=0)
        self._drawFrame =  frameon




    def set_child(self, child):
        "set the child to be anchored"
        self._child = child

    def get_child(self):
        "return the child"
        return self._child

    def get_children(self):
        "return the list of children"
        return [self._child]


    def get_extent(self, renderer):
        """
        return the extent of the artist. The extent of the child
        added with the pad is returned
        """
        w, h, xd, yd =  self.get_child().get_extent(renderer)
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        pad = self.pad * fontsize

        return w+2*pad, h+2*pad, xd+pad, yd+pad


    def get_bbox_to_anchor(self):
        """
        return the bbox that the legend will be anchored
        """
        if self._bbox_to_anchor is None:
            return self.axes.bbox
        else:
            transform = self._bbox_to_anchor_transform
            if transform is None:
                return self._bbox_to_anchor
            else:
                return TransformedBbox(self._bbox_to_anchor,
                                       transform)
                



    def set_bbox_to_anchor(self, bbox, transform=None):
        """
        set the bbox that the child will be anchored.

        *bbox* can be a Bbox instance, a list of [left, bottom, width,
        height], or a list of [left, bottom] where the width and
        height will be assumed to be zero. The bbox will be
        transformed to display coordinate by the given transform. 
        """
        if bbox is None or isinstance(bbox, BboxBase):
            self._bbox_to_anchor = bbox
        else:
            try:
                l = len(bbox)
            except TypeError:
                raise ValueError("Invalid argument for bbox : %s" % str(bbox))

            if l == 2:
                bbox = [bbox[0], bbox[1], 0, 0]

            self._bbox_to_anchor = Bbox.from_bounds(*bbox)

        self._bbox_to_anchor_transform = transform


    def get_window_extent(self, renderer):
        '''
        get the bounding box in display space.
        '''
        self._update_offset_func(renderer)
        w, h, xd, yd = self.get_extent(renderer)
        ox, oy = self.get_offset(w, h, xd, yd)
        return Bbox.from_bounds(ox-xd, oy-yd, w, h)


    def _update_offset_func(self, renderer, fontsize=None):
        """
        Update the offset func which depends on the dpi of the
        renderer (because of the padding).
        """
        if fontsize is None:
            fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())

        def _offset(w, h, xd, yd, fontsize=fontsize, self=self):
            bbox = Bbox.from_bounds(0, 0, w, h)
            borderpad = self.borderpad*fontsize
            bbox_to_anchor = self.get_bbox_to_anchor()

            x0, y0 = self._get_anchored_bbox(self.loc,
                                             bbox,
                                             bbox_to_anchor,
                                             borderpad)
            return x0+xd, y0+yd

        self.set_offset(_offset)


    def draw(self, renderer):
        "draw the artist"

        if not self.get_visible(): return

        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        self._update_offset_func(renderer, fontsize)

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
        """
        return the position of the bbox anchored at the parentbbox
        with the loc code, with the borderpad.
        """
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
