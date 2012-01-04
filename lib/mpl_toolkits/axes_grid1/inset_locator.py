from matplotlib.offsetbox import AnchoredOffsetbox
#from matplotlib.transforms import IdentityTransform

import matplotlib.transforms as mtrans
#from matplotlib.axes import Axes
from mpl_axes import Axes

from matplotlib.transforms import Bbox, TransformedBbox, IdentityTransform

from matplotlib.patches import Patch
from matplotlib.path import Path

from matplotlib.patches import Rectangle


class InsetPosition(object):
   def __init__(self, parent, lbwh):
       self.parent = parent
       self.lbwh = lbwh # position of the inset axes in the normalized coordinate of the parent axes

   def __call__(self, ax, renderer):
       bbox_parent = self.parent.get_position(original=False)
       trans = mtrans.BboxTransformTo(bbox_parent)
       bbox_inset = mtrans.Bbox.from_bounds(*self.lbwh)
       bb = mtrans.TransformedBbox(bbox_inset, trans)
       return bb


class AnchoredLocatorBase(AnchoredOffsetbox):
   def __init__(self, bbox_to_anchor, offsetbox, loc,
                borderpad=0.5, bbox_transform=None):

       super(AnchoredLocatorBase, self).__init__(loc,
                                                 pad=0., child=None,
                                                 borderpad=borderpad,
                                                 bbox_to_anchor=bbox_to_anchor,
                                                 bbox_transform=bbox_transform)


   def draw(self, renderer):
       raise RuntimeError("No draw method should be called")


   def __call__(self, ax, renderer):

       fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
       self._update_offset_func(renderer, fontsize)

       width, height, xdescent, ydescent = self.get_extent(renderer)

       px, py = self.get_offset(width, height, 0, 0, renderer)
       bbox_canvas = mtrans.Bbox.from_bounds(px, py, width, height)
       tr = ax.figure.transFigure.inverted()
       bb = mtrans.TransformedBbox(bbox_canvas, tr)

       return bb




import axes_size as Size

class AnchoredSizeLocator(AnchoredLocatorBase):
   def __init__(self, bbox_to_anchor, x_size, y_size, loc,
                borderpad=0.5, bbox_transform=None):

      self.axes = None
      self.x_size = Size.from_any(x_size)
      self.y_size = Size.from_any(y_size)

      super(AnchoredSizeLocator, self).__init__(bbox_to_anchor, None, loc,
                                                borderpad=borderpad,
                                                bbox_transform=bbox_transform)

   def get_extent(self, renderer):

      x, y, w, h = self.get_bbox_to_anchor().bounds

      dpi = renderer.points_to_pixels(72.)

      r, a = self.x_size.get_size(renderer)
      width = w*r + a*dpi

      r, a = self.y_size.get_size(renderer)
      height = h*r + a*dpi
      xd, yd = 0, 0

      fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
      pad = self.pad * fontsize

      return width+2*pad, height+2*pad, xd+pad, yd+pad


   def __call__(self, ax, renderer):

      self.axes = ax
      return super(AnchoredSizeLocator, self).__call__(ax, renderer)


class AnchoredZoomLocator(AnchoredLocatorBase):
   def __init__(self, parent_axes, zoom, loc,
                borderpad=0.5,
                bbox_to_anchor=None,
                bbox_transform=None):

      self.parent_axes = parent_axes
      self.zoom = zoom

      if bbox_to_anchor is None:
         bbox_to_anchor = parent_axes.bbox

      super(AnchoredZoomLocator, self).__init__(bbox_to_anchor, None, loc,
                                                borderpad=borderpad,
                                                bbox_transform=bbox_transform)

      self.axes = None


   def get_extent(self, renderer):

      bb = mtrans.TransformedBbox(self.axes.viewLim, self.parent_axes.transData)

      x, y, w, h = bb.bounds

      xd, yd = 0, 0

      fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
      pad = self.pad * fontsize

      return w*self.zoom+2*pad, h*self.zoom+2*pad, xd+pad, yd+pad


   def __call__(self, ax, renderer):

      self.axes = ax
      return super(AnchoredZoomLocator, self).__call__(ax, renderer)






class BboxPatch(Patch):
   def __init__(self, bbox, **kwargs):
        if "transform" in kwargs:
           raise ValueError("transform should not be set")

        kwargs["transform"] = IdentityTransform()
        Patch.__init__(self, **kwargs)
        self.bbox = bbox

   def get_path(self):
       x0, y0, x1, y1 = self.bbox.extents

       verts = [(x0, y0),
                (x1, y0),
                (x1, y1),
                (x0, y1),
                (x0, y0),
                (0,0)]

       codes = [Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY]

       return Path(verts, codes)




class BboxConnector(Patch):

    @staticmethod
    def get_bbox_edge_pos(bbox, loc):
      x0, y0, x1, y1 = bbox.extents
      if loc==1:
         return x1, y1
      elif loc==2:
         return x0, y1
      elif loc==3:
         return x0, y0
      elif loc==4:
         return x1, y0

    @staticmethod
    def connect_bbox(bbox1, bbox2, loc1, loc2=None):
       if isinstance(bbox1, Rectangle):
          transform = bbox1.get_transfrom()
          bbox1 = Bbox.from_bounds(0, 0, 1, 1)
          bbox1 = TransformedBbox(bbox1, transform)

       if isinstance(bbox2, Rectangle):
          transform = bbox2.get_transform()
          bbox2 = Bbox.from_bounds(0, 0, 1, 1)
          bbox2 = TransformedBbox(bbox2, transform)

       if loc2 is None:
          loc2 = loc1

       x1, y1 = BboxConnector.get_bbox_edge_pos(bbox1, loc1)
       x2, y2 = BboxConnector.get_bbox_edge_pos(bbox2, loc2)

       verts = [[x1, y1], [x2,y2]]
       #Path()

       codes = [Path.MOVETO, Path.LINETO]

       return Path(verts, codes)


    def __init__(self, bbox1, bbox2, loc1, loc2=None, **kwargs):
        """
        *path* is a :class:`matplotlib.path.Path` object.

        Valid kwargs are:
        %(Patch)s

        .. seealso::

            :class:`Patch`
                For additional kwargs

        """
        if "transform" in kwargs:
           raise ValueError("transform should not be set")

        kwargs["transform"] = IdentityTransform()
        Patch.__init__(self, **kwargs)
        self.bbox1 = bbox1
        self.bbox2 = bbox2
        self.loc1 = loc1
        self.loc2 = loc2


    def get_path(self):
       return self.connect_bbox(self.bbox1, self.bbox2,
                                self.loc1, self.loc2)


class BboxConnectorPatch(BboxConnector):

    def __init__(self, bbox1, bbox2, loc1a, loc2a, loc1b, loc2b, **kwargs):
        if "transform" in kwargs:
            raise ValueError("transform should not be set")
        BboxConnector.__init__(self, bbox1, bbox2, loc1a, loc2a, **kwargs)
        self.loc1b = loc1b
        self.loc2b = loc2b

    def get_path(self):
        path1 = self.connect_bbox(self.bbox1, self.bbox2, self.loc1, self.loc2)
        path2 = self.connect_bbox(self.bbox2, self.bbox1, self.loc2b, self.loc1b)
        path_merged = list(path1.vertices) + list (path2.vertices) + [path1.vertices[0]]
        return Path(path_merged)



def _add_inset_axes(parent_axes, inset_axes):
   parent_axes.figure.add_axes(inset_axes)
   inset_axes.set_navigate(False)


def inset_axes(parent_axes, width, height, loc=1,
               bbox_to_anchor=None, bbox_transform=None,
               axes_class=None,
               axes_kwargs=None,
               **kwargs):

   if axes_class is None:
      axes_class = Axes

   if axes_kwargs is None:
      inset_axes = axes_class(parent_axes.figure, parent_axes.get_position())
   else:
      inset_axes = axes_class(parent_axes.figure, parent_axes.get_position(),
                              **axes_kwargs)

   if bbox_to_anchor is None:
      bbox_to_anchor = parent_axes.bbox

   axes_locator = AnchoredSizeLocator(bbox_to_anchor,
                                      width, height,
                                      loc=loc,
                                      bbox_transform=bbox_transform,
                                      **kwargs)

   inset_axes.set_axes_locator(axes_locator)

   _add_inset_axes(parent_axes, inset_axes)

   return inset_axes


def zoomed_inset_axes(parent_axes, zoom, loc=1,
                      bbox_to_anchor=None, bbox_transform=None,
                      axes_class=None,
                      axes_kwargs=None,
                      **kwargs):

   if axes_class is None:
      axes_class = Axes

   if axes_kwargs is None:
      inset_axes = axes_class(parent_axes.figure, parent_axes.get_position())
   else:
      inset_axes = axes_class(parent_axes.figure, parent_axes.get_position(),
                              **axes_kwargs)

   axes_locator = AnchoredZoomLocator(parent_axes, zoom=zoom, loc=loc,
                                      bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform,
                                      **kwargs)
   inset_axes.set_axes_locator(axes_locator)

   _add_inset_axes(parent_axes, inset_axes)

   return inset_axes


def mark_inset(parent_axes, inset_axes, loc1, loc2, **kwargs):
   rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

   pp = BboxPatch(rect, **kwargs)
   parent_axes.add_patch(pp)

   p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1, **kwargs)
   inset_axes.add_patch(p1)
   p1.set_clip_on(False)
   p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2, **kwargs)
   inset_axes.add_patch(p2)
   p2.set_clip_on(False)

   return pp, p1, p2



