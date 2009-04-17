from matplotlib.offsetbox import AnchoredOffsetbox


import matplotlib.transforms as mtrans

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
   def __init__(self, parent_bbox, offsetbox, loc, **kwargs):

       for k in ["parent_bbox", "child", "pad"]:
           if kwargs.has_key(k):
               raise ValueError("%s paramter should not be provided" % (k,))

       kwargs["pad"] = 0.
       kwargs["child"] = None
       kwargs["parent_bbox"] = parent_bbox

       super(AnchoredLocatorBase, self).__init__(loc, **kwargs)


   def draw(self, renderer):
       raise RuntimeError("No draw method should be called")


   def __call__(self, ax, renderer):

       fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
       self._update_offset_func(renderer, fontsize)

       width, height, xdescent, ydescent = self.get_extent(renderer)

       px, py = self.get_offset(width, height, 0, 0)
       bbox_canvas = mtrans.Bbox.from_bounds(px, py, width, height)
       tr = ax.figure.transFigure.inverted()
       bb = mtrans.TransformedBbox(bbox_canvas, tr)

       return bb


class AnchoredOffsetBoxLocator(AnchoredLocatorBase):
   def __init__(self, parent_bbox, offsetbox, loc, **kwargs):

       for k in ["parent_bbox", "child", "pad"]:
           if kwargs.has_key(k):
               raise ValueError("%s paramter should not be provided" % (k,))

       kwargs["pad"] = 0.
       kwargs["child"] = offsetbox
       kwargs["parent_bbox"] = parent_bbox

       super(AnchoredOffsetBoxLocator, self).__init__(loc, **kwargs)



from mpl_toolkits.axes_grid.axes_divider import Size

class AnchoredSizeLocator(AnchoredLocatorBase):
   def __init__(self, parent_bbox, x_size, y_size,
                 loc, **kwargs):

      self.axes = None
      self.x_size = Size.from_any(x_size)
      self.y_size = Size.from_any(y_size)

      super(AnchoredSizeLocator, self).__init__(parent_bbox, None, loc, **kwargs)

   def get_extent(self, renderer):

      x, y, w, h = self._parent_bbox.bounds

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
   def __init__(self, parent_axes, zoom,
                 loc, **kwargs):

      self.parent_axes = parent_axes
      self.zoom = zoom

      super(AnchoredZoomLocator, self).__init__(parent_axes.bbox, None, loc, **kwargs)

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


# class AnchoredAxesBoxLocator(AnchoredOffsetBoxLocator):
#     def __init__(self, ax, x0, x1, y0, y1, zoom, zoomy=None,
#                  parent_bbox, width_inch, height_inch,
#                  loc, **kwargs):

#         self.width_inch = width_inch
#         self.height_inch = height_inch

#         super(AnchoredFixedBoxLocator, self).__init__(parent_bbox, None, loc, **kwargs)

#     def get_extent(self, renderer):

#         w =self.width_inch * renderer.points_to_pixels(72.)
#         h =self.height_inch * renderer.points_to_pixels(72.)
#         xd, yd = 0, 0

#         fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
#         pad = self.pad * fontsize

#         return w+2*pad, h+2*pad, xd+pad, yd+pad


if __name__ == "__main__":

   import matplotlib.pyplot as plt

   fig = plt.figure(1)
   ax = fig.add_subplot(1,2,1)
   ax.set_aspect(1.)

   # width : 30% of parent_bbox (ax.bbox)
   # height : 1 inch
   axes_locator = AnchoredSizeLocator(ax.bbox, "30%", 1, loc=1)

   axins = fig.add_axes([0, 0, 1, 1], label="inset1")
   axins.set_axes_locator(axes_locator)




   ax = fig.add_subplot(1,2,2)
   ax.set_aspect(1.)

   # inset axes has a data scale of the parent axes multiplied by a zoom factor
   axes_locator = AnchoredZoomLocator(ax, zoom=0.5, loc=1)

   axins = fig.add_axes([0, 0, 0.5, 1], label="inset2")
   #axins = plt.axes([0, 0, 1, 1])
   axins.set_axes_locator(axes_locator)



   


   #locator = AnchoredBoxLocator(parent_bbox, Fixed(1.), Scaled(0.2), loc=1)
