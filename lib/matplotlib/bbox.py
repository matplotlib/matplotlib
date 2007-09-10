"""
A convenience class for handling bounding boxes

2007 Michael Droettboom
"""

import numpy as N

class Interval:
    def __init__(self, bounds):
	self._bounds = N.array(bounds, N.float_)

    def contains(self, value):
	bounds = self._bounds
	return value >= bounds[0] and value <= bounds[1]

    def contains_open(self, value):
	bounds = self._bounds
	return value > bounds[0] and value < bounds[1]

    def get_bounds(self):
	return self._bounds

    def set_bounds(self, lower, upper):
	self._bounds = lower, upper

    def span(self):
	bounds = self._bounds
	return bounds[1] - bounds[0]
	
class Bbox:
    def __init__(self, points):
	self._points = N.array(points, N.float_)

    #@staticmethod
    def unit():
	return Bbox([[0,0], [1,1]])
    unit = staticmethod(unit)

    #@staticmethod
    def from_lbwh(left, bottom, width, height):
	return Bbox([[left, bottom], [left + width, bottom + height]])
    from_lbwh = staticmethod(from_lbwh)

    #@staticmethod
    def from_lbrt(left, bottom, right, top):
	return Bbox([[left, bottom], [right, top]])
    from_lbrt = staticmethod(from_lbrt)

    def copy(self):
	return Bbox(self._points.copy())
    
    # MGDTODO: Probably a more efficient ways to do this...
    def xmin(self):
	return self._points[0,0]

    def ymin(self):
	return self._points[0,1]

    def xmax(self):
	return self._points[1,0]
    
    def ymax(self):
	return self._points[1,1]

    def width(self):
	return self.xmax() - self.xmin()

    def height(self):
	return self.ymax() - self.ymin()

    def transform(self, transform):
	return Bbox(transform(self._points))

    def inverse_transform(self, transform):
	return Bbox(transform.inverted()(self._points))

    def get_bounds(self):
	return (self.xmin(), self.ymin(),
		self.xmax() - self.xmin(), self.ymax() - self.ymin())

    def intervalx(self):
	return Interval(self._points[0])

    def intervaly(self):
	return Interval(self._points[1])

    def scaled(self, sw, sh):
	width = self.width()
	height = self.height()
	deltaw = (sw * width - width) / 2.0
	deltah = (sh * height - height) / 2.0
	a = N.array([[-deltaw, -deltah], [deltaw, deltah]])
	return Bbox(self._points + a)
	
def lbwh_to_bbox(left, bottom, width, height):
    return Bbox([[left, bottom], [left + width, bottom + height]])
    
def bbox_union(bboxes):
    """
    Return the Bbox that bounds all bboxes
    """
    assert(len(bboxes))

    if len(bboxes) == 1:
	return bboxes[0]

    bbox = bboxes[0]
    xmin = bbox.xmin()
    ymin = bbox.ymin()
    xmax = bbox.xmax()
    ymax = bbox.ymax()

    for bbox in bboxes[1:]:
	xmin = min(xmin, bbox.xmin())
	ymin = min(ymin, bbox.ymin())
	xmax = max(xmax, bbox.xmax())
	ymax = max(ymax, bbox.ymax())

    return Bbox.from_lbrt(xmin, ymin, xmax, ymax)

# MGDTODO: There's probably a better place for this
def nonsingular(vmin, vmax, expander=0.001, tiny=1e-15, increasing=True):
    '''
    Ensure the endpoints of a range are not too close together.

    "too close" means the interval is smaller than 'tiny' times
            the maximum absolute value.

    If they are too close, each will be moved by the 'expander'.
    If 'increasing' is True and vmin > vmax, they will be swapped,
    regardless of whether they are too close.
    '''
    swapped = False
    if vmax < vmin:
        vmin, vmax = vmax, vmin
        swapped = True
    if vmax - vmin <= max(abs(vmin), abs(vmax)) * tiny:
        if vmin==0.0:
            vmin = -expander
            vmax = expander
        else:
            vmin -= expander*abs(vmin)
            vmax += expander*abs(vmax)
    if swapped and not increasing:
        vmin, vmax = vmax, vmin
    return vmin, vmax
