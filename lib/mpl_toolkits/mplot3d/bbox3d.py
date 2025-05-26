from matplotlib.transforms import Bbox

class _Bbox3d:
    """
    A helper class to represent a 3D bounding box.

    This class stores the minimum and maximum extents of data in 3D space
    (xmin, xmax, ymin, ymax, zmin, zmax). It provides methods to convert
    these extents into 2D bounding boxes (`Bbox`) for compatibility with
    existing matplotlib functionality.

    Attributes
    ----------
    xmin, xmax : float
        The minimum and maximum extents along the x-axis.
    ymin, ymax : float
        The minimum and maximum extents along the y-axis.
    zmin, zmax : float
        The minimum and maximum extents along the z-axis.

    Methods
    -------
    to_bbox_xy():
        Converts the x and y extents into a 2D `Bbox`.
    to_bbox_zz():
        Converts the z extents into a 2D `Bbox`, with the y-component unused.
    """
    def __init__(self, points):
        ((self.xmin, self.xmax),
         (self.ymin, self.ymax),
         (self.zmin, self.zmax)) = points

    def to_bbox_xy(self):
        return Bbox(((self.xmin, self.xmax), (self.ymin, self.ymax)))

    def to_bbox_zz(self):
        # first component contains z, second is unused
        return Bbox(((self.zmin, self.zmax), (0, 0)))