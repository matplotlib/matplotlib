import warnings

from matplotlib import path

def pnpoly(x, y, xyverts):
    """
    inside = pnpoly(x, y, xyverts)

    Return 1 if x,y is inside the polygon, 0 otherwise.

    *xyverts*
        a sequence of x,y vertices.

    A point on the boundary may be treated as inside or outside.

    Deprecated: Use `matplotlib.path.Path.contains_point` instead.
    """
    warings.warn(
        DeprecationWarning,
        "nxutils is deprecated.  Use matplotlib.path.Path.contains_point instead.")

    p = path.Path(xyverts)
    return p.contains_point(x, y)

def points_inside_poly(xypoints, xyverts):
    """
    mask = points_inside_poly(xypoints, xyverts)

    Returns a boolean ndarray, True for points inside the polygon.

    *xypoints*
        a sequence of N x,y pairs.

    *xyverts*
        sequence of x,y vertices of the polygon.

    A point on the boundary may be treated as inside or outside.

    Deprecated: Use `matplotlib.path.Path.contains_points` instead.
    """
    warnings.warn(
        DeprecationWarning,
        "nxutils is deprecated.  Use matplotlib.path.Path.contains_points instead.")

    p = path.Path(xyverts)
    return p.contains_points(xypoints)
