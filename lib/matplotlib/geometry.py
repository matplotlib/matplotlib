"""
Geometry utilities
"""
import numpy as np

def orientation(a, b, c):
    """
    Return the orientation of the three points.
    The result is 1 if *a*, *b*, *c* are oriented counterclockwise
    (i.e. *c* is to the left of *b* as seen from *a*),
    -1 if they are oriented clockwise, 0 if they are collinear.
    """
    return np.sign(np.linalg.det(np.c_[[a,b,c], [1,1,1]]))

def project_point_on_line_through(p0, p1, p2, require_convex=False):
    """
    Project the point *p0* onto the line through points *p1* and *p2*.
    Return the coordinates of the projection. All points are 2D.

    If *require_convex* is true and the projection is outside
    the segment from *p1* and *p2*, raise ValueError.
    """
    len_p1_p2 = np.sqrt(np.dot(p2-p1, p2-p1))
    unitvec = (p2 - p1) / len_p1_p2
    dot = np.dot(p0 - p1, unitvec)
    if require_convex and not 0 <= dot <= len_p1_p2:
        raise ValueError, "projection not between p1 and p2"
    return p1 + dot * unitvec
