"""
A module providing some utility functions regarding Bezier path manipulation.
"""

import math

import numpy as np

import matplotlib.cbook as cbook
from matplotlib.path import Path


class NonIntersectingPathException(ValueError):
    pass

# some functions


def get_intersection(cx1, cy1, cos_t1, sin_t1,
                     cx2, cy2, cos_t2, sin_t2):
    """
    Return the intersection between the line through (*cx1*, *cy1*) at angle
    *t1* and the line through (*cx2*, *cy2*) at angle *t2*.
    """

    # line1 => sin_t1 * (x - cx1) - cos_t1 * (y - cy1) = 0.
    # line1 => sin_t1 * x + cos_t1 * y = sin_t1*cx1 - cos_t1*cy1

    line1_rhs = sin_t1 * cx1 - cos_t1 * cy1
    line2_rhs = sin_t2 * cx2 - cos_t2 * cy2

    # rhs matrix
    a, b = sin_t1, -cos_t1
    c, d = sin_t2, -cos_t2

    ad_bc = a * d - b * c
    if abs(ad_bc) < 1e-12:
        raise ValueError("Given lines do not intersect. Please verify that "
                         "the angles are not equal or differ by 180 degrees.")

    # rhs_inverse
    a_, b_ = d, -b
    c_, d_ = -c, a
    a_, b_, c_, d_ = [k / ad_bc for k in [a_, b_, c_, d_]]

    x = a_ * line1_rhs + b_ * line2_rhs
    y = c_ * line1_rhs + d_ * line2_rhs

    return x, y


def get_normal_points(cx, cy, cos_t, sin_t, length):
    """
    For a line passing through (*cx*, *cy*) and having an angle *t*, return
    locations of the two points located along its perpendicular line at the
    distance of *length*.
    """

    if length == 0.:
        return cx, cy, cx, cy

    cos_t1, sin_t1 = sin_t, -cos_t
    cos_t2, sin_t2 = -sin_t, cos_t

    x1, y1 = length * cos_t1 + cx, length * sin_t1 + cy
    x2, y2 = length * cos_t2 + cx, length * sin_t2 + cy

    return x1, y1, x2, y2


# BEZIER routines

# subdividing bezier curve
# http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html


def _de_casteljau1(beta, t):
    next_beta = beta[:-1] * (1 - t) + beta[1:] * t
    return next_beta


def split_de_casteljau(beta, t):
    """
    Split a Bezier segment defined by its control points *beta* into two
    separate segments divided at *t* and return their control points.
    """
    beta = np.asarray(beta)
    beta_list = [beta]
    while True:
        beta = _de_casteljau1(beta, t)
        beta_list.append(beta)
        if len(beta) == 1:
            break
    left_beta = [beta[0] for beta in beta_list]
    right_beta = [beta[-1] for beta in reversed(beta_list)]

    return left_beta, right_beta


def find_bezier_t_intersecting_with_closedpath(
        bezier_point_at_t, inside_closedpath, t0=0., t1=1., tolerance=0.01):
    """
    Find the intersection of the Bezier curve with a closed path.

    The intersection point *t* is approximated by two parameters *t0*, *t1*
    such that *t0* <= *t* <= *t1*.

    Search starts from *t0* and *t1* and uses a simple bisecting algorithm
    therefore one of the end points must be inside the path while the other
    doesn't. The search stops when the distance of the points parametrized by
    *t0* and *t1* gets smaller than the given *tolerance*.

    Parameters
    ----------
    bezier_point_at_t : callable
        A function returning x, y coordinates of the Bezier at parameter *t*.
        It must have the signature::

            bezier_point_at_t(t: float) -> Tuple[float, float]

    inside_closedpath : callable
        A function returning True if a given point (x, y) is inside the
        closed path. It must have the signature::

            inside_closedpath(point: Tuple[float, float]) -> bool

    t0, t1 : float
        Start parameters for the search.

    tolerance : float
        Maximal allowed distance between the final points.

    Returns
    -------
    t0, t1 : float
        The Bezier path parameters.
    """
    start = bezier_point_at_t(t0)
    end = bezier_point_at_t(t1)

    start_inside = inside_closedpath(start)
    end_inside = inside_closedpath(end)

    if start_inside == end_inside and start != end:
        raise NonIntersectingPathException(
            "Both points are on the same side of the closed path")

    while True:

        # return if the distance is smaller than the tolerance
        if np.hypot(start[0] - end[0], start[1] - end[1]) < tolerance:
            return t0, t1

        # calculate the middle point
        middle_t = 0.5 * (t0 + t1)
        middle = bezier_point_at_t(middle_t)
        middle_inside = inside_closedpath(middle)

        if start_inside ^ middle_inside:
            t1 = middle_t
            end = middle
            end_inside = middle_inside
        else:
            t0 = middle_t
            start = middle
            start_inside = middle_inside


class BezierSegment:
    """
    A D-dimensional Bezier segment.

    Parameters
    ----------
    control_points : (N, D) array
        Location of the *N* control points.
    """

    def __init__(self, control_points):
        self.cpoints = np.asarray(control_points)
        self.n, self.d = self.cpoints.shape
        self._orders = np.arange(self.n)
        coeff = [math.factorial(self.n - 1)
                 // (math.factorial(i) * math.factorial(self.n - 1 - i))
                 for i in range(self.n)]
        self._px = self.cpoints.T * coeff

    def point_at_t(self, t):
        """Return the point on the Bezier curve for parameter *t*."""
        return tuple(
            self._px @ (((1 - t) ** self._orders)[::-1] * t ** self._orders))

    @property
    def tan_in(self):
        if self.n < 2:
            raise ValueError("Need at least two control points to get tangent "
                             "vector!")
        return self.cpoints[1] - self.cpoints[0]

    @property
    def tan_out(self):
        if self.n < 2:
            raise ValueError("Need at least two control points to get tangent "
                             "vector!")
        return self.cpoints[-1] - self.cpoints[-2]

    @property
    def interior_extrema(self):
        if self.n <= 2: # a line's extrema are always its tips
            return np.array([]), np.array([])
        elif self.n == 3: # quadratic curve
            # the bezier curve in standard form is
            # cp[0] * (1 - t)^2 + cp[1] * 2t(1-t) + cp[2] * t^2
            # can be re-written as
            # cp[0] + 2 (cp[1] - cp[0]) t + (cp[2] - 2 cp[1] + cp[0]) t^2
            # which has simple derivative
            # 2*(cp[2] - 2*cp[1] + cp[0]) t + 2*(cp[1] - cp[0])
            num = 2*(self.cpoints[2] - 2*self.cpoints[1] + self.cpoints[0])
            denom = self.cpoints[1] - self.cpoints[0]
            mask = ~np.isclose(denom, 0)
            zeros = num[mask]/denom[mask]
            dims = np.arange(self.d)[mask]
            in_range = (0 <= zeros) & (zeros <= 1)
            return dims[in_range], zeros[in_range]
        elif self.n == 4: # cubic curve
            # derivative of cubic bezier curve has coefficients
            a = 3*(points[3] - 3*points[2] + 3*points[1] - points[0])
            b = 6*(points[2] - 2*points[1] + points[0])
            c = 3*(points[1] - points[0])
            under_sqrt = b**2 - 4*a*c
            dims = []
            zeros = []
            for i in range(d):
                if under_sqrt[i] < 0:
                    continue
                roots = [(-b + np.sqrt(under_sqrt))/2/a,
                        (-b - np.sqrt(under_sqrt))/2/a]
                for root in roots:
                    if 0 <= root <= 1:
                        dims.append(i)
                        zeros.append(root)
            return np.asarray(dims), np.asarray(zeros)
        else: # self.n > 4:
            raise NotImplementedError("Zero finding only implemented up to "
                                      "cubic curves.")


def split_bezier_intersecting_with_closedpath(
        bezier, inside_closedpath, tolerance=0.01):
    """
    Split a Bezier curve into two at the intersection with a closed path.

    Parameters
    ----------
    bezier : array-like(N, 2)
        Control points of the Bezier segment. See `.BezierSegment`.
    inside_closedpath : callable
        A function returning True if a given point (x, y) is inside the
        closed path. See also `.find_bezier_t_intersecting_with_closedpath`.
    tolerance : float
        The tolerance for the intersection. See also
        `.find_bezier_t_intersecting_with_closedpath`.

    Returns
    -------
    left, right
        Lists of control points for the two Bezier segments.
    """

    bz = BezierSegment(bezier)
    bezier_point_at_t = bz.point_at_t

    t0, t1 = find_bezier_t_intersecting_with_closedpath(
        bezier_point_at_t, inside_closedpath, tolerance=tolerance)

    _left, _right = split_de_casteljau(bezier, (t0 + t1) / 2.)
    return _left, _right


# matplotlib specific


def split_path_inout(path, inside, tolerance=0.01, reorder_inout=False):
    """
    Divide a path into two segments at the point where ``inside(x, y)`` becomes
    False.
    """
    path_iter = path.iter_segments()

    ctl_points, command = next(path_iter)
    begin_inside = inside(ctl_points[-2:])  # true if begin point is inside

    ctl_points_old = ctl_points

    concat = np.concatenate

    iold = 0
    i = 1

    for ctl_points, command in path_iter:
        iold = i
        i += len(ctl_points) // 2
        if inside(ctl_points[-2:]) != begin_inside:
            bezier_path = concat([ctl_points_old[-2:], ctl_points])
            break
        ctl_points_old = ctl_points
    else:
        raise ValueError("The path does not intersect with the patch")

    bp = bezier_path.reshape((-1, 2))
    left, right = split_bezier_intersecting_with_closedpath(
        bp, inside, tolerance)
    if len(left) == 2:
        codes_left = [Path.LINETO]
        codes_right = [Path.MOVETO, Path.LINETO]
    elif len(left) == 3:
        codes_left = [Path.CURVE3, Path.CURVE3]
        codes_right = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    elif len(left) == 4:
        codes_left = [Path.CURVE4, Path.CURVE4, Path.CURVE4]
        codes_right = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    else:
        raise AssertionError("This should never be reached")

    verts_left = left[1:]
    verts_right = right[:]

    if path.codes is None:
        path_in = Path(concat([path.vertices[:i], verts_left]))
        path_out = Path(concat([verts_right, path.vertices[i:]]))

    else:
        path_in = Path(concat([path.vertices[:iold], verts_left]),
                       concat([path.codes[:iold], codes_left]))

        path_out = Path(concat([verts_right, path.vertices[i:]]),
                        concat([codes_right, path.codes[i:]]))

    if reorder_inout and not begin_inside:
        path_in, path_out = path_out, path_in

    return path_in, path_out


def inside_circle(cx, cy, r):
    """
    Return a function that checks whether a point is in a circle with center
    (*cx*, *cy*) and radius *r*.

    The returned function has the signature::

        f(xy: Tuple[float, float]) -> bool
    """
    r2 = r ** 2

    def _f(xy):
        x, y = xy
        return (x - cx) ** 2 + (y - cy) ** 2 < r2
    return _f


CornerInfo = namedtuple('CornerInfo', 'apex incidence_angle corner_angle')
r"""Used to have a universal way to account for how much the bounding box of a
shape will grow as we increase its `markeredgewidth`.

Attributes
----------
    `apex` : float
        the vertex that marks the "tip" of the corner
    `incidence_angle` : float
        the angle that the corner bisector makes with the box edge (where
        top/bottom box edges are horizontal, left/right box edges are
        vertical).
    `corner_angle` : float
        the internal angle of the corner, where np.pi is a straight line, and 0
        is retracing exactly the way you came. None can be used to signify that
        the line ends there (i.e. no corner).

Notes
-----
$\pi$ and 0 are equivalent for `corner_angle`. Both $\theta$ and $\pi - \theta$
are equivalent for `incidence_angle` by symmetry."""


def _incidence_corner_from_angles(angle_1, angle_2):
    """Gets CornerInfo from direction of lines making up corner.

    This function expects angle_1 and angle_2 (in radians) to be
    the orientation of lines 1 and 2 (arbitrarily chosen to point
    towards the corner where they meet) relative to the coordinate
    system.

    Helper function for iter_corners.

    Returns
    -------
    incidence_angle : float in [0, 2*pi]
        as described in CornerInfo docs
    corner_angle : float in [0, pi]
        as described in CornerInfo docs
    """
    # get "interior" angle between tangents to joined curves' tips
    corner_angle = np.abs(angle_1 - angle_2)
    if corner_angle > np.pi:
        corner_angle = 2*np.pi - corner_angle
    # since [-pi, pi], we need to sort to avoid modulo
    smaller_angle = min(angle_1, angle_2)
    larger_angle = max(angle_1, angle_2)
    if np.isclose(smaller_angle + corner_angle, larger_angle):
        incident_angle = smaller_angle + corner_angle/2
    else:
        incident_angle = smaller_angle - corner_angle/2
    return incident_angle, corner_angle


def iter_corners(path, **kwargs):
    """Iterate over a mpl.path.Path object and return information about every
    cap and corner.

    Parameters
    ----------
    path : mpl.path.Path
        the path to extract corners from
    kwargs : Dict[str, object]
        passed onto Path.iter_curves

    Yields
    ------
    corner : CornerInfo
        Measure of the corner's position, orientation, and angle. Useful in
        order to determine how the corner affects the bbox of the curve.
    """
    first_tan_angle = None
    first_vertex = None
    prev_tan_angle = None
    prev_vertex = None
    for bcurve, code in test_path.iter_curves(**kwargs):
        bcurve = BezierSegment(bcurve)
        if code == Path.MOVETO:
            # deal with capping ends of previous polyline, if it exists
            if prev_tan_angle is not None and is_capped:
                for cap_angle, cap_vertex in [(first_tan_angle, first_vertex),
                                              (prev_tan_angle, prev_vertex)]:
                    yield CornerInfo(cap_vertex, cap_angle, None)
            first_tan_angle = None
            prev_tan_angle = None
            first_vertex = bcurve.cpoints[0]
            prev_vertex = first_vertex
            # lines will end in a cap by default unless a CLOSEPOLY is observed
            is_capped = True
            continue
        if code == Path.CLOSEPOLY:
            is_capped = False
            if prev_tan_angle is None:
                raise ValueError("Misformed path, cannot close poly with single vertex!")
            tan_in = prev_vertex - first_vertex
            # often CLOSEPOLY is used when the curve has already reached it's initial point
            # in order to prevent there from being a stray straight line segment
            # if it's used this way, then we more or less ignore the current bcurve
            if np.isclose(np.linalg.norm(tan_in), 0):
                incident_angle, corner_angle = _incidence_corner_from_angles(
                        prev_tan_angle, first_tan_angle)
                yield CornerInfo(prev_vertex, incident_angle, corner_angle)
                continue
            # otherwise, we have to calculate both the corner from the
            # previous line segment to the current straight line, and from the current straight
            # line to the original starting line. The former is taken care of by the
            # non-special-case code below. the latter looks like:
            tan_out = bcurve.tan_out
            angle_end = np.arctan2(tan_out[1], tan_out[0])
            incident_angle, corner_angle = _incidence_corner_from_angles(
                    angle_end, first_tan_angle)
            yield CornerInfo(first_vertex, incident_angle, corner_angle)
        # finally, usual case is when two curves meet at an angle
        tan_in = -bcurve.tan_in
        angle_in = np.arctan2(tan_in[1], tan_in[0])
        if first_tan_angle is None:
            first_tan_angle = angle_in
        if prev_tan_angle is not None:
            incident_angle, corner_angle = _incidence_corner_from_angles(
                    angle_in, prev_tan_angle)
            yield CornerInfo(prev_vertex, incident_angle, corner_angle)
        tan_out = bcurve.tan_out
        prev_tan_angle = np.arctan2(tan_out[1], tan_out[0])
        prev_vertex = bcurve.cpoints[-1]
    if prev_tan_angle is not None and is_capped:
        for cap_angle, cap_vertex in [(first_tan_angle, first_vertex),
                                      (prev_tan_angle, prev_vertex)]:
            yield CornerInfo(cap_vertex, cap_angle, None)

# quadratic Bezier lines

def get_cos_sin(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    d = (dx * dx + dy * dy) ** .5
    # Account for divide by zero
    if d == 0:
        return 0.0, 0.0
    return dx / d, dy / d


def check_if_parallel(dx1, dy1, dx2, dy2, tolerance=1.e-5):
    """
    Check if two lines are parallel.

    Parameters
    ----------
    dx1, dy1, dx2, dy2 : float
        The gradients *dy*/*dx* of the two lines.
    tolerance : float
        The angular tolerance in radians up to which the lines are considered
        parallel.

    Returns
    -------
    is_parallel
        - 1 if two lines are parallel in same direction.
        - -1 if two lines are parallel in opposite direction.
        - False otherwise.
    """
    theta1 = np.arctan2(dx1, dy1)
    theta2 = np.arctan2(dx2, dy2)
    dtheta = abs(theta1 - theta2)
    if dtheta < tolerance:
        return 1
    elif abs(dtheta - np.pi) < tolerance:
        return -1
    else:
        return False


def get_parallels(bezier2, width):
    """
    Given the quadratic Bezier control points *bezier2*, returns
    control points of quadratic Bezier lines roughly parallel to given
    one separated by *width*.
    """

    # The parallel Bezier lines are constructed by following ways.
    #  c1 and c2 are control points representing the begin and end of the
    #  Bezier line.
    #  cm is the middle point

    c1x, c1y = bezier2[0]
    cmx, cmy = bezier2[1]
    c2x, c2y = bezier2[2]

    parallel_test = check_if_parallel(c1x - cmx, c1y - cmy,
                                      cmx - c2x, cmy - c2y)

    if parallel_test == -1:
        cbook._warn_external(
            "Lines do not intersect. A straight line is used instead.")
        cos_t1, sin_t1 = get_cos_sin(c1x, c1y, c2x, c2y)
        cos_t2, sin_t2 = cos_t1, sin_t1
    else:
        # t1 and t2 is the angle between c1 and cm, cm, c2.  They are
        # also a angle of the tangential line of the path at c1 and c2
        cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
        cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c2x, c2y)

    # find c1_left, c1_right which are located along the lines
    # through c1 and perpendicular to the tangential lines of the
    # Bezier path at a distance of width. Same thing for c2_left and
    # c2_right with respect to c2.
    c1x_left, c1y_left, c1x_right, c1y_right = (
        get_normal_points(c1x, c1y, cos_t1, sin_t1, width)
    )
    c2x_left, c2y_left, c2x_right, c2y_right = (
        get_normal_points(c2x, c2y, cos_t2, sin_t2, width)
    )

    # find cm_left which is the intersecting point of a line through
    # c1_left with angle t1 and a line through c2_left with angle
    # t2. Same with cm_right.
    if parallel_test != 0:
        # a special case for a straight line, i.e., angle between two
        # lines are smaller than some (arbitrary) value.
        cmx_left, cmy_left = (
            0.5 * (c1x_left + c2x_left), 0.5 * (c1y_left + c2y_left)
        )
        cmx_right, cmy_right = (
            0.5 * (c1x_right + c2x_right), 0.5 * (c1y_right + c2y_right)
        )
    else:
        cmx_left, cmy_left = get_intersection(c1x_left, c1y_left, cos_t1,
                                              sin_t1, c2x_left, c2y_left,
                                              cos_t2, sin_t2)

        cmx_right, cmy_right = get_intersection(c1x_right, c1y_right, cos_t1,
                                                sin_t1, c2x_right, c2y_right,
                                                cos_t2, sin_t2)

    # the parallel Bezier lines are created with control points of
    # [c1_left, cm_left, c2_left] and [c1_right, cm_right, c2_right]
    path_left = [(c1x_left, c1y_left),
                 (cmx_left, cmy_left),
                 (c2x_left, c2y_left)]
    path_right = [(c1x_right, c1y_right),
                  (cmx_right, cmy_right),
                  (c2x_right, c2y_right)]

    return path_left, path_right


def find_control_points(c1x, c1y, mmx, mmy, c2x, c2y):
    """
    Find control points of the Bezier curve passing through (*c1x*, *c1y*),
    (*mmx*, *mmy*), and (*c2x*, *c2y*), at parametric values 0, 0.5, and 1.
    """
    cmx = .5 * (4 * mmx - (c1x + c2x))
    cmy = .5 * (4 * mmy - (c1y + c2y))
    return [(c1x, c1y), (cmx, cmy), (c2x, c2y)]


def make_wedged_bezier2(bezier2, width, w1=1., wm=0.5, w2=0.):
    """
    Being similar to get_parallels, returns control points of two quadratic
    Bezier lines having a width roughly parallel to given one separated by
    *width*.
    """

    # c1, cm, c2
    c1x, c1y = bezier2[0]
    cmx, cmy = bezier2[1]
    c3x, c3y = bezier2[2]

    # t1 and t2 is the angle between c1 and cm, cm, c3.
    # They are also a angle of the tangential line of the path at c1 and c3
    cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
    cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c3x, c3y)

    # find c1_left, c1_right which are located along the lines
    # through c1 and perpendicular to the tangential lines of the
    # Bezier path at a distance of width. Same thing for c3_left and
    # c3_right with respect to c3.
    c1x_left, c1y_left, c1x_right, c1y_right = (
        get_normal_points(c1x, c1y, cos_t1, sin_t1, width * w1)
    )
    c3x_left, c3y_left, c3x_right, c3y_right = (
        get_normal_points(c3x, c3y, cos_t2, sin_t2, width * w2)
    )

    # find c12, c23 and c123 which are middle points of c1-cm, cm-c3 and
    # c12-c23
    c12x, c12y = (c1x + cmx) * .5, (c1y + cmy) * .5
    c23x, c23y = (cmx + c3x) * .5, (cmy + c3y) * .5
    c123x, c123y = (c12x + c23x) * .5, (c12y + c23y) * .5

    # tangential angle of c123 (angle between c12 and c23)
    cos_t123, sin_t123 = get_cos_sin(c12x, c12y, c23x, c23y)

    c123x_left, c123y_left, c123x_right, c123y_right = (
        get_normal_points(c123x, c123y, cos_t123, sin_t123, width * wm)
    )

    path_left = find_control_points(c1x_left, c1y_left,
                                    c123x_left, c123y_left,
                                    c3x_left, c3y_left)
    path_right = find_control_points(c1x_right, c1y_right,
                                     c123x_right, c123y_right,
                                     c3x_right, c3y_right)

    return path_left, path_right


def make_path_regular(p):
    """
    If the ``codes`` attribute of `.Path` *p* is None, return a copy of *p*
    with ``codes`` set to (MOVETO, LINETO, LINETO, ..., LINETO); otherwise
    return *p* itself.
    """
    c = p.codes
    if c is None:
        c = np.full(len(p.vertices), Path.LINETO, dtype=Path.code_type)
        c[0] = Path.MOVETO
        return Path(p.vertices, c)
    else:
        return p


def concatenate_paths(paths):
    """Concatenate a list of paths into a single path."""
    vertices = np.concatenate([p.vertices for p in paths])
    codes = np.concatenate([make_path_regular(p).codes for p in paths])
    return Path(vertices, codes)
