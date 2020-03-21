"""
A module providing some utility functions regarding Bezier path manipulation.
"""

import math
import warnings
from collections import deque

import numpy as np

import matplotlib.cbook as cbook

# same algorithm as 3.8's math.comb
@np.vectorize
def _comb(n, k):
    k = min(k, n - k)
    i = np.arange(1, k + 1)
    return np.prod((n + 1 - i)/i).astype(int)


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


@cbook.deprecated("3.3", alternative="Path.split_path_inout()")
def split_path_inout(path, inside, tolerance=0.01, reorder_inout=False):
    """
    Divide a path into two segments at the point where ``inside(x, y)``
    becomes False.
    """
    return path.split_path_inout(inside, tolerance, reorder_inout)


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
    A d-dimensional Bezier segment.

    Parameters
    ----------
    control_points : (N, d) array
        Location of the *N* control points.
    """

    def __init__(self, control_points):
        self._cpoints = np.asarray(control_points)
        self._N, self._d = self._cpoints.shape
        self._orders = np.arange(self._N)
        coeff = [math.factorial(self._N - 1)
                 // (math.factorial(i) * math.factorial(self._N - 1 - i))
                 for i in range(self._N)]
        self._px = (self._cpoints.T * coeff).T

    def __call__(self, t):
        t = np.array(t)
        orders_shape = (1,)*t.ndim + self._orders.shape
        t_shape = t.shape + (1,)  # self._orders.ndim == 1
        orders = np.reshape(self._orders, orders_shape)
        rev_orders = np.reshape(self._orders[::-1], orders_shape)
        t = np.reshape(t, t_shape)
        return ((1 - t)**rev_orders * t**orders) @ self._px

    def point_at_t(self, t):
        """Return the point on the Bezier curve for parameter *t*."""
        return tuple(self(t))

    def split_at_t(self, t):
        """Split into two Bezier curves using de casteljau's algorithm.

        Parameters
        ----------
        t : float
            Point in [0,1] at which to split into two curves

        Returns
        -------
        B1, B2 : BezierSegment
            The two sub-curves.
        """
        new_cpoints = split_de_casteljau(self._cpoints, t)
        return BezierSegment(new_cpoints[0]), BezierSegment(new_cpoints[1])

    def control_net_length(self):
        """Sum of lengths between control points"""
        L = 0
        N, d = self._cpoints.shape
        for i in range(N - 1):
            L += np.linalg.norm(self._cpoints[i+1] - self._cpoints[i])
        return L

    def arc_length(self, rtol=None, atol=None):
        """Estimate the length using iterative refinement.

        Our estimate is just the average between the length of the chord and
        the length of the control net.

        Since the chord length and control net give lower and upper bounds
        (respectively) on the length, this maximum possible error is tested
        against an absolute tolerance threshold at each subdivision.

        However, sometimes this estimator converges much faster than this error
        esimate would suggest. Therefore, the relative change in the length
        estimate between subdivisions is compared to a relative error tolerance
        after each set of subdivisions.

        Parameters
        ----------
        rtol : float, default 1e-4
            If :code:`abs(est[i+1] - est[i]) <= rtol * est[i+1]`, we return
            :code:`est[i+1]`.
        atol : float, default 1e-6
            If the distance between chord length and control length at any
            point falls below this number, iteration is terminated.
        """
        if rtol is None:
            rtol = 1e-4
        if atol is None:
            atol = 1e-6

        chord = np.linalg.norm(self._cpoints[-1] - self._cpoints[0])
        net = self.control_net_length()
        max_err = (net - chord)/2
        curr_est = chord + max_err
        # early exit so we don't try to "split" paths of zero length
        if max_err < atol:
            return curr_est

        prev_est = np.inf
        curves = deque([self])
        errs = deque([max_err])
        lengths = deque([curr_est])
        while np.abs(curr_est - prev_est) > rtol * curr_est:
            # subdivide the *whole* curve before checking relative convergence
            # again
            prev_est = curr_est
            num_curves = len(curves)
            for i in range(num_curves):
                curve = curves.popleft()
                new_curves = curve.split_at_t(0.5)
                max_err -= errs.popleft()
                curr_est -= lengths.popleft()
                for ncurve in new_curves:
                    chord = np.linalg.norm(
                            ncurve._cpoints[-1] - ncurve._cpoints[0])
                    net = ncurve.control_net_length()
                    nerr = (net - chord)/2
                    nlength = chord + nerr
                    max_err += nerr
                    curr_est += nlength
                    curves.append(ncurve)
                    errs.append(nerr)
                    lengths.append(nlength)
                if max_err < atol:
                    return curr_est
        return curr_est

    def arc_center_of_mass(self):
        r"""
        Center of mass of the (even-odd-rendered) area swept out by the ray
        from the origin to the path.

        Summing this vector for each segment along a closed path will produce
        that area's center of mass.

        Returns
        -------
        r_cm : (2,) np.array<float>
            the "arc's center of mass"

        Notes
        -----
        A simple analytical form can be derived for general Bezier curves.
        Suppose the curve was closed, so :math:`B(0) = B(1)`. Call the area
        enclosed by :math:`B(t)` :math:`B_\text{int}`. The center of mass of
        :math:`B_\text{int}` is defined by the expected value of the position
        vector `\vec{r}`

        .. math::

            \vec{R}_\text{cm} = \int_{B_\text{int}} \vec{r} \left( \frac{1}{
            \int_{B_\text{int}}} d\vec{r} \right) d\vec{r}

        where :math:`(1/\text{Area}(B_\text{int})` can be interpreted as a
        probability density.

        In order to compute this integral, we choose two functions
        :math:`F_0(x,y) = [x^2/2, 0]` and :math:`F_1(x,y) = [0, y^2/2]` such
        that :math:`[\div \cdot F_0, \div \cdot F_1] = \vec{r}`. Then, applying
        the divergence integral (componentwise), we get that

        .. math::
            \vec{R}_\text{cm} &= \oint_{B(t)} F \cdot \vec{n} dt \\
            &= \int_0^1 \left[ \begin{array}{1}
                B^{(0)}(t) \frac{dB^{(1)}(t)}{dt}  \\
              - B^{(1)}(t) \frac{dB^{(0)}(t)}{dt}  \end{array} \right] dt

        After expanding in Berstein polynomials and moving the integral inside
        all the sums, we get that

        .. math::
            \vec{R}_\text{cm} = \frac{1}{6} \sum_{i,j=0}^n\sum_{k=0}^{n-1}
                \frac{{n \choose i}{n \choose j}{{n-1} \choose k}}
                     {{3n - 1} \choose {i + j + k}}
                \left(\begin{array}{1}
                    P^{(0)}_i P^{(0)}_j (P^{(1)}_{k+1} - P^{(1)}_k)
                  - P^{(1)}_i P^{(1)}_j (P^{(0)}_{k+1} - P^{(0)}_k)
                \right) \end{array}

        where :math:`P_i = [P^{(0)}_i, P^{(1)}_i]` is the :math:`i`'th control
        point of the curve and :math:`n` is the degree of the curve.
        """
        n = self.degree
        r_cm = np.zeros(2)
        P = self.control_points
        dP = np.diff(P, axis=0)
        Pn = np.array([[1, -1]])*dP[:, ::-1]  # n = [y, -x]
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n):
                    r_cm += _comb(n, i) * _comb(n, j) * _comb(n - 1, k) \
                            * P[i]*P[j]*Pn[k] / _comb(3*n - 1, i + j + k)
        return r_cm/6

    def arc_area(self):
        r"""
        (Signed) area swept out by ray from origin to curve.

        Notes
        -----
        A simple, analytical formula is possible for arbitrary bezier curves.

        Given a bezier curve B(t), in order to calculate the area of the arc
        swept out by the ray from the origin to the curve, we simply need to
        compute :math:`\frac{1}{2}\int_0^1 B(t) \cdot n(t) dt`, where
        :math:`n(t) = u^{(1)}(t) \hat{x}_0 - u{(0)}(t) \hat{x}_1` is the normal
        vector oriented away from the origin and :math:`u^{(i)}(t) =
        \frac{d}{dt} B^{(i)}(t)` is the :math:`i`th component of the curve's
        tangent vector.  (This formula can be found by applying the divergence
        theorem to :math:`F(x,y) = [x, y]/2`, and calculates the *signed* area
        for a counter-clockwise curve, by the right hand rule).

        The control points of the curve are just its coefficients in a
        Bernstein expansion, so if we let :math:`P_i = [P^{(0)}_i, P^{(1)}_i]`
        be the :math:`i`'th control point, then

        .. math::

            \frac{1}{2}\int_0^1 B(t) \cdot n(t) dt
                   &= \frac{1}{2}\int_0^1 B^{(0)}(t) \frac{d}{dt} B^{(1)}(t)
                                        - B^{(1)}(t) \frac{d}{dt} B^{(0)}(t)
                                        dt \\
                   &= \frac{1}{2}\int_0^1
                        \left( \sum_{j=0}^n P_j^{(0)} b_{j,n} \right)
                        \left( n \sum_{k=0}^{n-1} (P_{k+1}^{(1)} -
                               P_{k}^{(1)}) b_{j,n} \right)
                      \\
                      &\hspace{1em} - \left( \sum_{j=0}^n P_j^{(1)} b_{j,n}
                        \right) \left( n \sum_{k=0}^{n-1} (P_{k+1}^{(0)}
                                     - P_{k}^{(0)}) b_{j,n} \right)
                      dt,

        where :math:`b_{\nu, n}(t) = {n \choose \nu} t^\nu {(1 - t)}^{n-\nu}`
        is the :math:`\nu`'th Bernstein polynomial of degree :math:`n`.

        Grouping :math:`t^l(1-t)^m` terms together for each :math:`l`,
        :math:`m`, we get that the integrand becomes

        .. math::

            \sum_{j=0}^n \sum_{k=0}^{n-1}
                {n \choose j} {{n - 1} \choose k}
                &\left[P_j^{(0)} (P_{k+1}^{(1)} - P_{k}^{(1)})
                    - P_j^{(1)} (P_{k+1}^{(0)} - P_{k}^{(0)})\right] \\
                &\hspace{1em}\times{}t^{j + k} {(1 - t)}^{2n - 1 - j - k}

        or just

        .. math::

            \sum_{j=0}^n \sum_{k=0}^{n-1}
                \frac{{n \choose j} {{n - 1} \choose k}}
                        {{{2n - 1} \choose {j+k}}}
                [P_j^{(0)} (P_{k+1}^{(1)} - P_{k}^{(1)})
                    - P_j^{(1)} (P_{k+1}^{(0)} - P_{k}^{(0)})]
                b_{j+k,2n-1}(t).

        Interchanging sum and integral, and using the fact that :math:`\int_0^1
        b_{\nu, n}(t) dt = \frac{1}{n + 1}`, we conclude that the
        original integral  can
        simply be written as

        .. math::

            \frac{1}{2}&\int_0^1 B(t) \cdot n(t) dt
            \\
            &= \frac{1}{4}\sum_{j=0}^n \sum_{k=0}^{n-1}
              \frac{{n \choose j} {{n - 1} \choose k}}
                    {{{2n - 1} \choose {j+k}}}
              [P_j^{(0)} (P_{k+1}^{(1)} - P_{k}^{(1)})
             - P_j^{(1)} (P_{k+1}^{(0)} - P_{k}^{(0)})]
        """
        n = self.degree
        area = 0
        P = self.control_points
        dP = np.diff(P, axis=0)
        for j in range(n + 1):
            for k in range(n):
                area += _comb(n, j)*_comb(n-1, k)/_comb(2*n - 1, j + k) \
                        * (P[j, 0]*dP[k, 1] - P[j, 1]*dP[k, 0])
        return (1/4)*area

    def center_of_mass(self):
        """Return the center of mass of the curve (not the filled curve!)

        Notes
        -----
        Computed as the mean of the control points.
        """
        return np.mean(self._cpoints, axis=0)

    @classmethod
    def differentiate(cls, B):
        """Return the derivative of a BezierSegment, itself a BezierSegment"""
        dcontrol_points = B.degree*np.diff(B.control_points, axis=0)
        return cls(dcontrol_points)

    @property
    def control_points(self):
        """The control points of the curve."""
        return self._cpoints

    @property
    def dimension(self):
        """The dimension of the curve."""
        return self._d

    @property
    def degree(self):
        """The number of control points in the curve."""
        return self._N - 1

    @property
    def polynomial_coefficients(self):
        r"""The polynomial coefficients of the Bezier curve.

        Returns
        -------
        coefs : float, (n+1, d) array_like
            Coefficients after expanding in polynomial basis, where :math:`n`
            is the degree of the bezier curve and :math:`d` its dimension.
            These are the numbers (:math:`C_j`) such that the curve can be
            written :math:`\sum_{j=0}^n C_j t^j`.

        Notes
        -----
        The coefficients are calculated as

        .. math::

            {n \choose j} \sum_{i=0}^j (-1)^{i+j} {j \choose i} P_i

        where :math:`P_i` are the control points of the curve.
        """
        n = self.degree
        # matplotlib uses n <= 4. overflow plausible starting around n = 15.
        if n > 10:
            warnings.warn("Polynomial coefficients formula unstable for high "
                          "order Bezier curves!", RuntimeWarning)
        d = self.dimension
        P = self.control_points
        coefs = np.zeros((n+1, d))
        for j in range(n+1):
            i = np.arange(j+1)
            prefactor = np.power(-1, i + j) * _comb(j, i)
            prefactor = np.tile(prefactor, (d, 1)).T
            coefs[j] = _comb(n, j) * np.sum(prefactor*P[i], axis=0)
        return coefs

    @property
    def axis_aligned_extrema(self):
        """
        Return the location along the curve's interior where its partial
        derivative is zero, along with the dimension along which it is zero for
        each such instance.

        Returns
        -------
        dims : int, array_like
            dimension :math:`i` along which the corresponding zero occurs
        dzeros : float, array_like
            of same size as dims. the :math:`t` such that :math:`d/dx_i B(t) =
            0`
        """
        n = self.degree
        Cj = self.polynomial_coefficients
        # much faster than .differentiate(self).polynomial_coefficients
        dCj = np.atleast_2d(np.arange(1, n+1)).T * Cj[1:]
        if len(dCj) == 0:
            return np.array([]), np.array([])
        dims = []
        roots = []
        for i, pi in enumerate(dCj.T):
            r = np.roots(pi[::-1])
            roots.append(r)
            dims.append(i*np.ones_like(r))
        roots = np.concatenate(roots)
        dims = np.concatenate(dims)
        in_range = np.isreal(roots) & (roots >= 0) & (roots <= 1)
        return dims[in_range], np.real(roots)[in_range]


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


@cbook.deprecated("3.3")
def inside_circle(cx, cy, r):
    """
    Return a function that checks whether a point is in a circle with center
    (*cx*, *cy*) and radius *r*.

    The returned function has the signature::

        f(xy: Tuple[float, float]) -> bool
    """
    from .patches import _inside_circle
    return _inside_circle(cx, cy, r)


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
    try:
        cmx_left, cmy_left = get_intersection(c1x_left, c1y_left, cos_t1,
                                              sin_t1, c2x_left, c2y_left,
                                              cos_t2, sin_t2)
        cmx_right, cmy_right = get_intersection(c1x_right, c1y_right, cos_t1,
                                                sin_t1, c2x_right, c2y_right,
                                                cos_t2, sin_t2)
    except ValueError:
        # Special case straight lines, i.e., angle between two lines is
        # less than the threshold used by get_intersection (we don't use
        # check_if_parallel as the threshold is not the same).
        cmx_left, cmy_left = (
            0.5 * (c1x_left + c2x_left), 0.5 * (c1y_left + c2y_left)
        )
        cmx_right, cmy_right = (
            0.5 * (c1x_right + c2x_right), 0.5 * (c1y_right + c2y_right)
        )

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


@cbook.deprecated(
    "3.3", alternative="Path.cleaned() and remove the final STOP if needed")
def make_path_regular(p):
    """
    If the ``codes`` attribute of `.Path` *p* is None, return a copy of *p*
    with ``codes`` set to (MOVETO, LINETO, LINETO, ..., LINETO); otherwise
    return *p* itself.
    """
    from .path import Path
    c = p.codes
    if c is None:
        c = np.full(len(p.vertices), Path.LINETO, dtype=Path.code_type)
        c[0] = Path.MOVETO
        return Path(p.vertices, c)
    else:
        return p


@cbook.deprecated("3.3", alternative="Path.make_compound_path()")
def concatenate_paths(paths):
    """Concatenate a list of paths into a single path."""
    from .path import Path
    return Path.make_compound_path(*paths)
