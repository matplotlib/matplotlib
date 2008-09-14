"""
A collection of utility functions that do various numerical or geometrical
manipulations.  
"""
import numpy as np
from numpy import ma
import matplotlib.cbook as cbook

##################################################
# Linear interpolation algorithms
##################################################
def less_simple_linear_interpolation( x, y, xi, extrap=False ):
    """
    This function provides simple (but somewhat less so than
    cbook.simple_linear_interpolation) linear interpolation.
    simple_linear_interpolation will give a list of point between a
    start and an end, while this does true linear interpolation at an
    arbitrary set of points.

    This is very inefficient linear interpolation meant to be used
    only for a small number of points in relatively non-intensive use
    cases.  For real linear interpolation, use scipy.
    """
    if cbook.is_scalar(xi): xi = [xi]

    x = np.asarray(x)
    y = np.asarray(y)
    xi = np.asarray(xi)

    s = list(y.shape)
    s[0] = len(xi)
    yi = np.tile( np.nan, s )

    for ii,xx in enumerate(xi):
        bb = x == xx
        if np.any(bb):
            jj, = np.nonzero(bb)
            yi[ii] = y[jj[0]]
        elif xx<x[0]:
            if extrap:
                yi[ii] = y[0]
        elif xx>x[-1]:
            if extrap:
                yi[ii] = y[-1]
        else:
            jj, = np.nonzero(x<xx)
            jj = max(jj)

            yi[ii] = y[jj] + (xx-x[jj])/(x[jj+1]-x[jj]) * (y[jj+1]-y[jj])

    return yi

def slopes(x,y):
    """
    SLOPES calculate the slope y'(x) Given data vectors X and Y SLOPES
    calculates Y'(X), i.e the slope of a curve Y(X). The slope is
    estimated using the slope obtained from that of a parabola through
    any three consecutive points.

    This method should be superior to that described in the appendix
    of A CONSISTENTLY WELL BEHAVED METHOD OF INTERPOLATION by Russel
    W. Stineman (Creative Computing July 1980) in at least one aspect:

    Circles for interpolation demand a known aspect ratio between x-
    and y-values.  For many functions, however, the abscissa are given
    in different dimensions, so an aspect ratio is completely
    arbitrary.

    The parabola method gives very similar results to the circle
    method for most regular cases but behaves much better in special
    cases

    Norbert Nemec, Institute of Theoretical Physics, University or
    Regensburg, April 2006 Norbert.Nemec at physik.uni-regensburg.de

    (inspired by a original implementation by Halldor Bjornsson,
    Icelandic Meteorological Office, March 2006 halldor at vedur.is)
    """
    # Cast key variables as float.
    x=np.asarray(x, np.float_)
    y=np.asarray(y, np.float_)

    yp=np.zeros(y.shape, np.float_)

    dx=x[1:] - x[:-1]
    dy=y[1:] - y[:-1]
    dydx = dy/dx
    yp[1:-1] = (dydx[:-1] * dx[1:] + dydx[1:] * dx[:-1])/(dx[1:] + dx[:-1])
    yp[0] = 2.0 * dy[0]/dx[0] - yp[1]
    yp[-1] = 2.0 * dy[-1]/dx[-1] - yp[-2]
    return yp


def stineman_interp(xi,x,y,yp=None):
    """
    STINEMAN_INTERP Well behaved data interpolation.  Given data
    vectors X and Y, the slope vector YP and a new abscissa vector XI
    the function stineman_interp(xi,x,y,yp) uses Stineman
    interpolation to calculate a vector YI corresponding to XI.

    Here's an example that generates a coarse sine curve, then
    interpolates over a finer abscissa:

      x = linspace(0,2*pi,20);  y = sin(x); yp = cos(x)
      xi = linspace(0,2*pi,40);
      yi = stineman_interp(xi,x,y,yp);
      plot(x,y,'o',xi,yi)

    The interpolation method is described in the article A
    CONSISTENTLY WELL BEHAVED METHOD OF INTERPOLATION by Russell
    W. Stineman. The article appeared in the July 1980 issue of
    Creative Computing with a note from the editor stating that while
    they were

      not an academic journal but once in a while something serious
      and original comes in adding that this was
      "apparently a real solution" to a well known problem.

    For yp=None, the routine automatically determines the slopes using
    the "slopes" routine.

    X is assumed to be sorted in increasing order

    For values xi[j] < x[0] or xi[j] > x[-1], the routine tries a
    extrapolation.  The relevance of the data obtained from this, of
    course, questionable...

    original implementation by Halldor Bjornsson, Icelandic
    Meteorolocial Office, March 2006 halldor at vedur.is

    completely reworked and optimized for Python by Norbert Nemec,
    Institute of Theoretical Physics, University or Regensburg, April
    2006 Norbert.Nemec at physik.uni-regensburg.de

    """

    # Cast key variables as float.
    x=np.asarray(x, np.float_)
    y=np.asarray(y, np.float_)
    assert x.shape == y.shape
    N=len(y)

    if yp is None:
        yp = slopes(x,y)
    else:
        yp=np.asarray(yp, np.float_)

    xi=np.asarray(xi, np.float_)
    yi=np.zeros(xi.shape, np.float_)

    # calculate linear slopes
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    s = dy/dx  #note length of s is N-1 so last element is #N-2

    # find the segment each xi is in
    # this line actually is the key to the efficiency of this implementation
    idx = np.searchsorted(x[1:-1], xi)

    # now we have generally: x[idx[j]] <= xi[j] <= x[idx[j]+1]
    # except at the boundaries, where it may be that xi[j] < x[0] or xi[j] > x[-1]

    # the y-values that would come out from a linear interpolation:
    sidx = s.take(idx)
    xidx = x.take(idx)
    yidx = y.take(idx)
    xidxp1 = x.take(idx+1)
    yo = yidx + sidx * (xi - xidx)

    # the difference that comes when using the slopes given in yp
    dy1 = (yp.take(idx)- sidx) * (xi - xidx)       # using the yp slope of the left point
    dy2 = (yp.take(idx+1)-sidx) * (xi - xidxp1) # using the yp slope of the right point

    dy1dy2 = dy1*dy2
    # The following is optimized for Python. The solution actually
    # does more calculations than necessary but exploiting the power
    # of numpy, this is far more efficient than coding a loop by hand
    # in Python
    yi = yo + dy1dy2 * np.choose(np.array(np.sign(dy1dy2), np.int32)+1,
                                 ((2*xi-xidx-xidxp1)/((dy1-dy2)*(xidxp1-xidx)),
                                  0.0,
                                  1/(dy1+dy2),))
    return yi

##################################################
# Code related to things in and around polygons
##################################################
def inside_poly(points, verts):
    """
    points is a sequence of x,y points
    verts is a sequence of x,y vertices of a poygon

    return value is a sequence of indices into points for the points
    that are inside the polygon
    """
    res, =  np.nonzero(nxutils.points_inside_poly(points, verts))
    return res

def poly_below(xmin, xs, ys):
    """
    given a sequence of xs and ys, return the vertices of a polygon
    that has a horzontal base at xmin and an upper bound at the ys.
    xmin is a scalar.

    intended for use with Axes.fill, eg
    xv, yv = poly_below(0, x, y)
    ax.fill(xv, yv)
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    Nx = len(xs)
    Ny = len(ys)
    assert(Nx==Ny)
    x = xmin*np.ones(2*Nx)
    y = np.ones(2*Nx)
    x[:Nx] = xs
    y[:Nx] = ys
    y[Nx:] = ys[::-1]
    return x, y


def poly_between(x, ylower, yupper):
    """
    given a sequence of x, ylower and yupper, return the polygon that
    fills the regions between them.  ylower or yupper can be scalar or
    iterable.  If they are iterable, they must be equal in length to x

    return value is x, y arrays for use with Axes.fill
    """
    Nx = len(x)
    if not cbook.iterable(ylower):
        ylower = ylower*np.ones(Nx)

    if not cbook.iterable(yupper):
        yupper = yupper*np.ones(Nx)

    x = np.concatenate( (x, x[::-1]) )
    y = np.concatenate( (yupper, ylower[::-1]) )
    return x,y

def is_closed_polygon(X):
    """
    Tests whether first and last object in a sequence are the same.  These are
    presumably coordinates on a polygonal curve, in which case this function
    tests if that curve is closed.

    """
    return np.all(X[0] == X[-1])

##################################################
# Vector and path length geometry calculations
##################################################
def vector_lengths( X, P=2., axis=None ):
    """
    Finds the length of a set of vectors in n dimensions.  This is
    like the numpy norm function for vectors, but has the ability to
    work over a particular axis of the supplied array or matrix.

    Computes (sum((x_i)^P))^(1/P) for each {x_i} being the elements of X along
    the given axis.  If *axis* is *None*, compute over all elements of X.
    """
    X = np.asarray(X)
    return (np.sum(X**(P),axis=axis))**(1./P)

def distances_along_curve( X ):
    """
    Computes the distance between a set of successive points in N dimensions.

    where X is an MxN array or matrix.  The distances between successive rows
    is computed.  Distance is the standard Euclidean distance.
    """
    X = np.diff( X, axis=0 )
    return vector_lengths(X,axis=1)

def path_length(X):
    """
    Computes the distance travelled along a polygonal curve in N dimensions.


    where X is an MxN array or matrix.  Returns an array of length M consisting
    of the distance along the curve at each point (i.e., the rows of X).
    """
    X = distances_along_curve(X)
    return np.concatenate( (np.zeros(1), np.cumsum(X)) )

def quad2cubic(q0x, q0y, q1x, q1y, q2x, q2y):
    """
    Converts a quadratic Bezier curve to a cubic approximation.

    The inputs are the x and y coordinates of the three control points
    of a quadratic curve, and the output is a tuple of x and y
    coordinates of the four control points of the cubic curve.
    """
    # c0x, c0y = q0x, q0y
    c1x, c1y = q0x + 2./3. * (q1x - q0x), q0y + 2./3. * (q1y - q0y)
    c2x, c2y = c1x + 1./3. * (q2x - q0x), c1y + 1./3. * (q2y - q0y)
    # c3x, c3y = q2x, q2y
    return q0x, q0y, c1x, c1y, c2x, c2y, q2x, q2y

