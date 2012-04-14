"""
Stacked area plot for 1D arrays inspired by Douglas Y'barbo's stackoverflow
answer:
http://stackoverflow.com/questions/2225995/how-can-i-create-stacked-line-graph-with-matplotlib

(http://stackoverflow.com/users/66549/doug)

"""
import numpy as np
import matplotlib

__all__ = ['stackplot']


def stackplot(axes, x, y):
    """Draws a stacked area plot.

    Parameters
    ----------
    *x* : 1d array of dimension N
    *y* : 2d array of dimension MxN. The data is assumed to be unstacked.

    Returns
    -------
    *r* : A list of `matplotlib.collections.PolyCollection`, one for each
          element in the stacked area plot.
    """

    y = np.atleast_2d(y)

    # Assume data passed has not been 'stacked', so stack it here.
    y_stack = np.cumsum(y, axis=0)

    r = []

    # Color between x = 0 and the first array.
    r.append(axes.fill_between(x, 0, y_stack[0,:], facecolor=axes._get_lines.color_cycle.next()))

    # Color between array i-1 and array i
    for i in xrange(len(y)-1):
        r.append(axes.fill_between(x, y_stack[i-1,:], y_stack[i,:], facecolor=axes._get_lines.color_cycle.next()))
    return r
