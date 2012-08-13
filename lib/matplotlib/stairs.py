"""
Stairstep plots.
"""

import numpy as np

__all__ = ['stairs']

def stairs(axes, *args, **kwargs):
    """Draws a stairstep plot

    Parameters
    ----------
    Takes either one or two arguments. Valid calls are:

    ax.stairs(y) # Make a stairstep plot of the values in *y*
    ax.stairs(x, y) # Stairstep plot of the values in *y* at points in *x*

    *x*, *y* : 1d arrays.

    Returns
    -------
    *lines* : :class:`~matplotlib.collections.LineCollection`
              Line collection defining all the steps in the stairstep plot
    """

    if len(args) == 1:
        y = np.asarray(args[0])
        x = np.arange(len(y))
    elif len(args) == 2:
        x = np.asarray(args[0])
        y = np.asarray(args[1])
    else:
        raise ValueError, "stairs takes either 1 or 2 arguments, %d given" % len(args)

    d = 0.5 * np.abs(np.diff(x))
    dm = np.append(d[0], d)
    dp = np.append(d, d[-1])

    xm = x - dm
    xp = x + dp
    x_all = np.dstack((xm, x, xp)).flatten()
    y_all = np.dstack((y, y, y)).flatten()

    return axes.plot(x_all, y_all, **kwargs)
