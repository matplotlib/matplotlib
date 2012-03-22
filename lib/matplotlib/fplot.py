"""
1D Callable function plotting.

"""
import numpy as np
import matplotlib


__all__ = ['fplot']


def fplot(axes, f, limits, *args, **kwargs):
    """Plots a callable function f.

    Parameters
    ----------
    *f* : Python callable, the function that is to be plotted.
    *limits* : 2-element array or list of limits: [xmin, xmax]. The function f
        is to to be plotted between xmin and xmax.

    Returns
    -------
    *lines* : `matplotlib.collections.LineCollection`
        Line collection with that describes the function *f* between xmin
        and xmax. all streamlines as a series of line segments.
    """

    # TODO: Check f is callable. If not callable, support array of callables.
    # TODO: Support y limits?

    # Some small number, usually close to machine epsilon
    eps = 1e-10

    # The scaling factor used to scale the step size
    # as a function of the domain length
    scale = max(1.0, abs(limits[1] - limits[0]))

    tol = kwargs.pop('tol', None)
    n = kwargs.pop('tol', None)

    if tol is None:
        # 0.2% absolute error
        tol = 2e-3

    if n is None:
        n = 50
    x = np.linspace(limits[0], limits[1], n)

    # Bisect abscissa until the gradient error changes by less than tol
    within_tol = False

    while not within_tol:
        within_tol = True
        new_pts = []
        for i in xrange(len(x)-1):
            # Make sure the step size is not pointlessly small.
            # This is a numerical check to prevent silly roundoff errors.
            #
            # The temporary variable is to ensure the step size is
            # represented properly in binary.
            min_step = np.sqrt(eps) * x[i] * scale

            tmp = x[i] + min_step

            # The multiplation by two is just to be conservative.
            min_step = 2*(tmp - x[i])

            # Subdivide
            x_new = (x[i+1] + x[i]) / 2.0

            # If the absicissa points are too close, don't bisect
            # since calculation of the gradient will produce mostly
            # nonsense values due to roundoff error.
            #
            # If the function values are too close, the payoff is
            # negligible, so skip them.
            if np.abs(x_new - x[i]) < min_step or np.abs(f(x_new) - f(x[i])) < min_step:
                continue

            # Compute gradient
            # FIXME: What if f(x[i]) is nan?
            grad = (f(x[i+1]) - f(x[i])) / (x[i+1] - x[i])

            # Compute gradients to the left and right of x_new
            grad_right = (f(x[i+1]) - f(x_new)) / (x[i+1] - x_new)
            grad_left = (f(x_new) - f(x[i])) / (x_new - x[i])

            # If the new gradients are not within the tolerance, store
            # the subdivision point for merging later
            if np.abs(grad_right - grad) > tol or np.abs(grad_left - grad) > tol:
                within_tol = False
                new_pts.append(x_new)

        if not within_tol:
                # Not sure this is the best way to do this...
                # Merge the subdivision points into the array of abscissae
                x = merge_pts(x, new_pts)

    return axes.plot(x, f(x))

def merge_pts(a, b):
    x = []
    ia = 0
    ib = 0
    while ib < len(b):
        if b[ib] < a[ia]:
            x.append(b[ib])
            ib += 1
        else:
            x.append(a[ia])
            ia += 1
    if ia < len(a):
        return np.append(x, a[ia::])
    else:
        return np.array(x)
