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

    # 0.2% absolute error
    tol = kwargs.pop('tol', 2e-3)
    n = kwargs.pop('tol', 50)

    x = np.linspace(limits[0], limits[1], n)
    f_vals = [f(xi) for xi in x]

    # Bisect abscissa until the gradient error changes by less than tol
    within_tol = False

    while not within_tol:
        within_tol = True
        new_pts = []
        new_f = []
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
            f_new = f(x_new)    # Used later, so store it
            if abs(x_new - x[i]) < min_step or abs(f_new - f_vals[i]) < min_step:
                continue

            # Compare gradients of actual f and linear approximation
            # FIXME: What if f(x[i]) or f(x[i+1]) is nan?
            dx = abs(x[i+1] - x[i])
            f_interp = (f_vals[i+1] + f_vals[i])

            # This line is the absolute error of the gradient
            grad_error = np.abs(f_interp - 2.0 * f_new) / dx

            # If the new gradient is not within the tolerance, store
            # the subdivision point for merging later
            if grad_error > tol:
                within_tol = False
                new_pts.append(x_new)
                new_f.append(f_new)

        if not within_tol:
                # Not sure this is the best way to do this...
                # Merge the subdivision points into the array of abscissae
                x, f_vals = merge_pts(x, new_pts, f_vals, new_f)

    return axes.plot(x, f_vals)

def merge_pts(xs, xs_sub, fs, fs_sub):
    x = []
    f = []
    ia = 0
    ib = 0
    while ib < len(xs_sub):
        if xs_sub[ib] < xs[ia]:
            x.append(xs_sub[ib])
            f.append(fs_sub[ib])
            ib += 1
        else:
            x.append(xs[ia])
            f.append(fs[ia])
            ia += 1
    if ia < len(xs):
        return np.append(x, xs[ia::]), np.append(f, fs[ia::])
    else:
        return np.array(x), np.array(f)
