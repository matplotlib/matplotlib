"""
1D Callable function plotting.
"""

import numpy as np


__all__ = ['fplot']


class FPlot(object):
    def __init__(self, axes, *args, **kwargs):
        self._process_args(*args, **kwargs)

        self.axes = axes
        self.axes.set_autoscale_on(False)

        self.n = kwargs.pop('res', 1000)

        self.x = np.linspace(self.limits[0], self.limits[1], self.n)
        self.f_vals = np.asarray([self.f(xi) for xi in self.x])

        self.fline, = self.axes.plot(self.x, self.f_vals)
        self._process_singularities()
        self.axes.set_xlim([self.x[0], self.x[-1]])
        mn, mx = np.nanmin(self.f_vals), np.nanmax(self.f_vals)
        self.axes.set_ylim([mn, mx])

        axes.callbacks.connect('xlim_changed', self._update)
        axes.callbacks.connect('ylim_changed', self._update)

    def _process_args(self, *args, **kwargs):
        # TODO: Check f is callable. If not callable, support array of callables.
        # TODO: Support y limits?
        self.f = args[0]
        self.limits = args[1]

    def _update(self, axes):
        # bounds is (l, b, w, h)
        bounds = axes.viewLim.bounds
        self.x = np.linspace(bounds[0], bounds[0] + bounds[2], self.n)
        self.f_vals = [self.f(xi) for xi in self.x]
        self._process_singularities()
        self.fline.set_data(self.x, self.f_vals)
        self.axes.figure.canvas.draw_idle()

    def _process_singularities(self):
        # Note:  d[i] == f_vals[i+1] - f_vals[i]
        d = np.diff(self.f_vals)

        # 80% is arbitrary.  Perhaps more control could be offered here?
        badness = np.where(d > 0.80 * self.axes.viewLim.bounds[3])[0]

        # We don't draw the signularities
        for b in badness:
            self.f_vals[b] = np.nan
            self.f_vals[b + 1] = np.nan


def fplot(ax, *args, **kwargs):
    """
    Plots a callable function f.

    Parameters
    ----------
    f : Python callable, the function that is to be plotted.
    limits : 2-element array or list of limits: [xmin, xmax]. The function f
        is to to be plotted between xmin and xmax.

    Returns
    -------
    lines : `matplotlib.collections.LineCollection`
        Line collection with that describes the function *f* between xmin
        and xmax. all streamlines as a series of line segments.
    """
    if not ax._hold:
        ax.cla()
    return FPlot(ax, *args, **kwargs)
