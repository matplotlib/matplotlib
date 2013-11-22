import matplotlib.pyplot as plt
from functools import wraps

Axes = matplotlib.axes.Axes
Figure = matplotlib.figure.Figure

WRAP_WITH_HOLD = set((
        'acorr',
        'angle_spectrum',
        'arrow',
        'axhline',
        'axhspan',
        'axvline',
        'axvspan',
        'bar',
        'barh',
        'broken_barh',
        'boxplot',
        'cohere',
        'clabel',
        'contour',
        'contourf',
        'csd',
        'errorbar',
        'eventplot',
        'fill',
        'fill_between',
        'fill_betweenx',
        'hexbin',
        'hist',
        'hist2d',
        'hlines',
        'imshow',
        'loglog',
        'magnitude_spectrum',
        'pcolor',
        'pcolormesh',
        'phase_spectrum',
        'pie',
        'plot',
        'plot_date',
        'psd',
        'quiver',
        'quiverkey',
        'scatter',
        'semilogx',
        'semilogy',
        'specgram',
        'stackplot',
        'stem',
        'step',
        'streamplot',
        'tricontour',
        'tricontourf',
        'tripcolor',
        'triplot',
        'vlines',
        'xcorr',
        'barbs',
        ))


def gca_wrapper(key):
    ax = plt.gca()
    f = getattr(ax, key)
    @wraps(f)
    def inner(*args, **kwargs):
        ret = f(*args, **kwargs)
        plt.draw_if_interactive()
        return ret

    return inner


def gca_wrapper_hold(key):
    ax = plt.gca()
    f = getattr(ax, key)
    @wraps(f)
    def inner(*args, hold=None, **kwargs):
        washold = ax.ishold()
        if hold is not None:
            ax.hold(hold)
        try:
            ret = f(*args, **kwargs)
            plt.draw_if_interactive()
        finally:
            ax.hold(washold)

        return ret

    return inner


def gcf_wrapper(key):
    fig = plt.gcf()
    f = getattr(fig, key)
    @wraps(f)
    def inner(*args, **kwargs):
        ret = f(*args, **kwargs)
        plt.draw_if_interactive()
        return ret

    return inner


class pyplotNG(object):

    def __getattr__(self, key):
        if hasattr(Axes, key):
            if key in WRAP_WITH_HOLD:
                return gca_wrapper_hold(key)
            return gca_wrapper(key)
        elif hasattr(Figure, key):
            return gcf_wrapper(key)
        else:
            return getattr(plt, key)
