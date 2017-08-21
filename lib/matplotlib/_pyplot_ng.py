from functools import wraps

import matplotlib.pyplot as plt
import matplotlib

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
    def inner(*args, **kwargs):
        hold = kwargs.pop('hold', None)
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


def set_defaults(cls, key, new_defaults):
    if hasattr(cls, '_orig_' + key):
        orig_fun = getattr(Axes, '_orig_' + key)
    else:
        orig_fun = getattr(cls, key)
    setattr(cls, '_orig_' + key, orig_fun)

    @wraps(orig_fun)
    def wrapper(*args, **kwargs):
        for k, v in new_defaults.iteritems():
            if k not in kwargs:
                kwargs[k] = v
        return orig_fun(*args, **kwargs)

    setattr(cls, key, wrapper)


def reset_defaults(cls, key):
    if hasattr(cls, '_orig_' + key):
        setattr(cls, key, getattr(Axes, '_orig_' + key))
