from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import matplotlib.pyplot as plt  # to grab figure management stuff
from matplotlib.axes import Axes
import functools


# apparently, raw object objects use slots.
class Dummy(object):
    pass

pyplot_nng = Dummy()
interactive = Dummy()


def interactive_wrapper(func):
    @functools.wraps(func)
    def inner(ax, *args, **kwargs):
        ret_list = func(ax, *args, **kwargs)
        ax.figure.canvas.draw()
        return ret_list

    return inner


def wrap_for_pyplot(func):
    def inner(*args, **kwargs):
        ax = plt.gca()
        art_list = func(ax, *args, **kwargs)
        ax.figure.canvas.draw()
        return art_list

    inner.__name__ = func.__name__
    # This should be modified to strip the docs
    # if the axes as the first argument is documented
    inner.__doc__ = func.__doc__
    return inner

funcs_to_wrap = [atr for atr in
                 (getattr(Axes, atr_name) for atr_name in dir(Axes)
                  if not atr_name.startswith('_'))
                 if callable(atr)]

for f in funcs_to_wrap:
    setattr(pyplot_nng, f.__name__, wrap_for_pyplot(f))
    setattr(interactive, f.__name__, interactive_wrapper(f))
