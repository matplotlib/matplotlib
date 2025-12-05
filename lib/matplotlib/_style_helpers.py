import collections.abc
import itertools

import matplotlib.cbook as cbook
import matplotlib.colors as mcolors
import matplotlib.lines as mlines



def check_non_empty(key, value):
    """Raise a TypeError if an empty sequence is passed"""
    if isinstance(value, collections.abc.Sized) and len(value) == 0:
        raise TypeError(f'{key} must not be an empty sequence')


def iterate_styles(kw):
    """
    Helper for handling style sequences (e.g. facecolor=['r', 'b', 'k']) within plotting
    methods that repeatedly call other plotting methods (e.g. hist, stackplot).

    Given a dictionary of keyword parameters, yield a series of copies of the
    dictionary.  Style parameters expressed as sequences are replaced in the copy with
    the next element of the sequence.

    Note 'color' is deliberately not handled since the calling methods have their own
    handling for that.
    """
    kw_iterators = {}
    remaining_kw = {}
    for key, value in kw.items():
        if cbook.is_scalar_or_string(value):
            # No iteration required
            remaining_kw[key] = value

        elif key in ['facecolor', 'edgecolor']:
            check_non_empty(key, value)
            kw_iterators[key] = itertools.cycle(mcolors.to_rgba_array(value))

        elif key in ['hatch', 'linewidth']:
            check_non_empty(key, value)
            kw_iterators[key] = itertools.cycle(value)

        elif key == 'linestyle':
            check_non_empty(key, value)
            kw_iterators[key] = itertools.cycle(mlines._get_dash_patterns(value))

        else:
            remaining_kw[key] = value


    while True:
        yield {key: next(val) for key, val in kw_iterators.items()} | remaining_kw
