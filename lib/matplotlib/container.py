from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from matplotlib.artist import Artist, allow_rasterization
import matplotlib.cbook as cbook


class Container(tuple, Artist):
    """
    Base class for containers.
    """
    _no_broadcast = ['label', 'visible', 'zorder', 'animated',
                     'agg_filter']

    def __repr__(self):
        return "<Container object of %d artists>" % (len(self))

    def __new__(cls, *kl, **kwargs):
        return tuple.__new__(cls, kl[0])

    def __init__(self, kl, label=None, **kwargs):
        # set up the artist details
        Artist.__init__(self, **kwargs)
        # for some reason we special case label
        self.set_label(label)

    def remove(self):
        # remove the children
        for c in self:
            c.remove()
        # call up to the Artist remove method
        super(Container, self).remove(self)

    def get_children(self):
        return list(cbook.flatten(self))

    def __getattribute__(self, key):

        # broadcast set_* and get_* methods across members
        # except for these explicitly not.
        if (('set' in key or 'get' in key) and
              all(k not in key for k in self._no_broadcast)):

            def inner(*args, **kwargs):
                return [getattr(a, key)(*args, **kwargs)
                        for a in self]
            inner.__name__ = key
            doc = getattr(self[0], key).__doc__
            inner.__doc__ = doc
            return inner
        else:
            return super(Container, self).__getattribute__(key)

    @allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        # just broadcast the draw down to children
        for a in self:
            a.draw(renderer, *args, **kwargs)


class BarContainer(Container):
    def __new__(cls, patches, errorbar=None, **kwargs):
        if errorbar is None:
            errorbar = tuple()
        else:
            errorbar = tuple(errorbar)
        patches = tuple(patches)
        return super(BarContainer, cls).__new__(patches + errorbar, **kwargs)

    def __init__(self, patches, errorbar=None, **kwargs):
        self.patches = patches
        self.errorbar = errorbar
        Container.__init__(self, patches, **kwargs)


class ErrorbarContainer(Container):

    def __init__(self, lines, has_xerr=False, has_yerr=False, **kwargs):
        self.lines = lines
        self.has_xerr = has_xerr
        self.has_yerr = has_yerr
        Container.__init__(self, lines, **kwargs)


class StemContainer(Container):

    def __init__(self, markerline_stemlines_baseline, **kwargs):
        markerline, stemlines, baseline = markerline_stemlines_baseline
        self.markerline = markerline
        self.stemlines = stemlines
        self.baseline = baseline
        Container.__init__(self, markerline_stemlines_baseline, **kwargs)
