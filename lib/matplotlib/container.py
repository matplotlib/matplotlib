from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from matplotlib.artist import Artist
import matplotlib.cbook as cbook


class Container(tuple, Artist):
    """
    Base class for containers.
    """

    def __repr__(self):
        return "<Container object of %d artists>" % (len(self))

    def __new__(cls, *kl, **kwargs):
        return tuple.__new__(cls, kl[0])

    def __init__(self, kl, label=None, **kwargs):
        Artist.__init__(self, **kwargs)
        self.set_label(label=label)

    def remove(self):
        # remove the children
        for c in self:
            c.remove()
        # call up to the Artist remove method
        super(Container, self).remove(self)

    def get_children(self):
        return list(cbook.flatten(self))

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
