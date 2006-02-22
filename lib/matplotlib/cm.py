"""
This module contains the instantiations of color mapping classes
"""

import colors
from matplotlib import verbose
from matplotlib import rcParams
from matplotlib.numerix import asarray
from numerix import nx
import numerix.ma as ma
from _cm import *



def get_cmap(name=None, lut=None):
    """
    Get a colormap instance, defaulting to rc values if name is None
    """
    if name is None: name = rcParams['image.cmap']
    if lut is None: lut = rcParams['image.lut']

    assert(name in datad.keys())
    return colors.LinearSegmentedColormap(name,  datad[name], lut)

class ScalarMappable:
    """
    This is a mixin class to support scalar -> RGBA mapping.  Handles
    normalization and colormapping
    """

    def __init__(self, norm=None, cmap=None):
        """
        norm is a colors.Norm instance to map luminance to 0-1
        cmap is a cm colormap instance
        """

        if cmap is None: cmap = get_cmap()
        if norm is None: norm = colors.normalize()

        self._A = None
        self.norm = norm
        self.cmap = cmap
        self.observers = []
        self.colorbar = None

    def set_colorbar(self, im, ax):
        'set the colorbar image and axes associated with mappable'
        self.colorbar = im, ax

    def to_rgba(self, x, alpha=1.0):
        # assume normalized rgb, rgba
        if len(x.shape)>2: return x
        x = ma.asarray(x)
        x = self.norm(x)
        x = self.cmap(x, alpha)
        return x

    def set_array(self, A):
        'Set the image array from numeric/numarray A'
        self._A = A.astype(nx.Float32)

    def get_array(self):
        'Return the array'
        return self._A

    def set_clim(self, vmin=None, vmax=None):
        'set the norm limits for image scaling'
        self.norm.vmin = vmin
        self.norm.vmax = vmax
        if self.colorbar is not None:
            im, ax = self.colorbar
            ax.set_ylim((vmin, vmax))
        self.changed()

    def set_cmap(self, cmap):
        'set the colormap for luminance data'
        if cmap is None: cmap = get_cmap()
        self.cmap = cmap
        self.changed()

    def set_norm(self, norm):
        'set the normalization instance'
        if norm is None: norm = colors.normalize()
        self.norm = norm
        self.changed()

    def autoscale(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        self.norm.autoscale(self._A)
        self.changed()

    def add_observer(self, mappable):
        """
        whenever the norm, clim or cmap is set, call the notify
        instance of the mappable observer with self.

        This is designed to allow one image to follow changes in the
        cmap of another image
        """
        self.observers.append(mappable)

    def notify(self, mappable):
        """
        If this is called then we are pegged to another mappable.
        Update the cmap, norm accordingly
        """
        self.set_cmap(mappable.cmap)
        self.set_norm(mappable.norm)

    def changed(self):
        """
        Call this whenever the mappable is changed so observers can
        update state
        """
        for observer in self.observers:
            observer.notify(self)
