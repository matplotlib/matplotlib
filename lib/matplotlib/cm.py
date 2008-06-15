"""
This module contains the instantiations of color mapping classes
"""

import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib._cm import *



def get_cmap(name=None, lut=None):
    """
    Get a colormap instance, defaulting to rc values if *name* is None
    """
    if name is None: name = mpl.rcParams['image.cmap']
    if lut is None: lut = mpl.rcParams['image.lut']

    assert(name in datad.keys())
    return colors.LinearSegmentedColormap(name,  datad[name], lut)

class ScalarMappable:
    """
    This is a mixin class to support scalar -> RGBA mapping.  Handles
    normalization and colormapping
    """

    def __init__(self, norm=None, cmap=None):
        """
        *norm* is an instance of :class:`colors.Normalize` or one of
        its subclasses, used to map luminance to 0-1. *cmap* is a
        :mod:`cm` colormap instance, for example :data:`cm.jet`
        """

        self.callbacksSM = cbook.CallbackRegistry((
                'changed',))

        if cmap is None: cmap = get_cmap()
        if norm is None: norm = colors.Normalize()

        self._A = None
        self.norm = norm
        self.cmap = cmap
        self.colorbar = None
        self.update_dict = {'array':False}

    def set_colorbar(self, im, ax):
        'set the colorbar image and axes associated with mappable'
        self.colorbar = im, ax

    def to_rgba(self, x, alpha=1.0, bytes=False):
        '''Return a normalized rgba array corresponding to *x*. If *x*
        is already an rgb array, insert *alpha*; if it is already
        rgba, return it unchanged. If *bytes* is True, return rgba as
        4 uint8s instead of 4 floats.
        '''
        try:
            if x.ndim == 3:
                if x.shape[2] == 3:
                    if x.dtype == np.uint8:
                        alpha = np.array(alpha*255, np.uint8)
                    m, n = x.shape[:2]
                    xx = np.empty(shape=(m,n,4), dtype = x.dtype)
                    xx[:,:,:3] = x
                    xx[:,:,3] = alpha
                elif x.shape[2] == 4:
                    xx = x
                else:
                    raise ValueError("third dimension must be 3 or 4")
                if bytes and xx.dtype != np.uint8:
                    xx = (xx * 255).astype(np.uint8)
                return xx
        except AttributeError:
            pass
        x = ma.asarray(x)
        x = self.norm(x)
        x = self.cmap(x, alpha=alpha, bytes=bytes)
        return x

    def set_array(self, A):
        'Set the image array from numpy array *A*'
        self._A = A
        self.update_dict['array'] = True

    def get_array(self):
        'Return the array'
        return self._A

    def get_cmap(self):
        'return the colormap'
        return self.cmap

    def get_clim(self):
        'return the min, max of the color limits for image scaling'
        return self.norm.vmin, self.norm.vmax

    def set_clim(self, vmin=None, vmax=None):
        """
        set the norm limits for image scaling; if *vmin* is a length2
        sequence, interpret it as ``(vmin, vmax)`` which is used to
        support setp

        ACCEPTS: a length 2 sequence of floats
        """
        if (vmin is not None and vmax is None and
                                cbook.iterable(vmin) and len(vmin)==2):
            vmin, vmax = vmin

        if vmin is not None: self.norm.vmin = vmin
        if vmax is not None: self.norm.vmax = vmax
        self.changed()

    def set_cmap(self, cmap):
        """
        set the colormap for luminance data

        ACCEPTS: a colormap
        """
        if cmap is None: cmap = get_cmap()
        self.cmap = cmap
        self.changed()

    def set_norm(self, norm):
        'set the normalization instance'
        if norm is None: norm = colors.Normalize()
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

    def autoscale_None(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        if self._A is None:
            raise TypeError('You must first set_array for mappable')
        self.norm.autoscale_None(self._A)
        self.changed()


    def add_checker(self, checker):
        """
        Add an entry to a dictionary of boolean flags
        that are set to True when the mappable is changed.
        """
        self.update_dict[checker] = False

    def check_update(self, checker):
        """
        If mappable has changed since the last check,
        return True; else return False
        """
        if self.update_dict[checker]:
            self.update_dict[checker] = False
            return True
        return False

    def changed(self):
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal
        """
        self.callbacksSM.process('changed', self)

        for key in self.update_dict:
            self.update_dict[key] = True
