"""
This module contains the instantiations of color mapping classes
"""

import colors
from matplotlib import verbose
from matplotlib import rcParams
from numerix import nx
LUTSIZE = rcParams['image.lut']

_gray_data =  {'red':   ((0., 0, 0), (1., 1, 1)),
               'green': ((0., 0, 0), (1., 1, 1)),
               'blue':  ((0., 0, 0), (1., 1, 1))}
_jet_data =   {'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89,1, 1), 
                         (1, 0.5, 0.5)),
               'green': ((0., 0, 0), (0.125,0, 0), (0.375,1, 1), (0.64,1, 1),
                         (0.91,0,0), (1, 0, 0)),   
               'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65,0, 0),
                         (1, 0, 0))}


datad = {
    'gray' : _gray_data,
    'jet' :  _jet_data,
    }
gray = colors.LinearSegmentedColormap('gray', _gray_data, LUTSIZE)
jet =  colors.LinearSegmentedColormap('jet',  _jet_data, LUTSIZE)

def get_cmap(name=None, lut=None):
    """
    Get a colormap instance, defaulting to rc values if name is None
    """
    if name is None: name = rcParams['image.cmap']
    if lut is None: lut = rcParams['image.lut']
    
    assert(name in datad.keys())
    return colors.LinearSegmentedColormap(name,  datad[name], lut)

# These are provided for backwards compat
import sys
def ColormapJet(N=LUTSIZE):
    verbose.report_error("ColormapJet deprecated, please use cm.jet instead")
    return colors.LinearSegmentedColormap('jet',  _jet_data, N)

def Grayscale(N=LUTSIZE):
    verbose.report_error("Grayscale deprecated, please use cm.jet instead")
    return colors.LinearSegmentedColormap('gray',  _gray_data, N)


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
        
    def to_rgba(self, x, alpha=1.0):
        # assume normalized rgb, rgba
        if len(x.shape)>2: return x
        x = self.norm(x)
        

        return self.cmap(x, alpha)
    
    def set_array(self, A):
        'Set the image array from numeric/numarray A'
        self._A = A.astype(nx.Float32)

    def set_clim(self, vmin=None, vmax=None):
        'set the norm limits for image scaling'
        self.norm.vmin = vmin
        self.norm.vmax = vmax
        self.changed()
        
    def set_cmap(self, cmap):
        'set the colormap for luminance data'
        if cmap is None: cmap = get_cmap()                
        self.cmap = cmap
        self.changed()
        
    def set_norm(self, norm):
        'set the colormap for luminance data'
        if norm is None: norm = colors.normalize()
        self.norm = norm
        self.changed()

    def autoscale(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        assert(self._A is not None)
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
        self.cmap = mappable.cmap
        self.norm = mappable.norm

    def changed(self):
        """
        Call this whenever the mappable is changed so observers can
        update state
        """
        for observer in self.observers:
            observer.notify(self)
