from __future__ import division
import sys
from cbook import True, False
from transforms import identity_transform

class Artist:
    """
    Abstract base class for someone who renders into a FigureCanvas
    """

    aname = 'Artist'
    def __init__(self):

        self.figure = None
        self._transform = identity_transform()
        self._transformSet = False
        self._visible = True
        self._alpha = 1.0
        self.clipbox = None
        self._clipon = False
        self._lod = False

    def is_transform_set(self):
        'Artist has transform explicity let'                
        return self._transformSet

    def set_transform(self, t):
        'set the Transformation instance used by this artist'        
        self._transform = t
        self._transformSet = True

    def get_transform(self):
        'return the Transformation instance used by this artist'
        return self._transform

    def is_figure_set(self):
        return self.figure is not None

    def set_figure(self, fig):
        """
        Set the figure instance the artist belong to
        """
        self.figure = fig
                
    def set_clip_box(self, clipbox):
        """
        Set the artist's clip Bbox
        """
        self.clipbox = clipbox
        self._clipon = clipbox is not None
        
    def get_alpha(self):
        """
        Return the alpha value used for blending - not supported on
        all backends
        """
        return self._alpha

    def get_visible(self):
        "return the artist's visiblity"
        return self._visible 

    def get_clip_on(self):
        'Return whether artist uses clipping'
        return self._clipon and self.clipbox is not None

    def set_clip_on(self, b):
        'Set  whether artist uses clipping'
        self._clipon = b

    def draw(self, renderer, *args, **kwargs):
        'Derived classes drawing method'
        pass

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on
        all backends
        """
        self._alpha = alpha


    def set_lod(self, on):
        """
        Set Level of Detail on or off.  If on, the artists may examine
        things like the pixel width of the axes and draw a subset of
        their contents accordingly
        """
        self._lod = on

    def set_visible(self, b):
        "set the artist's visiblity"
        self._visible = b
