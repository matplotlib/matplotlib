from __future__ import division
import sys
from transforms import identity_transform

## Note, matplotlib artists use the doc strings for set and get
# methods to enable the introspection methods of set and get in the
# matlab interface Every set_ to be controlled by the set function
# method should have a docstring containing the line
#
# ACCEPTS: [ legal | values ]
#
# and aliases for setters and getters should have a docstring that
# starts with 'alias for ', as in 'alias for set_somemethod'
#
# You may wonder why we use so much boiler-plate manually defining the
# set_alias and get_alias functions, rather than using some clever
# python trick.  The answer is that I need to be able to manipulate
# the docstring, and there is no clever way to do that in python 2.2,
# as far as I can see - see
# http://groups.google.com/groups?hl=en&lr=&threadm=mailman.5090.1098044946.5135.python-list%40python.org&rnum=1&prev=/groups%3Fq%3D__doc__%2Bauthor%253Ajdhunter%2540ace.bsd.uchicago.edu%26hl%3Den%26btnG%3DGoogle%2BSearch


class Artist:
    """
    Abstract base class for someone who renders into a FigureCanvas
    """

    aname = 'Artist'
    zorder = 0
    def __init__(self):

        self.figure = None
        self._transform = identity_transform()
        self._transformSet = False
        self._visible = True
        self._alpha = 1.0
        self.clipbox = None
        self._clipon = False
        self._lod = False
        self._label = ''
        
    def is_transform_set(self):
        'Artist has transform explicity let'                
        return self._transformSet

    def set_transform(self, t):
        """
set the Transformation instance used by this artist

ACCEPTS: a matplotlib.transform transformation instance
"""
        self._transform = t
        self._transformSet = True

    def get_transform(self):
        'return the Transformation instance used by this artist'
        return self._transform

    def is_figure_set(self):
        return self.figure is not None

    def get_figure(self):
        'return the figure instance'
        return self.figure
    
    def set_figure(self, fig):
        """
Set the figure instance the artist belong to

ACCEPTS: a matplotlib.figure.Figure instance
        """
        self.figure = fig
                
    def set_clip_box(self, clipbox):        
        """
Set the artist's clip Bbox

ACCEPTS: a matplotlib.transform.Bbox instance
        """
        self.clipbox = clipbox
        self._clipon = clipbox is not None
        
    def get_alpha(self):
        """
Return the alpha value used for blending - not supported on all
backends
        """
        return self._alpha

    def get_visible(self):
        "return the artist's visiblity"
        return self._visible 

    def get_clip_on(self):
        'Return whether artist uses clipping'
        return self._clipon and self.clipbox is not None

    def set_clip_on(self, b):
        """
Set  whether artist uses clipping

ACCEPTS: [True | False]
"""
        self._clipon = b
        if not b: self.clipbox = None
        
    def draw(self, renderer, *args, **kwargs):
        'Derived classes drawing method'
        if not self.get_visible(): return 

    def set_alpha(self, alpha):
        """
Set the alpha value used for blending - not supported on
all backends

ACCEPTS: float
        """
        self._alpha = alpha


    def set_lod(self, on):
        """
Set Level of Detail on or off.  If on, the artists may examine
things like the pixel width of the axes and draw a subset of
their contents accordingly

ACCEPTS: [True | False]
        """
        self._lod = on

    def set_visible(self, b):
        """
set the artist's visiblity

ACCEPTS: [True | False]
"""
        self._visible = b

    def update(self, props):
        for k,v in props.items():
            func = getattr(self, 'set_'+k, None)
            if func is None or not callable(func):
                raise AttributeError('Unknown property %s'%k)
            func(v)

    def get_label(self): return self._label

    def set_label(self, s):
        """
Set the line label to s for auto legend

ACCEPTS: any string
"""
        self._label = s

    def get_zorder(self): return self.zorder

    def set_zorder(self, level):
        """
Set the zorder for the artist

ACCEPTS: any number
"""
        self.zorder = level
    
