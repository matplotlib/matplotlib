"""
This is artist.py implemented with Traits
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from collections import OrderedDict, namedtuple

import re
import warnings
import inspect
import numpy as np
import matplotlib
import matplotlib.cbook as cbook
from matplotlib.cbook import mplDeprecation
from matplotlib import docstring, rcParams
from matplotlib.transforms import (Bbox, IdentityTransform, TransformedBbox,
                         TransformedPatchPath, TransformedPath, Transform)
from matplotlib.path import Path
from functools import wraps
from contextlib import contextmanager

from traits import TraitProxy, Perishable

class Artist(HasTraits):

    aname = Unicode('Artist')
    zorder = Int(default_value = 0)
    #_prop_order = dict(color=-1)
    prop_order = Dict()

    stale = Bool(default_value = True)
    # stale_callback = Callable(allow_none = True, default_value = True)
    axes = Instance('matplotlib.axes.Axes', allow_none = True, default_value = None)
    figure = Instance('matplotlib.figure.Figure', allow_none = True, default_value = None)
    transform = Instance('matplotlib.transform.Transform', allow_none = True, default_value = None)
    transformSet = Bool(default_value = False )
    visible = Bool(default_value = True)
    animated = Bool(default_value = False)
    alpha = Float(default_value = None ,allow_none = True)
    clipbox = Instance('matplotlib.transforms.Bbox', allow_none = True, default_value = None)
    #clippath
    clipon = Boolean(default_value = True)
    label = Unicode(allow_none = True, default_value = '')
    picker = Union(Float, Boolean, Callable, allow_none = True, default_value = None)
    contains = List(default_value=None)
    rasterized = Perishable(Boolean(allow_none = True, default_value = None))
    agg_filter = Unicode(allow_none = True, default_value = None) #set agg_filter function
    mouseover = Boolean(default_value = False)
    eventson = Boolean(default_value = False)
    oid = Int(allow_none = True, default_value = 0)
    propobservers = Dict()
    url = Unicode(allow_none = True,default_value = None)
    gid = Unicode(allow_none = True, default_value = None)
    snap = Perishable(Boolean(allow_none = True, default_value = None))
    sketch = Tuple(Float(), Float(), Float(), default_value = rcParams['path.sketch'])
    path_effects = List(Instance('matplotlib.patheffect._Base'), default_value = rcParams['path.effects'])
    #_XYPair = namedtuple("_XYPair", "x y")
    #sticky_edges

    #the axes bounding box in display space
    #TO DO: window_extent -> Bbox([[0, 0], [0, 0]])

    # axes = Axes('matplotlib.axes.Axes', allow_none = True)
    # figure = Figure('matplotlib.figure.Figure', allow_none = True)
    # transform = Transform('matplotlib.transform.Transform', allow_none = True)
    #TO DO: create  PICKER trait that takes in a None, float, boolean, callable
    #TO DO: not sure to create snap ?
    #group id

"""
_______________________________________________________________________________
"""

    #stale default
    @default("stale")
    def stale_default(self):
        pass
    #stale validate
    @validate("stale")
    def stale_validate(self, proposal):
        pass
    #stale observer
    @observe("stale", type = change)
    def stale_observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #axes default
    @default("axes")
    def axes_default(self):
        pass
    #axes validate
    @validate("axes")
    def axes_validate(self, proposal):
        pass
    #axes observer
    @observe("axes", type = change)
    def axes_observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #figure default
    @default("")
    def _default(self):
        pass
    #figure validate
    @validate("")
    def _validate(self, proposal):
        pass
    #figure observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #transform default
    @default("")
    def _default(self):
        pass
    #transform validate
    @validate("")
    def _validate(self, proposal):
        pass
    #transform observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #transformSet default
    @default("")
    def _default(self):
        pass
    #transformSet validate
    @validate("")
    def _validate(self, proposal):
        pass
    #transformSet observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #visible default
    @default("")
    def _default(self):
        pass
    #visible validate
    @validate("")
    def _validate(self, proposal):
        pass
    #visible observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #animated default
    @default("")
    def _default(self):
        pass
    #animated validate
    @validate("")
    def _validate(self, proposal):
        pass
    #animated observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #alpha default
    @default("")
    def _default(self):
        pass
    #alpha validate
    @validate("")
    def _validate(self, proposal):
        pass
    #alpha observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #clipbox default
    @default("")
    def _default(self):
        pass
    #clipbox validate
    @validate("")
    def _validate(self, proposal):
        pass
    #clipbox observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #To do: create either a clippath trait or modify the get and set functions
    #for now i have comments down for default, validate and observer decortors
    #clippath default
    @default("")
    def _default(self):
        pass
    #clippath validate
    @validate("")
    def _validate(self, proposal):
        pass
    #clippath observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #clipon default
    @default("")
    def _default(self):
        pass
    #clipon validate
    @validate("")
    def _validate(self, proposal):
        pass
    #clipon observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #label default
    @default("")
    def _default(self):
        pass
    #label validate
    @validate("")
    def _validate(self, proposal):
        pass
    #label observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #picker default
    @default("")
    def _default(self):
        pass
    #picker validate
    @validate("")
    def _validate(self, proposal):
        pass
    #picker observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #contains default
    @default("")
    def _default(self):
        pass
    #contains validate
    @validate("")
    def _validate(self, proposal):
        pass
    #contains observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #rasterized default
    @default("")
    def _default(self):
        pass
    #rasterized validate
    @validate("")
    def _validate(self, proposal):
        pass
    #rasterized observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #agg_filter default
    @default("")
    def _default(self):
        pass
    #agg_filter validate
    @validate("")
    def _validate(self, proposal):
        pass
    #agg_filter observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #mouseover default
    @default("")
    def _default(self):
        pass
    #mouseover validate
    @validate("")
    def _validate(self, proposal):
        pass
    #mouseover observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #eventson default
    @default("")
    def _default(self):
        pass
    #eventson validate
    @validate("")
    def _validate(self, proposal):
        pass
    #eventson observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #oid default
    @default("")
    def _default(self):
        pass
    #oid validate
    @validate("")
    def _validate(self, proposal):
        pass
    #oid observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #propobservers default
    @default("")
    def _default(self):
        pass
    #propobservers validate
    @validate("")
    def _validate(self, proposal):
        pass
    #propobservers observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #url default
    @default("")
    def _default(self):
        pass
    #url validate
    @validate("")
    def _validate(self, proposal):
        pass
    #url observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #gid default
    @default("")
    def _default(self):
        pass
    #gid validate
    @validate("")
    def _validate(self, proposal):
        pass
    #gid observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #snap default
    @default("")
    def _default(self):
        pass
    #snap validate
    @validate("")
    def _validate(self, proposal):
        pass
    #snap observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #sketch default
    @default("")
    def _default(self):
        pass
    #sketch validate
    @validate("")
    def _validate(self, proposal):
        pass
    #sketch observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #path_effects default
    @default("")
    def _default(self):
        pass
    #path_effects validate
    @validate("")
    def _validate(self, proposal):
        pass
    #path_effects observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""

    #sticky_edges default
    @default("")
    def _default(self):
        pass
    #sticky_edges validate
    @validate("")
    def _validate(self, proposal):
        pass
    #sticky_edges observer
    @observe("", type = change)
    def _observe(self, change):
        pass

"""
_______________________________________________________________________________
"""
