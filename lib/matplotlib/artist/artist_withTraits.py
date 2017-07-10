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
    def stale_observer(self, change):

    #axes default
    @default("axes")
    def axes_default(self):
        pass
    #axes validate
    @validate("axes")
    def axes_validate(self, proposal):
        pass
    #axes observer
    @observer("axes", type = change)
    def axes_observer(self, change):
        pass

    #figure default
    @default("")
    def _default(self):
        pass
    #figure validate
    @validate("")
    def _validate(self, proposal):
        pass
    #figure observer

    #transform default
    @default("")
    def _default(self):
        pass
    #transform validate
    #transform observer

    #transformSet default
    @default("")
    def _default(self):
        pass
    #transformSet validate
    #transformSet observer

    #visible default
    @default("")
    def _default(self):
        pass
    #visible validate
    #visible observer

    #animated default
    @default("")
    def _default(self):
        pass
    #animated validate
    #animated observer

    #alpha default
    @default("")
    def _default(self):
        pass
    #alpha validate
    #alpha observer

    #clipbox default
    @default("")
    def _default(self):
        pass
    #clipbox validate
    #clipbox observer

    #To do: create either a clippath trait or modify the get and set functions
    #for now i have comments down for default, validate and observer decortors
    #clippath default
    @default("")
    def _default(self):
        pass
    #clippath validate
    #clippath observer

    #clipon default
    @default("")
    def _default(self):
        pass
    #clipon validate
    #clipon observer

    #label default
    @default("")
    def _default(self):
        pass
    #label validate
    #label observer

    #picker default
    @default("")
    def _default(self):
        pass
    #picker validate
    #picker observer

    #contains default
    @default("")
    def _default(self):
        pass
    #contains validate
    #contains observer

    #rasterized default
    @default("")
    def _default(self):
        pass
    #rasterized validate
    #rasterized observer

    #agg_filter default
    @default("")
    def _default(self):
        pass
    #agg_filter validate
    #agg_filter observer

    #mouseover default
    @default("")
    def _default(self):
        pass
    #mouseover validate
    #mouseover observer

    #eventson default
    @default("")
    def _default(self):
        pass
    #eventson validate
    #eventson observer

    #oid default
    @default("")
    def _default(self):
        pass
    #oid validate
    #oid observer

    #propobservers default
    @default("")
    def _default(self):
        pass
    #propobservers validate
    #propobservers observer

    #url default
    @default("")
    def _default(self):
        pass
    #url validate
    #url observer

    #gid default
    @default("")
    def _default(self):
        pass
    #gid validate
    #gid observer

    #snap default
    @default("")
    def _default(self):
        pass
    #snap validate
    #snap observer

    #sketch default
    @default("")
    def _default(self):
        pass
    #sketch validate
    #sketch observer

    #path_effects default
    @default("")
    def _default(self):
        pass
    #path_effects validate
    #path_effects observer

    #sticky_edges default
    @default("")
    def _default(self):
        pass
    #sticky_edges validate
    #sticky_edges observer

    #default
    #validate
    #observer

    #default
    #validate
    #observer

    #default
    #validate
    #observer

    #default
    #validate
    #observer

    #default
    #validate
    #observer

    #default
    #validate
    #observer

    #default
    #validate
    #observer
