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
        print("generating default stale value")
        return True
    #stale validate
    @validate("stale")
    def stale_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #stale observer
    @observe("stale", type = change)
    def stale_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #axes default
    @default("axes")
    def axes_default(self):
        print("generating default stale value")

    #axes validate
    @validate("axes")
    def axes_validate(self, proposal):

    #axes observer
    @observe("axes", type = change)
    def axes_observe(self, change):


"""
_______________________________________________________________________________
"""

    #figure default
    @default("figure")
    def figure_default(self):
        print("generating default stale value")

    #figure validate
    @validate("figure")
    def figure_validate(self, proposal):

    #figure observer
    @observe("figure", type = change)
    def figure_observe(self, change):


"""
_______________________________________________________________________________
"""

    #transform default
    @default("transform")
    def transform_default(self):
        print("generating default stale value")

    #transform validate
    @validate("transform")
    def transform_validate(self, proposal):

    #transform observer
    @observe("transform", type = change)
    def transform_observe(self, change):


"""
_______________________________________________________________________________
"""

    #transformSet default
    @default("transformSet")
    def transformSet_default(self):
        print("generating default stale value")

    #transformSet validate
    @validate("transformSet")
    def transformSet_validate(self, proposal):

    #transformSet observer
    @observe("transformSet", type = change)
    def transformSet_observe(self, change):


"""
_______________________________________________________________________________
"""

    #visible default
    @default("visible")
    def visible_default(self):
        print("generating default stale value")

    #visible validate
    @validate("visible")
    def visible_validate(self, proposal):

    #visible observer
    @observe("visible", type = change)
    def visible_observe(self, change):


"""
_______________________________________________________________________________
"""

    #animated default
    @default("animated")
    def animated_default(self):
        print("generating default stale value")

    #animated validate
    @validate("animated")
    def animated_validate(self, proposal):

    #animated observer
    @observe("animated", type = change)
    def animated_observe(self, change):


"""
_______________________________________________________________________________
"""

    #alpha default
    @default("alpha")
    def alpha_default(self):
        print("generating default stale value")

    #alpha validate
    @validate("alpha")
    def alpha_validate(self, proposal):

    #alpha observer
    @observe("alpha", type = change)
    def alpha_observe(self, change):


"""
_______________________________________________________________________________
"""

    #clipbox default
    @default("clipbox")
    def clipbox_default(self):
        print("generating default stale value")

    #clipbox validate
    @validate("clipbox")
    def clipbox_validate(self, proposal):

    #clipbox observer
    @observe("clipbox", type = change)
    def clipbox_observe(self, change):


"""
_______________________________________________________________________________
"""

    #To do: create either a clippath trait or modify the get and set functions
    #for now i have comments down for default, validate and observer decortors
    #clippath default
    @default("clippath")
    def clippath_default(self):

    #clippath validate
    @validate("clippath")
    def clippath_validate(self, proposal):

    #clippath observer
    @observe("clippath", type = change)
    def clippath_observe(self, change):


"""
_______________________________________________________________________________
"""

    #clipon default
    @default("clipon")
    def clipon_default(self):

    #clipon validate
    @validate("clipon")
    def clipon_validate(self, proposal):

    #clipon observer
    @observe("clipon", type = change)
    def clipon_observe(self, change):


"""
_______________________________________________________________________________
"""

    #label default
    @default("label")
    def label_default(self):

    #label validate
    @validate("label")
    def label_validate(self, proposal):

    #label observer
    @observe("label", type = change)
    def label_observe(self, change):


"""
_______________________________________________________________________________
"""

    #picker default
    @default("picker")
    def picker_default(self):

    #picker validate
    @validate("picker")
    def picker_validate(self, proposal):

    #picker observer
    @observe("picker", type = change)
    def picker_observe(self, change):


"""
_______________________________________________________________________________
"""

    #contains default
    @default("contains")
    def contains_default(self):

    #contains validate
    @validate("contains")
    def contains_validate(self, proposal):

    #contains observer
    @observe("contains", type = change)
    def contains_observe(self, change):


"""
_______________________________________________________________________________
"""

    #rasterized default
    @default("rasterized")
    def rasterized_default(self):

    #rasterized validate
    @validate("rasterized")
    def rasterized_validate(self, proposal):

    #rasterized observer
    @observe("rasterized", type = change)
    def rasterized_observe(self, change):


"""
_______________________________________________________________________________
"""

    #agg_filter default
    @default("agg_filter")
    def agg_filter_default(self):

    #agg_filter validate
    @validate("agg_filter")
    def agg_filter_validate(self, proposal):

    #agg_filter observer
    @observe("agg_filter", type = change)
    def agg_filter_observe(self, change):


"""
_______________________________________________________________________________
"""

    #mouseover default
    @default("mouseover")
    def mouseover_default(self):

    #mouseover validate
    @validate("mouseover")
    def mouseover_validate(self, proposal):

    #mouseover observer
    @observe("mouseover", type = change)
    def mouseover_observe(self, change):


"""
_______________________________________________________________________________
"""

    #eventson default
    @default("eventson")
    def eventson_default(self):

    #eventson validate
    @validate("eventson")
    def eventson_validate(self, proposal):

    #eventson observer
    @observe("eventson", type = change)
    def eventson_observe(self, change):


"""
_______________________________________________________________________________
"""

    #oid default
    @default("oid")
    def oid_default(self):

    #oid validate
    @validate("oid")
    def oid_validate(self, proposal):

    #oid observer
    @observe("oid", type = change)
    def oid_observe(self, change):


"""
_______________________________________________________________________________
"""

    #propobservers default
    @default("propobservers")
    def propobservers_default(self):

    #propobservers validate
    @validate("propobservers")
    def propobservers_validate(self, proposal):

    #propobservers observer
    @observe("propobservers", type = change)
    def propobservers_observe(self, change):


"""
_______________________________________________________________________________
"""

    #url default
    @default("url")
    def url_default(self):

    #url validate
    @validate("url")
    def url_validate(self, proposal):

    #url observer
    @observe("url", type = change)
    def url_observe(self, change):


"""
_______________________________________________________________________________
"""

    #gid default
    @default("gid")
    def gid_default(self):

    #gid validate
    @validate("gid")
    def gid_validate(self, proposal):

    #gid observer
    @observe("gid", type = change)
    def gid_observe(self, change):


"""
_______________________________________________________________________________
"""

    #snap default
    @default("snap")
    def snap_default(self):

    #snap validate
    @validate("snap")
    def snap_validate(self, proposal):

    #snap observer
    @observe("snap", type = change)
    def snap_observe(self, change):


"""
_______________________________________________________________________________
"""

    #sketch default
    @default("sketch")
    def sketch_default(self):

    #sketch validate
    @validate("sketch")
    def sketch_validate(self, proposal):

    #sketch observer
    @observe("sketch", type = change)
    def sketch_observe(self, change):


"""
_______________________________________________________________________________
"""

    #path_effects default
    @default("path_effects")
    def path_effects_default(self):
        print("generating default stale value")

    #path_effects validate
    @validate("path_effects")
    def path_effects_validate(self, proposal):

    #path_effects observer
    @observe("path_effects", type = change)
    def path_effects_observe(self, change):


"""
_______________________________________________________________________________
"""

    #sticky_edges default
    @default("sticky_edges")
    def sticky_edges_default(self):
        print("generating default stale value")

    #sticky_edges validate
    @validate("sticky_edges")
    def sticky_edges_validate(self, proposal):

    #sticky_edges observer
    @observe("sticky_edges", type = change)
    def sticky_edges_observe(self, change):


"""
_______________________________________________________________________________
"""
