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

class Artist(HasTraits):

    stale = Bool(default_value = True)
    #the axes bounding box in display space
    #TO DO: window_extent -> Bbox([[0, 0], [0, 0]])

    axes = Axes('matplotlib.axes.Axes', allow_none = True)
    figure = Figure('matplotlib.figure.Figure', allow_none = True)
    transform = Transform('matplotlib.transform.Transform', allow_none = True)
    transformSet = Bool(default_value = False )
    children = List()
    #TO DO: create  PICKER trait that takes in a None, float, boolean, callable
    #TO DO: not sure to create snap ?

    alpha = Int(default_value = None ,allow_none = True)
    visible = Bool(default_value = True)
    animated = Bool(default_value = False)
    url = Unicode(default_value = None)
    #group id
    gid = Unicode(default_value = None)

    zorder = Int(default_value = 0)

    """3-tuple (scale, length or 128.0, randomness or 16.0):
    scale: amplitude of the wiggle perpendicular to the
      source line
    length: length of the wiggle along the line
    randomness: scale factor by which the length is
      shrunken or expanded
    """
    # sketch_Params = Tuple()

    clip_on = Bool(default_value = True)
    label = Unicode()
    # propobservers = {}  # a dict from oids to funcs
    propobservers = Dict()
    #observer id
    oid = Int()

    #Bbox as own TraitType and clip box is an instance of BBpx


    # clip path is anattribute
    #that will take in either a Trait
    #matplotlib.patches.Patch
    #matplotlib.path.Path
    #matplotlib.transforms.Transform
