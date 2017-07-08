from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from traitlets import *

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
from .transforms import (Bbox, IdentityTransform, TransformedBbox,
                         TransformedPatchPath, TransformedPath, Transform)
from .path import Path
from functools import wraps
from contextlib import contextmanager

"""matplotlib.axes.Axes
"""
class AxesTrait(TypeCast):

    allow_none = True
    default_value = None
    klass = matplotlib.axes.Axes

    def validate(self, obj, value):
        value = super(Axes, self).validate(obj, value)
        if value not in (getattr(obj, self.name), None):
            raise ValueError("Can not reset the axes. You are "
                "probably trying to re-use an artist in more "
                "than one Axes which is not supported.")
        if value is not None and value is not self:
            obj.stale_callback = _stale_axes_callback
        return value

"""matplotlib.figure.Figure
"""
class FigureTrait(TypeCast):

    allow_none = True
    default_value = None
    klass = matplotlib.figure.Figure

    def validate(self, obj, value):
        value(Figure, self).validate(obj, value)
        if value not in (getattr(obj, self.name), None):
            raise RuntimeError("Can not put single artist in "
                               "more than one figure")
        if value is not None and value is not self:
            self.pchanged()
        self.stale = True
        return value

#
# class TransformTrait(TypeCast):
#
#     allow_none = True
#     default_value = None
#     klass = matplotlib.transforms.Transform
#
#     def validate(self, obj, value):
#         value(Transform, self).validate(obj, value)
#         if value

"""BboxTrait -> will be used to create
    1. window_extent
    2. clip box
"""
# class BboxTrait(TypeCast):
#
#     allow_none = True
#     default_value = None
#     klass = matplotlib.transforms.Bbox
#
#     def validate(self, obj, value):
#         value.(Bbox, self).validate(obj, value)

# class PathTrait(TypeCast):

class PatchTrait(TypeCast):

    allow_none = True
    default_value = None
    klass = matplotlib.path.Path

    def validate(self, obj, value)
