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


class TraitProxy(TraitType):

    def __init__(self, trait):
        self.__trait = trait

    def instance_init(self, obj):
        self.__trait.instance_init(obj)

    def class_init(self, cls, name):
        self.__trait.class_init(cls, name)

    def set(self, obj, val):
        self.__trait.set(obj, val)

    def get(self, obj, cls):
        return self.__trait.get(obj, cls)

    def __getattr__(self, name):
        return getattr(self.__trait, name)

class Perishable(TraitProxy):

    def set(self, obj, val):
        super(Perishable, self).set(obj, val)
        obj.stale = True


# class ClipPathTrait(TraitType):
#
#     def __init__(self, trait):
#         #not sure if this going to work: needs testing
#         self.__trait = trait
#         pass
#
#     # def instance_init(self, obj):
#     #     pass
#     #
#     # def class_init(self, cls, name):
#     #     pass
#
#     def set(self, obj, val):
#
#         pass
#
#     def get(self, obj, cls):
#         return self.__trait.get(obj, cls)
#         pass
#
#     def __getattr__(self, name):
#         return getattr(self.__trait, name)

"""
def set_clip_path(self, path, transform=None):

    Set the artist's clip path, which may be:

      * a :class:`~matplotlib.patches.Patch` (or subclass) instance

      * a :class:`~matplotlib.path.Path` instance, in which case
         an optional :class:`~matplotlib.transforms.Transform`
         instance may be provided, which will be applied to the
         path before using it for clipping.

      * *None*, to remove the clipping path

    For efficiency, if the path happens to be an axis-aligned
    rectangle, this method will set the clipping box to the
    corresponding rectangle and set the clipping path to *None*.

    ACCEPTS: [ (:class:`~matplotlib.path.Path`,
    :class:`~matplotlib.transforms.Transform`) |
    :class:`~matplotlib.patches.Patch` | None ]

    from matplotlib.patches import Patch, Rectangle

    success = False
    if transform is None:
        if isinstance(path, Rectangle):
            self.clipbox = TransformedBbox(Bbox.unit(),
                                           path.get_transform())
            self._clippath = None
            success = True
        elif isinstance(path, Patch):
            self._clippath = TransformedPatchPath(path)
            success = True
        elif isinstance(path, tuple):
            path, transform = path

    if path is None:
        self._clippath = None
        success = True
    elif isinstance(path, Path):
        self._clippath = TransformedPath(path, transform)
        success = True
    elif isinstance(path, TransformedPatchPath):
        self._clippath = path
        success = True
    elif isinstance(path, TransformedPath):
        self._clippath = path
        success = True

    if not success:
        print(type(path), type(transform))
        raise TypeError("Invalid arguments to set_clip_path")
    # this may result in the callbacks being hit twice, but grantees they
    # will be hit at least once
    self.pchanged()
    self.stale = True
"""

"""
    def get_clip_path(self):
        'Return artist clip path'
        return self._clippath
"""
