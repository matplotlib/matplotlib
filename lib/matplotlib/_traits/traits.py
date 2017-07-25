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

class TransformTrait(TraitType):

    default_value = None
    allow_none = True
    info_text = "TransformTrait"

    #Question: why use the get_transform function as the validate?
    # I understand that there is a logic involving how to handle None and
    #returning IdentityTransform() if there is None, but at the time,
    #how does that validate it?
    def validate(self, obj, value):
        if value is None:
            return IdentityTransform()
    elif (not isinstance(value, Transform)
          and hasattr(value, '_as_mpl_transform')):

        # self._transform = self._transform._as_mpl_transform(self.axes)
    return value
#
# class PathTrait(TraitType):
#
#     default_value = None
#     allow_none = True
#     info_text = "PathTrait"
#
# class PatchTrait(TraitType):
#


class ClipPathTrait(TraitType):


    """
    value[0] = matplotlib.path.Path
    value[1] = TransformTrait: if None is input TransformTrait validation logic will return Idenity Trait which
    is a trait
    """
    def validate(self, obj, value):
        if isinstance(value, tuple):
            if len(value)==2:
                if isinstance(value[0], 'matplotlib.path.Path') and isinstance(value[1], TransformTrait):
                    path = value[0]
                    transform = value[1]
                    return value

    def __init__(self, trait):
        pass
        # self.__trait = trait

    def instance_init(self, obj):
        pass
        # self.__trait.instance_init(obj)

    def class_init(self, cls, name):
        pass
        # self.__trait.class_init(cls, name)

    # reference artist.py set_clip_path function
    def set(self, obj, value):

        #import statements
        from matplotlib.patches import Patch, Rectangle
        from matplotlib.path import Path

        #extract values from value
        path = value[0]
        transform = value[1]

        #used for error checking
        success = False

        if transform is None:
            if isinstance(path, Rectangle):
                #we have to input a clipbox as well...
                self.__trait.set(obj, None)
                # self.clipbox = TransformedBbox(Bbox.unit(), path.get_transform())
                # self._clippath = None
                success = True
            elif isinstance(path, Patch):
                self.__trait.set(obj, TransformedPatchPath(path))
                # self._clippath = TransformedPatchPath(path)
                success = True
            elif isinstance(path, tuple):

                path, transform = path

        if path is None:
            # self._clippath = None
            success = True
        elif isinstance(path, Path):
            # self._clippath = TransformedPath(path, transform)
            success = True
        elif isinstance(path, TransformedPatchPath):
            # self._clippath = path
            success = True
        elif isinstance(path, TransformedPath):
            # self._clippath = path
            success = True

        if not success:
            print(type(path), type(transform))
            raise self.error("Invalid arguments to set_clip_path")


        # pass
        # self.__trait.set(obj, val)

    def get(self, obj, cls):
        pass
        # return self.__trait.get(obj, cls)

    def __getattr__(self, name):
        pass
        # return getattr(self.__trait, name)



#start of the install_traits() function
# def install_traits():
#     import matplotlib.lines
#     matplotlib.lines.Line2D  = Line2DWithTraits
