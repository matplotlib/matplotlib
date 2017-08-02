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
import matplotlib.artist as _artist #original artist base class
import matplotlib.cbook as cbook
from matplotlib.cbook import mplDeprecation
from matplotlib import docstring, rcParams
from matplotlib.transforms import (Bbox, IdentityTransform, TransformedBbox,
                         TransformedPatchPath, TransformedPath, Transform)
from matplotlib.path import Path
from functools import wraps
from contextlib import contextmanager

from traitlets import HasTraits, Unicode, Int, Dict, Bool, Instance, Tuple, List, default, validate, observe
from traitlets import Callable
from traits import TraitProxy, Perishable, ClipPathTrait

#this is for sticky_edges but im thinking we can just use a tuple trait...?
_XYPair = namedtuple("_XYPair", "x y")

def allow_rasterization(draw):
    """
    Decorator for Artist.draw method. Provides routines
    that run before and after the draw call. The before and after functions
    are useful for changing artist-dependent renderer attributes or making
    other setup function calls, such as starting and flushing a mixed-mode
    renderer.
    """
    @contextmanager
    def with_rasterized(artist, renderer):

        if artist.get_rasterized():
            renderer.start_rasterizing()

        if artist.get_agg_filter() is not None:
            renderer.start_filter()

        try:
            yield
        finally:
            if artist.get_agg_filter() is not None:
                renderer.stop_filter(artist.get_agg_filter())

            if artist.get_rasterized():
                renderer.stop_rasterizing()

    # the axes class has a second argument inframe for its draw method.
    @wraps(draw)
    def draw_wrapper(artist, renderer, *args, **kwargs):
        with with_rasterized(artist, renderer):
            return draw(artist, renderer, *args, **kwargs)

    draw_wrapper._supports_rasterization = True
    return draw_wrapper

#class Artist(_artist.Artist)
class Artist(HasTraits, _artist.Artist):

    aname=Unicode('Artist')
    zorder=Int(default_value=0)
    #_prop_order = dict(color=-1)
    prop_order=Dict() #asked thomas question about this asstribute
    # pchanged = Bool(default_value = False)
    stale=Bool(default_value=True)
    stale_callback=Callable(allow_none=True, default_value=True)
    axes=Instance('matplotlib.axes.Axes', allow_none=True, default_value=None)
    figure=Instance('matplotlib.figure.Figure', allow_none=True, default_value=None)
    #not sure if this would be the correct way to call TransformTrait
    transform = TransformTrait(allow_none=True, default_value=None)
    # transform=Instance('matplotlib.transform.Transform', allow_none=True, default_value=None)
    transformSet=Bool(default_value=False)
    visible=Bool(default_value=True)
    animated=Bool(default_value=False)
    alpha=Float(default_value=None, allow_none=True)
    clipbox=Instance('matplotlib.transforms.Bbox', allow_none=True, default_value=None)

    # clippath=ClipPathTrait((Tuple(Instance('matplotlib.path.Path'), TransformTrait(allow_none = True, default_value = None)), allow_none=True, default_value=None))
    clippath = Union([Instance(TransformedPath),Instance("matplotlib.patches.Patch")], allow_none=True, default_value=None)

    clipon=Boolean(default_value=True)
    label=Unicode(allow_none=True, default_value='')
    picker=Union(Float,Boolean,Callable, allow_none=True, default_value=None)
    contains=List(default_value=None)
    rasterized=Perishable(Boolean(allow_none=True, default_value=None))
    agg_filter=Unicode(allow_none=True, default_value=None) #set agg_filter function
    mouseover=Boolean(default_value=False)
    eventson=Boolean(default_value=False)
    oid=Int(allow_none=True, default_value=0)
    propobservers=Dict(default_value={}) #this may or may not work o/w leave alone and see what happens
    remove_method = Any(allow_none = True, default_value = None)
    url=Unicode(allow_none=True, default_value=None)
    gid=Unicode(allow_none=True, default_value=None)
    snap=Perishable(Boolean(allow_none=True, default_value=None))
    sketch=Tuple(Float(),Float(),Float(), default_value=rcParams['path.sketch'])
    path_effects=List(Instance('matplotlib.patheffect._Base'), default_value=rcParams['path.effects'])

    #_XYPair = namedtuple("_XYPair", "x y")
    #sticky_edges is a tuple with lists of floats
    #the first element of this tuple represents x
    #and the second element of sticky_edges represents y
    sticky_edges=Tuple(List(trait=Float()),List(trait=Float()))

    #stale default
    @default("stale")
    def _stale_default(self):
        print("generating default stale value")
        return True
    #stale validate: reference @stale.setter
    @validate("stale")
    def _stale_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        self.stale=proposal.value
        if self.animated is True:
            return proposal.value
        if proposal.value and self.stale_callback is not None:
            self.stale_callback(self, proposal.value)
            #not sure if I should return anything here
        # return proposal.value
    #stale observer
    @observe("stale", type=change)
    def _stale_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #stale_callback default
    @default("stale_callback")
    def _stale_callback_default(self):
        print("generating default stale_callback value")
        return None
    #stale_callback validate
    @validate("stale_callback")
    def _stale_callback_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #stale_callback observer
    @observe("stale_callback", type=change)
    def _stale_callback_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #axes default
    @default("axes")
    def _axes_default(self):
        print("generating default axes value")
        return None
    #axes validate: reference @axes.setter
    @validate("axes")
    def _axes_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        if (proposal.value is not None and
                (self.axes is not None and proposal.value != self.axes)):
            raise ValueError("Can not reset the axes.  You are "
                             "probably trying to re-use an artist "
                             "in more than one Axes which is not "
                             "supported")

        self.axes = proposal.value
        if proposal.value is not None and proposal.value is not self:
            self.stale_callback = _stale_axes_callback #this line needs testing
        return proposal.value
    #axes observer
    @observe("axes", type = change)
    def _axes_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #figure default
    @default("figure")
    def _figure_default(self):
        print("generating default figure value")
        return None
    #figure validate: reference set_figure
    @validate("figure")
    def _figure_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        # if this is a no-op just return
        if self.figure is proposal.value:
            return
        # if we currently have a figure (the case of both `self.figure`
        # and `fig` being none is taken care of above) we then user is
        # trying to change the figure an artist is associated with which
        # is not allowed for the same reason as adding the same instance
        # to more than one Axes
        if self.figure is not None:
            raise RuntimeError("Can not put single artist in "
                               "more than one figure")
        self.figure = proposal.value
        #what does this line even mean?
        if self.figure and self.figure is not self:
            self.pchanged()
        self.stale = True
        # return proposal.value
    #figure observer
    @observe("figure", type = change)
    def _figure_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))


    #TO DO: make a transform trait with the proper validation logic
    #transform default
    @default("transform")
    def _transform_default(self):
        print("generating default transform value")
        return None
    #transform validate
    @validate("transform")
    def _transform_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #transform observer: reference set_transform
    @observe("transform", type = change)
    def _transform_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.transformSet = True
        print("set _transformSet: %r" % self.transformSet)
        self.stale = True
        print("set stale: %r" % self.stale)



    #transformSet default
    @default("transformSet")
    def _transformSet_default(self):
        print("generating default transformSet value")
        return False
    #transformSet validate
    @validate("transformSet")
    def _transformSet_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #transformSet observer
    @observe("transformSet", type = change)
    def _transformSet_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #visible default
    @default("visible")
    def _visible_default(self):
        print("generating default visible value")
        return True
    #visible validate
    @validate("visible")
    def _visible_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #visible observer: reference set_visible
    @observe("visible", type = change)
    def _visible_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.pchanged()
        print("called self.pchanged()")
        self.stale = True
        print("set stale: %r" % self.stale)


    #animated default
    @default("animated")
    def _animated_default(self):
        print("generating default animated value")
        return False
    #animated validate: reference set_animated
    @validate("animated")
    def _animated_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        if self._animated is not proposal.value:
            self.pchanged()
            print("called self.pchanged()")
            return proposal.value
        return self._animated
    #animated observer
    @observe("animated", type = change)
    def _animated_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #alpha default
    @default("alpha")
    def _alpha_default(self):
        print("generating default alpha value")
        return None
    #alpha validate
    @validate("alpha")
    def _alpha_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #alpha observer: reference set_alpha
    @observe("alpha", type = change)
    def _alpha_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.pchanged()
        print("called self.pchanged()")
        self.stale = True
        print("set stale: %r" % self.stale)


    #clipbox default
    @default("clipbox")
    def _clipbox_default(self):
        print("generating default clipbox value")
        return None
    #clipbox validate
    @validate("clipbox")
    def _clipbox_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #clipbox observer: reference set_clip_box
    @observe("clipbox", type = change)
    def _clipbox_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.pchanged()
        print("called self.pchanged()")
        self.stale = True
        print("set stale: %r" % self.stale)


    # Note: this is for if Clippath is a Union Trait
    @validate("clippath")
    def _validate_clippath(self, proposal):
        value = proposal.value
        from matplotlib.patches import Patch
        if isinstance(value, Patch):
            value = TransformedPatchPath(value)
        return value

    def set_clip_path(self, path, transform):
        from matplotlib.patches import Rectangle, Patch

        success = False
        if transform is None:
            if isinstance(path, Rectangle):
                self.clipbox = TransformedBbox(Bbox.unit(), path.get_transform())
                self.clippath = None
            elif isinstance(path, Patch):
                self.clippath = path
            elif isinstance(path, tuple):
                path, transform = path

        if path is None:
            self.clippath = None
        elif isinstance(path, Path):
            self.clippath = TransformedPath(path, transform)
        elif isinstance(path, TransformedPath):
            # TransformedPatchPath is a subclass of TransformedPath
            self.clippath = path

        if not success:
            print(type(path), type(transform))
            raise TypeError("Invalid arguments to set_clip_path")


    #To do: create either a clippath trait or modify the get and set functions
    #for now i have comments down for default, validate and observer decortors
    #clippath default
    # @default("clippath")
    # def _clippath_default(self):
    #     print("generating default clippath value")
    #     return None
    # #clippath validate: reference set_clip_path
    # @validate("clippath")
    # def _clippath_validate(self, proposal):
    #     print("cross validating %r" % proposal.value)
    #     from matplotlib.patches import Patch, Rectangle
    #     success = False
    #     #note sure how to go about the validation yet but taking a shot
    #     #referring to the set_clip_path function
    #
    #     #validate the TransformTrait here: Needs to be determined if it is None here
    #     #if this statement is true:
    #     if proposal.TransformTrait()
    #     if isinstance(path, Rectangle):
    #         self.clipbox = TransformedBbox(Bbox.unit(), path.get_transform())
    #         self._clippath = None
    #         success = True
    #     elif isinstance(path, Patch):
    #         self._clippath = TransformedPatchPath(path)
    #         success = True
    #     elif isinstance(path, tuple):
    #         path, transform = path
    #
    #     if path is None:
    #         self._clippath = None
    #         success = True
    #     elif isinstance(path, Path):
    #         self._clippath = TransformedPath(path, transform)
    #         success = True
    #     elif isinstance(path, TransformedPatchPath):
    #         self._clippath = path
    #         success = True
    #     elif isinstance(path, TransformedPath):
    #         self._clippath = path
    #         success = True
    #
    # # if setting the clip path is not a success
    # if not success:
    #     print(type(path), type(transform))
    #     raise TypeError("Invalid arguments to set_clip_path")


    #clippath observer
    @observe("clippath", type = change)
    def _clippath_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.pchanged()
        print("called self.pchanged()")
        self.stale = True
        print("set stale: %r" % self.stale)



    #clipon default
    @default("clipon")
    def _clipon_default(self):
        print("generating default clipon value")
        return True
    #clipon validate
    @validate("clipon")
    def _clipon_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #clipon observer
    @observe("clipon", type = change)
    def _clipon_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.pchanged()
        print("called self.pchanged()")
        self.stale = True
        print("set stale: %r" % self.stale)


    #label default
    @default("label")
    def _label_default(self):
        print("generating default label value")
        return None
    #label validate
    @validate("label")
    def _label_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        if proposal.value is not None:
            return proposal.value
        return proposal.value
    #label observer
    @observe("label", type = change)
    def _label_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.pchanged()
        print("called self.pchanged()")
        self.stale = True
        print("set stale: %r" % self.stale)


    #picker default
    @default("picker")
    def _picker_default(self):
        print("generating default picker value")
        return None
    #picker validate
    @validate("picker")
    def _picker_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #picker observer
    @observe("picker", type = change)
    def _picker_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #contains default
    @default("contains")
    def _contains_default(self):
        print("generating default contains value")
        return None
    #contains validate
    @validate("contains")
    def _contains_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #contains observer
    @observe("contains", type = change)
    def _contains_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #rasterized default
    @default("rasterized")
    def _rasterized_default(self):
        print("generating default rasterized value")
        return None
    #rasterized validate
    @validate("rasterized")
    def _rasterized_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        if proposal.value and not hasattr(self.draw, "_supports_rasterization"):
            warnings.warn("Rasterization of '%s' will be ignored" % self)
        return proposal.value
    #rasterized observer
    @observe("rasterized", type = change)
    def _rasterized_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #agg_filter default
    @default("agg_filter")
    def _agg_filter_default(self):
        print("generating default agg_filter value")
        return None
    #agg_filter validate
    @validate("agg_filter")
    def _agg_filter_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #agg_filter observer
    @observe("agg_filter", type = change)
    def _agg_filter_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.stale = True
        print("set stale: %r" % self.stale)



    #mouseover default
    @default("mouseover")
    def _mouseover_default(self):
        print("generating default mouseover value")
        return False
    #mouseover validate: reference @mouseover.setter
    @validate("mouseover")
    def _mouseover_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        val = bool(proposal.value)
        #val is the returned value
        return val
    #mouseover observer
    @observe("mouseover", type = change)
    def _mouseover_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        print("adding or discarding from axes.mouseover_set")
        ax = self.axes
        if ax:
            if val:
                ax.mouseover_set.add(self)
            else:
                ax.mouseover_set.discard(self)



    #eventson default
    @default("eventson")
    def _eventson_default(self):
        print("generating default eventson value")
        return False
    #eventson validate
    @validate("eventson")
    def _eventson_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #eventson observer
    @observe("eventson", type = change)
    def _eventson_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #oid default
    @default("oid")
    def _oid_default(self):
        print("generating default oid (observer id) value")
        return 0
    #oid validate
    @validate("oid")
    def _oid_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #oid observer
    @observe("oid", type = change)
    def _oid_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #propobservers default
    @default("propobservers")
    def _propobservers_default(self):
        print("generating default propobservers value")
        return {}
    #propobservers validate
    @validate("propobservers")
    def _propobservers_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #propobservers observer
    @observe("propobservers", type = change)
    def _propobservers_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #url default
    @default("url")
    def _url_default(self):
        print("generating default url value")
        return None
    #url validate
    @validate("url")
    def _url_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #url observer
    @observe("url", type = change)
    def _url_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #gid default
    @default("gid")
    def _gid_default(self):
        print("generating default gid (group id) value")
        return None
    #gid validate
    @validate("gid")
    def _gid_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #gid observer
    @observe("gid", type = change)
    def _gid_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #snap default
    @default("snap")
    def _snap_default(self):
        print("generating default snap value")
        return None
    #snap validate
    @validate("snap")
    def _snap_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #snap observer
    @observe("snap", type = change)
    def _snap_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.stale = True
        print("set stale: %r" % self.stale)




    # This may not work, I may or may not have to work around rcParams['path.sketch']
    # but I am not sure yet
    # this may also have to be a trait?

    #sketch default
    @default("sketch")
    def _sketch_default(self):
        print("generating default sketch value")
        return rcParams['path.sketch']
    #sketch validate
    @validate("sketch")
    def _sketch_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        if proposal.scale is None:
            return None
        else:
            #(scale, length or 128.0, randomness or 16.0)
            #not sure if this is how to go about this?
            return (proposal.scale, proposal.length or 128.0, proposal.randomness or 16.0)
        # return proposal.value
    #sketch observer
    @observe("sketch", type = change)
    def _sketch_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.stale = True
        print("set stale: %r" % self.stale)


        """
        same note as sketch:
        This may not work, I may or may not have to work around rcParams['path.effects']
        but I am not sure yet
        """

    #path_effects default
    @default("path_effects")
    def _path_effects_default(self):
        print("generating default path_effects value")
        return rcParams['path.effects']
    #path_effects validate
    @validate("path_effects")
    def _path_effects_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #path_effects observer
    @observe("path_effects", type = change)
    def _path_effects_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.stale = True
        print("set stale: %r" % self.stale)



    """
    Note: This may or may not work, I have to work around the
    _XYPair([], [])
    """

    #sticky_edges default
    @default("sticky_edges")
    def _sticky_edges_default(self):
        print("generating default sticky_edges value")
        #(x,y) where x & yare both List(trait=Float())
        #Tuple(List(trait=Float()), List(trait=Float()))
        return ([], [])
    #sticky_edges validate
    @validate("sticky_edges")
    def _sticky_edges_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        return proposal.value
    #sticky_edges observer
    @observe("sticky_edges", type = change)
    def _sticky_edges_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))



    #ARTIST FUNCTIONS
    #
    # def __init__(self):
    #     pass
    #
    # def __getstate__(self):
        # d = self.__dict__.copy()
        # # remove the unpicklable remove method, this will get re-added on load
        # # (by the axes) if the artist lives on an axes.
        # d['_remove_method'] = None
        # d['stale_callback'] = None
        # return d
    #     pass

    def remove(self):
        """
        Remove the artist from the figure if possible.  The effect
        will not be visible until the figure is redrawn, e.g., with
        :meth:`matplotlib.axes.Axes.draw_idle`.  Call
        :meth:`matplotlib.axes.Axes.relim` to update the axes limits
        if desired.

        Note: :meth:`~matplotlib.axes.Axes.relim` will not see
        collections even if the collection was added to axes with
        *autolim* = True.

        Note: there is no support for removing the artist's legend entry.
        """

        # There is no method to set the callback.  Instead the parent should
        # set the _remove_method attribute directly.  This would be a
        # protected attribute if Python supported that sort of thing.  The
        # callback has one parameter, which is the child to be removed.
        if self.remove_method is not None:
            self.remove_method(self)
            # clear stale callback
            self.stale_callback = None
            #purpose of ax_flag?
            ax_flag = False
            if hasattr(self, 'axes') and self.axes:
                # remove from the mouse hit list
                self.axes.mouseover_set.discard(self)
                # mark the axes as stale
                self.axes.stale = True
                # decouple the artist from the axes
                self.axes = None
                ax_flag = True

            if self.figure:
                self.figure = None
                if not ax_flag:
                    self.figure = True

        else:
            raise NotImplementedError('cannot remove artist')
        # TODO: the fix for the collections relim problem is to move the
        # limits calculation into the artist itself, including the property of
        # whether or not the artist should affect the limits.  Then there will
        # be no distinction between axes.add_line, axes.add_patch, etc.
        # TODO: add legend support

        """
        These following functions are copied and pasted from the original Artist class.
        This is because I feel as if they can be altered to their respective traits.
        """

    def have_units(self):
        ax = self.axes
        if ax is None or ax.xaxis is None:
            return False
        return ax.xaxis.have_units() or ax.yaxis.have_units()

    def convert_xunits(self, x):
        """For artists in an axes, if the xaxis has units support,
        convert *x* using xaxis unit type
        """
        ax = getattr(self, 'axes', None)
        if ax is None or ax.xaxis is None:
            return x
        return ax.xaxis.convert_units(x)

    def convert_yunits(self, y):
        """For artists in an axes, if the yaxis has units support,
        convert *y* using yaxis unit type
        """
        ax = getattr(self, 'axes', None)
        if ax is None or ax.yaxis is None:
            return y
        return ax.yaxis.convert_units(y)

    def get_window_extent(self,renderer):
        """
        Get the axes bounding box in display space.
        Subclasses should override for inclusion in the bounding box
        "tight" calculation. Default is to return an empty bounding
        box at 0, 0.

        Be careful when using this function, the results will not update
        if the artist window extent of the artist changes.  The extent
        can change due to any changes in the transform stack, such as
        changing the axes limits, the figure size, or the canvas used
        (as is done when saving a figure).  This can lead to unexpected
        behavior where interactive figures will look fine on the screen,
        but will save incorrectly.
        """
        return Bbox([[0, 0], [0, 0]])

    def add_callback(self,func):
        """
        Adds a callback function that will be called whenever one of
        the :class:`Artist`'s properties changes.

        Returns an *id* that is useful for removing the callback with
        :meth:`remove_callback` later.
        """
        oid = self.oid
        self.propobservers[oid] = func
        self.oid += 1
        return oid

    def remove_callback(self,oid):
        """
        Remove a callback based on its *id*.

        .. seealso::

            :meth:`add_callback`
               For adding callbacks

        """
        try:
            del self.propobservers[oid]
        except KeyError:
            pass

    #this will stay for now to say the least bit because gives access to registered callbacks
    def pchanged(self):
        """
        Fire an event when property changed, calling all of the
        registered callbacks.
        """
        for oid, func in six.iteritems(self.propobservers):
            func(self)

    #leave for backwards compatability
    def is_transform_set(self):
        """
        Returns *True* if :class:`Artist` has a transform explicitly
        set.
        """
        return self.transformSet

    #leave for backwards compatability
    def is_figure_set(self):
        """
        Returns True if the artist is assigned to a
        :class:`~matplotlib.figure.Figure`.
        """
        return self.figure is not None

    def hitlist(self, event):
        """
        List the children of the artist which contain the mouse event *event*.
        """
        L = []
        try:
            hascursor, info = self.contains(event)
            if hascursor:
                L.append(self)
        except:
            import traceback
            traceback.print_exc()
            print("while checking", self.__class__)

        for a in self.get_children():
            L.extend(a.hitlist(event))
        return L

    def contains(self, mouseevent):
        """Test whether the artist contains the mouse event.

        Returns the truth value and a dictionary of artist specific details of
        selection, such as which points are contained in the pick radius.  See
        individual artists for details.
        """
        if callable(self.contains):
            return self.contains(self, mouseevent)
        warnings.warn("'%s' needs 'contains' method" % self.__class__.__name__)
        return False, {}

    def pickable(self):
        'Return *True* if :class:`Artist` is pickable.'
        return (self.figure is not None and
                self.figure.canvas is not None and
                self.picker is not None)

    def pick(self, mouseevent):
        """
        Process pick event

        each child artist will fire a pick event if *mouseevent* is over
        the artist and the artist has picker set
        """
        # Pick self
        if self.pickable():
            # picker = self.get_picker()
            picker = self.picker
            if callable(picker):
                inside, prop = picker(self, mouseevent)
            else:
                inside, prop = self.contains(mouseevent)
            if inside:
                self.figure.canvas.pick_event(mouseevent, self, **prop)

        # Pick children
        # for a in self.get_children():
        #if in the case this does not work, bring over the get_children function for contains
        for a in self.contains:
            # make sure the event happened in the same axes
            ax = getattr(a, 'axes', None)
            if (mouseevent.inaxes is None or ax is None
                    or mouseevent.inaxes == ax):
                # we need to check if mouseevent.inaxes is None
                # because some objects associated with an axes (e.g., a
                # tick label) can be outside the bounding box of the
                # axes and inaxes will be None
                # also check that ax is None so that it traverse objects
                # which do no have an axes property but children might
                a.pick(mouseevent)

    # need to look into this
    def get_transformed_clip_path_and_affine(self):
        '''
        Return the clip path with the non-affine part of its
        transformation applied, and the remaining affine part of its
        transformation.
        '''
        if self.clippath is not None:
            return self.clippath.get_transformed_path_and_affine()
        return None, None

    def _set_gc_clip(self, gc):
        'Set the clip properly for the gc'
        if self.clipon:
            if self.clipbox is not None:
                gc.set_clip_rectangle(self.clipbox)
            gc.set_clip_path(self.clippath)
        else:
            gc.set_clip_rectangle(None)
            gc.set_clip_path(None)

    def draw(self, renderer, *args, **kwargs):
        'Derived classes drawing method'
        # if not self.get_visible():
        #if this does not work, we will create a get function for visible trait
        if not self.visible:
            return
        self.stale = False

    #note there is a def update function in artist and I do not think I need
    #to implement it due to the migration to traitlets

    #note there is a def update_from function and I do think I need to implement
    #and observer for Artist; The update_from function is frivolous
    #I need to look into observe Artist, but I am not sure


    # In order for this to work I need to implement ArtistInspector with respect
    #to traitlets
    # def properties(self):
    #     """
    #     return a dictionary mapping property name -> value for all Artist props
    #     """
    #     return ArtistInspector(self).properties()

    #there is a set function but I do not think I need it

    #for now just bringing this over from Artist, but need to inspect it
    def findobj(self, match=None, include_self=True):
        """
        Find artist objects.

        Recursively find all :class:`~matplotlib.artist.Artist` instances
        contained in self.

        *match* can be

          - None: return all objects contained in artist.

          - function with signature ``boolean = match(artist)``
            used to filter matches

          - class instance: e.g., Line2D.  Only return artists of class type.

        If *include_self* is True (default), include self in the list to be
        checked for a match.

        """
        if match is None:  # always return True
            def matchfunc(x):
                return True
        elif isinstance(match, type) and issubclass(match, Artist):
            def matchfunc(x):
                return isinstance(x, match)
        elif callable(match):
            matchfunc = match
        else:
            raise ValueError('match must be None, a matplotlib.artist.Artist '
                             'subclass, or a callable')

        # artists = sum([c.findobj(matchfunc) for c in self.get_children()], [])
        #in theory, self.contains should get the list of children in contains list
        artists = sum([c.findobj(matchfunc) for c in self.contains], [])
        if include_self and matchfunc(self):
            artists.append(self)
        return artists

    #what is the purpose of this function if it returns None?
    def get_cursor_data(self, event):
        """
        Get the cursor data for a given event.
        """
        return None

    #why return None in get_cursor_data if we are setting cursor data here?
    def format_cursor_data(self, data):
        """
        Return *cursor data* string formatted.
        """
        try:
            data[0]
        except (TypeError, IndexError):
            data = [data]
        return ', '.join('{:0.3g}'.format(item) for item in data if
                isinstance(item, (np.floating, np.integer, int, float)))


# outside of the artist/artist inspector
#ArtistInspector needed for the following functions below
# def getp(obj, property=None):
#     """
#     Return the value of object's property.  *property* is an optional string
#     for the property you want to return
#
#     Example usage::
#
#         getp(obj)  # get all the object properties
#         getp(obj, 'linestyle')  # get the linestyle property
#
#     *obj* is a :class:`Artist` instance, e.g.,
#     :class:`~matplotllib.lines.Line2D` or an instance of a
#     :class:`~matplotlib.axes.Axes` or :class:`matplotlib.text.Text`.
#     If the *property* is 'somename', this function returns
#
#       obj.get_somename()
#
#     :func:`getp` can be used to query all the gettable properties with
#     ``getp(obj)``. Many properties have aliases for shorter typing, e.g.
#     'lw' is an alias for 'linewidth'.  In the output, aliases and full
#     property names will be listed as:
#
#       property or alias = value
#
#     e.g.:
#
#       linewidth or lw = 2
#     """
#     if property is None:
#         insp = ArtistInspector(obj)
#         ret = insp.pprint_getters()
#         print('\n'.join(ret))
#         return
#
#     func = getattr(obj, 'get_' + property)
#     return func()

# alias
# get = getp


# def setp(obj, *args, **kwargs):
#     """
#     Set a property on an artist object.
#
#     matplotlib supports the use of :func:`setp` ("set property") and
#     :func:`getp` to set and get object properties, as well as to do
#     introspection on the object.  For example, to set the linestyle of a
#     line to be dashed, you can do::
#
#       >>> line, = plot([1,2,3])
#       >>> setp(line, linestyle='--')
#
#     If you want to know the valid types of arguments, you can provide
#     the name of the property you want to set without a value::
#
#       >>> setp(line, 'linestyle')
#           linestyle: [ '-' | '--' | '-.' | ':' | 'steps' | 'None' ]
#
#     If you want to see all the properties that can be set, and their
#     possible values, you can do::
#
#       >>> setp(line)
#           ... long output listing omitted
#
#     You may specify another output file to `setp` if `sys.stdout` is not
#     acceptable for some reason using the `file` keyword-only argument::
#
#       >>> with fopen('output.log') as f:
#       >>>     setp(line, file=f)
#
#     :func:`setp` operates on a single instance or a iterable of
#     instances. If you are in query mode introspecting the possible
#     values, only the first instance in the sequence is used. When
#     actually setting values, all the instances will be set.  e.g.,
#     suppose you have a list of two lines, the following will make both
#     lines thicker and red::
#
#       >>> x = arange(0,1.0,0.01)
#       >>> y1 = sin(2*pi*x)
#       >>> y2 = sin(4*pi*x)
#       >>> lines = plot(x, y1, x, y2)
#       >>> setp(lines, linewidth=2, color='r')
#
#     :func:`setp` works with the MATLAB style string/value pairs or
#     with python kwargs.  For example, the following are equivalent::
#
#       >>> setp(lines, 'linewidth', 2, 'color', 'r')  # MATLAB style
#       >>> setp(lines, linewidth=2, color='r')        # python style
#     """
#
#     if not cbook.iterable(obj):
#         objs = [obj]
#     else:
#         objs = list(cbook.flatten(obj))
#
#     if not objs:
#         return
#
#     insp = ArtistInspector(objs[0])
#
#     # file has to be popped before checking if kwargs is empty
#     printArgs = {}
#     if 'file' in kwargs:
#         printArgs['file'] = kwargs.pop('file')
#
#     if not kwargs and len(args) < 2:
#         if args:
#             print(insp.pprint_setters(prop=args[0]), **printArgs)
#         else:
#             print('\n'.join(insp.pprint_setters()), **printArgs)
#         return
#
#     if len(args) % 2:
#         raise ValueError('The set args must be string, value pairs')
#
#     # put args into ordereddict to maintain order
#     funcvals = OrderedDict()
#     for i in range(0, len(args) - 1, 2):
#         funcvals[args[i]] = args[i + 1]
#
#     ret = [o.update(funcvals) for o in objs]
#     ret.extend([o.set(**kwargs) for o in objs])
#     return [x for x in cbook.flatten(ret)]
#
#
# def kwdoc(a):
#     hardcopy = matplotlib.rcParams['docstring.hardcopy']
#     if hardcopy:
#         return '\n'.join(ArtistInspector(a).pprint_setters_rest(
#                          leadingspace=2))
#     else:
#         return '\n'.join(ArtistInspector(a).pprint_setters(leadingspace=2))
#
# docstring.interpd.update(Artist=kwdoc(Artist))

# _get_axes_msg = """{0} has been deprecated in mpl 1.5, please use the
# axes property.  A removal date has not been set."""

#monkey patching
_artist.Artist = Artist
