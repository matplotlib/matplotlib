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

from traitlets import HasTraits, Unicode, Int, Dict, Callable, Bool, Instance, Tuple, List, default, validate, observe
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


class Artist(HasTraits):

    aname = Unicode('Artist')
    zorder = Int(default_value = 0)
    #_prop_order = dict(color=-1)
    _prop_order = Dict() #asked thomas question about this asstribute
    # pchanged = Bool(default_value = False)
    _stale = Bool(default_value = True)
    stale_callback = Callable(allow_none = True, default_value = True)
    _axes = Instance('matplotlib.axes.Axes', allow_none = True, default_value = None)
    figure = Instance('matplotlib.figure.Figure', allow_none = True, default_value = None)
    _transform = Instance('matplotlib.transform.Transform', allow_none = True, default_value = None)
    _transformSet = Bool(default_value = False )
    _visible = Bool(default_value = True)
    _animated = Bool(default_value = False)
    _alpha = Float(default_value = None ,allow_none = True)
    clipbox = Instance('matplotlib.transforms.Bbox', allow_none = True, default_value = None)
    """
    Notes from Documentation:
    Union([Float(), Bool(), Int()]) attempts to
    validate the provided values with the validation function of Float, then Bool, and finally Int.
    """
    _clippath = Perishable(Union([Instance('matplotlib.path.Path'), Instance('matplotlib.transforms.Transform'), Instance('matplotlib.patches.Patch')], allow_none = True, default_value = None))
    _clipon = Boolean(default_value = True)
    _label = Unicode(allow_none = True, default_value = '')
    _picker = Union(Float, Boolean, Callable, allow_none = True, default_value = None)
    _contains = List(default_value=None)
    _rasterized = Perishable(Boolean(allow_none = True, default_value = None))
    _agg_filter = Unicode(allow_none = True, default_value = None) #set agg_filter function
    _mouseover = Boolean(default_value = False)
    eventson = Boolean(default_value = False)
    _oid = Int(allow_none = True, default_value = 0)
    _propobservers = Dict(default_value = {}) #this may or may not work o/w leave alone and see what happens
    # _remove_method =  #have to look into this
    _url = Unicode(allow_none = True,default_value = None)
    _gid = Unicode(allow_none = True, default_value = None)
    _snap = Perishable(Boolean(allow_none = True, default_value = None))
    _sketch = Tuple(Float(), Float(), Float(), default_value = rcParams['path.sketch'])
    _path_effects = List(Instance('matplotlib.patheffect._Base'), default_value = rcParams['path.effects'])
    #_XYPair = namedtuple("_XYPair", "x y")
    #sticky_edges is a tuple with lists of floats
    #the first element of this tuple represents x
    #and the second element of sticky_edges represents y
    _sticky_edges = Tuple(List(trait=Float()), List(trait=Float()))

    #the axes bounding box in display space
    #TO DO: window_extent -> Bbox([[0, 0], [0, 0]])

"""
_______________________________________________________________________________
"""
    #
    # #pchanged default
    # @default("pchanged")
    # def pchanged_default(self):
    #     print("generating default stale value")
    #     return True
    # #pchanged validate
    # @validate("pchanged")
    # def pchanged_validate(self, proposal):
    #     print("cross validating %r" % proposal.value")
    #     return proposal.value
    # #pchanged observer
    # @observe("pchanged", type = change)
    # def pchanged_observe(self, change):
    #     print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #stale default
    @default("stale")
    def _stale_default(self):
        print("generating default stale value")
        return True
    #stale validate: reference @stale.setter
    @validate("stale")
    def _stale_validate(self, proposal):
        print("cross validating %r" % proposal.value)
        self._stale = proposal.value
        if self._animated is True:
            return proposal.value
        if proposal.value and self.stale_callback is not None:
            self.stale_callback(self, proposal.value)
            #not sure if I should return anything here
        # return proposal.value
    #stale observer
    @observe("stale", type = change)
    def _stale_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #stale_callback default
    @default("stale_callback")
    def _stale_callback_default(self):
        print("generating default stale_callback value")
        return None
    #stale_callback validate
    @validate("stale_callback")
    def _stale_callback_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #stale_callback observer
    @observe("stale_callback", type = change)
    def _stale_callback_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #axes default
    @default("axes")
    def _axes_default(self):
        print("generating default axes value")
        return None
    #axes validate: reference @axes.setter
    @validate("axes")
    def _axes_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        if (proposal.value is not None and
                (self._axes is not None and proposal.value != self._axes)):
            raise ValueError("Can not reset the axes.  You are "
                             "probably trying to re-use an artist "
                             "in more than one Axes which is not "
                             "supported")

        self._axes = proposal.value
        if proposal.value is not None and proposal.value is not self:
            self.stale_callback = _stale_axes_callback #this line needs testing
        return proposal.value
    #axes observer
    @observe("axes", type = change)
    def _axes_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #figure default
    @default("figure")
    def _figure_default(self):
        print("generating default figure value")
        return None
    #figure validate: reference set_figure
    @validate("figure")
    def _figure_validate(self, proposal):
        print("cross validating %r" % proposal.value")
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

"""
_______________________________________________________________________________
"""

    #transform default
    @default("transform")
    def _transform_default(self):
        print("generating default transform value")
        return None
    #transform validate
    @validate("transform")
    def _transform_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #transform observer: reference set_transform
    @observe("transform", type = change)
    def _transform_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self._transformSet = True
        print("set _transformSet: %r" self._transformSet)
        self.stale = True
        print("set stale: %r" self.stale)

"""
_______________________________________________________________________________
"""

    #transformSet default
    @default("transformSet")
    def _transformSet_default(self):
        print("generating default transformSet value")
        return False
    #transformSet validate
    @validate("transformSet")
    def _transformSet_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #transformSet observer
    @observe("transformSet", type = change)
    def _transformSet_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #visible default
    @default("visible")
    def _visible_default(self):
        print("generating default visible value")
        return True
    #visible validate
    @validate("visible")
    def _visible_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #visible observer: reference set_visible
    @observe("visible", type = change)
    def _visible_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.stale = True
        print("set stale: %r" self.stale)
"""
_______________________________________________________________________________
"""

    #animated default
    @default("animated")
    def _animated_default(self):
        print("generating default animated value")
        return False
    #animated validate: reference set_animated
    @validate("animated")
    def _animated_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        if self._animated is not proposal.value:
            return proposal.value
        return self._animated
    #animated observer
    @observe("animated", type = change)
    def _animated_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #alpha default
    @default("alpha")
    def _alpha_default(self):
        print("generating default alpha value")
        return None
    #alpha validate
    @validate("alpha")
    def _alpha_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #alpha observer: reference set_alpha
    @observe("alpha", type = change)
    def _alpha_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.stale = True
        print("set stale: %r" self.stale)
"""
_______________________________________________________________________________
"""

    #clipbox default
    @default("clipbox")
    def _clipbox_default(self):
        print("generating default clipbox value")
        return None
    #clipbox validate
    @validate("clipbox")
    def _clipbox_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #clipbox observer: reference set_clip_box
    @observe("clipbox", type = change)
    def _clipbox_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.stale = True
        print("set stale: %r" self.stale)


"""
_______________________________________________________________________________
"""

    #To do: create either a clippath trait or modify the get and set functions
    #for now i have comments down for default, validate and observer decortors
    #clippath default
    @default("clippath")
    def _clippath_default(self):
        print("generating default clippath value")
        return None
    #clippath validate
    @validate("clippath")
    def _clippath_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #clippath observer
    @observe("clippath", type = change)
    def _clippath_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #clipon default
    @default("clipon")
    def _clipon_default(self):
        print("generating default clipon value")
        return True
    #clipon validate
    @validate("clipon")
    def _clipon_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #clipon observer
    @observe("clipon", type = change)
    def _clipon_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #label default
    @default("label")
    def _label_default(self):
        print("generating default label value")
        return None
    #label validate
    @validate("label")
    def _label_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #label observer
    @observe("label", type = change)
    def _label_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #picker default
    @default("picker")
    def _picker_default(self):
        print("generating default picker value")
        return None
    #picker validate
    @validate("picker")
    def _picker_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #picker observer
    @observe("picker", type = change)
    def _picker_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #contains default
    @default("contains")
    def _contains_default(self):
        print("generating default contains value")
        return None
    #contains validate
    @validate("contains")
    def _contains_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #contains observer
    @observe("contains", type = change)
    def _contains_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #rasterized default
    @default("rasterized")
    def _rasterized_default(self):
        print("generating default rasterized value")
        return None
    #rasterized validate
    @validate("rasterized")
    def _rasterized_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #rasterized observer
    @observe("rasterized", type = change)
    def _rasterized_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #agg_filter default
    @default("agg_filter")
    def _agg_filter_default(self):
        print("generating default agg_filter value")
        return None
    #agg_filter validate
    @validate("agg_filter")
    def _agg_filter_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #agg_filter observer
    @observe("agg_filter", type = change)
    def _agg_filter_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #mouseover default
    @default("mouseover")
    def _mouseover_default(self):
        print("generating default mouseover value")
        return False
    #mouseover validate
    @validate("mouseover")
    def _mouseover_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #mouseover observer
    @observe("mouseover", type = change)
    def _mouseover_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #eventson default
    @default("eventson")
    def _eventson_default(self):
        print("generating default eventson value")
        return False
    #eventson validate
    @validate("eventson")
    def _eventson_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #eventson observer
    @observe("eventson", type = change)
    def _eventson_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #oid default
    @default("oid")
    def _oid_default(self):
        print("generating default oid (observer id) value")
        return 0
    #oid validate
    @validate("oid")
    def _oid_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #oid observer
    @observe("oid", type = change)
    def _oid_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #propobservers default
    @default("propobservers")
    def _propobservers_default(self):
        print("generating default propobservers value")
        return {}
    #propobservers validate
    @validate("propobservers")
    def _propobservers_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #propobservers observer
    @observe("propobservers", type = change)
    def _propobservers_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #url default
    @default("url")
    def _url_default(self):
        print("generating default url value")
        return None
    #url validate
    @validate("url")
    def _url_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #url observer
    @observe("url", type = change)
    def _url_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #gid default
    @default("gid")
    def _gid_default(self):
        print("generating default gid (group id) value")
        return None
    #gid validate
    @validate("gid")
    def _gid_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #gid observer
    @observe("gid", type = change)
    def _gid_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #snap default
    @default("snap")
    def _snap_default(self):
        print("generating default snap value")
        return None
    #snap validate
    @validate("snap")
    def _snap_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #snap observer
    @observe("snap", type = change)
    def _snap_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))
        self.stale = True
        print("set stale: %r" self.stale)
"""
_______________________________________________________________________________
"""

    """
    This may not work, I may or may not have to work around rcParams['path.sketch']
    but I am not sure yet
    """

    #sketch default
    @default("sketch")
    def _sketch_default(self):
        print("generating default sketch value")
        return rcParams['path.sketch']
    #sketch validate
    @validate("sketch")
    def _sketch_validate(self, proposal):
        print("cross validating %r" % proposal.value")
        return proposal.value
    #sketch observer
    @observe("sketch", type = change)
    def _sketch_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

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
        print("cross validating %r" % proposal.value")
        return proposal.value
    #path_effects observer
    @observe("path_effects", type = change)
    def _path_effects_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

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
        print("cross validating %r" % proposal.value")
        return proposal.value
    #sticky_edges observer
    @observe("sticky_edges", type = change)
    def _sticky_edges_observe(self, change):
        print("observed a change from %r to %r" % (change.old, change.new))

"""
_______________________________________________________________________________
"""

    #ARTIST FUNCTIONS

    def __init__(self):
        pass

    def __getstate__(self):
        pass

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
        oid = self._oid
        self._propobservers[oid] = func
        self._oid += 1
        return oid

    def remove_callback(self,oid):
        """
        Remove a callback based on its *id*.

        .. seealso::

            :meth:`add_callback`
               For adding callbacks

        """
        try:
            del self._propobservers[oid]
        except KeyError:
            pass

    #this will stay for now to say the least bit because gives access to registered callbacks
    def pchanged(self):
        """
        Fire an event when property changed, calling all of the
        registered callbacks.
        """
        for oid, func in six.iteritems(self._propobservers):
            func(self)

    def is_transform_set(self):
        """
        Returns *True* if :class:`Artist` has a transform explicitly
        set.
        """
        return self._transformSet

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

    # Note sure if this is how to go about a contains mouseevent function
    def contains(self, mouseevent):
        """Test whether the artist contains the mouse event.

        Returns the truth value and a dictionary of artist specific details of
        selection, such as which points are contained in the pick radius.  See
        individual artists for details.
        """
        if callable(self._contains):
            return self._contains(self, mouseevent)
        warnings.warn("'%s' needs 'contains' method" % self.__class__.__name__)
        return False, {}

    def pickable(self):
        'Return *True* if :class:`Artist` is pickable.'
        return (self.figure is not None and
                self.figure.canvas is not None and
                self._picker is not None)

    def pick(self, mouseevent):
        """
        Process pick event

        each child artist will fire a pick event if *mouseevent* is over
        the artist and the artist has picker set
        """
        # Pick self
        if self.pickable():
            picker = self.get_picker()
            if callable(picker):
                inside, prop = picker(self, mouseevent)
            else:
                inside, prop = self.contains(mouseevent)
            if inside:
                self.figure.canvas.pick_event(mouseevent, self, **prop)

        # Pick children
        for a in self.get_children():
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

    #leave for backwards compatability
    """
    use
    """
    def is_figure_set(self):
        """
        Returns True if the artist is assigned to a
        :class:`~matplotlib.figure.Figure`.
        """
        return self.figure is not None

    # need to look into this
    def get_transformed_clip_path_and_affine(self):
        '''
        Return the clip path with the non-affine part of its
        transformation applied, and the remaining affine part of its
        transformation.
        '''
        if self._clippath is not None:
            return self._clippath.get_transformed_path_and_affine()
        return None, None

    def _set_gc_clip(self, gc):
        'Set the clip properly for the gc'
        if self._clipon:
            if self.clipbox is not None:
                gc.set_clip_rectangle(self.clipbox)
            gc.set_clip_path(self._clippath)
        else:
            gc.set_clip_rectangle(None)
            gc.set_clip_path(None)

    def draw(self, renderer, *args, **kwargs):
        'Derived classes drawing method'
        if not self.get_visible():
            return
        self.stale = False

    #note there is a def update function in artist and I do not think I need
    #to implement it due to the migration to traitlets

    #note there is a def update_from function and I do think I need to implement
    #and observer for Artist; The update_from function is frivolous
    #I need to look into observe Artist, but I am not sure

    # In order for this to work I need to implement ArtistInspector with respect
    #to traitlets
    def properties(self):
        """
        return a dictionary mapping property name -> value for all Artist props
        """
        return ArtistInspector(self).properties()

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

        artists = sum([c.findobj(matchfunc) for c in self.get_children()], [])
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
