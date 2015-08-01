from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import types

import re
import warnings
import inspect
import matplotlib
import matplotlib.cbook as cbook
from matplotlib.cbook import mplDeprecation
from matplotlib import docstring, rcParams
from .transforms import (Bbox, IdentityTransform, TransformedBbox,
                         TransformedPath, Transform)
from .path import Path

from .traitlets import (Configurable, Unicode, Bool, Int, Float, Bool, Tuple,
                        Dict, List, Instance, Union, Callable, oInstance, Undefined)

# Note, matplotlib artists use the doc strings for set and get
# methods to enable the introspection methods of setp and getp.  Every
# set_* method should have a docstring containing the line
#
# ACCEPTS: [ legal | values ]s
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


def allow_rasterization(draw):
    """
    Decorator for Artist.draw method. Provides routines
    that run before and after the draw call. The before and after functions
    are useful for changing artist-dependant renderer attributes or making
    other setup function calls, such as starting and flushing a mixed-mode
    renderer.
    """
    def before(artist, renderer):
        if artist.get_rasterized():
            renderer.start_rasterizing()

        if artist.get_agg_filter() is not None:
            renderer.start_filter()

    def after(artist, renderer):

        if artist.get_agg_filter() is not None:
            renderer.stop_filter(artist.get_agg_filter())

        if artist.get_rasterized():
            renderer.stop_rasterizing()

    # the axes class has a second argument inframe for its draw method.
    def draw_wrapper(artist, renderer, *args, **kwargs):
        before(artist, renderer)
        draw(artist, renderer, *args, **kwargs)
        after(artist, renderer)

    # "safe wrapping" to exactly replicate anything we haven't overridden above
    draw_wrapper.__name__ = draw.__name__
    draw_wrapper.__dict__ = draw.__dict__
    draw_wrapper.__doc__ = draw.__doc__
    draw_wrapper._supports_rasterization = True
    return draw_wrapper


def _stale_figure_callback(self):
    self.figure.stale = True

def _stale_axes_callback(self):
    self.axes.stale = True

class Artist(Configurable):
    """
    Abstract base class for someone who renders into a
    :class:`FigureCanvas`.
    """

    aname = 'Artist'
    zorder = 0

    # warn on all : check whether serialize is/isn't required.

    # perishable=True ==> set stale = True
    _transformSet = Bool(False, serialize=True, config=True)
    # warn : oInstance used, new TraitType?
    transform = oInstance('matplotlib.transforms.Transform',
                        serialize=True, perishable=True, config=True)
    axes = Instance('matplotlib.axes._axes.Axes',allow_none=True,
                        serialize=True, config=True)
    contains = Callable(allow_none=True, config=True)
    figure = Instance('matplotlib.figure.Figure', allow_none=True,
                    serialize=True, perishable=True, config=True)
    visible = Bool(True, perishable=True, serialize=True, config=True)
    animated = Bool(False, perishable=True, serialize=True, config=True)
    alpha = Float(None, allow_none=True, perishable=True, serialize=True, config=True)
    url = Unicode(allow_none=True, serialize=True, config=True)
    gid = Unicode(allow_none=True, serialize=True, config=True)
    clipbox = Instance('matplotlib.transforms.BboxBase', allow_none=True,
                        perishable=True, serialize=True, config=True)
    snap = Bool(allow_none=True, perishable=True, config=True)
    clipon = Bool(True, perishable=True, config=True)
    # * setter and getter methods for `self._clippath` could be refactored
    # using TraitTypes potentially ==> clippath = ?
    label = Union([Unicode(''),Instance('matplotlib.text.Text'),Int()],
                allow_none=True, perishable=True, config=True)
    rasterized = Bool(allow_none=True, config=True)
    _agg_filter = Callable(None,allow_none=True, perishable=True, config=True)
    eventson = Bool(True, config=True)
    _sketch = Tuple(rcParams['path.sketch'], allow_none=True,
                    perishable=True,serialize=True, config=True)
    _path_effects = List(trait=Instance('matplotlib.patheffects.AbstractPathEffect'),
                        allow_none=True, perishable=True, serialize=True, config=True)
    _propobservers = Dict({}, config=True) # a dict from oids to funcs
    _oid = Int(0, config=True) # an observer id

    # sketch = mpltr.Tuple(allow_none=True)
    # path_effects = mpltr.

    def __init__(self, config=None, parent=None):

        super(Artist, self).__init__(config=config, parent=parent)

        pnames = self.trait_names(perishable=True)
        self.on_trait_change(self._fire_callbacks, pnames)

        self.stale = True
        self._pickable = False
        self._clippath = None
        self._picker = None
        self._remove_method = None

        self._sketch = rcParams['path.sketch']
        self._path_effects = rcParams['path.effects']

    def __getstate__(self):
        d = self.__dict__.copy()
        # remove the unpicklable remove method, this will get re-added on load
        # (by the axes) if the artist lives on an axes.
        d['_remove_method'] = None
        return d

    # handled by _fire_callbacks
    def pchanged(self):
        # add warn
        self._fire_callbacks()

    # can be superseded by on_trait_change or _%_changed methods
    def _fire_callbacks(self):
        """Set as stale and fire the registered callbacks."""
        self.stale = True
        for oid, func in six.iteritems(self._propobservers):
            func(self)

    # can be superseded by on_trait_change or _%_changed methods
    def add_callback(self, func):
        """
        Adds a callback function that will be called whenever one of
        the :class:`Artist`'s "perishable" properties changes.

        Returns an *id* that is useful for removing the callback with
        :meth:`remove_callback` later.
        """
        oid = self._oid
        self._propobservers[oid] = func
        self._oid += 1
        return self._oid

    # can be superseded by on_trait_change or _%_changed methods
    def remove_callback(self, oid):
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

    # - - - - - - - - - - - - -
    # traitlet change handlers
    # - - - - - - - - - - - - -

    def _transform_changed(self, name, new):
        self._transformSet = True

    def _transform_overload(self, trait, value):
        if value is None:
            return IdentityTransform()
        elif (not isinstance(value, Transform)
            and hasattr(value, '_as_mpl_transform')):
            return value._as_mpl_transform(self.axes)
        trait.error(self, value)

    def _axes_changed(self, name, old, new):
        if old not in [Undefined, None]:
            # old != true already checked in `TraitType._validate`
            raise ValueError("Can not reset the axes.  You are "
                             "probably trying to re-use an artist "
                             "in more than one Axes which is not "
                             "supported")
        self.axes = new
        if new is not None and new is not self:
            self.add_callback(_stale_axes_callback)

    def _contains_changed(self, name, new):
        self._trait_values[name] = types.MethodType(new,self)

    def _contains_default(self):
        def contains_defualt(*args, **kwargs):
            warnings.warn("'%s' obj needs 'contains' method" % self.__class__.__name__)
            return False, {}
        return contains_default

    def _figure_changed(self, name, new):
        self.add_callback(_stale_figure_callback)

    def _snap_changed(self, name, new):
        if not rcParams['path.snap']:
            self._trait_values[name] = False

    def _rasterized_changed(self, name, new):
        if new and not hasattr(self.draw, "_supports_rasterization"):
            warnings.warn("Rasterization of '%s' will be ignored" % self)

    def _eventson_changed(self):
        # add warn
        # this feature will be removed
        # it's handled by configurables
        pass

    def _picker_changed(self, name, new):
        if new is None:
            self._pickable = False
        self._pickable = True

    # - - - - - - - - - - - - - - -
    # warned setters and getters
    # - - - - - - - - - - - - - - -

    @property
    def _contains(self):
        return self.contains
    @_contains.setter
    def _contains(self, value):
        self.contains = value

    @property
    def _transform(self):
        #add warn
        return self.transform
    @_transform.setter
    def _transform(self, value):
        # add warn
        self.transform = value

    def get_transform(self):
        # add warn
        return self.transform

    def set_transform(self, t):
        # add warn
        self.transform = t

    def get_figure(self):
        """
        Return the :class:`~matplotlib.figure.Figure` instance the
        artist belongs to.
        """
        # add warn
        return self.figure

    def set_figure(self, fig):
        """
        Set the :class:`~matplotlib.figure.Figure` instance the artist
        belongs to.

        ACCEPTS: a :class:`matplotlib.figure.Figure` instance
        """
        # add warn
        self.figure = fig


    @property
    def _url(self):
        #add warn
        return self.url
    @_url.setter
    def _url(self, value):
        # add warn
        self.url = value

    def get_url(self):
        """
        Returns the url
        """
        # add warn
        return self.url

    def set_url(self, url):
        """
        Sets the url for the artist

        ACCEPTS: a url string
        """
        # add warn
        self.url = url

    @property
    def _alpha(self):
        #add warn
        return self.alpha
    @_alpha.setter
    def _alpha(self, value):
        # add warn
        self.alpha = value

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on
        all backends.

        ACCEPTS: float (0.0 transparent through 1.0 opaque)
        """
        # add warn
        self.alpha = alpha

    def get_alpha(self):
        """
        Return the alpha value used for blending - not supported on all
        backends
        """
        # add warn
        return self.alpha

    @property
    def _gid(self):
        #add warn
        return self.gid
    @_gid.setter
    def _gid(self, value):
        # add warn
        self.gid = value

    def get_gid(self):
        """
        Returns the group id
        """
        # add warn
        return self.gid

    def set_gid(self, gid):
        """
        Sets the (group) id for the artist

        ACCEPTS: an id string
        """
        # add warn
        self.gid = gid

    @property
    def _clipbox(self):
        #add warn
        return self.clipbox
    @_clipbox.setter
    def _clipbox(self, value):
        # add warn
        self.clipbox = value

    def set_clip_box(self, clipbox):
        """
        Set the artist's clip :class:`~matplotlib.transforms.Bbox`.

        ACCEPTS: a :class:`matplotlib.transforms.Bbox` instance
        """
        # add warn
        self.clipbox = clipbox

    def get_clip_box(self):
        'Return artist clipbox'
        # add warn
        return self.clipbox

    @property
    def _snap(self):
        #add warn
        return self.snap
    @_snap.setter
    def _snap(self, value):
        # add warn
        self.snap = value

    def get_snap(self):
        """
        Returns the snap setting which may be:

          * True: snap vertices to the nearest pixel center

          * False: leave vertices as-is

          * None: (auto) If the path contains only rectilinear line
            segments, round to the nearest pixel center

        Only supported by the Agg and MacOSX backends.
        """
        # add warn
        return self.snap

    def set_snap(self, snap):
        """
        Sets the snap setting which may be:

          * True: snap vertices to the nearest pixel center

          * False: leave vertices as-is

          * None: (auto) If the path contains only rectilinear line
            segments, round to the nearest pixel center

        Only supported by the Agg and MacOSX backends.
        """
        # add warn
        self.snap = snap

    # temp properties
    @property
    def _clipon(self):
        # add warn
        return self.clipon
    @_clipon.setter
    def _clipon(self, value):
        # add warn
        self.clipon = value

    def set_clip_on(self, b):
        """
        Set whether artist uses clipping.

        When False artists will be visible out side of the axes which
        can lead to unexpected results.

        ACCEPTS: [True | False]
        """
        # add warn

        # This may result in the callbacks being hit twice, but ensures they
        # are hit at least once
        self.clipon = b

    def get_clip_on(self):
        'Return whether artist uses clipping'
        # add warn
        return self.clipon

    @property
    def _label(self):
        # add warn
        return self.label
    @_label.setter
    def _label(self, value):
        # add warn
        self.label = value

    def set_label(self, s):
        """
        Set the label to *s* for auto legend.

        ACCEPTS: string or anything printable with '%s' conversion.
        """
        # add warn
        self.label = s

    def get_label(self):
        """
        Get the label used for this artist in the legend.
        """
        # add warn
        return self.label

    def set_rasterized(self, rasterized):
        """
        Force rasterized (bitmap) drawing in vector backend output.

        Defaults to None, which implies the backend's default behavior

        ACCEPTS: [True | False | None]
        """
        # add warn
        self.rasterized = rasterized

    def get_rasterized(self):
        "return True if the artist is to be rasterized"
        # add warn
        return self.rasterized

    # temp properties
    @property
    def _axes(self):
        # add warn
        return self.axes
    @_axes.setter
    def _axes(self, value):
        # add warn
        self.axes = value
    @_axes.deleter
    def _axes(self):
        # add warn
        self._trait_values.pop('axes',None)

    def set_animated(self, b):
        """
        Set the artist's animation state.

        ACCEPTS: [True | False]
        """
        # add warn
        self.animated = b

    def get_animated(self):
        "Return the artist's animated state"
        # add warn
        return self.animated

    @property
    def _visible(self):
        # add warn
        return self.visible
    @_visible.setter
    def _visible(self, value):
        # add warn
        self.visible = value

    def set_visible(self, b):
        """
        Set the artist's visiblity.

        ACCEPTS: [True | False]
        """
        # add warn
        self.visible = b

    def get_visible(self):
        "Return the artist's visiblity"
        # add warn
        return self.visible

    def set_axes(self, axes):
        """
        Set the :class:`~matplotlib.axes.Axes` instance in which the
        artist resides, if any.

        This has been deprecated in mpl 1.5, please use the
        axes property.  Will be removed in 1.7 or 2.0.

        ACCEPTS: an :class:`~matplotlib.axes.Axes` instance
        """
        warnings.warn(_get_axes_msg, mplDeprecation, stacklevel=1)
        self.axes = axes

    def get_axes(self):
        """
        Return the :class:`~matplotlib.axes.Axes` instance the artist
        resides in, or *None*.

        This has been deprecated in mpl 1.5, please use the
        axes property.  Will be removed in 1.7 or 2.0.
        """
        warnings.warn(_get_axes_msg, mplDeprecation, stacklevel=1)
        return self.axes

    # - - - - - - - - - - - - - -
    # generic getters and setters
    # - - - - - - - - - - - - - -

    def set_agg_filter(self, filter_func):
        """
        set agg_filter fuction.
        """
        self._agg_filter = filter_func

    def set_picker(self, picker):
        """
        Set the epsilon for picking used by this artist

        *picker* can be one of the following:

          * *None*: picking is disabled for this artist (default)

          * A boolean: if *True* then picking will be enabled and the
            artist will fire a pick event if the mouse event is over
            the artist

          * A float: if picker is a number it is interpreted as an
            epsilon tolerance in points and the artist will fire
            off an event if it's data is within epsilon of the mouse
            event.  For some artists like lines and patch collections,
            the artist may provide additional data to the pick event
            that is generated, e.g., the indices of the data within
            epsilon of the pick event

          * A function: if picker is callable, it is a user supplied
            function which determines whether the artist is hit by the
            mouse event::

              hit, props = picker(artist, mouseevent)

            to determine the hit test.  if the mouse event is over the
            artist, return *hit=True* and props is a dictionary of
            properties you want added to the PickEvent attributes.

        ACCEPTS: [None|float|boolean|callable]
        """
        self._picker = picker

    def get_picker(self):
        'Return the picker object used by this artist'
        return self._picker

    def pickable(self):
        """Return *True* if :class:`Artist` is pickable.

        Truth value is updated by traitlets change handlers"""
        return self._pickable

    def set_clip_path(self, path, transform=None):
        """
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
        """
        from matplotlib.patches import Patch, Rectangle

        success = False
        if transform is None:
            if isinstance(path, Rectangle):
                self.clipbox = TransformedBbox(Bbox.unit(), path.get_transform())
                self._clippath = None
                success = True
            elif isinstance(path, Patch):
                self._clippath = TransformedPath(
                    path.get_path(),
                    path.get_transform())
                success = True
            elif isinstance(path, tuple):
                path, transform = path

        if path is None:
            self._clippath = None
            success = True
        elif isinstance(path, Path):
            self._clippath = TransformedPath(path, transform)
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

    def get_clip_path(self):
        'Return artist clip path'
        return self._clippath

    def get_sketch_params(self):
        """
        Returns the sketch parameters for the artist.

        Returns
        -------
        sketch_params : tuple or `None`

        A 3-tuple with the following elements:

          * `scale`: The amplitude of the wiggle perpendicular to the
            source line.

          * `length`: The length of the wiggle along the line.

          * `randomness`: The scale factor by which the length is
            shrunken or expanded.

        May return `None` if no sketch parameters were set.
        """
        return self._sketch

    def set_sketch_params(self, scale=None, length=None, randomness=None):
        """
        Sets the sketch parameters.

        Parameters
        ----------

        scale : float, optional
            The amplitude of the wiggle perpendicular to the source
            line, in pixels.  If scale is `None`, or not provided, no
            sketch filter will be provided.

        length : float, optional
             The length of the wiggle along the line, in pixels
             (default 128.0)

        randomness : float, optional
            The scale factor by which the length is shrunken or
            expanded (default 16.0)
        """
        if scale is None:
            self._sketch = None
        else:
            self._sketch = (scale, length or 128.0, randomness or 16.0)

    def set_path_effects(self, path_effects):
        """
        set path_effects, which should be a list of instances of
        matplotlib.patheffect._Base class or its derivatives.
        """
        self._path_effects = path_effects

    def get_path_effects(self):
        return self._path_effects

    # - - - - - - - - - - - - -
    # general member functions
    # - - - - - - - - - - - - -

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

        # There is no method to set the callback. Instead the parent should
        # set the _remove_method attribute directly.  This would be a
        # protected attribute if Python supported that sort of thing.  The
        # callback has one parameter, which is the child to be removed.
        if self._remove_method is not None:
            self._remove_method(self)
        else:
            raise NotImplementedError('cannot remove artist')
        # TODO: the fix for the collections relim problem is to move the
        # limits calculation into the artist itself, including the property of
        # whether or not the artist should affect the limits.  Then there will
        # be no distinction between axes.add_line, axes.add_patch, etc.
        # TODO: add legend support

    def pick(self, mouseevent):
        """
        call signature::

          pick(mouseevent)

        each child artist will fire a pick event if *mouseevent* is over
        the artist and the artist has picker set
        """
        # Pick self
        if self.pickable():
            picker = self.get_picker()
            if six.callable(picker):
                inside, prop = picker(self, mouseevent)
            else:
                inside, prop = self.contains(mouseevent)
            if inside:
                self.figure.canvas.pick_event(mouseevent, self, **prop)

        # Pick children
        for a in self.get_children():
            # make sure the event happened in the same axes
            ax = getattr(a, 'axes', None)
            if mouseevent.inaxes is None or ax is None or \
                    mouseevent.inaxes == ax:
                # we need to check if mouseevent.inaxes is None
                # because some objects associated with an axes (e.g., a
                # tick label) can be outside the bounding box of the
                # axes and inaxes will be None
                # also check that ax is None so that it traverse objects
                # which do no have an axes property but children might
                a.pick(mouseevent)

    def have_units(self):
        'Return *True* if units are set on the *x* or *y* axes'
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

    def is_transform_set(self):
        """
        Returns *True* if :class:`Artist` has a transform explicitly
        set.
        """
        return self._transformSet

    def get_window_extent(self, renderer):
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

    # should be superseded by `on_trait_change` methods in
    # `__init__` constructor of inherited classes
    def get_children(self):
        """
        Return a list of the child :class:`Artist`s this
        :class:`Artist` contains.
        """
        return []

    def is_figure_set(self):
        """
        Returns True if the artist is assigned to a
        :class:`~matplotlib.figure.Figure`.
        """
        return self.figure is not None

    def set_clip_path(self, path, transform=None):
        """
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
        """
        from matplotlib.patches import Patch, Rectangle

        success = False
        if transform is None:
            if isinstance(path, Rectangle):
                self.clipbox = TransformedBbox(Bbox.unit(), path.get_transform())
                self._clippath = None
                success = True
            elif isinstance(path, Patch):
                self._clippath = TransformedPath(
                    path.get_path(),
                    path.get_transform())
                success = True
            elif isinstance(path, tuple):
                path, transform = path

        if path is None:
            self._clippath = None
            success = True
        elif isinstance(path, Path):
            self._clippath = TransformedPath(path, transform)
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

    def get_agg_filter(self):
        "return filter function to be used for agg filter"
        return self._agg_filter

    def draw(self, renderer, *args, **kwargs):
        'Derived classes drawing method'
        if not self.get_visible():
            return
        self.stale = False

    def update(self, props):
        """
        Update the properties of this :class:`Artist` from the
        dictionary *prop*.
        """
        # all can be handleded by configurable
        # self.update_config(config)

        store = self.eventson
        self.eventson = False
        changed = False

        for k, v in six.iteritems(props):
            if k in ['axes']:
                setattr(self, k, v)
            else:
                func = getattr(self, 'set_' + k, None)
                if func is None or not six.callable(func):
                    raise AttributeError('Unknown property %s' % k)
                func(v)
            changed = True
        self.eventson = store

    def get_zorder(self):
        """
        Return the :class:`Artist`'s zorder.
        """
        return self.zorder

    def set_zorder(self, level):
        """
        Set the zorder for the artist.  Artists with lower zorder
        values are drawn first.

        ACCEPTS: any number
        """
        self.zorder = level
        self.pchanged()
        self.stale = True

    def update_from(self, other):
        'Copy properties from *other* to *self*.'
        self.transform = other.transform
        self._transformSet = other._transformSet
        self._visible = other._visible
        self._alpha = other._alpha
        self.clipbox = other.clipbox
        self._clipon = other._clipon
        self._clippath = other._clippath
        self._label = other._label
        self._sketch = other._sketch
        self._path_effects = other._path_effects
        self.pchanged()
        self.stale = True

    def properties(self):
        """
        return a dictionary mapping property name -> value for all Artist props
        """
        return ArtistInspector(self).properties()

    def set(self, **kwargs):
        """
        A tkstyle set command, pass *kwargs* to set properties
        """
        ret = []
        for k, v in six.iteritems(kwargs):
            k = k.lower()
            funcName = "set_%s" % k
            func = getattr(self, funcName)
            ret.extend([func(v)])
        return ret

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
        elif cbook.issubclass_safe(match, Artist):
            def matchfunc(x):
                return isinstance(x, match)
        elif six.callable(match):
            matchfunc = match
        else:
            raise ValueError('match must be None, a matplotlib.artist.Artist '
                             'subclass, or a callable')

        artists = []

        for c in self.get_children():
            if matchfunc(c):
                artists.append(c)
            artists.extend([thisc for thisc in
                            c.findobj(matchfunc, include_self=False)
                            if matchfunc(thisc)])

        if include_self and matchfunc(self):
            artists.append(self)
        return artists

    def get_cursor_data(self, event):
        """
        Get the cursor data for a given event.
        """
        return None

    def format_cursor_data(self, data):
        """
        Return *cursor data* string formatted.
        """
        try:
            data[0]
        except (TypeError, IndexError):
            data = [data]
        return ', '.join('{:0.3g}'.format(item) for item in data)


class ArtistInspector(object):
    """
    A helper class to inspect an :class:`~matplotlib.artist.Artist`
    and return information about it's settable properties and their
    current values.
    """
    def __init__(self, o):
        """
        Initialize the artist inspector with an
        :class:`~matplotlib.artist.Artist` or sequence of :class:`Artists`.
        If a sequence is used, we assume it is a homogeneous sequence (all
        :class:`Artists` are of the same type) and it is your responsibility
        to make sure this is so.
        """
        if cbook.iterable(o) and len(o):
            o = o[0]

        self.oorig = o
        if not isinstance(o, type):
            o = type(o)
        self.o = o

        self.aliasd = self.get_aliases()

    def get_aliases(self):
        """
        Get a dict mapping *fullname* -> *alias* for each *alias* in
        the :class:`~matplotlib.artist.ArtistInspector`.

        e.g., for lines::

          {'markerfacecolor': 'mfc',
           'linewidth'      : 'lw',
          }

        """
        names = [name for name in dir(self.o) if
                 (name.startswith('set_') or name.startswith('get_'))
                 and six.callable(getattr(self.o, name))]
        aliases = {}
        for name in names:
            func = getattr(self.o, name)
            if not self.is_alias(func):
                continue
            docstring = func.__doc__
            fullname = docstring[10:]
            aliases.setdefault(fullname[4:], {})[name[4:]] = None
        return aliases

    _get_valid_values_regex = re.compile(
        r"\n\s*ACCEPTS:\s*((?:.|\n)*?)(?:$|(?:\n\n))"
    )

    def get_valid_values(self, attr):
        """
        Get the legal arguments for the setter associated with *attr*.

        This is done by querying the docstring of the function *set_attr*
        for a line that begins with ACCEPTS:

        e.g., for a line linestyle, return
        "[ ``'-'`` | ``'--'`` | ``'-.'`` | ``':'`` | ``'steps'`` | ``'None'``
        ]"
        """

        name = 'set_%s' % attr
        if not hasattr(self.o, name):
            raise AttributeError('%s has no function %s' % (self.o, name))
        func = getattr(self.o, name)

        docstring = func.__doc__
        if docstring is None:
            return 'unknown'

        if docstring.startswith('alias for '):
            return None

        match = self._get_valid_values_regex.search(docstring)
        if match is not None:
            return match.group(1).replace('\n', ' ')
        return 'unknown'

    def _get_setters_and_targets(self):
        """
        Get the attribute strings and a full path to where the setter
        is defined for all setters in an object.
        """

        setters = []
        for name in dir(self.o):
            if not name.startswith('set_'):
                continue
            o = getattr(self.o, name)
            if not six.callable(o):
                continue
            if len(inspect.getargspec(o)[0]) < 2:
                continue
            func = o
            if self.is_alias(func):
                continue
            source_class = self.o.__module__ + "." + self.o.__name__
            for cls in self.o.mro():
                if name in cls.__dict__:
                    source_class = cls.__module__ + "." + cls.__name__
                    break
            setters.append((name[4:], source_class + "." + name))
        return setters

    def get_setters(self):
        """
        Get the attribute strings with setters for object.  e.g., for a line,
        return ``['markerfacecolor', 'linewidth', ....]``.
        """

        return [prop for prop, target in self._get_setters_and_targets()]

    def is_alias(self, o):
        """
        Return *True* if method object *o* is an alias for another
        function.
        """
        ds = o.__doc__
        if ds is None:
            return False
        return ds.startswith('alias for ')

    def aliased_name(self, s):
        """
        return 'PROPNAME or alias' if *s* has an alias, else return
        PROPNAME.

        e.g., for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'
        """

        if s in self.aliasd:
            return s + ''.join([' or %s' % x
                                for x
                                in six.iterkeys(self.aliasd[s])])
        else:
            return s

    def aliased_name_rest(self, s, target):
        """
        return 'PROPNAME or alias' if *s* has an alias, else return
        PROPNAME formatted for ReST

        e.g., for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'
        """

        if s in self.aliasd:
            aliases = ''.join([' or %s' % x
                               for x
                               in six.iterkeys(self.aliasd[s])])
        else:
            aliases = ''
        return ':meth:`%s <%s>`%s' % (s, target, aliases)

    def pprint_setters(self, prop=None, leadingspace=2):
        """
        If *prop* is *None*, return a list of strings of all settable properies
        and their valid values.

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of property : valid
        values.
        """
        if leadingspace:
            pad = ' ' * leadingspace
        else:
            pad = ''
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return '%s%s: %s' % (pad, prop, accepts)

        attrs = self._get_setters_and_targets()
        attrs.sort()
        lines = []

        for prop, path in attrs:
            accepts = self.get_valid_values(prop)
            name = self.aliased_name(prop)

            lines.append('%s%s: %s' % (pad, name, accepts))
        return lines

    def pprint_setters_rest(self, prop=None, leadingspace=2):
        """
        If *prop* is *None*, return a list of strings of all settable properies
        and their valid values.  Format the output for ReST

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of property : valid
        values.
        """
        if leadingspace:
            pad = ' ' * leadingspace
        else:
            pad = ''
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return '%s%s: %s' % (pad, prop, accepts)

        attrs = self._get_setters_and_targets()
        attrs.sort()
        lines = []

        ########
        names = [self.aliased_name_rest(prop, target)
                 for prop, target
                 in attrs]
        accepts = [self.get_valid_values(prop) for prop, target in attrs]

        col0_len = max([len(n) for n in names])
        col1_len = max([len(a) for a in accepts])
        table_formatstr = pad + '=' * col0_len + '   ' + '=' * col1_len

        lines.append('')
        lines.append(table_formatstr)
        lines.append(pad + 'Property'.ljust(col0_len + 3) +
                     'Description'.ljust(col1_len))
        lines.append(table_formatstr)

        lines.extend([pad + n.ljust(col0_len + 3) + a.ljust(col1_len)
                      for n, a in zip(names, accepts)])

        lines.append(table_formatstr)
        lines.append('')
        return lines
        ########

        for prop, path in attrs:
            accepts = self.get_valid_values(prop)
            name = self.aliased_name_rest(prop, path)

            lines.append('%s%s: %s' % (pad, name, accepts))
        return lines

    def properties(self):
        """
        return a dictionary mapping property name -> value
        """
        o = self.oorig
        getters = [name for name in dir(o)
                   if name.startswith('get_')
                   and six.callable(getattr(o, name))]
        getters.sort()
        d = dict()
        for name in getters:
            func = getattr(o, name)
            if self.is_alias(func):
                continue

            try:
                val = func()
            except:
                continue
            else:
                d[name[4:]] = val

        return d

    def pprint_getters(self):
        """
        Return the getters and actual values as list of strings.
        """

        d = self.properties()
        names = list(six.iterkeys(d))
        names.sort()
        lines = []
        for name in names:
            val = d[name]
            if getattr(val, 'shape', ()) != () and len(val) > 6:
                s = str(val[:6]) + '...'
            else:
                s = str(val)
            s = s.replace('\n', ' ')
            if len(s) > 50:
                s = s[:50] + '...'
            name = self.aliased_name(name)
            lines.append('    %s = %s' % (name, s))
        return lines

    def findobj(self, match=None):
        """
        Recursively find all :class:`matplotlib.artist.Artist`
        instances contained in *self*.

        If *match* is not None, it can be

          - function with signature ``boolean = match(artist)``

          - class instance: e.g., :class:`~matplotlib.lines.Line2D`

        used to filter matches.
        """
        if match is None:  # always return True
            def matchfunc(x):
                return True
        elif issubclass(match, Artist):
            def matchfunc(x):
                return isinstance(x, match)
        elif six.callable(match):
            matchfunc = func
        else:
            raise ValueError('match must be None, an '
                             'matplotlib.artist.Artist '
                             'subclass, or a callable')

        artists = []

        for c in self.get_children():
            if matchfunc(c):
                artists.append(c)
            artists.extend([thisc
                            for thisc
                            in c.findobj(matchfunc)
                            if matchfunc(thisc)])

        if matchfunc(self):
            artists.append(self)
        return artists


def getp(obj, property=None):
    """
    Return the value of object's property.  *property* is an optional string
    for the property you want to return

    Example usage::

        getp(obj)  # get all the object properties
        getp(obj, 'linestyle')  # get the linestyle property

    *obj* is a :class:`Artist` instance, e.g.,
    :class:`~matplotllib.lines.Line2D` or an instance of a
    :class:`~matplotlib.axes.Axes` or :class:`matplotlib.text.Text`.
    If the *property* is 'somename', this function returns

      obj.get_somename()

    :func:`getp` can be used to query all the gettable properties with
    ``getp(obj)``. Many properties have aliases for shorter typing, e.g.
    'lw' is an alias for 'linewidth'.  In the output, aliases and full
    property names will be listed as:

      property or alias = value

    e.g.:

      linewidth or lw = 2
    """
    if property is None:
        insp = ArtistInspector(obj)
        ret = insp.pprint_getters()
        print('\n'.join(ret))
        return

    func = getattr(obj, 'get_' + property)
    return func()

# alias
get = getp


def setp(obj, *args, **kwargs):
    """
    Set a property on an artist object.

    matplotlib supports the use of :func:`setp` ("set property") and
    :func:`getp` to set and get object properties, as well as to do
    introspection on the object.  For example, to set the linestyle of a
    line to be dashed, you can do::

      >>> line, = plot([1,2,3])
      >>> setp(line, linestyle='--')

    If you want to know the valid types of arguments, you can provide the
    name of the property you want to set without a value::

      >>> setp(line, 'linestyle')
          linestyle: [ '-' | '--' | '-.' | ':' | 'steps' | 'None' ]

    If you want to see all the properties that can be set, and their
    possible values, you can do::

      >>> setp(line)
          ... long output listing omitted

    :func:`setp` operates on a single instance or a list of instances.
    If you are in query mode introspecting the possible values, only
    the first instance in the sequence is used.  When actually setting
    values, all the instances will be set.  e.g., suppose you have a
    list of two lines, the following will make both lines thicker and
    red::

      >>> x = arange(0,1.0,0.01)
      >>> y1 = sin(2*pi*x)
      >>> y2 = sin(4*pi*x)
      >>> lines = plot(x, y1, x, y2)
      >>> setp(lines, linewidth=2, color='r')

    :func:`setp` works with the MATLAB style string/value pairs or
    with python kwargs.  For example, the following are equivalent::

      >>> setp(lines, 'linewidth', 2, 'color', 'r')  # MATLAB style
      >>> setp(lines, linewidth=2, color='r')        # python style
    """

    insp = ArtistInspector(obj)

    if len(kwargs) == 0 and len(args) == 0:
        print('\n'.join(insp.pprint_setters()))
        return

    if len(kwargs) == 0 and len(args) == 1:
        print(insp.pprint_setters(prop=args[0]))
        return

    if not cbook.iterable(obj):
        objs = [obj]
    else:
        objs = cbook.flatten(obj)

    if len(args) % 2:
        raise ValueError('The set args must be string, value pairs')

    funcvals = []
    for i in range(0, len(args) - 1, 2):
        funcvals.append((args[i], args[i + 1]))
    funcvals.extend(kwargs.items())

    ret = []
    for o in objs:
        for s, val in funcvals:
            s = s.lower()
            funcName = "set_%s" % s
            func = getattr(o, funcName)
            ret.extend([func(val)])
    return [x for x in cbook.flatten(ret)]

Artist.ps = []
Artist.s = []

def kwdoc(a):
    hardcopy = matplotlib.rcParams['docstring.hardcopy']
    if hardcopy:
        return '\n'.join(ArtistInspector(a).pprint_setters_rest(
                         leadingspace=2))
    else:
        return '\n'.join(ArtistInspector(a).pprint_setters(leadingspace=2))

docstring.interpd.update(Artist=kwdoc(Artist))

_get_axes_msg = """This has been deprecated in mpl 1.5, please use the
axes property.  A removal date has not been set."""
