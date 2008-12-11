from __future__ import division
import re, warnings
import matplotlib
import matplotlib.cbook as cbook
from transforms import Bbox, IdentityTransform, TransformedBbox, TransformedPath
from path import Path

## Note, matplotlib artists use the doc strings for set and get
# methods to enable the introspection methods of setp and getp.  Every
# set_* method should have a docstring containing the line
#
# ACCEPTS: [ legal | values ]
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


class Artist(object):
    """
    Abstract base class for someone who renders into a
    :class:`FigureCanvas`.
    """

    aname = 'Artist'
    zorder = 0
    def __init__(self):
        self.figure = None

        self._transform = None
        self._transformSet = False
        self._visible = True
        self._animated = False
        self._alpha = 1.0
        self.clipbox = None
        self._clippath = None
        self._clipon = True
        self._lod = False
        self._label = ''
        self._picker = None
        self._contains = None

        self.eventson = False  # fire events only if eventson
        self._oid = 0  # an observer id
        self._propobservers = {} # a dict from oids to funcs
        self.axes = None
        self._remove_method = None
        self._url = None
        self.x_isdata = True  # False to avoid updating Axes.dataLim with x
        self.y_isdata = True  #                                      with y
        self._snap = None

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

        # There is no method to set the callback.  Instead the parent should set
        # the _remove_method attribute directly.  This would be a protected
        # attribute if Python supported that sort of thing.  The callback
        # has one parameter, which is the child to be removed.
        if self._remove_method != None:
            self._remove_method(self)
        else:
            raise NotImplementedError('cannot remove artist')
        # TODO: the fix for the collections relim problem is to move the
        # limits calculation into the artist itself, including the property
        # of whether or not the artist should affect the limits.  Then there
        # will be no distinction between axes.add_line, axes.add_patch, etc.
        # TODO: add legend support

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
            #print 'artist.convert_xunits no conversion: ax=%s'%ax
            return x
        return ax.xaxis.convert_units(x)

    def convert_yunits(self, y):
        """For artists in an axes, if the yaxis has units support,
        convert *y* using yaxis unit type
        """
        ax = getattr(self, 'axes', None)
        if ax is None or ax.yaxis is None: return y
        return ax.yaxis.convert_units(y)

    def set_axes(self, axes):
        """
        Set the :class:`~matplotlib.axes.Axes` instance in which the
        artist resides, if any.

        ACCEPTS: an :class:`~matplotlib.axes.Axes` instance
        """
        self.axes = axes

    def get_axes(self):
        """
        Return the :class:`~matplotlib.axes.Axes` instance the artist
        resides in, or *None*
        """
        return self.axes

    def add_callback(self, func):
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

    def remove_callback(self, oid):
        """
        Remove a callback based on its *id*.

        .. seealso::
            :meth:`add_callback`
        """
        try: del self._propobservers[oid]
        except KeyError: pass

    def pchanged(self):
        """
        Fire an event when property changed, calling all of the
        registered callbacks.
        """
        for oid, func in self._propobservers.items():
            func(self)

    def is_transform_set(self):
        """
        Returns *True* if :class:`Artist` has a transform explicitly
        set.
        """
        return self._transformSet

    def set_transform(self, t):
        """
        Set the :class:`~matplotlib.transforms.Transform` instance
        used by this artist.

        ACCEPTS: :class:`~matplotlib.transforms.Transform` instance
        """
        self._transform = t
        self._transformSet = True
        self.pchanged()

    def get_transform(self):
        """
        Return the :class:`~matplotlib.transforms.Transform`
        instance used by this artist.
        """
        if self._transform is None:
            self._transform = IdentityTransform()
        return self._transform

    def hitlist(self, event):
        """
        List the children of the artist which contain the mouse event *event*.
        """
        import traceback
        L = []
        try:
            hascursor,info = self.contains(event)
            if hascursor:
                L.append(self)
        except:
            traceback.print_exc()
            print "while checking",self.__class__


        for a in self.get_children():
            L.extend(a.hitlist(event))
        return L

    def get_children(self):
        """
        Return a list of the child :class:`Artist`s this
        :class:`Artist` contains.
        """
        return []

    def contains(self, mouseevent):
        """Test whether the artist contains the mouse event.

        Returns the truth value and a dictionary of artist specific details of
        selection, such as which points are contained in the pick radius.  See
        individual artists for details.
        """
        if callable(self._contains): return self._contains(self,mouseevent)
        #raise NotImplementedError,str(self.__class__)+" needs 'contains' method"
        warnings.warn("'%s' needs 'contains' method" % self.__class__.__name__)
        return False,{}

    def set_contains(self,picker):
        """
        Replace the contains test used by this artist. The new picker
        should be a callable function which determines whether the
        artist is hit by the mouse event::

            hit, props = picker(artist, mouseevent)

        If the mouse event is over the artist, return *hit* = *True*
        and *props* is a dictionary of properties you want returned
        with the contains test.

        ACCEPTS: a callable function
        """
        self._contains = picker

    def get_contains(self):
        """
        Return the _contains test used by the artist, or *None* for default.
        """
        return self._contains

    def pickable(self):
        'Return *True* if :class:`Artist` is pickable.'
        return (self.figure is not None and
                self.figure.canvas is not None and
                self._picker is not None)

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
            if callable(picker):
                inside,prop = picker(self,mouseevent)
            else:
                inside,prop = self.contains(mouseevent)
            if inside:
                self.figure.canvas.pick_event(mouseevent, self, **prop)

        # Pick children
        for a in self.get_children():
            a.pick(mouseevent)

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
            that is generated, e.g. the indices of the data within
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

    def is_figure_set(self):
        """
        Returns True if the artist is assigned to a
        :class:`~matplotlib.figure.Figure`.
        """
        return self.figure is not None

    def get_url(self):
        """
        Returns the url
        """
        return self._url

    def set_url(self, url):
        """
        Sets the url for the artist
        """
        self._url = url

    def get_snap(self):
        """
        Returns the snap setting which may be:

          * True: snap vertices to the nearest pixel center

          * False: leave vertices as-is

          * None: (auto) If the path contains only rectilinear line
            segments, round to the nearest pixel center

        Only supported by the Agg backends.
        """
        return self._snap

    def set_snap(self, snap):
        """
        Sets the snap setting which may be:

          * True: snap vertices to the nearest pixel center

          * False: leave vertices as-is

          * None: (auto) If the path contains only rectilinear line
            segments, round to the nearest pixel center

        Only supported by the Agg backends.
        """
        self._snap = snap

    def get_figure(self):
        """
        Return the :class:`~matplotlib.figure.Figure` instance the
        artist belongs to.
        """
        return self.figure

    def set_figure(self, fig):
        """
        Set the :class:`~matplotlib.figure.Figure` instance the artist
        belongs to.

        ACCEPTS: a :class:`matplotlib.figure.Figure` instance
        """
        self.figure = fig
        self.pchanged()

    def set_clip_box(self, clipbox):
        """
        Set the artist's clip :class:`~matplotlib.transforms.Bbox`.

        ACCEPTS: a :class:`matplotlib.transforms.Bbox` instance
        """
        self.clipbox = clipbox
        self.pchanged()

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
        from patches import Patch, Rectangle

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

        if path is None:
            self._clippath = None
            success = True
        elif isinstance(path, Path):
            self._clippath = TransformedPath(path, transform)
            success = True

        if not success:
            print type(path), type(transform)
            raise TypeError("Invalid arguments to set_clip_path")

        self.pchanged()

    def get_alpha(self):
        """
        Return the alpha value used for blending - not supported on all
        backends
        """
        return self._alpha

    def get_visible(self):
        "Return the artist's visiblity"
        return self._visible

    def get_animated(self):
        "Return the artist's animated state"
        return self._animated

    def get_clip_on(self):
        'Return whether artist uses clipping'
        return self._clipon

    def get_clip_box(self):
        'Return artist clipbox'
        return self.clipbox

    def get_clip_path(self):
        'Return artist clip path'
        return self._clippath

    def get_transformed_clip_path_and_affine(self):
        '''
        Return the clip path with the non-affine part of its
        transformation applied, and the remaining affine part of its
        transformation.
        '''
        if self._clippath is not None:
            return self._clippath.get_transformed_path_and_affine()
        return None, None

    def set_clip_on(self, b):
        """
        Set whether artist uses clipping.

        ACCEPTS: [True | False]
        """
        self._clipon = b
        self.pchanged()

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
        if not self.get_visible(): return

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on
        all backends

        ACCEPTS: float (0.0 transparent through 1.0 opaque)
        """
        self._alpha = alpha
        self.pchanged()

    def set_lod(self, on):
        """
        Set Level of Detail on or off.  If on, the artists may examine
        things like the pixel width of the axes and draw a subset of
        their contents accordingly

        ACCEPTS: [True | False]
        """
        self._lod = on
        self.pchanged()

    def set_visible(self, b):
        """
        Set the artist's visiblity.

        ACCEPTS: [True | False]
        """
        self._visible = b
        self.pchanged()


    def set_animated(self, b):
        """
        Set the artist's animation state.

        ACCEPTS: [True | False]
        """
        self._animated = b
        self.pchanged()

    def update(self, props):
        """
        Update the properties of this :class:`Artist` from the
        dictionary *prop*.
        """
        store = self.eventson
        self.eventson = False
        changed = False
        for k,v in props.items():
            func = getattr(self, 'set_'+k, None)
            if func is None or not callable(func):
                raise AttributeError('Unknown property %s'%k)
            func(v)
            changed = True
        self.eventson = store
        if changed: self.pchanged()


    def get_label(self):
        """
        Get the label used for this artist in the legend.
        """
        return self._label

    def set_label(self, s):
        """
        Set the label to *s* for auto legend.

        ACCEPTS: any string
        """
        self._label = s
        self.pchanged()



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

    def update_from(self, other):
        'Copy properties from *other* to *self*.'
        self._transform = other._transform
        self._transformSet = other._transformSet
        self._visible = other._visible
        self._alpha = other._alpha
        self.clipbox = other.clipbox
        self._clipon = other._clipon
        self._clippath = other._clippath
        self._lod = other._lod
        self._label = other._label
        self.pchanged()


    def set(self, **kwargs):
        """
        A tkstyle set command, pass *kwargs* to set properties
        """
        ret = []
        for k,v in kwargs.items():
            k = k.lower()
            funcName = "set_%s"%k
            func = getattr(self,funcName)
            ret.extend( [func(v)] )
        return ret

    def findobj(self, match=None):
        """
        pyplot signature:
          findobj(o=gcf(), match=None)

        Recursively find all :class:matplotlib.artist.Artist instances
        contained in self.

        *match* can be

          - None: return all objects contained in artist (including artist)

          - function with signature ``boolean = match(artist)`` used to filter matches

          - class instance: eg Line2D.  Only return artists of class type

        .. plot:: mpl_examples/pylab_examples/findobj_demo.py
        """

        if match is None: # always return True
            def matchfunc(x): return True
        elif cbook.issubclass_safe(match, Artist):
            def matchfunc(x):
                return isinstance(x, match)
        elif callable(match):
            matchfunc = match
        else:
            raise ValueError('match must be None, an matplotlib.artist.Artist subclass, or a callable')


        artists = []

        for c in self.get_children():
            if matchfunc(c):
                artists.append(c)
            artists.extend([thisc for thisc in c.findobj(matchfunc) if matchfunc(thisc)])

        if matchfunc(self):
            artists.append(self)
        return artists






class ArtistInspector:
    """
    A helper class to inspect an :class:`~matplotlib.artist.Artist`
    and return information about it's settable properties and their
    current values.
    """
    def __init__(self, o):
        """
        Initialize the artist inspector with an
        :class:`~matplotlib.artist.Artist` or sequence of
        :class:`Artists`.  If a sequence is used, we assume it is a
        homogeneous sequence (all :class:`Artists` are of the same
        type) and it is your responsibility to make sure this is so.
        """
        if cbook.iterable(o) and len(o): o = o[0]

        self.oorig = o
        if not isinstance(o, type):
            o = type(o)
        self.o = o

        self.aliasd = self.get_aliases()

    def get_aliases(self):
        """
        Get a dict mapping *fullname* -> *alias* for each *alias* in
        the :class:`~matplotlib.artist.ArtistInspector`.

        Eg., for lines::

          {'markerfacecolor': 'mfc',
           'linewidth'      : 'lw',
          }

        """
        names = [name for name in dir(self.o) if
                 (name.startswith('set_') or name.startswith('get_'))
                 and callable(getattr(self.o,name))]
        aliases = {}
        for name in names:
            func = getattr(self.o, name)
            if not self.is_alias(func): continue
            docstring = func.__doc__
            fullname = docstring[10:]
            aliases.setdefault(fullname[4:], {})[name[4:]] = None
        return aliases

    _get_valid_values_regex = re.compile(r"\n\s*ACCEPTS:\s*((?:.|\n)*?)(?:$|(?:\n\n))")
    def get_valid_values(self, attr):
        """
        Get the legal arguments for the setter associated with *attr*.

        This is done by querying the docstring of the function *set_attr*
        for a line that begins with ACCEPTS:

        Eg., for a line linestyle, return
        [ '-' | '--' | '-.' | ':' | 'steps' | 'None' ]
        """

        name = 'set_%s'%attr
        if not hasattr(self.o, name):
            raise AttributeError('%s has no function %s'%(self.o,name))
        func = getattr(self.o, name)

        docstring = func.__doc__
        if docstring is None: return 'unknown'

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
            if not name.startswith('set_'): continue
            o = getattr(self.o, name)
            if not callable(o): continue
            func = o
            if self.is_alias(func): continue
            source_class = self.o.__module__ + "." + self.o.__name__
            for cls in self.o.mro():
                if name in cls.__dict__:
                    source_class = cls.__module__ + "." + cls.__name__
                    break
            setters.append((name[4:], source_class + "." + name))
        return setters

    def get_setters(self):
        """
        Get the attribute strings with setters for object.  Eg., for a line,
        return ``['markerfacecolor', 'linewidth', ....]``.
        """

        return [prop for prop, target in self._get_setters_and_targets()]

    def is_alias(self, o):
        """
        Return *True* if method object *o* is an alias for another
        function.
        """
        ds = o.__doc__
        if ds is None: return False
        return ds.startswith('alias for ')

    def aliased_name(self, s):
        """
        return 'PROPNAME or alias' if *s* has an alias, else return
        PROPNAME.

        E.g. for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'
        """

        if s in self.aliasd:
            return s + ''.join([' or %s' % x for x in self.aliasd[s].keys()])
        else:
            return s


    def aliased_name_rest(self, s, target):
        """
        return 'PROPNAME or alias' if *s* has an alias, else return
        PROPNAME formatted for ReST

        E.g. for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'
        """

        if s in self.aliasd:
            aliases = ''.join([' or %s' % x for x in self.aliasd[s].keys()])
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
            pad = ' '*leadingspace
        else:
            pad  = ''
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return '%s%s: %s' %(pad, prop, accepts)

        attrs = self._get_setters_and_targets()
        attrs.sort()
        lines = []

        for prop, path in attrs:
            accepts = self.get_valid_values(prop)
            name = self.aliased_name(prop)

            lines.append('%s%s: %s' %(pad, name, accepts))
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
            pad = ' '*leadingspace
        else:
            pad  = ''
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return '%s%s: %s' %(pad, prop, accepts)

        attrs = self._get_setters_and_targets()
        attrs.sort()
        lines = []

        ########
        names = [self.aliased_name_rest(prop, target) for prop, target in attrs]
        accepts = [self.get_valid_values(prop) for prop, target in attrs]

        col0_len = max([len(n) for n in names])
        col1_len = max([len(a) for a in accepts])
        table_formatstr = pad + '='*col0_len + '   ' + '='*col1_len

        lines.append('')
        lines.append(table_formatstr)
        lines.append(pad + 'Property'.ljust(col0_len+3) + \
                     'Description'.ljust(col1_len))
        lines.append(table_formatstr)

        lines.extend([pad + n.ljust(col0_len+3) + a.ljust(col1_len)
                      for n, a in zip(names, accepts)])

        lines.append(table_formatstr)
        lines.append('')
        return lines
        ########

        for prop, path in attrs:
            accepts = self.get_valid_values(prop)
            name = self.aliased_name_rest(prop, path)

            lines.append('%s%s: %s' %(pad, name, accepts))
        return lines

    def pprint_getters(self):
        """
        Return the getters and actual values as list of strings.
        """

        o = self.oorig
        getters = [name for name in dir(o)
                   if name.startswith('get_')
                   and callable(getattr(o, name))]
        #print getters
        getters.sort()
        lines = []
        for name in getters:
            func = getattr(o, name)
            if self.is_alias(func): continue

            try: val = func()
            except: continue
            if getattr(val, 'shape', ()) != () and len(val)>6:
                s = str(val[:6]) + '...'
            else:
                s = str(val)
            s = s.replace('\n', ' ')
            if len(s)>50:
                s = s[:50] + '...'
            name = self.aliased_name(name[4:])
            lines.append('    %s = %s' %(name, s))
        return lines



    def findobj(self, match=None):
        """
        Recursively find all :class:`matplotlib.artist.Artist`
        instances contained in *self*.

        If *match* is not None, it can be

          - function with signature ``boolean = match(artist)``

          - class instance: eg :class:`~matplotlib.lines.Line2D`

        used to filter matches.
        """

        if match is None: # always return True
            def matchfunc(x): return True
        elif issubclass(match, Artist):
            def matchfunc(x):
                return isinstance(x, match)
        elif callable(match):
            matchfunc = func
        else:
            raise ValueError('match must be None, an matplotlib.artist.Artist subclass, or a callable')


        artists = []

        for c in self.get_children():
            if matchfunc(c):
                artists.append(c)
            artists.extend([thisc for thisc in c.findobj(matchfunc) if matchfunc(thisc)])

        if matchfunc(self):
            artists.append(self)
        return artists






def getp(o, property=None):
    """
    Return the value of handle property.  property is an optional string
    for the property you want to return

    Example usage::

        getp(o)  # get all the object properties
        getp(o, 'linestyle')  # get the linestyle property

    *o* is a :class:`Artist` instance, eg
    :class:`~matplotllib.lines.Line2D` or an instance of a
    :class:`~matplotlib.axes.Axes` or :class:`matplotlib.text.Text`.
    If the *property* is 'somename', this function returns

      o.get_somename()

    :func:`getp` can be used to query all the gettable properties with
    ``getp(o)``. Many properties have aliases for shorter typing, e.g.
    'lw' is an alias for 'linewidth'.  In the output, aliases and full
    property names will be listed as:

      property or alias = value

    e.g.:

      linewidth or lw = 2
    """

    insp = ArtistInspector(o)

    if property is None:
        ret = insp.pprint_getters()
        print '\n'.join(ret)
        return

    func = getattr(o, 'get_' + property)

    return func()

# alias
get = getp

def setp(h, *args, **kwargs):
    """
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
    values, all the instances will be set.  E.g., suppose you have a
    list of two lines, the following will make both lines thicker and
    red::

      >>> x = arange(0,1.0,0.01)
      >>> y1 = sin(2*pi*x)
      >>> y2 = sin(4*pi*x)
      >>> lines = plot(x, y1, x, y2)
      >>> setp(lines, linewidth=2, color='r')

    :func:`setp` works with the matlab(TM) style string/value pairs or
    with python kwargs.  For example, the following are equivalent::

      >>> setp(lines, 'linewidth', 2, 'color', r')  # matlab style

      >>> setp(lines, linewidth=2, color='r')       # python style
    """

    insp = ArtistInspector(h)

    if len(kwargs)==0 and len(args)==0:
        print '\n'.join(insp.pprint_setters())
        return

    if len(kwargs)==0 and len(args)==1:
        print insp.pprint_setters(prop=args[0])
        return

    if not cbook.iterable(h): h = [h]
    else: h = cbook.flatten(h)


    if len(args)%2:
        raise ValueError('The set args must be string, value pairs')

    funcvals = []
    for i in range(0, len(args)-1, 2):
        funcvals.append((args[i], args[i+1]))
    funcvals.extend(kwargs.items())

    ret = []
    for o in h:
        for s, val in funcvals:
            s = s.lower()
            funcName = "set_%s"%s
            func = getattr(o,funcName)
            ret.extend( [func(val)] )
    return [x for x in cbook.flatten(ret)]

def kwdoc(a):
    hardcopy = matplotlib.rcParams['docstring.hardcopy']
    if hardcopy:
        return '\n'.join(ArtistInspector(a).pprint_setters_rest(leadingspace=2))
    else:
        return '\n'.join(ArtistInspector(a).pprint_setters(leadingspace=2))

kwdocd = dict()
kwdocd['Artist'] = kwdoc(Artist)
