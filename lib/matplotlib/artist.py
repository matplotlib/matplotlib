from __future__ import division
import sys, re
from cbook import iterable, flatten
from transforms import identity_transform
import matplotlib.units as units

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


class Artist:
    """
    Abstract base class for someone who renders into a FigureCanvas
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
        self._clipon = False
        self._lod = False
        self._label = ''
        self._picker = None
        self._contains = None

        self.eventson = False  # fire events only if eventson
        self._oid = 0  # an observer id
        self._propobservers = {} # a dict from oids to funcs
        self.axes = None
        self._remove_method = None

    def remove(self):
        """
        Remove the artist from the figure if possible.  The effect will not
        be visible until the figure is redrawn, e.g., with ax.draw_idle().
        Call ax.relim() to update the axes limits if desired.

        Note: relim() will not see collections even if the collection
        was added to axes with autolim=True.

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
        'return True if units are set on the x or y axes'
        ax = self.axes
        if ax is None or ax.xaxis is None:
            return False
        return ax.xaxis.have_units() or ax.yaxis.have_units()

    def convert_xunits(self, x):
        """for artists in an axes, if the xaxis as units support,
        convert x using xaxis unit type
        """
        ax = getattr(self, 'axes', None)
        if ax is None or ax.xaxis is None:
            #print 'artist.convert_xunits no conversion: ax=%s'%ax
            return x
        return ax.xaxis.convert_units(x)

    def convert_yunits(self, y):
        """for artists in an axes, if the yaxis as units support,
        convert y using yaxis unit type
        """
        ax = getattr(self, 'axes', None)
        if ax is None or ax.yaxis is None: return y
        return ax.yaxis.convert_units(y)

    def set_axes(self, axes):
        """
        set the axes instance the artist resides in, if any

        ACCEPTS: an axes instance
        """
        self.axes = axes

    def get_axes(self):
        'return the axes instance the artist resides in, or None'
        return self.axes

    def add_callback(self, func):
        oid = self._oid
        self._propobservers[oid] = func
        self._oid += 1
        return oid

    def remove_callback(self, oid):
        try: del self._propobservers[oid]
        except KeyError: pass

    def pchanged(self):
        'fire event when property changed'
        for oid, func in self._propobservers.items():
            func(self)

    def is_transform_set(self):
        'Artist has transform explicity let'
        return self._transformSet

    def set_transform(self, t):
        """
        set the Transformation instance used by this artist

        ACCEPTS: a matplotlib.transform transformation instance
        """
        self._transform = t
        self._transformSet = True
        self.pchanged()

    def get_transform(self):
        'return the Transformation instance used by this artist'
        if self._transform is None:
            self._transform = identity_transform()
        return self._transform

    def hitlist(self,event):
        """List the children of the artist which contain the mouse event"""
        import traceback
        L = []
        try:
            hascursor,info = self.contains(event)
            if hascursor:
                L.append(self)
        except:
            traceback.print_exc()
            print "while checking",self.__class__

        if hasattr(self,'get_children'):
            for a in self.get_children(): L.extend(a.hitlist(event))
        return L

    def contains(self,mouseevent):
        """Test whether the artist contains the mouse event.

        Returns the truth value and a dictionary of artist specific details of
        selection, such as which points are contained in the pick radius.  See
        individual artists for details.
        """
        if callable(self._contains): return self._contains(self,mouseevent)
        #raise NotImplementedError,str(self.__class__)+" needs 'contains' method"
        print str(self.__class__)+" needs 'contains' method"
        return False,{}

    def set_contains(self,picker):
        """Replace the contains test used by this artist. The new picker should
        be a callable function which determines whether the artist is hit by the
        mouse event:

            hit, props = picker(artist, mouseevent)

        If the mouse event is over the artist, return hit=True and props
        is a dictionary of properties you want returned with the contains test.
        """
        self._contains = picker

    def get_contains(self):
        'return the _contains test used by the artist, or None for default.'
        return self._contains

    def pickable(self):
        'return True if self is pickable'
        return (self.figure is not None and
                self.figure.canvas is not None and
                self._picker is not None)

    def pick(self, mouseevent):
        """
        pick(mouseevent)

        each child artist will fire a pick event if mouseevent is over
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
        if hasattr(self,'get_children'):
            for a in self.get_children(): a.pick(mouseevent)

    def set_picker(self, picker):
        """
        set the epsilon for picking used by this artist

        picker can be one of the following:

          None -  picking is disabled for this artist (default)

          boolean - if True then picking will be enabled and the
            artist will fire a pick event if the mouse event is over
            the artist

          float - if picker is a number it is interpreted as an
            epsilon tolerance in points and the the artist will fire
            off an event if it's data is within epsilon of the mouse
            event.  For some artists like lines and patch collections,
            the artist may provide additional data to the pick event
            that is generated, eg the indices of the data within
            epsilon of the pick event

          function - if picker is callable, it is a user supplied
            function which determines whether the artist is hit by the
            mouse event.

              hit, props = picker(artist, mouseevent)

            to determine the hit test.  if the mouse event is over the
            artist, return hit=True and props is a dictionary of
            properties you want added to the PickEvent attributes

        ACCEPTS: [None|float|boolean|callable]
        """
        self._picker = picker

    def get_picker(self):
        'return the Pickeration instance used by this artist'
        return self._picker


    def is_figure_set(self):
        return self.figure is not None

    def get_figure(self):
        'return the figure instance'
        return self.figure

    def set_figure(self, fig):
        """
        Set the figure instance the artist belong to

        ACCEPTS: a matplotlib.figure.Figure instance
        """
        self.figure = fig
        self.pchanged()

    def set_clip_box(self, clipbox):
        """
        Set the artist's clip Bbox

        ACCEPTS: a matplotlib.transform.Bbox instance
        """
        self.clipbox = clipbox
        self._clipon = clipbox is not None or self._clippath is not None
        self.pchanged()

    def set_clip_path(self, path):
        """
        Set the artist's clip path

        ACCEPTS: an agg.path_storage instance
        """
        self._clippath = path
        self._clipon = self.clipbox is not None or path is not None
        self.pchanged()


    def get_alpha(self):
        """
        Return the alpha value used for blending - not supported on all
        backends
        """
        return self._alpha

    def get_visible(self):
        "return the artist's visiblity"
        return self._visible

    def get_animated(self):
        "return the artist's animated state"
        return self._animated

    def get_clip_on(self):
        'Return whether artist uses clipping'
        return self._clipon and (self.clipbox is not None or self._clippath is not None)

    def get_clip_box(self):
        'Return artist clipbox'
        return self.clipbox

    def get_clip_path(self):
        'Return artist clip path'
        return self._clippath

    def set_clip_on(self, b):
        """
        Set  whether artist uses clipping

        ACCEPTS: [True | False]
        """
        self._clipon = b
        if not b:
            self.clipbox = None
            self._clippath = None
        self.pchanged()

    def _set_gc_clip(self, gc):
        'set the clip properly for the gc'
        if self.clipbox is not None:
            gc.set_clip_rectangle(self.clipbox.get_bounds())
        gc.set_clip_path(self._clippath)

    def draw(self, renderer, *args, **kwargs):
        'Derived classes drawing method'
        if not self.get_visible(): return

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on
        all backends

        ACCEPTS: float
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
        set the artist's visiblity

        ACCEPTS: [True | False]
        """
        self._visible = b
        self.pchanged()


    def set_animated(self, b):
        """
        set the artist's animation state

        ACCEPTS: [True | False]
        """
        self._animated = b
        self.pchanged()

    def update(self, props):
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
        return self._label

    def set_label(self, s):
        """
        Set the line label to s for auto legend

        ACCEPTS: any string
        """
        self._label = s
        self.pchanged()



    def get_zorder(self): return self.zorder

    def set_zorder(self, level):
        """
        Set the zorder for the artist

        ACCEPTS: any number
        """
        self.zorder = level
        self.pchanged()

    def update_from(self, other):
        'copy properties from other to self'
        self._transform = other._transform
        self._transformSet = other._transformSet
        self._visible = other._visible
        self._alpha = other._alpha
        self.clipbox = other.clipbox
        self._clipon = other._clipon
        self._lod = other._lod
        self._label = other._label
        self.pchanged()


    def set(self, **kwargs):
        """
        A tkstyle set command, pass kwargs to set properties
        """
        ret = []
        for k,v in kwargs.items():
            k = k.lower()
            funcName = "set_%s"%k
            func = getattr(self,funcName)
            ret.extend( [func(v)] )
        return ret


class ArtistInspector:
    """
    A helper class to inspect an Artist and return information about
    it's settable properties and their current values
    """
    def __init__(self, o):
        """
        Initialize the artist inspector with an artist or sequence of
        artists.  Id a sequence is used, we assume it is a homogeneous
        sequence (all Artists are of the same type) and it is your
        responsibility to make sure this is so.
        """
        if iterable(o) and len(o): o = o[0]
        self.o = o
        self.aliasd = self.get_aliases()

    def get_aliases(self):
        """
        get a dict mapping fullname -> alias for each alias in o.
        Eg for lines: {'markerfacecolor': 'mfc',
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
            aliases[fullname[4:]] = name[4:]
        return aliases

    _get_valid_values_regex = re.compile(r"\n\s*ACCEPTS:\s*(.*)\n")
    def get_valid_values(self, attr):
        """
        get the legal arguments for the setter associated with attr

        This is done by querying the doc string of the function set_attr
        for a line that begins with ACCEPTS:

        Eg, for a line linestyle, return
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
            return match.group(1)
        return 'unknown'

    def get_setters(self):
        """
        Get the attribute strings with setters for object h.  Eg, for a line,
        return ['markerfacecolor', 'linewidth', ....]
        """

        setters = []
        for name in dir(self.o):
            if not name.startswith('set_'): continue
            o = getattr(self.o,name)
            if not callable(o): continue
            func = o
            if self.is_alias(func): continue
            setters.append(name[4:])
        return setters

    def is_alias(self, o):
        'return true if method object o is an alias for another function'
        ds = o.__doc__
        if ds is None: return False
        return ds.startswith('alias for ')

    def aliased_name(self, s):
        """
        return 'PROPNAME or alias' if s has an alias, else return
        PROPNAME.

        Eg for the line markerfacecolor property, which has an alias,
        return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'
        """
        if self.aliasd.has_key(s):
            return '%s or %s' % (s, self.aliasd[s])
        else: return s

    def pprint_setters(self, prop=None, leadingspace=2):
        """
        if prop is None, return a list of strings of all settable properies
        and their valid values

        if prop is not None, it is a valid property name and that
        property will be returned as a string of property : valid
        values
        """
        if leadingspace:
            pad = ' '*leadingspace
        else:
            pad  = ''
        if prop is not None:
            accepts = self.get_valid_values(prop)
            return '%s%s: %s' %(pad, prop, accepts)

        attrs = self.get_setters()
        attrs.sort()
        lines = []

        for prop in attrs:
            accepts = self.get_valid_values(prop)
            name = self.aliased_name(prop)

            lines.append('%s%s: %s' %(pad, name, accepts))
        return lines

    def pprint_getters(self):
        """
        return the getters and actual values as list of strings'
        """
        getters = [name for name in dir(self.o)
                   if name.startswith('get_')
                   and callable(getattr(self.o, name))]
        getters.sort()
        lines = []
        for name in getters:
            func = getattr(self.o, name)
            if self.is_alias(func): continue
            try: val = func()
            except: continue
            if getattr(val, 'shape', ()) != () and len(val)>6:
                s = str(val[:6]) + '...'
            else:
                s = str(val)
            name = self.aliased_name(name[4:])
            lines.append('    %s = %s' %(name, s))
        return lines


def getp(o, *args):
    """
    Return the value of handle property s

    h is an instance of a class, eg a Line2D or an Axes or Text.
    if s is 'somename', this function returns

      o.get_somename()

    getp can be used to query all the gettable properties with getp(o)
    Many properties have aliases for shorter typing, eg 'lw' is an
    alias for 'linewidth'.  In the output, aliases and full property
    names will be listed as

      property or  alias = value

    eg

      linewidth or lw = 2
    """

    insp = ArtistInspector(o)

    if len(args)==0:
        print '\n'.join(insp.pprint_getters())
        return

    name = args[0]
    func = getattr(o, 'get_' + name)
    return func()

def get(o, *args, **kwargs):
    return getp(o, *args, **kwargs)
get.__doc__ = getp.__doc__


def setp(h, *args, **kwargs):
    """
    matplotlib supports the use of setp ("set property") and getp to set
    and get object properties, as well as to do introspection on the
    object For example, to set the linestyle of a line to be dashed, you
    can do

      >>> line, = plot([1,2,3])
      >>> setp(line, linestyle='--')

    If you want to know the valid types of arguments, you can provide the
    name of the property you want to set without a value

      >>> setp(line, 'linestyle')
          linestyle: [ '-' | '--' | '-.' | ':' | 'steps' | 'None' ]

    If you want to see all the properties that can be set, and their
    possible values, you can do


      >>> setp(line)
          ... long output listing omitted'

    setp operates on a single instance or a list of instances.  If you
    are in query mode introspecting the possible values, only the first
    instance in the sequence is used.  When actually setting values,
    all the instances will be set.  Eg, suppose you have a list of two
    lines, the following will make both lines thicker and red

        >>> x = arange(0,1.0,0.01)
        >>> y1 = sin(2*pi*x)
        >>> y2 = sin(4*pi*x)
        >>> lines = plot(x, y1, x, y2)
        >>> setp(lines, linewidth=2, color='r')

    setp works with the matlab(TM) style string/value pairs or with
    python kwargs.  For example, the following are equivalent

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

    if not iterable(h): h = [h]
    else: h = flatten(h)


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
    return [x for x in flatten(ret)]

def kwdoc(a):
    return '\n'.join(ArtistInspector(a).pprint_setters(leadingspace=4))

kwdocd = dict()
kwdocd['Artist'] = kwdoc(Artist)
