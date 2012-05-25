"""
The figure module provides the top-level
:class:`~matplotlib.artist.Artist`, the :class:`Figure`, which
contains all the plot elements.  The following classes are defined

:class:`SubplotParams`
    control the default spacing of the subplots

:class:`Figure`
    top level container for all plot elements


"""
import numpy as np

import artist
from artist import Artist, allow_rasterization
from axes import Axes, SubplotBase, subplot_class_factory
from cbook import flatten, allequal, Stack, iterable, is_string_like
import _image
import colorbar as cbar
from image import FigureImage
from matplotlib import rcParams
from patches import Rectangle
from text import Text, _process_text_args

from legend import Legend
from transforms import Affine2D, Bbox, BboxTransformTo, TransformedBbox
from projections import projection_factory, get_projection_names, \
    get_projection_class
from matplotlib.blocking_input import BlockingMouseInput, BlockingKeyMouseInput

import matplotlib.cbook as cbook
from matplotlib import docstring

from operator import itemgetter
import os.path

docstring.interpd.update(projection_names = get_projection_names())

class AxesStack(Stack):
    """
    Specialization of the Stack to handle all
    tracking of Axes in a Figure.  This requires storing
    key, (ind, axes) pairs. The key is based on the args and kwargs
    used in generating the Axes. ind is a serial number for tracking
    the order in which axes were added.
    """
    def __init__(self):
        Stack.__init__(self)
        self._ind = 0

    def as_list(self):
        """
        Return a list of the Axes instances that have been added to the figure
        """
        ia_list = [a for k, a in self._elements]
        ia_list.sort()
        return [a for i, a in ia_list]

    def get(self, key):
        """
        Return the Axes instance that was added with *key*.
        If it is not present, return None.
        """
        item = dict(self._elements).get(key)
        if item is None:
            return None
        return item[1]

    def _entry_from_axes(self, e):
        ind, k = dict([(a, (ind, k)) for (k, (ind, a)) in self._elements])[e]
        return (k, (ind, e))

    def remove(self, a):
        Stack.remove(self, self._entry_from_axes(a))

    def bubble(self, a):
        return Stack.bubble(self, self._entry_from_axes(a))

    def add(self, key, a):
        """
        Add Axes *a*, with key *key*, to the stack, and return the stack.

        If *a* is already on the stack, don't add it again, but
        return *None*.
        """
        # All the error checking may be unnecessary; but this method
        # is called so seldom that the overhead is negligible.
        if not isinstance(a, Axes):
            raise ValueError("second argument, %s, is not an Axes" % a)
        try:
            hash(key)
        except TypeError:
            raise ValueError("first argument, %s, is not a valid key" % key)

        a_existing = self.get(key)
        if a_existing is not None:
            Stack.remove(self, (key, a_existing))
            warnings.Warn(
                    "key %s already existed; Axes is being replaced" % key)
            # I don't think the above should ever happen.

        if a in self:
            return None
        self._ind += 1
        return Stack.push(self, (key, (self._ind, a)))

    def __call__(self):
        if not len(self._elements):
            return self._default
        else:
            return self._elements[self._pos][1][1]

    def __contains__(self, a):
        return a in self.as_list()

class SubplotParams:
    """
    A class to hold the parameters for a subplot
    """
    def __init__(self, left=None, bottom=None, right=None, top=None,
                 wspace=None, hspace=None):
        """
        All dimensions are fraction of the figure width or height.
        All values default to their rc params

        The following attributes are available

        *left*  : 0.125
            The left side of the subplots of the figure

        *right* : 0.9
            The right side of the subplots of the figure

        *bottom* : 0.1
            The bottom of the subplots of the figure

        *top* : 0.9
            The top of the subplots of the figure

        *wspace* : 0.2
            The amount of width reserved for blank space between subplots

        *hspace* : 0.2
            The amount of height reserved for white space between subplots
        """

        self.validate = True
        self.update(left, bottom, right, top, wspace, hspace)

    def update(self,left=None, bottom=None, right=None, top=None,
               wspace=None, hspace=None):
        """
        Update the current values.  If any kwarg is None, default to
        the current value, if set, otherwise to rc

        """

        thisleft = getattr(self, 'left', None)
        thisright = getattr(self, 'right', None)
        thistop = getattr(self, 'top', None)
        thisbottom = getattr(self, 'bottom', None)
        thiswspace = getattr(self, 'wspace', None)
        thishspace = getattr(self, 'hspace', None)


        self._update_this('left', left)
        self._update_this('right', right)
        self._update_this('bottom', bottom)
        self._update_this('top', top)
        self._update_this('wspace', wspace)
        self._update_this('hspace', hspace)

        def reset():
            self.left = thisleft
            self.right = thisright
            self.top = thistop
            self.bottom = thisbottom
            self.wspace = thiswspace
            self.hspace = thishspace

        if self.validate:
            if self.left>=self.right:
                reset()
                raise ValueError('left cannot be >= right')

            if self.bottom>=self.top:
                reset()
                raise ValueError('bottom cannot be >= top')



    def _update_this(self, s, val):
        if val is None:
            val = getattr(self, s, None)
            if val is None:
                key = 'figure.subplot.' + s
                val = rcParams[key]

        setattr(self, s, val)

class Figure(Artist):

    """
    The Figure instance supports callbacks through a *callbacks*
    attribute which is a :class:`matplotlib.cbook.CallbackRegistry`
    instance.  The events you can connect to are 'dpi_changed', and
    the callback will be called with ``func(fig)`` where fig is the
    :class:`Figure` instance.

    *patch*
       The figure patch is drawn by a
       :class:`matplotlib.patches.Rectangle` instance

    *suppressComposite*
       For multiple figure images, the figure will make composite
       images depending on the renderer option_image_nocomposite
       function.  If suppressComposite is True|False, this will
       override the renderer
    """

    def __str__(self):
        return "Figure(%gx%g)" % tuple(self.bbox.size)

    def __init__(self,
                 figsize   = None,  # defaults to rc figure.figsize
                 dpi       = None,  # defaults to rc figure.dpi
                 facecolor = None,  # defaults to rc figure.facecolor
                 edgecolor = None,  # defaults to rc figure.edgecolor
                 linewidth = 0.0,   # the default linewidth of the frame
                 frameon = True,    # whether or not to draw the figure frame
                 subplotpars = None, # default to rc
                 ):
        """
        *figsize*
            w,h tuple in inches

        *dpi*
            Dots per inch

        *facecolor*
            The figure patch facecolor; defaults to rc ``figure.facecolor``

        *edgecolor*
            The figure patch edge color; defaults to rc ``figure.edgecolor``

        *linewidth*
            The figure patch edge linewidth; the default linewidth of the frame

        *frameon*
            If *False*, suppress drawing the figure frame

        *subplotpars*
            A :class:`SubplotParams` instance, defaults to rc
        """
        Artist.__init__(self)

        self.callbacks = cbook.CallbackRegistry()

        if figsize is None  : figsize   = rcParams['figure.figsize']
        if dpi is None      : dpi       = rcParams['figure.dpi']
        if facecolor is None: facecolor = rcParams['figure.facecolor']
        if edgecolor is None: edgecolor = rcParams['figure.edgecolor']

        self.dpi_scale_trans = Affine2D()
        self.dpi = dpi
        self.bbox_inches = Bbox.from_bounds(0, 0, *figsize)
        self.bbox = TransformedBbox(self.bbox_inches, self.dpi_scale_trans)

        self.frameon = frameon

        self.transFigure = BboxTransformTo(self.bbox)

        # the figurePatch name is deprecated
        self.patch = self.figurePatch = Rectangle(
            xy=(0,0), width=1, height=1,
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth,
            )
        self._set_artist_props(self.patch)
        self.patch.set_aa(False)

        self._hold = rcParams['axes.hold']
        self.canvas = None

        if subplotpars is None:
            subplotpars = SubplotParams()

        self.subplotpars = subplotpars

        self._axstack = AxesStack()  # track all figure axes and current axes
        self.clf()
        self._cachedRenderer = None

    def _get_axes(self):
        return self._axstack.as_list()

    axes = property(fget=_get_axes, doc="Read-only: list of axes in Figure")

    def _get_dpi(self):
        return self._dpi
    def _set_dpi(self, dpi):
        self._dpi = dpi
        self.dpi_scale_trans.clear().scale(dpi, dpi)
        self.callbacks.process('dpi_changed', self)
    dpi = property(_get_dpi, _set_dpi)

    def autofmt_xdate(self, bottom=0.2, rotation=30, ha='right'):
        """
        Date ticklabels often overlap, so it is useful to rotate them
        and right align them.  Also, a common use case is a number of
        subplots with shared xaxes where the x-axis is date data.  The
        ticklabels are often long, and it helps to rotate them on the
        bottom subplot and turn them off on other subplots, as well as
        turn off xlabels.

        *bottom*
            The bottom of the subplots for :meth:`subplots_adjust`

        *rotation*
            The rotation of the xtick labels

        *ha*
            The horizontal alignment of the xticklabels
        """
        allsubplots = np.alltrue([hasattr(ax, 'is_last_row') for ax in self.axes])
        if len(self.axes)==1:
            for label in ax.get_xticklabels():
                label.set_ha(ha)
                label.set_rotation(rotation)
        else:
            if allsubplots:
                for ax in self.get_axes():
                    if ax.is_last_row():
                        for label in ax.get_xticklabels():
                            label.set_ha(ha)
                            label.set_rotation(rotation)
                    else:
                        for label in ax.get_xticklabels():
                            label.set_visible(False)
                        ax.set_xlabel('')

        if allsubplots:
            self.subplots_adjust(bottom=bottom)

    def get_children(self):
        'get a list of artists contained in the figure'
        children = [self.patch]
        children.extend(self.artists)
        children.extend(self.axes)
        children.extend(self.lines)
        children.extend(self.patches)
        children.extend(self.texts)
        children.extend(self.images)
        children.extend(self.legends)
        return children

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred on the figure.

        Returns True,{}
        """
        if callable(self._contains): return self._contains(self,mouseevent)
        #inside = mouseevent.x >= 0 and mouseevent.y >= 0
        inside = self.bbox.contains(mouseevent.x,mouseevent.y)

        return inside,{}

    def get_window_extent(self, *args, **kwargs):
        'get the figure bounding box in display space; kwargs are void'
        return self.bbox

    def suptitle(self, t, **kwargs):
        """
        Add a centered title to the figure.

        kwargs are :class:`matplotlib.text.Text` properties.  Using figure
        coordinates, the defaults are:

          *x* : 0.5
            The x location of the text in figure coords

          *y* : 0.98
            The y location of the text in figure coords

          *horizontalalignment* : 'center'
            The horizontal alignment of the text

          *verticalalignment* : 'top'
            The vertical alignment of the text

        A :class:`matplotlib.text.Text` instance is returned.

        Example::

          fig.suptitle('this is the figure title', fontsize=12)
        """
        x = kwargs.pop('x', 0.5)
        y = kwargs.pop('y', 0.98)
        if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
            kwargs['horizontalalignment'] = 'center'

        if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
            kwargs['verticalalignment'] = 'top'

        t = self.text(x, y, t, **kwargs)
        return t

    def set_canvas(self, canvas):
        """
        Set the canvas the contains the figure

        ACCEPTS: a FigureCanvas instance
        """
        self.canvas = canvas

    def hold(self, b=None):
        """
        Set the hold state.  If hold is None (default), toggle the
        hold state.  Else set the hold state to boolean value b.

        Eg::

            hold()      # toggle hold
            hold(True)  # hold is on
            hold(False) # hold is off
        """
        if b is None: self._hold = not self._hold
        else: self._hold = b

    def figimage(self, X,
                 xo=0,
                 yo=0,
                 alpha=None,
                 norm=None,
                 cmap=None,
                 vmin=None,
                 vmax=None,
                 origin=None,
                 **kwargs):
        """
        call signatures::

          figimage(X, **kwargs)

        adds a non-resampled array *X* to the figure.

        ::

          figimage(X, xo, yo)

        with pixel offsets *xo*, *yo*,

        *X* must be a float array:

        * If *X* is MxN, assume luminance (grayscale)
        * If *X* is MxNx3, assume RGB
        * If *X* is MxNx4, assume RGBA

        Optional keyword arguments:

          =========   ==========================================================
          Keyword     Description
          =========   ==========================================================
          xo or yo    An integer, the *x* and *y* image offset in pixels
          cmap        a :class:`matplotlib.colors.Colormap` instance, eg cm.jet.
                      If *None*, default to the rc ``image.cmap`` value
          norm        a :class:`matplotlib.colors.Normalize` instance. The
                      default is normalization().  This scales luminance -> 0-1
          vmin|vmax   are used to scale a luminance image to 0-1.  If either is
                      *None*, the min and max of the luminance values will be
                      used.  Note if you pass a norm instance, the settings for
                      *vmin* and *vmax* will be ignored.
          alpha       the alpha blending value, default is *None*
          origin      [ 'upper' | 'lower' ] Indicates where the [0,0] index of
                      the array is in the upper left or lower left corner of
                      the axes. Defaults to the rc image.origin value
          =========   ==========================================================

        figimage complements the axes image
        (:meth:`~matplotlib.axes.Axes.imshow`) which will be resampled
        to fit the current axes.  If you want a resampled image to
        fill the entire figure, you can define an
        :class:`~matplotlib.axes.Axes` with size [0,1,0,1].

        An :class:`matplotlib.image.FigureImage` instance is returned.

        .. plot:: mpl_examples/pylab_examples/figimage_demo.py


        Additional kwargs are Artist kwargs passed on to
        :class:`~matplotlib.image.FigureImage`
        """

        if not self._hold: self.clf()

        im = FigureImage(self, cmap, norm, xo, yo, origin, **kwargs)
        im.set_array(X)
        im.set_alpha(alpha)
        if norm is None:
            im.set_clim(vmin, vmax)
        self.images.append(im)
        return im

    def set_size_inches(self, *args, **kwargs):
        """
        set_size_inches(w,h, forward=False)

        Set the figure size in inches

        Usage::

             fig.set_size_inches(w,h)  # OR
             fig.set_size_inches((w,h) )

        optional kwarg *forward=True* will cause the canvas size to be
        automatically updated; eg you can resize the figure window
        from the shell

        ACCEPTS: a w,h tuple with w,h in inches
        """

        forward = kwargs.get('forward', False)
        if len(args)==1:
            w,h = args[0]
        else:
            w,h = args

        dpival = self.dpi
        self.bbox_inches.p1 = w, h

        if forward:
            dpival = self.dpi
            canvasw = w*dpival
            canvash = h*dpival
            manager = getattr(self.canvas, 'manager', None)
            if manager is not None:
                manager.resize(int(canvasw), int(canvash))

    def get_size_inches(self):
        return self.bbox_inches.p1

    def get_edgecolor(self):
        'Get the edge color of the Figure rectangle'
        return self.patch.get_edgecolor()

    def get_facecolor(self):
        'Get the face color of the Figure rectangle'
        return self.patch.get_facecolor()

    def get_figwidth(self):
        'Return the figwidth as a float'
        return self.bbox_inches.width

    def get_figheight(self):
        'Return the figheight as a float'
        return self.bbox_inches.height

    def get_dpi(self):
        'Return the dpi as a float'
        return self.dpi

    def get_frameon(self):
        'get the boolean indicating frameon'
        return self.frameon

    def set_edgecolor(self, color):
        """
        Set the edge color of the Figure rectangle

        ACCEPTS: any matplotlib color - see help(colors)
        """
        self.patch.set_edgecolor(color)

    def set_facecolor(self, color):
        """
        Set the face color of the Figure rectangle

        ACCEPTS: any matplotlib color - see help(colors)
        """
        self.patch.set_facecolor(color)

    def set_dpi(self, val):
        """
        Set the dots-per-inch of the figure

        ACCEPTS: float
        """
        self.dpi = val

    def set_figwidth(self, val):
        """
        Set the width of the figure in inches

        ACCEPTS: float
        """
        self.bbox_inches.x1 = val

    def set_figheight(self, val):
        """
        Set the height of the figure in inches

        ACCEPTS: float
        """
        self.bbox_inches.y1 = val

    def set_frameon(self, b):
        """
        Set whether the figure frame (background) is displayed or invisible

        ACCEPTS: boolean
        """
        self.frameon = b

    def delaxes(self, a):
        'remove a from the figure and update the current axes'
        self._axstack.remove(a)
        for func in self._axobservers: func(self)

    def _make_key(self, *args, **kwargs):
        'make a hashable key out of args and kwargs'

        def fixitems(items):
            #items may have arrays and lists in them, so convert them
            # to tuples for the key
            ret = []
            for k, v in items:
                if iterable(v): v = tuple(v)
                ret.append((k,v))
            return tuple(ret)

        def fixlist(args):
            ret = []
            for a in args:
                if iterable(a): a = tuple(a)
                ret.append(a)
            return tuple(ret)

        key = fixlist(args), fixitems(kwargs.items())
        return key

    @docstring.dedent_interpd
    def add_axes(self, *args, **kwargs):
        """
        Add an axes at position *rect* [*left*, *bottom*, *width*,
        *height*] where all quantities are in fractions of figure
        width and height.  kwargs are legal
        :class:`~matplotlib.axes.Axes` kwargs plus *projection* which
        sets the projection type of the axes.  (For backward
        compatibility, ``polar=True`` may also be provided, which is
        equivalent to ``projection='polar'``).  Valid values for
        *projection* are: %(projection_names)s.  Some of these
        projections support  additional kwargs, which may be provided
        to :meth:`add_axes`. Typical usage::

            rect = l,b,w,h
            fig.add_axes(rect)
            fig.add_axes(rect, frameon=False, axisbg='g')
            fig.add_axes(rect, polar=True)
            fig.add_axes(rect, projection='polar')
            fig.add_axes(ax)

        If the figure already has an axes with the same parameters,
        then it will simply make that axes current and return it.  If
        you do not want this behavior, e.g. you want to force the
        creation of a new Axes, you must use a unique set of args and
        kwargs.  The axes :attr:`~matplotlib.axes.Axes.label`
        attribute has been exposed for this purpose.  Eg., if you want
        two axes that are otherwise identical to be added to the
        figure, make sure you give them unique labels::

            fig.add_axes(rect, label='axes1')
            fig.add_axes(rect, label='axes2')

        In rare circumstances, add_axes may be called with a single
        argument, an Axes instance already created in the present
        figure but not in the figure's list of axes.  For example,
        if an axes has been removed with :meth:`delaxes`, it can
        be restored with::

            fig.add_axes(ax)

        In all cases, the :class:`~matplotlib.axes.Axes` instance
        will be returned.

        In addition to *projection*, the following kwargs are supported:

        %(Axes)s
        """
        if not len(args): return

        key = self._make_key(*args, **kwargs)
        ax = self._axstack.get(key)
        if ax is not None:
            self.sca(ax)
            return ax

        if isinstance(args[0], Axes):
            a = args[0]
            assert(a.get_figure() is self)
        else:
            rect = args[0]
            ispolar = kwargs.pop('polar', False)
            projection = kwargs.pop('projection', None)
            if ispolar:
                if projection is not None and projection != 'polar':
                    raise ValueError(
                        "polar=True, yet projection='%s'. " +
                        "Only one of these arguments should be supplied." %
                        projection)
                projection = 'polar'

            a = projection_factory(projection, self, rect, **kwargs)

        self._axstack.add(key, a)
        self.sca(a)
        return a

    @docstring.dedent_interpd
    def add_subplot(self, *args, **kwargs):
        """
        Add a subplot.  Examples::

            fig.add_subplot(111)
            fig.add_subplot(1,1,1)            # equivalent but more general
            fig.add_subplot(212, axisbg='r')  # add subplot with red background
            fig.add_subplot(111, polar=True)  # add a polar subplot
            fig.add_subplot(sub)              # add Subplot instance sub

        *kwargs* are legal :class:`~matplotlib.axes.Axes` kwargs plus
        *projection*, which chooses a projection type for the axes.
        (For backward compatibility, *polar=True* may also be
        provided, which is equivalent to *projection='polar'*). Valid
        values for *projection* are: %(projection_names)s.  Some of
        these projections
        support additional *kwargs*, which may be provided to
        :meth:`add_axes`.

        The :class:`~matplotlib.axes.Axes` instance will be returned.

        If the figure already has a subplot with key (*args*,
        *kwargs*) then it will simply make that subplot current and
        return it.

        The following kwargs are supported:

        %(Axes)s
        """
        if not len(args): return

        if len(args) == 1 and isinstance(args[0], int):
            args = tuple([int(c) for c in str(args[0])])

        if isinstance(args[0], SubplotBase):
            a = args[0]
            assert(a.get_figure() is self)
            key = self._make_key(*args, **kwargs)
        else:
            kwargs = kwargs.copy()
            ispolar = kwargs.pop('polar', False)
            projection = kwargs.pop('projection', None)
            if ispolar:
                if projection is not None and projection != 'polar':
                    raise ValueError(
                        "polar=True, yet projection='%s'. " +
                        "Only one of these arguments should be supplied." %
                        projection)
                projection = 'polar'

            projection_class = get_projection_class(projection)

            # Remake the key without projection kwargs:
            key = self._make_key(*args, **kwargs)
            ax = self._axstack.get(key)
            if ax is not None:
                if isinstance(ax, projection_class):
                    self.sca(ax)
                    return ax
                else:
                    self._axstack.remove(ax)
                    # Undocumented convenience behavior:
                    # subplot(111); subplot(111, projection='polar')
                    # will replace the first with the second.
                    # Without this, add_subplot would be simpler and
                    # more similar to add_axes.

            a = subplot_class_factory(projection_class)(self, *args, **kwargs)
        self._axstack.add(key, a)
        self.sca(a)
        return a

    def clf(self, keep_observers=False):
        """
        Clear the figure.

        Set *keep_observers* to True if, for example,
        a gui widget is tracking the axes in the figure.
        """
        self.suppressComposite = None
        self.callbacks = cbook.CallbackRegistry()

        for ax in tuple(self.axes):  # Iterate over the copy.
            ax.cla()
            self.delaxes(ax)         # removes ax from self._axstack

        toolbar = getattr(self.canvas, 'toolbar', None)
        if toolbar is not None:
            toolbar.update()
        self._axstack.clear()
        self.artists = []
        self.lines = []
        self.patches = []
        self.texts=[]
        self.images = []
        self.legends = []
        if not keep_observers:
            self._axobservers = []

    def clear(self):
        """
        Clear the figure -- synonym for :meth:`clf`.
        """
        self.clf()

    @allow_rasterization
    def draw(self, renderer):
        """
        Render the figure using :class:`matplotlib.backend_bases.RendererBase`
        instance *renderer*.
        """
        # draw the figure bounding box, perhaps none for white figure
        if not self.get_visible(): return
        renderer.open_group('figure')

        if self.frameon: self.patch.draw(renderer)

        # a list of (zorder, func_to_call, list_of_args)
        dsu = []

        for a in self.patches:
            dsu.append( (a.get_zorder(), a, a.draw, [renderer]))

        for a in self.lines:
            dsu.append( (a.get_zorder(), a, a.draw, [renderer]))

        for a in self.artists:
            dsu.append( (a.get_zorder(), a, a.draw, [renderer]))

        # override the renderer default if self.suppressComposite
        # is not None
        not_composite = renderer.option_image_nocomposite()
        if self.suppressComposite is not None:
            not_composite = self.suppressComposite

        if len(self.images)<=1 or not_composite or \
                not allequal([im.origin for im in self.images]):
            for a in self.images:
                dsu.append( (a.get_zorder(), a, a.draw, [renderer]))
        else:
            # make a composite image blending alpha
            # list of (_image.Image, ox, oy)
            mag = renderer.get_image_magnification()
            ims = [(im.make_image(mag), im.ox, im.oy)
                   for im in self.images]

            im = _image.from_images(self.bbox.height * mag,
                                    self.bbox.width * mag,
                                    ims)

            im.is_grayscale = False
            l, b, w, h = self.bbox.bounds

            def draw_composite():
                gc = renderer.new_gc()
                gc.set_clip_rectangle(self.bbox)
                gc.set_clip_path(self.get_clip_path())
                renderer.draw_image(gc, l, b, im)
                gc.restore()

            dsu.append((self.images[0].get_zorder(), self.images[0], draw_composite, []))

        # render the axes
        for a in self.axes:
            dsu.append( (a.get_zorder(), a, a.draw, [renderer]))

        # render the figure text
        for a in self.texts:
            dsu.append( (a.get_zorder(), a, a.draw, [renderer]))

        for a in self.legends:
            dsu.append( (a.get_zorder(), a, a.draw, [renderer]))

        dsu = [row for row in dsu if not row[1].get_animated()]
        dsu.sort(key=itemgetter(0))
        for zorder, a, func, args in dsu:
            func(*args)

        renderer.close_group('figure')

        self._cachedRenderer = renderer

        self.canvas.draw_event(renderer)

    def draw_artist(self, a):
        """
        draw :class:`matplotlib.artist.Artist` instance *a* only --
        this is available only after the figure is drawn
        """
        assert self._cachedRenderer is not None
        a.draw(self._cachedRenderer)

    def get_axes(self):
        return self.axes

    def legend(self, handles, labels, *args, **kwargs):
        """
        Place a legend in the figure.  Labels are a sequence of
        strings, handles is a sequence of
        :class:`~matplotlib.lines.Line2D` or
        :class:`~matplotlib.patches.Patch` instances, and loc can be a
        string or an integer specifying the legend location

        USAGE::

          legend( (line1, line2, line3),
                  ('label1', 'label2', 'label3'),
                  'upper right')

        The *loc* location codes are::

          'best' : 0,          (currently not supported for figure legends)
          'upper right'  : 1,
          'upper left'   : 2,
          'lower left'   : 3,
          'lower right'  : 4,
          'right'        : 5,
          'center left'  : 6,
          'center right' : 7,
          'lower center' : 8,
          'upper center' : 9,
          'center'       : 10,

        *loc* can also be an (x,y) tuple in figure coords, which
        specifies the lower left of the legend box.  figure coords are
        (0,0) is the left, bottom of the figure and 1,1 is the right,
        top.

        Keyword arguments:

          *prop*: [ *None* | FontProperties | dict ]
            A :class:`matplotlib.font_manager.FontProperties`
            instance. If *prop* is a dictionary, a new instance will be
            created with *prop*. If *None*, use rc settings.

          *numpoints*: integer
            The number of points in the legend line, default is 4

          *scatterpoints*: integer
            The number of points in the legend line, default is 4

          *scatteroffsets*: list of floats
            a list of yoffsets for scatter symbols in legend

          *markerscale*: [ *None* | scalar ]
            The relative size of legend markers vs. original. If *None*, use rc
            settings.

          *fancybox*: [ *None* | *False* | *True* ]
            if *True*, draw a frame with a round fancybox.  If *None*, use rc

          *shadow*: [ *None* | *False* | *True* ]
            If *True*, draw a shadow behind legend. If *None*, use rc settings.

          *ncol* : integer
            number of columns. default is 1

          *mode* : [ "expand" | *None* ]
            if mode is "expand", the legend will be horizontally expanded
            to fill the axes area (or *bbox_to_anchor*)

          *title* : string
            the legend title

        Padding and spacing between various elements use following keywords
        parameters. The dimensions of these values are given as a fraction
        of the fontsize. Values from rcParams will be used if None.

        ================   ==================================================================
        Keyword            Description
        ================   ==================================================================
        borderpad          the fractional whitespace inside the legend border
        labelspacing       the vertical space between the legend entries
        handlelength       the length of the legend handles
        handletextpad      the pad between the legend handle and text
        borderaxespad      the pad between the axes and legend border
        columnspacing      the spacing between columns
        ================   ==================================================================

        .. Note:: Not all kinds of artist are supported by the legend.
                  See LINK (FIXME) for details.

        **Example:**

        .. plot:: mpl_examples/pylab_examples/figlegend_demo.py
        """
        l = Legend(self, handles, labels, *args, **kwargs)
        self.legends.append(l)
        return l

    @docstring.dedent_interpd
    def text(self, x, y, s, *args, **kwargs):
        """
        Call signature::

          text(x, y, s, fontdict=None, **kwargs)

        Add text to figure at location *x*, *y* (relative 0-1
        coords). See :func:`~matplotlib.pyplot.text` for the meaning
        of the other arguments.

        kwargs control the :class:`~matplotlib.text.Text` properties:

        %(Text)s
        """

        override = _process_text_args({}, *args, **kwargs)
        t = Text(
            x=x, y=y, text=s,
            )

        t.update(override)
        self._set_artist_props(t)
        self.texts.append(t)
        return t

    def _set_artist_props(self, a):
        if a!= self:
            a.set_figure(self)
        a.set_transform(self.transFigure)

    @docstring.dedent_interpd
    def gca(self, **kwargs):
        """
        Return the current axes, creating one if necessary

        The following kwargs are supported
        %(Axes)s
        """
        ax = self._axstack()
        if ax is not None:
            ispolar = kwargs.get('polar', False)
            projection = kwargs.get('projection', None)
            if ispolar:
                if projection is not None and projection != 'polar':
                    raise ValueError(
                        "polar=True, yet projection='%s'. " +
                        "Only one of these arguments should be supplied." %
                        projection)
                projection = 'polar'

            projection_class = get_projection_class(projection)
            if isinstance(ax, projection_class):
                return ax
        return self.add_subplot(111, **kwargs)

    def sca(self, a):
        'Set the current axes to be a and return a'
        self._axstack.bubble(a)
        for func in self._axobservers: func(self)
        return a

    def _gci(self):
        """
        helper for :func:`~matplotlib.pyplot.gci`;
        do not use elsewhere.
        """
        for ax in reversed(self.axes):
            im = ax._gci()
            if im is not None:
                return im
        return None

    def add_axobserver(self, func):
        'whenever the axes state change, ``func(self)`` will be called'
        self._axobservers.append(func)


    def savefig(self, *args, **kwargs):
        """
        Call signature::

          savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                  orientation='portrait', papertype=None, format=None,
                  transparent=False, bbox_inches=None, pad_inches=0.1):

        Save the current figure.

        The output formats available depend on the backend being used.

        Arguments:

          *fname*:
            A string containing a path to a filename, or a Python
            file-like object, or possibly some backend-dependent object
            such as :class:`~matplotlib.backends.backend_pdf.PdfPages`.

            If *format* is *None* and *fname* is a string, the output
            format is deduced from the extension of the filename. If
            the filename has no extension, the value of the rc parameter
            ``savefig.extension`` is used. If that value is 'auto',
            the backend determines the extension.

            If *fname* is not a string, remember to specify *format* to
            ensure that the correct backend is used.

        Keyword arguments:

          *dpi*: [ *None* | ``scalar > 0`` ]
            The resolution in dots per inch.  If *None* it will default to
            the value ``savefig.dpi`` in the matplotlibrc file.

          *facecolor*, *edgecolor*:
            the colors of the figure rectangle

          *orientation*: [ 'landscape' | 'portrait' ]
            not supported on all backends; currently only on postscript output

          *papertype*:
            One of 'letter', 'legal', 'executive', 'ledger', 'a0' through
            'a10', 'b0' through 'b10'. Only supported for postscript
            output.

          *format*:
            One of the file extensions supported by the active
            backend.  Most backends support png, pdf, ps, eps and svg.

          *transparent*:
            If *True*, the axes patches will all be transparent; the
            figure patch will also be transparent unless facecolor
            and/or edgecolor are specified via kwargs.
            This is useful, for example, for displaying
            a plot on top of a colored background on a web page.  The
            transparency of these patches will be restored to their
            original values upon exit of this function.

          *bbox_inches*:
            Bbox in inches. Only the given portion of the figure is
            saved. If 'tight', try to figure out the tight bbox of
            the figure.

          *pad_inches*:
            Amount of padding around the figure when bbox_inches is
            'tight'.

          *bbox_extra_artists*:
            A list of extra artists that will be considered when the
            tight bbox is calculated.

        """

        kwargs.setdefault('dpi', rcParams['savefig.dpi'])

        extension = rcParams['savefig.extension']
        if args and is_string_like(args[0]) and '.' not in os.path.splitext(args[0])[-1] and extension != 'auto':
            fname = args[0] + '.' + extension
            args = (fname,) + args[1:]

        transparent = kwargs.pop('transparent', False)
        if transparent:
            kwargs.setdefault('facecolor', 'none')
            kwargs.setdefault('edgecolor', 'none')
            original_axes_colors = []
            for ax in self.axes:
                patch = ax.patch
                original_axes_colors.append((patch.get_facecolor(),
                                             patch.get_edgecolor()))
                patch.set_facecolor('none')
                patch.set_edgecolor('none')
        else:
            kwargs.setdefault('facecolor', rcParams['savefig.facecolor'])
            kwargs.setdefault('edgecolor', rcParams['savefig.edgecolor'])

        self.canvas.print_figure(*args, **kwargs)

        if transparent:
            for ax, cc in zip(self.axes, original_axes_colors):
                ax.patch.set_facecolor(cc[0])
                ax.patch.set_edgecolor(cc[1])

    @docstring.dedent_interpd
    def colorbar(self, mappable, cax=None, ax=None, **kw):
        """
        Create a colorbar for a ScalarMappable instance, *mappable*.

        Documentation for the pylab thin wrapper:
        %(colorbar_doc)s
        """
        if ax is None:
            ax = self.gca()
        use_gridspec = kw.pop("use_gridspec", False)
        if cax is None:
            if use_gridspec and isinstance(ax, SubplotBase):
                cax, kw = cbar.make_axes_gridspec(ax, **kw)
            else:
                cax, kw = cbar.make_axes(ax, **kw)
        cax.hold(True)
        cb = cbar.Colorbar(cax, mappable, **kw)

        def on_changed(m):
            #print 'calling on changed', m.get_cmap().name
            cb.set_cmap(m.get_cmap())
            cb.set_clim(m.get_clim())
            cb.update_normal(m)

        self.cbid = mappable.callbacksSM.connect('changed', on_changed)
        mappable.set_colorbar(cb, cax)
        self.sca(ax)
        return cb

    def subplots_adjust(self, *args, **kwargs):
        """
        Call signature::

          subplots_adjust(left=None, bottom=None, right=None, top=None,
                              wspace=None, hspace=None)

        Update the :class:`SubplotParams` with *kwargs* (defaulting to rc when
        *None*) and update the subplot locations

        """
        self.subplotpars.update(*args, **kwargs)
        import matplotlib.axes
        for ax in self.axes:
            if not isinstance(ax, matplotlib.axes.SubplotBase):
                # Check if sharing a subplots axis
                if ax._sharex is not None and isinstance(ax._sharex, matplotlib.axes.SubplotBase):
                    ax._sharex.update_params()
                    ax.set_position(ax._sharex.figbox)
                elif ax._sharey is not None and isinstance(ax._sharey, matplotlib.axes.SubplotBase):
                    ax._sharey.update_params()
                    ax.set_position(ax._sharey.figbox)
            else:
                ax.update_params()
                ax.set_position(ax.figbox)

    def ginput(self, n=1, timeout=30, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2):
        """
        Call signature::

          ginput(self, n=1, timeout=30, show_clicks=True,
                 mouse_add=1, mouse_pop=3, mouse_stop=2)

        Blocking call to interact with the figure.

        This will wait for *n* clicks from the user and return a list of the
        coordinates of each click.

        If *timeout* is zero or negative, does not timeout.

        If *n* is zero or negative, accumulate clicks until a middle click
        (or potentially both mouse buttons at once) terminates the input.

        Right clicking cancels last input.

        The buttons used for the various actions (adding points, removing
        points, terminating the inputs) can be overriden via the
        arguments *mouse_add*, *mouse_pop* and *mouse_stop*, that give
        the associated mouse button: 1 for left, 2 for middle, 3 for
        right.

        The keyboard can also be used to select points in case your mouse
        does not have one or more of the buttons.  The delete and backspace
        keys act like right clicking (i.e., remove last point), the enter key
        terminates input and any other key (not already used by the window
        manager) selects a point.
        """

        blocking_mouse_input = BlockingMouseInput(self, mouse_add =mouse_add,
                                                        mouse_pop =mouse_pop,
                                                        mouse_stop=mouse_stop)
        return blocking_mouse_input(n=n, timeout=timeout,
                                    show_clicks=show_clicks)

    def waitforbuttonpress(self, timeout=-1):
        """
        Call signature::

          waitforbuttonpress(self, timeout=-1)

        Blocking call to interact with the figure.

        This will return True is a key was pressed, False if a mouse
        button was pressed and None if *timeout* was reached without
        either being pressed.

        If *timeout* is negative, does not timeout.
        """

        blocking_input = BlockingKeyMouseInput(self)
        return blocking_input(timeout=timeout)


    def get_default_bbox_extra_artists(self):
        bbox_extra_artists = [t for t in self.texts if t.get_visible()]
        for ax in self.axes:
            if ax.get_visible():
                bbox_extra_artists.extend(ax.get_default_bbox_extra_artists())
        return bbox_extra_artists


    def get_tightbbox(self, renderer):
        """
        Return a (tight) bounding box of the figure in inches.

        It only accounts axes title, axis labels, and axis
        ticklabels. Needs improvement.
        """

        bb = []
        for ax in self.axes:
            if ax.get_visible():
                bb.append(ax.get_tightbbox(renderer))

        _bbox = Bbox.union([b for b in bb if b.width!=0 or b.height!=0])

        bbox_inches = TransformedBbox(_bbox,
                                      Affine2D().scale(1./self.dpi))

        return bbox_inches


    def tight_layout(self, renderer=None, pad=1.2, h_pad=None, w_pad=None):
        """
        Adjust subplot parameters to give specified padding.

        Parameters:

          *pad* : float
            padding between the figure edge and the edges of subplots,
            as a fraction of the font-size.
          *h_pad*, *w_pad* : float
            padding (height/width) between edges of adjacent subplots.
            Defaults to `pad_inches`.
        """

        from tight_layout import auto_adjust_subplotpars, get_renderer

        if renderer is None:
            renderer = get_renderer(self)

        subplotspec_list = []
        subplot_list = []
        nrows_list = []
        ncols_list = []
        ax_bbox_list = []

        subplot_dict = {} # for axes_grid1, multiple axes can share
                          # same subplot_interface. Thus we need to
                          # join them together.

        for ax in self.axes:
            locator = ax.get_axes_locator()
            if hasattr(locator, "get_subplotspec"):
                subplotspec = locator.get_subplotspec().get_topmost_subplotspec()
            elif hasattr(ax, "get_subplotspec"):
                subplotspec = ax.get_subplotspec().get_topmost_subplotspec()
            else:
                continue

            if (subplotspec is None) or \
                   subplotspec.get_gridspec().locally_modified_subplot_params():
                continue

            subplots = subplot_dict.setdefault(subplotspec, [])

            if not subplots:
                myrows, mycols, _, _ = subplotspec.get_geometry()
                nrows_list.append(myrows)
                ncols_list.append(mycols)
                subplotspec_list.append(subplotspec)
                subplot_list.append(subplots)
                ax_bbox_list.append(subplotspec.get_position(self))

            subplots.append(ax)

        max_nrows = max(nrows_list)
        max_ncols = max(ncols_list)

        num1num2_list = []
        for subplotspec in subplotspec_list:
            rows, cols, num1, num2 = subplotspec.get_geometry()
            div_row, mod_row = divmod(max_nrows, rows)
            div_col, mod_col = divmod(max_ncols, cols)
            if (mod_row != 0) or (mod_col != 0):
                raise RuntimeError("")

            rowNum1, colNum1 =  divmod(num1, cols)
            if num2 is None:
                rowNum2, colNum2 =  rowNum1, colNum1
            else:
                rowNum2, colNum2 =  divmod(num2, cols)

            num1num2_list.append((rowNum1*div_row*max_ncols + colNum1*div_col,
                                  ((rowNum2+1)*div_row-1)*max_ncols + (colNum2+1)*div_col-1))


        kwargs = auto_adjust_subplotpars(self, renderer,
                                         nrows_ncols=(max_nrows, max_ncols),
                                         num1num2_list=num1num2_list,
                                         subplot_list=subplot_list,
                                         ax_bbox_list=ax_bbox_list,
                                         pad=pad, h_pad=h_pad, w_pad=w_pad)

        self.subplots_adjust(**kwargs)



def figaspect(arg):
    """
    Create a figure with specified aspect ratio.  If *arg* is a number,
    use that aspect ratio.  If *arg* is an array, figaspect will
    determine the width and height for a figure that would fit array
    preserving aspect ratio.  The figure width, height in inches are
    returned.  Be sure to create an axes with equal with and height,
    eg

    Example usage::

      # make a figure twice as tall as it is wide
      w, h = figaspect(2.)
      fig = Figure(figsize=(w,h))
      ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
      ax.imshow(A, **kwargs)


      # make a figure with the proper aspect for an array
      A = rand(5,3)
      w, h = figaspect(A)
      fig = Figure(figsize=(w,h))
      ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
      ax.imshow(A, **kwargs)

    Thanks to Fernando Perez for this function
    """

    isarray = hasattr(arg, 'shape')


    # min/max sizes to respect when autoscaling.  If John likes the idea, they
    # could become rc parameters, for now they're hardwired.
    figsize_min = np.array((4.0,2.0)) # min length for width/height
    figsize_max = np.array((16.0,16.0)) # max length for width/height
    #figsize_min = rcParams['figure.figsize_min']
    #figsize_max = rcParams['figure.figsize_max']

    # Extract the aspect ratio of the array
    if isarray:
        nr,nc = arg.shape[:2]
        arr_ratio = float(nr)/nc
    else:
        arr_ratio = float(arg)

    # Height of user figure defaults
    fig_height = rcParams['figure.figsize'][1]

    # New size for the figure, keeping the aspect ratio of the caller
    newsize = np.array((fig_height/arr_ratio,fig_height))

    # Sanity checks, don't drop either dimension below figsize_min
    newsize /= min(1.0,*(newsize/figsize_min))

    # Avoid humongous windows as well
    newsize /= max(1.0,*(newsize/figsize_max))

    # Finally, if we have a really funky aspect ratio, break it but respect
    # the min/max dimensions (we don't want figures 10 feet tall!)
    newsize = np.clip(newsize,figsize_min,figsize_max)
    return newsize

docstring.interpd.update(Figure=artist.kwdoc(Figure))
