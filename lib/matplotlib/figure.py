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
import time

import artist
from artist import Artist
from axes import Axes, SubplotBase, subplot_class_factory
from cbook import flatten, allequal, Stack, iterable, dedent
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

        *left*  = 0.125
            the left side of the subplots of the figure
        *right* = 0.9
            the right side of the subplots of the figure
        *bottom* = 0.1
            the bottom of the subplots of the figure
        *top* = 0.9
            the top of the subplots of the figure
        *wspace* = 0.2
            the amount of width reserved for blank space between subplots
        *hspace* = 0.2
            the amount of height reserved for white space between subplots
        *validate*
            make sure the params are in a legal state (*left*<*right*, etc)
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

    The figure patch is drawn by a the attribute

    *patch*
       a :class:`matplotlib.patches.Rectangle` instance

    *suppressComposite*
       for multiple figure images, the figure will make composite
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
                 linewidth = 1.0,   # the default linewidth of the frame
                 frameon = True,    # whether or not to draw the figure frame
                 subplotpars = None, # default to rc
                 ):
        """
        *figsize*
            w,h tuple in inches
        *dpi*
            dots per inch
        *facecolor*
            the figure patch facecolor; defaults to rc ``figure.facecolor``
        *edgecolor*
            the figure patch edge color; defaults to rc ``figure.edgecolor``
        *linewidth*
            the figure patch edge linewidth; the default linewidth of the frame
        *frameon*
            if False, suppress drawing the figure frame
        *subplotpars*
            a :class:`SubplotParams` instance, defaults to rc
        """
        Artist.__init__(self)

        self.callbacks = cbook.CallbackRegistry(('dpi_changed', ))

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

        self._hold = rcParams['axes.hold']
        self.canvas = None

        if subplotpars is None:
            subplotpars = SubplotParams()

        self.subplotpars = subplotpars

        self._axstack = Stack()  # maintain the current axes
        self.axes = []
        self.clf()
        self._cachedRenderer = None

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
            the bottom of the subplots for :meth:`subplots_adjust`
        *rotation*
            the rotation of the xtick labels
        *ha*
            the horizontal alignment of the xticklabels
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

          - *x* = 0.5
              the x location of text in figure coords

          - *y* = 0.98
              the y location of the text in figure coords

          - *horizontalalignment* = 'center'
              the horizontal alignment of the text

          - *verticalalignment* = 'top'
              the vertical alignment of the text

        A :class:`matplotlib.text.Text` instance is returned.

        Example::

          fig.subtitle('this is the figure title', fontsize=12)
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
                 alpha=1.0,
                 norm=None,
                 cmap=None,
                 vmin=None,
                 vmax=None,
                 origin=None):
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
          cmap        a :class:`matplotlib.cm.ColorMap` instance, eg cm.jet.
                      If None, default to the rc ``image.cmap`` value
          norm        a :class:`matplotlib.colors.Normalize` instance. The
                      default is normalization().  This scales luminance -> 0-1
          vmin|vmax   are used to scale a luminance image to 0-1.  If either is
                      None, the min and max of the luminance values will be
                      used.  Note if you pass a norm instance, the settings for
                      *vmin* and *vmax* will be ignored.
          alpha       the alpha blending value, default is 1.0
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

        """

        if not self._hold: self.clf()

        im = FigureImage(self, cmap, norm, xo, yo, origin)
        im.set_array(X)
        im.set_alpha(alpha)
        if norm is None:
            im.set_clim(vmin, vmax)
        self.images.append(im)
        return im

    def set_figsize_inches(self, *args, **kwargs):
        import warnings
        warnings.warn('Use set_size_inches instead!', DeprecationWarning)
        self.set_size_inches(*args, **kwargs)

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

        WARNING: forward=True is broken on all backends except GTK*
        and WX*

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
        self.axes.remove(a)
        self._axstack.remove(a)
        keys = []
        for key, thisax in self._seen.items():
            if a==thisax: del self._seen[key]
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

    def add_axes(self, *args, **kwargs):
        """
        Add an a axes with axes rect [*left*, *bottom*, *width*,
        *height*] where all quantities are in fractions of figure
        width and height.  kwargs are legal
        :class:`~matplotlib.axes.Axes` kwargs plus *projection* which
        sets the projection type of the axes.  (For backward
        compatibility, ``polar=True`` may also be provided, which is
        equivalent to ``projection='polar'``).  Valid values for
        *projection* are: %(list)s.  Some of these projections support
        additional kwargs, which may be provided to :meth:`add_axes`::

            rect = l,b,w,h
            fig.add_axes(rect)
            fig.add_axes(rect, frameon=False, axisbg='g')
            fig.add_axes(rect, polar=True)
            fig.add_axes(rect, projection='polar')
            fig.add_axes(ax)   # add an Axes instance

        If the figure already has an axes with the same parameters,
        then it will simply make that axes current and return it.  If
        you do not want this behavior, eg. you want to force the
        creation of a new axes, you must use a unique set of args and
        kwargs.  The axes :attr:`~matplotlib.axes.Axes.label`
        attribute has been exposed for this purpose.  Eg., if you want
        two axes that are otherwise identical to be added to the
        figure, make sure you give them unique labels::

            fig.add_axes(rect, label='axes1')
            fig.add_axes(rect, label='axes2')

        The :class:`~matplotlib.axes.Axes` instance will be returned.

        The following kwargs are supported:

        %(Axes)s
        """

        key = self._make_key(*args, **kwargs)

        if key in self._seen:
            ax = self._seen[key]
            self.sca(ax)
            return ax

        if not len(args): return
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

        self.axes.append(a)
        self._axstack.push(a)
        self.sca(a)
        self._seen[key] = a
        return a

    add_axes.__doc__ = dedent(add_axes.__doc__) % \
        {'list': (", ".join(get_projection_names())),
         'Axes': artist.kwdocd['Axes']}

    def add_subplot(self, *args, **kwargs):
        """
        Add a subplot.  Examples:

            fig.add_subplot(111)
            fig.add_subplot(1,1,1)            # equivalent but more general
            fig.add_subplot(212, axisbg='r')  # add subplot with red background
            fig.add_subplot(111, polar=True)  # add a polar subplot
            fig.add_subplot(sub)              # add Subplot instance sub

        *kwargs* are legal :class:`!matplotlib.axes.Axes` kwargs plus
        *projection*, which chooses a projection type for the axes.
        (For backward compatibility, *polar=True* may also be
        provided, which is equivalent to *projection='polar'*). Valid
        values for *projection* are: %(list)s.  Some of these projections
        support additional *kwargs*, which may be provided to
        :meth:`add_axes`.

        The :class:`~matplotlib.axes.Axes` instance will be returned.

        If the figure already has a subplot with key (*args*,
        *kwargs*) then it will simply make that subplot current and
        return it.

        The following kwargs are supported:

        %(Axes)s
        """

        kwargs = kwargs.copy()

        if not len(args): return

        if isinstance(args[0], SubplotBase):
            a = args[0]
            assert(a.get_figure() is self)
        else:
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

            key = self._make_key(*args, **kwargs)
            if key in self._seen:
                ax = self._seen[key]
                if isinstance(ax, projection_class):
                    self.sca(ax)
                    return ax
                else:
                    self.axes.remove(ax)
                    self._axstack.remove(ax)

            a = subplot_class_factory(projection_class)(self, *args, **kwargs)
            self._seen[key] = a
        self.axes.append(a)
        self._axstack.push(a)
        self.sca(a)
        return a
    add_subplot.__doc__ = dedent(add_subplot.__doc__) % {
        'list': ", ".join(get_projection_names()),
        'Axes': artist.kwdocd['Axes']}

    def clf(self):
        """
        Clear the figure
        """
        self.suppressComposite = None
        self.callbacks = cbook.CallbackRegistry(('dpi_changed', ))

        for ax in tuple(self.axes):  # Iterate over the copy.
            ax.cla()
            self.delaxes(ax)         # removes ax from self.axes

        toolbar = getattr(self.canvas, 'toolbar', None)
        if toolbar is not None:
            toolbar.update()
        self._axstack.clear()
        self._seen = {}
        self.artists = []
        self.lines = []
        self.patches = []
        self.texts=[]
        self.images = []
        self.legends = []
        self._axobservers = []

    def clear(self):
        """
        Clear the figure -- synonym for fig.clf
        """
        self.clf()

    def draw(self, renderer):
        """
        Render the figure using :class:`matplotlib.backend_bases.RendererBase` instance renderer
        """
        # draw the figure bounding box, perhaps none for white figure
        #print 'figure draw'
        if not self.get_visible(): return
        renderer.open_group('figure')

        if self.frameon: self.patch.draw(renderer)

        # todo: respect zorder
        for p in self.patches: p.draw(renderer)
        for l in self.lines: l.draw(renderer)
        for a in self.artists: a.draw(renderer)

        # override the renderer default if self.suppressComposite
        # is not None
        composite = renderer.option_image_nocomposite()
        if self.suppressComposite is not None:
            composite = self.suppressComposite

        if len(self.images)<=1 or composite or not allequal([im.origin for im in self.images]):
            for im in self.images:
                im.draw(renderer)
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
            clippath, affine = self.get_transformed_clip_path_and_affine()
            renderer.draw_image(l, b, im, self.bbox,
                                clippath, affine)

        # render the axes
        for a in self.axes: a.draw(renderer)

        # render the figure text
        for t in self.texts: t.draw(renderer)

        for legend in self.legends:
            legend.draw(renderer)

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

        The legend instance is returned.  The following kwargs are supported

        *loc*
            the location of the legend
        *numpoints*
            the number of points in the legend line
        *prop*
            a :class:`matplotlib.font_manager.FontProperties` instance
        *pad*
            the fractional whitespace inside the legend border
        *markerscale*
            the relative size of legend markers vs. original
        *shadow*
            if True, draw a shadow behind legend
        *labelsep*
            the vertical space between the legend entries
        *handlelen*
            the length of the legend lines
        *handletextsep*
            the space between the legend line and legend text
        *axespad*
            the border between the axes and legend edge

        .. plot:: mpl_examples/pylab_examples/figlegend_demo.py
        """
        handles = flatten(handles)
        l = Legend(self, handles, labels, *args, **kwargs)
        self.legends.append(l)
        return l

    def text(self, x, y, s, *args, **kwargs):
        """
        Call signature::

          figtext(x, y, s, fontdict=None, **kwargs)

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
    text.__doc__ = dedent(text.__doc__) % artist.kwdocd

    def _set_artist_props(self, a):
        if a!= self:
            a.set_figure(self)
        a.set_transform(self.transFigure)

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
    gca.__doc__ = dedent(gca.__doc__) % artist.kwdocd

    def sca(self, a):
        'Set the current axes to be a and return a'
        self._axstack.bubble(a)
        for func in self._axobservers: func(self)
        return a

    def add_axobserver(self, func):
        'whenever the axes state change, func(self) will be called'
        self._axobservers.append(func)


    def savefig(self, *args, **kwargs):
        """
        call signature::

          savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                  orientation='portrait', papertype=None, format=None,
                  transparent=False):

        Save the current figure.

        The output formats available depend on the backend being used.

        Arguments:

          *fname*:
            A string containing a path to a filename, or a Python file-like object.

            If *format* is *None* and *fname* is a string, the output
            format is deduced from the extension of the filename.

        Keyword arguments:

          *dpi*: [ None | scalar > 0 ]
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
            If *True*, the figure patch and axes patches will all be
            transparent.  This is useful, for example, for displaying
            a plot on top of a colored background on a web page.  The
            transparency of these patches will be restored to their
            original values upon exit of this function.
        """

        for key in ('dpi', 'facecolor', 'edgecolor'):
            if key not in kwargs:
                kwargs[key] = rcParams['savefig.%s'%key]

        transparent = kwargs.pop('transparent', False)
        if transparent:
            original_figure_alpha = self.patch.get_alpha()
            self.patch.set_alpha(0.0)
            original_axes_alpha = []
            for ax in self.axes:
                patch = ax.patch
                original_axes_alpha.append(patch.get_alpha())
                patch.set_alpha(0.0)

        self.canvas.print_figure(*args, **kwargs)

        if transparent:
            self.patch.set_alpha(original_figure_alpha)
            for ax, alpha in zip(self.axes, original_axes_alpha):
                ax.patch.set_alpha(alpha)

    def colorbar(self, mappable, cax=None, ax=None, **kw):
        if ax is None:
            ax = self.gca()
        if cax is None:
            cax, kw = cbar.make_axes(ax, **kw)
        cax.hold(True)
        cb = cbar.Colorbar(cax, mappable, **kw)

        def on_changed(m):
            #print 'calling on changed', m.get_cmap().name
            cb.set_cmap(m.get_cmap())
            cb.set_clim(m.get_clim())
            cb.update_bruteforce(m)

        self.cbid = mappable.callbacksSM.connect('changed', on_changed)
        mappable.set_colorbar(cb, cax)
        self.sca(ax)
        return cb
    colorbar.__doc__ =  '''
        Create a colorbar for a ScalarMappable instance.

        Documentation for the pylab thin wrapper:
        %s

        '''% cbar.colorbar_doc

    def subplots_adjust(self, *args, **kwargs):
        """
        fig.subplots_adjust(left=None, bottom=None, right=None, wspace=None, hspace=None)

        Update the :class:`SubplotParams` with *kwargs* (defaulting to rc where
        None) and update the subplot locations

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

    def ginput(self, n=1, timeout=30, show_clicks=True):
        """
        call signature::

          ginput(self, n=1, timeout=30, show_clicks=True)

        Blocking call to interact with the figure.

        This will wait for *n* clicks from the user and return a list of the
        coordinates of each click.

        If *timeout* is zero or negative, does not timeout.

        If *n* is zero or negative, accumulate clicks until a middle click
        (or potentially both mouse buttons at once) terminates the input.

        Right clicking cancels last input.

        The keyboard can also be used to select points in case your mouse
        does not have one or more of the buttons.  The delete and backspace
        keys act like right clicking (i.e., remove last point), the enter key
        terminates input and any other key (not already used by the window
        manager) selects a point.
        """

        blocking_mouse_input = BlockingMouseInput(self)
        return blocking_mouse_input(n=n, timeout=timeout,
                                    show_clicks=show_clicks)

    def waitforbuttonpress(self, timeout=-1):
        """
        call signature::

          waitforbuttonpress(self, timeout=-1)

        Blocking call to interact with the figure.

        This will return True is a key was pressed, False if a mouse
        button was pressed and None if *timeout* was reached without
        either being pressed.

        If *timeout* is negative, does not timeout.
        """

        blocking_input = BlockingKeyMouseInput(self)
        return blocking_input(timeout=timeout)


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

artist.kwdocd['Figure'] = artist.kwdoc(Figure)
