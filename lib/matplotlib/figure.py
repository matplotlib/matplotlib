"""
Figure class -- add docstring here!
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

        The following attributes are available:

        left : the left side of the subplots of the figure
        right : the right side of the subplots of the figure
        bottom : the bottom of the subplots of the figure
        top : the top of the subplots of the figure
        wspace : the amount of width reserved for blank space between subplots
        hspace : the amount of height reserved for white space between subplots

        validate : make sure the params are in a legal state
        (left<right, etc)
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


class BlockingMouseInput(object):
    """ Class that creates a callable object to retrieve mouse clicks in a
        blocking way.
    """
    def __init__(self, fig):
        self.fig = fig


    def on_click(self, event):
        """ Event handler that will be passed to the current figure to
            retrieve clicks.
        """
        if event.button == 3:
            # If it's a right click, pop the last coordinates.
            if len(self.clicks) > 0:
                self.clicks.pop()
                if self.show_clicks:
                    mark = self.marks.pop()
                    mark.remove()
                    self.fig.canvas.draw()
        elif event.button == 2 and self.n < 0:
            # If it's a middle click, and we are in infinite mode, finish
            self.done = True
        elif event.inaxes:
            # If it's a valid click, append the coordinates to the list
            self.clicks.append((event.xdata, event.ydata))
            if self.verbose:
                print "input %i: %f,%f" % (len(self.clicks),
                                    event.xdata, event.ydata)
            if self.show_clicks:
                self.marks.extend(
                    event.inaxes.plot([event.xdata,], [event.ydata,], 'r+') )
                self.fig.canvas.draw()
            if self.n > 0 and len(self.clicks) >= self.n:
                self.done = True


    def __call__(self, n=1, timeout=30, verbose=False, show_clicks=True):
        """ Blocking call to retrieve n coordinate pairs through mouse
            clicks.
        """
        self.verbose     = verbose
        self.done        = False
        self.clicks      = []
        self.show_clicks = True
        self.marks       = []

        assert isinstance(n, int), "Requires an integer argument"
        self.n = n

        # Ensure that the figure is shown
        self.fig.show()
        # connect the click events to the on_click function call
        self.callback = self.fig.canvas.mpl_connect('button_press_event',
                                                    self.on_click)
        # wait for n clicks
        counter = 0
        while not self.done:
            self.fig.canvas.flush_events()
            time.sleep(0.01)

            # check for a timeout
            counter += 1
            if timeout > 0 and counter > timeout/0.01:
                print "ginput timeout";
                break;

        # Disconnect the event, clean the figure, and return what we have
        self.fig.canvas.mpl_disconnect(self.callback)
        self.callback = None
        if self.show_clicks:
            for mark in self.marks:
                mark.remove()
            self.fig.canvas.draw()
        return self.clicks


class Figure(Artist):

    """
    The Figure instance supports callbacks through a callbacks
    attribute which is a cbook.CallbackRegistry instance.  The events
    you can connect to are 'dpi_changed', and the callback will be
    called with func(fig) where fig is the Figure instance
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
        figsize is a w,h tuple in inches
        dpi is dots per inch
        subplotpars is a SubplotParams instance, defaults to rc
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

        self.figurePatch = Rectangle(
            xy=(0,0), width=1, height=1,
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth,
            )
        self._set_artist_props(self.figurePatch)

        self._hold = rcParams['axes.hold']
        self.canvas = None

        if subplotpars is None:
            subplotpars = SubplotParams()

        self.subplotpars = subplotpars

        self._axstack = Stack()  # maintain the current axes
        self.axes = []
        self.clf()

        self._cachedRenderer = None
        self._autoLayout = rcParams['figure.autolayout']

    def _get_dpi(self):
        return self._dpi
    def _set_dpi(self, dpi):
        self._dpi = dpi
        self.dpi_scale_trans.clear().scale(dpi, dpi)
        self.callbacks.process('dpi_changed', self)
    dpi = property(_get_dpi, _set_dpi)

    def enable_auto_layout(self, setting=True):
        self._autoLayout = setting

    def autofmt_xdate(self, bottom=0.2, rotation=30, ha='right'):
        """
        Date ticklabels often overlap, so it is useful to rotate them
        and right align them.  Also, a common use case is a number of
        subplots with shared xaxes where the x-axis is date data.  The
        ticklabels are often long, and it helps to rotate them on the
        bottom subplot and turn them off on other subplots, as well as
        turn off xlabels.


        bottom : the bottom of the subplots for subplots_adjust
        rotation: the rotation of the xtick labels
        ha : the horizontal alignment of the xticklabels
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

        if allsubplots and not self._autoLayout:
            self.subplots_adjust(bottom=bottom)

    def get_children(self):
        'get a list of artists contained in the figure'
        children = [self.figurePatch]
        children.extend(self.axes)
        children.extend(self.lines)
        children.extend(self.patches)
        children.extend(self.texts)
        children.extend(self.images)
        children.extend(self.legends)
        return children

    def contains(self, mouseevent):
        """Test whether the mouse event occurred on the figure.

        Returns True,{}
        """
        if callable(self._contains): return self._contains(self,mouseevent)
        #inside = mouseevent.x >= 0 and mouseevent.y >= 0
        inside = self.bbox.contains(mouseevent.x,mouseevent.y)

        return inside,{}

    def get_window_extent(self, *args, **kwargs):
        'get the figure bounding box in display space; kwargs are void'
        return self.bbox

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

        Eg
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
        FIGIMAGE(X) # add non-resampled array to figure

        FIGIMAGE(X, xo, yo) # with pixel offsets

        FIGIMAGE(X, **kwargs) # control interpolation ,scaling, etc

        Add a nonresampled figure to the figure from array X.  xo and yo are
        offsets in pixels

        X must be a float array

            If X is MxN, assume luminance (grayscale)
            If X is MxNx3, assume RGB
            If X is MxNx4, assume RGBA

        The following kwargs are allowed:

          * cmap is a cm colormap instance, eg cm.jet.  If None, default to
            the rc image.cmap valuex

          * norm is a matplotlib.colors.Normalize instance; default is
            normalization().  This scales luminance -> 0-1

          * vmin and vmax are used to scale a luminance image to 0-1.  If
            either is None, the min and max of the luminance values will be
            used.  Note if you pass a norm instance, the settings for vmin and
            vmax will be ignored.

          * alpha = 1.0 : the alpha blending value

          * origin is either 'upper' or 'lower', which indicates where the [0,0]
            index of the array is in the upper left or lower left corner of
            the axes.  Defaults to the rc image.origin value

        This complements the axes image (Axes.imshow) which will be resampled
        to fit the current axes.  If you want a resampled image to fill the
        entire figure, you can define an Axes with size [0,1,0,1].

        A image.FigureImage instance is returned.
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

        Usage: set_size_inches(self, w,h)  OR
               set_size_inches(self, (w,h) )

        optional kwarg forward=True will cause the canvas size to be
        automatically updated; eg you can resize the figure window
        from the shell

        WARNING: forward=True is broken on all backends except GTK*

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
        return self.figurePatch.get_edgecolor()

    def get_facecolor(self):
        'Get the face color of the Figure rectangle'
        return self.figurePatch.get_facecolor()

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
        self.figurePatch.set_edgecolor(color)

    def set_facecolor(self, color):
        """
        Set the face color of the Figure rectangle

        ACCEPTS: any matplotlib color - see help(colors)
        """
        self.figurePatch.set_facecolor(color)

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
        Add an a axes with axes rect [left, bottom, width, height] where all
        quantities are in fractions of figure width and height.  kwargs are
        legal Axes kwargs plus "projection" which sets the projection type
        of the axes.  (For backward compatibility, polar=True may also be
        provided, which is equivalent to projection='polar').
        Valid values for "projection" are: %s.  Some of these projections
        support additional kwargs, which may be provided to add_axes.

            rect = l,b,w,h
            add_axes(rect)
            add_axes(rect, frameon=False, axisbg='g')
            add_axes(rect, polar=True)
            add_axes(rect, projection='polar')
            add_axes(ax)   # add an Axes instance


        If the figure already has an axes with key *args, *kwargs then it will
        simply make that axes current and return it.  If you do not want this
        behavior, eg you want to force the creation of a new axes, you must
        use a unique set of args and kwargs.  The artist "label" attribute has
        been exposed for this purpose.  Eg, if you want two axes that are
        otherwise identical to be added to the figure, make sure you give them
        unique labels:

            add_axes(rect, label='axes1')
            add_axes(rect, label='axes2')

        The Axes instance will be returned

        The following kwargs are supported:
        %s
        """ % (", ".join(get_projection_names()), '%(Axes)s')

        key = self._make_key(*args, **kwargs)

        if self._seen.has_key(key):
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

    add_axes.__doc__ = dedent(add_axes.__doc__) % artist.kwdocd

    def add_subplot(self, *args, **kwargs):
        """
        Add a subplot.  Examples

            add_subplot(111)
            add_subplot(1,1,1)            # equivalent but more general
            add_subplot(212, axisbg='r')  # add subplot with red background
            add_subplot(111, polar=True)  # add a polar subplot
            add_subplot(sub)              # add Subplot instance sub

        kwargs are legal Axes kwargs plus "projection", which chooses
        a projection type for the axes.  (For backward compatibility,
        polar=True may also be provided, which is equivalent to
        projection='polar').  Valid values for "projection" are: %s.
        Some of these projections support additional kwargs, which may
        be provided to add_axes.

        The Axes instance will be returned.

        If the figure already has a subplot with key *args, *kwargs then it will
        simply make that subplot current and return it

        The following kwargs are supported:
        %s
        """ % (", ".join(get_projection_names()), "%(Axes)s")

        key = self._make_key(*args, **kwargs)
        if self._seen.has_key(key):
            ax = self._seen[key]
            self.sca(ax)
            return ax


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
            a = subplot_class_factory(projection_class)(self, *args, **kwargs)

        self.axes.append(a)
        self._axstack.push(a)
        self.sca(a)
        self._seen[key] = a
        return a
    add_subplot.__doc__ = dedent(add_subplot.__doc__) % artist.kwdocd

    def clf(self):
        """
        Clear the figure
        """

        self.callbacks = cbook.CallbackRegistry(('dpi_changed', ))

        for ax in tuple(self.axes):  # Iterate over the copy.
            ax.cla()
            self.delaxes(ax)         # removes ax from self.axes

        toolbar = getattr(self.canvas, 'toolbar', None)
        if toolbar is not None:
            toolbar.update()
        self._axstack.clear()
        self._seen = {}
        self.lines = []
        self.patches = []
        self.texts=[]
        self.images = []
        self.legends = []
        self._axobservers = []

    def clear(self):
        """
        Clear the figure
        """
        self.clf()

    def draw(self, renderer):
        """
        Render the figure using Renderer instance renderer
        """
        # draw the figure bounding box, perhaps none for white figure
        #print 'figure draw'
        if not self.get_visible(): return
        renderer.open_group('figure')

        if self.frameon: self.figurePatch.draw(renderer)

        for p in self.patches: p.draw(renderer)
        for l in self.lines: l.draw(renderer)

        if len(self.images)<=1 or renderer.option_image_nocomposite() or not allequal([im.origin for im in self.images]):
            for im in self.images:
                im.draw(renderer)
        else:
            # make a composite image blending alpha
            # list of (_image.Image, ox, oy)

            mag = renderer.get_image_magnification()
            ims = [(im.make_image(mag), im.ox*mag, im.oy*mag)
                   for im in self.images]
            im = _image.from_images(self.bbox.height * mag,
                                    self.bbox.width * mag,
                                    ims)
            im.is_grayscale = False
            l, b, w, h = self.bbox.bounds
            renderer.draw_image(l, b, im, self.bbox,
                                *self.get_transformed_clip_path_and_affine())

        # update the positions of the axes
        # This gives each of the axes the opportunity to resize itself
        # based on the tick and axis labels etc., and then makes sure
        # that any axes that began life aligned to another axes remains
        # aligned after these adjustments
        if self._autoLayout and len(self.axes) > 1:
            aligned_positions = [{}, {}, {}, {}]
            sizes = [{}, {}]
            for a in self.axes:
                a.update_layout(renderer)
                orig_pos = a.get_position(True)
                curr_pos = a.get_position()
                for pos, orig, curr in zip(aligned_positions,
                                           orig_pos.get_points().flatten(),
                                           curr_pos.get_points().flatten()):
                    if orig in pos:
                        pos[orig][0].append(a)
                        pos[orig][1].add(curr)
                    else:
                        pos[orig] = [[a], set([curr])]
                for size, orig, curr in zip(sizes,
                                            orig_pos.size,
                                            curr_pos.size):
                    orig = round(orig * 1000.0) / 1000.0
                    if orig in size:
                        size[orig][0].append(a)
                        size[orig][1].add(curr)
                    else:
                        size[orig] = [[a], set([curr])]

            for i, pos in enumerate(aligned_positions):
                for axes, places in pos.values():
                    if len(places) > 1:
                        if i < 2:
                            curr = max(places)
                        else:
                            curr = min(places)
                        for a in axes:
                            curr_pos = a.get_position().frozen()
                            curr_pos.get_points()[i/2, i%2] = curr
                            a.set_position(curr_pos, 'active')

            for i, size in enumerate(sizes):
                for axes, dims in size.values():
                    new = min(dims)
                    for a in axes:
                        curr_pos = a.get_position().frozen()
                        curr = curr_pos.size[i]
                        if curr > new:
                            extra = (curr - new) * 0.5
                            curr_pos.get_points()[0, i] += extra
                            curr_pos.get_points()[1, i] -= extra
                            a.set_position(curr_pos, 'active')
        elif self._autoLayout:
            for a in self.axes: a.update_layout(renderer)

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
        'draw artist only -- this is available only after the figure is drawn'
        assert self._cachedRenderer is not None
        a.draw(self._cachedRenderer)

    def get_axes(self):
        return self.axes

    def legend(self, handles, labels, *args, **kwargs):
        """
        Place a legend in the figure.  Labels are a sequence of
        strings, handles is a sequence of line or patch instances, and
        loc can be a string or an integer specifying the legend
        location

        USAGE:
          legend( (line1, line2, line3),
                  ('label1', 'label2', 'label3'),
                  'upper right')

        The LOC location codes are

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

        loc can also be an (x,y) tuple in figure coords, which
        specifies the lower left of the legend box.  figure coords are
        (0,0) is the left, bottom of the figure and 1,1 is the right,
        top.

        The legend instance is returned.  The following kwargs are supported:

        loc = "upper right" #
        numpoints = 4         # the number of points in the legend line
        prop = FontProperties(size='smaller')  # the font property
        pad = 0.2             # the fractional whitespace inside the legend border
        markerscale = 0.6     # the relative size of legend markers vs. original
        shadow                # if True, draw a shadow behind legend
        labelsep = 0.005     # the vertical space between the legend entries
        handlelen = 0.05     # the length of the legend lines
        handletextsep = 0.02 # the space between the legend line and legend text
        axespad = 0.02       # the border between the axes and legend edge

        """


        handles = flatten(handles)
        l = Legend(self, handles, labels, *args, **kwargs)
        self.legends.append(l)
        return l

    def text(self, x, y, s, *args, **kwargs):
        """
        Add text to figure at location x,y (relative 0-1 coords) See
        the help for Axis text for the meaning of the other arguments

        kwargs control the Text properties:
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
        if ax is not None: return ax
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
        SAVEFIG(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None):

        Save the current figure.

        fname - the filename to save the current figure to.  The
                output formats supported depend on the backend being
                used.  and are deduced by the extension to fname.
                Possibilities are eps, jpeg, pdf, png, ps, svg.  fname
                can also be a file or file-like object - cairo backend
                only.

        dpi - is the resolution in dots per inch.  If
              None it will default to the value savefig.dpi in the
              matplotlibrc file

        facecolor and edgecolor are the colors of the figure rectangle

        orientation is either 'landscape' or 'portrait' - not supported on
        all backends; currently only on postscript output

        papertype is is one of 'letter', 'legal', 'executive', 'ledger', 'a0'
        through 'a10', or 'b0' through 'b10' - only supported for postscript
        output

        format - one of the file extensions supported by the active backend.
        """

        for key in ('dpi', 'facecolor', 'edgecolor'):
            if not kwargs.has_key(key):
                kwargs[key] = rcParams['savefig.%s'%key]

        self.canvas.print_figure(*args, **kwargs)

    def colorbar(self, mappable, cax=None, ax=None, **kw):
        if ax is None:
            ax = self.gca()
        if cax is None:
            cax, kw = cbar.make_axes(ax, **kw)
        cax.hold(True)
        cb = cbar.Colorbar(cax, mappable, **kw)
        mappable.add_observer(cb)
        mappable.set_colorbar(cb, cax)
        self.sca(ax)
        return cb
    colorbar.__doc__ =  '''
        Create a colorbar for a ScalarMappable instance.

        Documentation for the pylab thin wrapper: %s
        '''% cbar.colorbar_doc

    def subplots_adjust(self, *args, **kwargs):
        """
        subplots_adjust(self, left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=None)
        fig.subplots_adjust(left=None, bottom=None, right=None, wspace=None, hspace=None):
        Update the SubplotParams with kwargs (defaulting to rc where
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

    def ginput(self, n=1, timeout=30, verbose=False, show_clicks=True):
        """
        ginput(self, n=1, timeout=30, verbose=False, show_clicks=True)

        Blocking call to interact with the figure.

        This will wait for n clicks from the user and return a list of the
        coordinates of each click. If timeout is negative, does not
        timeout. If n is negative, accumulate clicks until a middle
        click terminates the input. Right clicking cancels last input.
        """

        blocking_mouse_input = BlockingMouseInput(self)
        return blocking_mouse_input(n=n, timeout=timeout,
                                          verbose=verbose, show_clicks=True)



def figaspect(arg):
    """
    Create a figure with specified aspect ratio.  If arg is a number,
    use that aspect ratio.  If arg is an array, figaspect will
    determine the width and height for a figure that would fit array
    preserving aspect ratio.  The figure width, height in inches are
    returned.  Be sure to create an axes with equal with and height,
    eg

    Example usage:

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
