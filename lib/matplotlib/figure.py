import sys
from artist import Artist
from axes import Axes, Subplot, PolarSubplot, PolarAxes
from cbook import flatten, allequal, dict_delall
import _image
from colors import normalize
from image import FigureImage
from matplotlib import rcParams
from patches import Rectangle
from text import Text, _process_text_args
from legend import Legend
from transforms import Bbox, Value, Point, get_bbox_transform, unit_bbox



class Figure(Artist):
    
    def __init__(self,
                 figsize   = None,  # defaults to rc figure.figsize
                 dpi       = None,  # defaults to rc figure.dpi
                 facecolor = None,  # defaults to rc figure.facecolor
                 edgecolor = None,  # defaults to rc figure.edgecolor
                 linewidth = 1.0,   # the default linewidth of the frame
                 frameon = True,
                 ):
        """
        paper size is a w,h tuple in inches
        DPI is dots per inch 
        """
        Artist.__init__(self)
        #self.set_figure(self)

        if figsize is None  : figsize   = rcParams['figure.figsize']
        if dpi is None      : dpi       = rcParams['figure.dpi']
        if facecolor is None: facecolor = rcParams['figure.facecolor']
        if edgecolor is None: edgecolor = rcParams['figure.edgecolor']
        
        self.dpi = Value(dpi)
        self.figwidth = Value(figsize[0])
        self.figheight = Value(figsize[1])
        self.ll = Point( Value(0), Value(0) )
        self.ur = Point( self.figwidth*self.dpi,
                         self.figheight*self.dpi )
        self.bbox = Bbox(self.ll, self.ur)
        self.frameon = frameon
        
        self.transFigure = get_bbox_transform( unit_bbox(), self.bbox) 


        
        self.figurePatch = Rectangle(
            xy=(0,0), width=1, height=1,
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth,
            )
        self._set_artist_props(self.figurePatch)

        self._hold = rcParams['axes.hold']
        self.clf()

        
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
        """\
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

  * norm is a matplotlib.colors.normalize instance; default is
    normalization().  This scales luminance -> 0-1

  * vmin and vmax are used to scale a luminance image to 0-1.  If
    either is None, the min and max of the luminance values will be
    used.  Note if you pass a norm instance, the settings for vmin and
    vmax will be ignored.

  * alpha = 1.0 : the alpha blending value

  * origin is either 'upper' or 'lower', which indicates where the [0,0]
    index of the array is in the upper left or lower left corner of
    the axes.  Defaults to the rc image.origin value

This complements the axes image which will be resampled to fit the
current axes.  If you want a resampled image to fill the entire
figure, you can define an Axes with size [0,1,0,1].

A image.FigureImage instance is returned.
"""        

        if not self._hold: self.clf()

        im = FigureImage(self, cmap, norm, xo, yo, origin)
        im.set_array(X)
        im.set_alpha(alpha)
        if norm is None:
            im.set_clim(vmin, vmax)
        self.images.append(im )
        return im

        
    def set_figsize_inches(self, *args):
        """
Set the figure size in inches

Usage: set_figsize_inches(self, w,h)  OR
       set_figsize_inches(self, (w,h) )

ACCEPTS: a w,h tuple with w,h in inches
"""
        if len(args)==1:
            w,h = args[0]
        else:
            w,h = args
        self.figwidth.set(w)
        self.figheight.set(h)

    def get_size_inches(self):
        return self.figwidth.get(), self.figheight.get()

    def get_edgecolor(self):
        'Get the edge color of the Figure rectangle' 
        return self.figurePatch.get_edgecolor()

    def get_facecolor(self):
        'Get the face color of the Figure rectangle'
        return self.figurePatch.get_facecolor()

    def set_edgecolor(self, color):
        """
Set the edge color of the Figure rectangle

ACCEPTS: any matplotlib color - see help(colors)"""
        self.figurePatch.set_edgecolor(color)

    def set_facecolor(self, color):
        """
Set the face color of the Figure rectangle

ACCEPTS: any matplotlib color - see help(colors)"""
        self.figurePatch.set_facecolor(color)

    def add_axis(self, *args, **kwargs):
        raise SystemExit("""\
matplotlib changed its axes creation API in 0.54.
Please see http://matplotlib.sourceforge.net/API_CHANGES for
instructions on how to port your code.
""")

        
    def add_axes(self, rect, axisbg=None, frameon=True, **kwargs):
        """
        Add an a axes with axes rect [left, bottom, width, height]
        where all quantities are in fractions of figure width and
        height.

        The Axes instance will be returned
        """
        if axisbg is None: axisbg=rcParams['axes.facecolor']
        ispolar = kwargs.get('polar', False)
        if ispolar:
            a = PolarAxes(self, rect, axisbg, frameon)
        else:
            a = Axes(self, rect, axisbg, frameon)            
        self.axes.append(a)
        return a

    def add_subplot(self, *args, **kwargs):
        """
        Add an a subplot, eg
        add_subplot(111) or add_subplot(212, axisbg='r')

        The Axes instance will be returned
        """
        ispolar = kwargs.get('polar', False)
        dict_delall(kwargs, ('polar', ) )
        if ispolar:
            a = PolarSubplot(self, *args, **kwargs)
        else:
            a = Subplot(self, *args, **kwargs)
        
        self.axes.append(a)

        return a
    
    def clf(self):
        """
        Clear the figure
        """
        self.axes = []
        self.lines = []
        self.patches = []
        self.texts=[]
        self.images = []
        self.legends = []

    def clear(self):
        """
        Clear the figure
        """
        self.clf()
        
    def draw(self, renderer):
        """
        Render the figure using RendererGD instance renderer
        """
        # draw the figure bounding box, perhaps none for white figure
        #print 'figure draw'
        renderer.open_group('figure')
        self.transFigure.freeze()  # eval the lazy objects
        if self.frameon: self.figurePatch.draw(renderer)

        for p in self.patches: p.draw(renderer)
        for l in self.lines: l.draw(renderer)

        if len(self.images)==1:
            im = self.images[0]
            im.draw(renderer)
        elif len(self.images)>1:
            # make a composite image blending alpha
            # list of (_image.Image, ox, oy)
            if not allequal([im.origin for im in self.images]):
                raise ValueError('Composite images with different origins not supported')
            else:
                origin = self.images[0].origin

            ims = [(im.make_image(), im.ox, im.oy) for im in self.images]
            im = _image.from_images(self.bbox.height(), self.bbox.width(), ims)
            im.is_grayscale = False
            l, b, w, h = self.bbox.get_bounds()
            renderer.draw_image(0, 0, im, origin, self.bbox)



        # render the axes
        for a in self.axes: a.draw(renderer)

        # render the figure text
        for t in self.texts: t.draw(renderer)

        for legend in self.legends:
            legend.draw(renderer)

        self.transFigure.thaw()  # release the lazy objects
        renderer.close_group('figure')

    def get_axes(self):
        return self.axes

    def legend(self, handles, labels, loc, **kwargs):
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

          'best' : 0,          (currently not supported, defaults to upper right)
          'upper right'  : 1,  (default)
          'upper left'   : 2,
          'lower left'   : 3,
          'lower right'  : 4,
          'right'        : 5,
          'center left'  : 6,
          'center right' : 7,
          'lower center' : 8,
          'upper center' : 9,
          'center'       : 10,

        The legend instance is returned
        """


        handles = flatten(handles)
        l = Legend(self, handles, labels, loc, isaxes=False, **kwargs)
        self._set_artist_props(l)
        self.legends.append(l)
        return l
    
    def text(self, x, y, s, *args, **kwargs):
        """
        Add text to figure at location x,y (relative 0-1 coords) See
        the help for Axis text for the meaning of the other arguments
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

    def get_width_height(self):
        'return the figure width and height in pixels'
        w = self.bbox.width()
        h = self.bbox.height()
        return w, h
