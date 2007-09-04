"""
Install instructions for traits 2.0

   # blow away old enthought
   rm -rf ~/dev/lib/python2.4/site-packages/enthought.*

   # get easy_install, if necessary
   wget  http://peak.telecommunity.com/dist/ez_setup.py
   sudo python sez_setup.py


    sudo rm -rf /usr/local/lib/python2.5/site-packages/enthought*
    sudo easy_install \
      -f http://code.enthought.com/enstaller/eggs/source/unstable \
      "enthought.traits < 3.0a"
    
"""

# see install instructions for enthrought traits2 in mtraits
import enthought.traits.api as traits
from enthought.traits.api import HasTraits, Instance, Trait, Float, Int, \
     Array, Tuple
from enthought.traits.trait_numeric import TraitArray
from matplotlib import agg
from matplotlib import colors as mcolors
from matplotlib import cbook
import numpy as npy

is_string_like = cbook.is_string_like

## begin core infrastructure


                    
class Affine(HasTraits):
    """
    An affine 3x3 matrix that supports matrix multiplication with
    other Affine instances or numpy arrays.

    a = Affine()
    a.translate = 10,20
    a.scale = 20, 40

    Be careful not to do *inplace* operations on the array components
    or the update callbacks will not be triggered, eg DO NOT 

      a.translate += 10, 20

    rather DO

      a.translate_delta(10, 20)

    Multiplication works as expected:

      a1 = Affine()
      a1.scale = 10, 20
      a2 = Affine()
      a2.scale = 4, 5
      print a1*a2

      x = numpy.random(3, 10)
      print a1*x

    All of the translate, scale, xlim, ylim and vec6 properties are
    simply views into the data matrix, and are updated by reference
    """
    # connect to the data_modified event if you want a callback
    data = Array('d', (3,3))

    translate = traits.Property(Array('d', (2,)))
    scale = traits.Property(Array('d', (2,)))    
    vec6 = traits.Property(Array('d', (6,)))

    xlim = traits.Property(Array('d', (2,)))
    ylim = traits.Property(Array('d', (2,)))            

    #data_modified = traits.Event 

    def _data_default(self):
        return npy.array([[1,0,0],[0,1,0],[0,0,1]], npy.float_)


    def _get_xlim(self):
        sx, b, tx = self.data[0]

        return self._get_lim(sx, tx)

    def _set_xlim(self, xlim):
        xmin, xmax = xlim

        oldsx, oldb, oldtx = self.data[0]
        
        sx = 1./(xmax-xmin)
        tx = -xmin*sx

        forward = oldsx!=sx or oldtx!=tx
        if forward:
            old = self.data.copy()
            self.data[0][0] = sx
            self.data[0][-1] = tx
            self._data_changed(old, self.data)

    def _get_ylim(self):
        c, sy, ty = self.data[1]

        return self._get_lim(sy, ty)

    def _set_ylim(self, ylim):
        ymin, ymax = ylim

        oldc, oldsy, oldty = self.data[1]
        
        sy = 1./(ymax-ymin)
        ty = -ymin*sy

        forward = oldsy!=sy or oldty!=ty
        if forward:
            old = self.data.copy()
            self.data[1][1] = sy
            self.data[1][-1] = ty
            self._data_changed(old, self.data)
            

    def _get_translate(self):
        return [self.data[0][-1], self.data[1][-1]]

    def _set_translate(self, s):
        oldtx = self.data[0][-1]
        oldty = self.data[1][-1]        
        
        tx, ty = s

        forward = tx!=oldtx or ty!=oldty
        if forward:
            old = self.data.copy()
            self.data[0][-1] = tx
            self.data[1][-1] = ty
            self._data_changed(old, self.data)

    def _get_scale(self):
        return [self.data[0][0], self.data[1][1]]

    def _set_scale(self, s):
        oldsx = self.data[0][0]
        oldsy = self.data[1][1]        
        
        sx, sy = s

        forward = sx!=oldsx or sy!=oldsy
        if forward:
            old = self.data.copy()
            self.data[0][0] = sx
            self.data[1][1] = sy
            self._data_changed(old, self.data)
            
    def _get_vec6(self):
        a,b,tx = self.data[0]
        c,d,ty = self.data[1]
        return [a,b,c,d,tx,ty]

    def _set_vec6(self, v):
        a,b,c,d,tx,ty = v

        olda, oldb, oldtx = self.data[0]
        oldc, oldd, oldty = self.data[1]

        forward = a!=olda or b!=oldb or c!=oldc or d!=oldd or tx!=oldtx or ty!=oldty
        if forward:
            old = self.data.copy()
            self.data[0] = a,b,tx
            self.data[1] = c,d,ty
            self._data_changed(old, self.data)
            

    def _get_lim(self, s, t):
        lmin = -t/s
        lmax = 1./s + lmin
        return lmin, lmax
    
    def _data_changed(self, old, new):
        # Make it known if the translate changed
        oldtx, oldty = old[0][-1], old[1][-1]
        tx, ty = new[0][-1], new[1][-1]

        oldsx, oldsy = old[0][0], old[1][1]
        sx, sy = new[0][0], new[1][1]


        oldb, oldc = old[0][1], old[1][0]
        b, c = new[0][1], new[1][0]

        tchanged = False
        schanged = False
        vchanged = False

        tchanged = oldtx!=tx or oldty!=ty
        schanged = oldsx!=sx or oldsy!=sy
        vchanged = tchanged or schanged or b!=oldb or c!=oldc
        xchanged = oldtx!=tx or oldsx!=sx
        ychanged = oldty!=ty or oldsy!=sy        
        
        if tchanged:
            self.trait_property_changed('translate', [oldtx, oldty], [tx, ty])

        if schanged:
            self.trait_property_changed('scale', [oldsx, oldsy], [sx, sy])

        if xchanged:
            oldxmin, oldxmax = self._get_lim(oldsx, oldtx)
            xmin, xmax = self._get_lim(sx, tx)            
            self.trait_property_changed('xlim', [oldxmin, oldxmax], [xmin, xmax])

        if ychanged:
            oldymin, oldymax = self._get_lim(oldsy, oldty)
            ymin, ymax = self._get_lim(sy, ty)            
            self.trait_property_changed('ylim', [oldymin, oldymax], [ymin, ymax])

        if vchanged:
            self.trait_property_changed(
                'vec6',
                [oldsx, oldb, oldc, oldsy, oldtx, oldty],
                [sx, b, c, sy, tx, ty])
            
        if tchanged or schanged or vchanged:
            #self._data_modified = True            
            self.trait_property_changed('data_modified', old, new)

        
    def follow(self, othervec6):
        self.vec6 = othervec6

        
    def __mul__(self, other):
        if isinstance(other, Affine):
            new = Affine()
            new.data = npy.dot(self.data, other.data)
            return new
        elif isinstance(other, npy.ndarray):
            return npy.dot(self.data, other)
        raise TypeError('Do not know how to multiply Affine by %s'%type(other))
                        

    def __repr__(self):
        return 'AFFINE: %s'%', '.join([str(val) for val in self.vec6])
        #return 'AFFINE:\n%s'%self.data


class Box(HasTraits):
    # left, bottom, width, height
    bounds = traits.Array('d', (4,))
    left = traits.Property(Float)
    bottom = traits.Property(Float)
    width = traits.Property(Float)  
    height = traits.Property(Float) 

    right = traits.Property(Float) # read only
    top = traits.Property(Float)   # read only

    def _bounds_default(self):
        return  [0.0, 0.0, 1.0, 1.0]

    def _get_left(self):
        return self.bounds[0]

    def _set_left(self, left):
        oldbounds = self.bounds[:]        
        self.bounds[0] = left
        self.trait_property_changed('bounds', oldbounds, self.bounds)

    def _get_bottom(self):
        return self.bounds[1]

    def _set_bottom(self, bottom):
        oldbounds = self.bounds[:]        
        self.bounds[1] = bottom
        self.trait_property_changed('bounds', oldbounds, self.bounds)

    def _get_width(self):
        return self.bounds[2]

    def _set_width(self, width):
        oldbounds = self.bounds[:]        
        self.bounds[2] = width
        self.trait_property_changed('bounds', oldbounds, self.bounds)

    def _get_height(self):
        return self.bounds[2]

    def _set_height(self, height):
        oldbounds = self.bounds[:]        
        self.bounds[2] = height
        self.trait_property_changed('bounds', oldbounds, self.bounds)

    def _get_right(self):
        return self.left + self.width

    def _get_top(self):
        return self.bottom + self.height

    def _bounds_changed(self, old, new):
        pass


## begin custom trait handlers
    
class TraitVertexArray(TraitArray):

    def __init__ ( self, typecode = None, shape = None, coerce = False ):
        TraitArray.__init__(self, typecode, shape, coerce)

    def validate(self, object, name, value):
        orig = value
        value = TraitArray.validate(self, object, name, value)
        if len(value.shape)!=2 or value.shape[1]!=2:
            return self.error(object, name, orig)

        return value

    def info(self):
        return 'an Nx2 array of doubles which are x,y vertices'

VertexArray = Trait(npy.array([[0,0], [1,1]], npy.float_),
                    TraitVertexArray('d'))
    
class ColorHandler(traits.TraitHandler):
    """
    This is a clever little traits mechanism -- users can specify the
    color as any mpl color, and the traited object will keep the
    original color, but will add a new attribute with a '_' postfix
    which is the color rgba tuple.

    Eg

    class C(HasTraits):
        facecolor = Trait('black', ColorHandler())

    c = C()
    c.facecolor = 'red'
    print c.facecolor    # prints red
    print c.facecolor_   # print (1,0,0,1)
    """
    is_mapped = True

    def post_setattr(self, object, name, value):
        object.__dict__[ name + '_' ] = self.mapped_value( value )

    def mapped_value(self, value ):
        if value is None: return None
        if is_string_like(value): value = value.lower()
        return mcolors.colorConverter.to_rgba(value)
       
    def validate(self, object, name, value):
        try:
            self.mapped_value(value)
        except ValueError:
            return self.error(object, name, value)
        else:            
            return value

    def info(self):
        return """\
any valid matplotlib color, eg an abbreviation like 'r' for red, a full
name like 'orange', a hex color like '#efefef', a grayscale intensity
like '0.5', or an RGBA tuple (1,0,0,1)"""


    


class MTraitsNamespace:
    DPI         = Float(72.)

    Alpha       = traits.Range(0., 1., 0.)
    Affine      = Trait(Affine())
    AntiAliased = traits.true
    Color       = Trait('black', ColorHandler())
    DPI         = Float(72.)
    Interval    = Array('d', (2,), npy.array([0.0, 1.0], npy.float_))
    LineStyle   = Trait('-', '--', '-.', ':', 'steps', None)
    LineWidth   = Float(1.0)
    Marker      = Trait(None, '.', ',', 'o', '^', 'v', '<', '>', 's',
                 '+', 'x', 'd', 'D', '|', '_', 'h', 'H',
                 'p', '1', '2', '3', '4')
    MarkerSize  = Float(6)
    Visible     = traits.true


mtraits = MTraitsNamespace()

def Alias(name):
    return Property(lambda obj: getattr(obj, name),
                    lambda obj, val: setattr(obj, name, val))


class IDGenerator:
    def __init__(self):
        self._id = 0

    def __call__(self):
        _id = self._id
        self._id += 1
        return _id




## begin backend API
# PATH CODES
STOP      = 0
MOVETO    = 1
LINETO    = 2
CURVE3    = 3
CURVE4    = 4
CURVEN    = 5
CATROM    = 6
UBSPLINE  = 7
CLOSEPOLY = 0x0F

class PathPrimitive(HasTraits):
    """
    The path is an object that talks to the backends, and is an
    intermediary between the high level path artists like Line and
    Polygon, and the backend renderer
    """

    
    color       = mtraits.Color('black')
    facecolor   = mtraits.Color('blue')
    alpha       = mtraits.Alpha(1.0)
    linewidth   = mtraits.LineWidth(1.0)
    antialiased = mtraits.AntiAliased
    pathdata     =Tuple(Array('b'), VertexArray)
    affine      = Instance(Affine, ())

    def _pathdata_default(self):
        return (npy.array([0,0], dtype=npy.uint8),
                npy.array([[0,0],[0,0]], npy.float_))
        

class MarkerPrimitive(HasTraits):
    locs  = Array('d')
    path   = Instance(PathPrimitive, ())       # marker path in points
    affine = Instance(Affine, ()) # transformation for the verts

    def _locs_default(self):
        return npy.array([[0,0],[0,0]], npy.float_)

class Renderer(HasTraits):
    dpi = mtraits.DPI
    size = traits.Tuple(Int(600), Int(400))

    adisplay = Instance(Affine, ()) 
    pathd = traits.Dict(Int, PathPrimitive)
    markerd = traits.Dict(Int, MarkerPrimitive)
    
    def __init__(self, size=(600,400)):
        self.pathd = dict()
        self.markerd = dict()
        self._size_changed(None, size)
        
    def _size_changed(self, old, new):
        width, height = new

        # almost all renderers assume 0,0 is left, upper, so we'll flip y here by default
        self.adisplay.translate = 0, height
        self.adisplay.scale = width, -height
        
        

    def render_path(self, pathid):
        pass
            

    def new_path_primitive(self):
        """
        return a PathPrimitive (or derived); these instances will be
        added and removed later through add_path and remove path
        """
        return PathPrimitive()

    def new_marker_primitive(self):
        """
        return a MarkerPrimitive (or derived); these instances will be
        added and removed later through add_maker and remove_marker
        """
        return MarkerPrimitive()
    

## begin backend agg    
class PathPrimitiveAgg(PathPrimitive):

    def __init__(self):
        self._pathdata_changed(None, self.pathdata)
        self._facecolor_changed(None, self.facecolor)
        self._color_changed(None, self.color)                

        
    @staticmethod
    def make_agg_path(pathdata):
        agg_path = agg.path_storage()

        codes, xy = pathdata

        Ncodes = len(codes)

        for i in range(Ncodes):
            x, y = xy[i]
            code = codes[i]
            #XXX handle other path codes here
            if code==MOVETO:
                agg_path.move_to(x, y)
            elif code==LINETO:
                agg_path.line_to(x, y)                
            elif code==CLOSEPOLY:
                agg_path.close_polygon()
        return agg_path

    def _pathdata_changed(self, olddata, newdata):        
        self.agg_path = PathPrimitiveAgg.make_agg_path(newdata)

        
    def _facecolor_changed(self, oldcolor, newcolor):        
        self.agg_facecolor = self.color_to_rgba8(self.facecolor_)

    def _color_changed(self, oldcolor, newcolor):                
        #print 'stroke color changed', newcolor
        c = self.color_to_rgba8(self.color_)
        self.agg_color = c

    def color_to_rgba8(self, color):
        if color is None: return None
        rgba = [int(255*c) for c in color]
        return agg.rgba8(*rgba)

class MarkerPrimitiveAgg(MarkerPrimitive):
    path   = Instance(PathPrimitiveAgg, ())



class RendererAgg(Renderer):
    gray = agg.rgba8(128,128,128,255)    
    white = agg.rgba8(255,255,255,255)
    blue = agg.rgba8(0,0,255,255)
    black = agg.rgba8(0,0,0,0)

    pathd = traits.Dict(Int, PathPrimitiveAgg)
    markerd = traits.Dict(Int, MarkerPrimitiveAgg)

    def _size_changed(self, old, new):
        Renderer._size_changed(self, old, new)

        width, height = self.size
        stride = width*4
        self.buf = buf = agg.buffer(width, height, stride)

        self.rbuf = rbuf = agg.rendering_buffer()
        rbuf.attachb(buf)

        self.pf = pf = agg.pixel_format_rgba(rbuf)
        self.rbase = rbase = agg.renderer_base_rgba(pf)
        rbase.clear_rgba8(self.white)

        # the antialiased renderers
        self.renderer =  agg.renderer_scanline_aa_solid_rgba(rbase);        
        self.rasterizer = agg.rasterizer_scanline_aa()
        self.scanline = agg.scanline_p8()
        self.trans = None

        # the aliased renderers
        self.rendererbin =  agg.renderer_scanline_bin_solid_rgba(rbase);
        self.scanlinebin = agg.scanline_bin()


    def new_path_primitive(self):
        'return a PathPrimitive (or derived)'
        return PathPrimitiveAgg()

    def new_marker_primitive(self):
        'return a MarkerPrimitive (or derived)'
        return MarkerPrimitiveAgg()

    def render_path(self, pathid):

        path = self.pathd[pathid]

        if path.antialiased:
            renderer = self.renderer
            scanline = self.scanline
            render_scanlines = agg.render_scanlines_rgba
        else:
            renderer = self.rendererbin
            scanline = self.scanlinebin
            render_scanlines = agg.render_scanlines_bin_rgba


        affine = self.adisplay * path.affine
        #print 'display affine:\n', self.adisplay
        #print 'path affine:\n', path.affine
        #print 'product affine:\n', affine
        aggaffine = agg.trans_affine(*affine.vec6)
        transpath = agg.conv_transform_path(path.agg_path, aggaffine)

        if path.facecolor is not None:
            #print 'render path', path.facecolor, path.agg_facecolor
            self.rasterizer.add_path(transpath)
            renderer.color_rgba8( path.agg_facecolor )
            render_scanlines(self.rasterizer, scanline, renderer);

        if path.color is not None:
            stroke = agg.conv_stroke_transpath(transpath)
            stroke.width(path.linewidth)
            self.rasterizer.add_path(stroke)
            renderer.color_rgba8( path.agg_color )        
            render_scanlines(self.rasterizer, scanline, renderer);

    def render_marker(self, markerid):
        marker = self.markerd[markerid]

        path = marker.path
        
        if path.antialiased:
            renderer = self.renderer
            scanline = self.scanline
            render_scanlines = agg.render_scanlines_rgba
        else:
            renderer = self.rendererbin
            scanline = self.scanlinebin
            render_scanlines = agg.render_scanlines_bin_rgba



        affinelocs = self.adisplay * marker.affine

        Nmarkers = marker.locs.shape[0]
        Locs = npy.ones((3, Nmarkers))
        Locs[0] = marker.locs[:,0]
        Locs[1] = marker.locs[:,1]        

        Locs = affinelocs * Locs
        

        dpiscale = self.dpi/72. # for some reason this is broken 
        # this will need to be highly optimized and hooked into some
        # extension code using cached marker rasters as we now do in
        # _backend_agg


        pathcodes, pathxy = marker.path.pathdata

        pathx = dpiscale*pathxy[:,0] 
        pathy = dpiscale*pathxy[:,1]

        Npath = len(pathcodes)
        XY = npy.ones((Npath, 2))



        for xv,yv,tmp in Locs.T:
            XY[:,0] = (pathx + xv).astype(int) + 0.5
            XY[:,1] = (pathy + yv).astype(int) + 0.5

            pathdata = pathcodes, XY
            aggpath = PathPrimitiveAgg.make_agg_path(pathdata)

            if path.facecolor is not None:
                self.rasterizer.add_path(aggpath)
                renderer.color_rgba8( path.agg_facecolor )
                render_scanlines(self.rasterizer, scanline, renderer);

            if path.color is not None:
                stroke = agg.conv_stroke_path(aggpath)
                stroke.width(path.linewidth)
                self.rasterizer.add_path(stroke)
                renderer.color_rgba8( path.agg_color )        
                render_scanlines(self.rasterizer, scanline, renderer);

    def show(self):
        # we'll cheat a little and use pylab for display

        X = npy.fromstring(self.buf.to_string(), npy.uint8)
        width, height = self.size
        X.shape = height, width, 4
        if 1:
            import pylab
            fig = pylab.figure()
            ax = fig.add_axes([0,0,1,1], xticks=[], yticks=[],
                              frameon=False, aspect='auto')
            ax.imshow(X, aspect='auto')
            pylab.show()


class Func(HasTraits):
    def __call__(self, X):
        'transform the numpy array with shape N,2'
        return X
    
    def invert(self, x, y):
        'invert the point x, y'
        return x, y

    def point(self, x, y):
        'transform the point x, y'
        return x, y
        

class Identity(Func):
    def __call__(self, X):
        'transform the numpy array with shape N,2'
        return X
    
    def invert(self, x, y):
        'invert the point x, y'
        return x, y

    def point(self, x, y):
        'transform the point x, y'
        return x, y


class Polar(Func):
    def __call__(self, X):
        'transform the numpy array with shape N,2'
        r = X[:,0]
        theta = X[:,1]
        x = r*npy.cos(theta)
        y = r*npy.sin(theta)
        return npy.array([x,y]).T
    
    def invert(self, x, y):
        'invert the point x, y'
        raise NotImplementedError

    def point(self, x, y):
        'transform the point x, y'
        raise NotImplementedError


mtraits.Model = Instance(Func, ())


    




## begin Artist layer
# coordinates:
#
#   artist model : a possibly nonlinear transformation (Func instance)
#     to a separable cartesian coordinate, eg for polar is takes r,
#     theta -> r*cos(theta), r*sin(theta)
#
#   AxesCoords.adata : an affine 3x3 matrix that takes model output and
#     transforms it to axes 0,1.  We are kind of stuck with the
#     mpl/matlab convention that 0,0 is the bottom left of the axes,
#     even though it contradicts pretty much every GUI layout in the
#     world
#
#   AxesCoords.aview: an affine 3x3 that transforms an axesview into figure
#     0,1 
#
#   Renderer.adisplay : takes an affine 3x3 and puts figure view into display.  0,
#      0 is left, top, which is the typical coordinate system of most
#      graphics formats

primitiveID = IDGenerator()
artistID = IDGenerator()


class Artist(HasTraits):
    zorder  = Float(1.0)
    alpha   = mtraits.Alpha()
    visible = mtraits.Visible()

    adata = Instance(Affine, ())  # the data affine
    aview = Instance(Affine, ())  # the view affine
    affine  = Instance(Affine, ()) # the product of the data and view affine

    renderer = Trait(None, Renderer)

    # every artist defines a string which is the name of the attr that
    # containers should put it into when added.  Eg, an Axes is an
    # Aritst container, and when you place a Line in to an Axes, the
    # Axes will store a reference to it in the sequence ax.lines where
    # Line.sequence = 'lines'
    sequence = 'artists'
    def __init__(self):
        self.artistid = artistID()

        # track affine as the product of the view and the data affines
        # -- this should be a property, but I had trouble making a
        # property on my custom class affine so this is a workaround
        def product(ignore):
            # modify in place
            self.affine.follow((self.aview * self.adata).vec6)
        product(None)  # force an affine product updated
        self.adata.on_trait_change(product, 'vec6')
        self.aview.on_trait_change(product, 'vec6')        


    def _get_affine(self):
        return self.aview * self.adata
    

    def draw(self):
        pass

class ArtistContainer(Artist):
    
    artistd = traits.Dict(Int, Artist)
    sequence = 'containers'
    def __init__(self):
        Artist.__init__(self)
        self.artistd = dict()


    
    def add_artist(self, artist, followdata=True, followview=True):        
        # this is a very interesting change from matplotlib -- every
        # artist acts as a container that can hold other artists, and
        # respects zorder drawing internally.  This makes zordering
        # much more flexibel
        self.artistd[artist.artistid] = artist
        self.__dict__.setdefault(artist.sequence, []).append(artist)
        
        artist.renderer = self.renderer
        self.sync_trait('renderer', artist, mutual=False)


        artist.followdata = followdata
        artist.followview = followview
        if followdata:
            # set the data affines to be the same
            artist.adata.follow(self.adata.vec6)
            self.adata.on_trait_change(artist.adata.follow, 'vec6')

        if followview:
            # set the view affines to be the same
            artist.aview.follow(self.aview.vec6)
            self.aview.on_trait_change(artist.aview.follow, 'vec6')



    def remove_artist(self, artist):

        if artist.followview:
            self.aview.on_trait_change(artist.aview.follow, 'vec6', remove=True)
            del artist.followview
            
        if artist.followdata:
            self.adata.on_trait_change(artist.adata.follow, 'vec6', remove=True)
            del artist.followdata

        self.sync_trait('renderer', artist, remove=True)
        del self.artistd[artist.artistid]
        self.__dict__[artist.sequence].remove(artist)

    def draw(self):
        if self.renderer is None or not self.visible: return
        
        dsu = [(artist.zorder, artist.artistid, artist) for artist in self.artistd.values()]
        dsu.sort()
        for zorder, artistid, artist in dsu:
            #print 'artist draw', self, artist, zorder
            artist.draw()


class Path(Artist):
    """
    An interface class between the higher level artists and the path
    primitive that needs to talk to the renderers
    """
    _path       = traits.Instance(PathPrimitive, ())
    antialiased = mtraits.AntiAliased()
    color       = mtraits.Color('blue')
    facecolor   = mtraits.Color('yellow')    
    linestyle   = mtraits.LineStyle('-')
    linewidth   = mtraits.LineWidth(1.0) 
    model       = mtraits.Model
    pathdata    = traits.Tuple(Array('b'), VertexArray)
    sequence    = 'paths'
    zorder      = Float(1.0)

    # why have an extra layer separating the PathPrimitive from the
    # Path artist?  The reasons are severalfold, but it is still not
    # clear if this is the better solution.  Doing it this way enables
    # the backends to create their own derived primitves (eg
    # RendererAgg creates PathPrimitiveAgg, and in that class sets up
    # trait listeners to create agg colors and agg paths when the
    # PathPrimitive traits change.  Another reason is that it allows
    # us to handle nonlinear transformation (the "model") at the top
    # layer w/o making the backends understand them.  The current
    # design is create a mapping between backend primitives and
    # primitive artists (Path, Text, Image, etc...) and all of the
    # higher level Artists (Line, Polygon, Axis) will use the
    # primitive artitsts. So only a few artists will need to know how
    # to talk to the backend.  The alternative is to make the backends
    # track and understand the primitive artists themselves.

    def __init__(self):
        """
        The model is a function taking Nx2->Nx2.  This is where the
        nonlinear transformation can be used
        """
        Artist.__init__(self)
        self._pathid = primitiveID()

    def _pathdata_default(self):
        return (npy.array([0,0], dtype=npy.uint8),
                npy.array([[0,0],[0,0]], npy.float_))
        
    def _update_path(self):
        'sync the Path traits with the path primitive'
        self.sync_trait('linewidth', self._path, mutual=False)
        self.sync_trait('color', self._path, mutual=False)
        self.sync_trait('facecolor', self._path, mutual=False)
        self.sync_trait('antialiased', self._path, mutual=False)

        # sync up the path affine
        self._path.affine.follow(self.affine.vec6)
        self.affine.on_trait_change(self._path.affine.follow, 'vec6')
        self._update_pathdata()

    def _update_pathdata(self):
        #print 'PATH: update pathdata'

        codes, xy = self.pathdata

        #print '    PATH: shapes', codes.shape, xy.shape
        if self.model is not None:
            xy = self.model(xy)

        pathdata = codes, xy

        self._path.pathdata = pathdata
        
    def draw(self):
        if self.renderer is None or not self.visible: return
        Artist.draw(self)
        self.renderer.render_path(self._pathid)
        

    def _renderer_changed(self, old, new):
        if old is not None:
            del old.pathd[self._pathid]
            
        if new is None: return 

        #print 'PATH renderer_changed; updating' 
        self._path = renderer.new_path_primitive()
        new.pathd[self._pathid] = self._path
        self._update_path()        
    
    def _model_changed(self, old, new):
        self._update_pathdata()

    def _pathdata_changed(self, old, new):
        #print 'PATH: pathdata changed'
        self._update_pathdata()


class Marker(Artist):
    """
    An interface class between the higher level artists and the marker
    primitive that needs to talk to the renderers
    """
    _marker       = traits.Instance(MarkerPrimitive, ())
    locs        = Array('d')
    path        = Instance(Path, ())    
    model       = mtraits.Model
    sequence    = 'markers'
    size        = Float(1.0) # size of the marker in points

    def __init__(self):
        """
        The model is a function taking Nx2->Nx2.  This is where the
        nonlinear transformation can be used
        """
        Artist.__init__(self)
        self._markerid = primitiveID()

    def _locs_default(self):
        return npy.array([[0,1],[0,1]], npy.float_)

        
    def _path_default(self):
        bounds = npy.array([-0.5, -0.5, 1, 1])*self.size
        return Rectangle().set(bounds=bounds)

    def _path_changed(self, old, new):
        if self.renderer is None:
            # we can't sync up to the underlying path yet
            return 
        print 'MARKER _path_changed', self.path._path.pathdata, self._marker.path.pathdata
        old.sync_trait('_path', self._marker, 'path', remove=True)
        new.sync_trait('_path', self._marker, 'path', mutual=False)

        
    def _update_marker(self):
        'sync the Marker traits with the marker primitive'
        if self.renderer is None:
            # we can't sync up to the underlying path yet
            return 

        # sync up the marker affine
        self.path.sync_trait('_path', self._marker, 'path', mutual=False)
        self._marker.affine.follow(self.affine.vec6)
        self.affine.on_trait_change(self._marker.affine.follow, 'vec6')
        self._update_locs()

        print 'MARKER _update_marker', self.path._path.pathdata, self._marker.path.pathdata

    def _update_locs(self):
        print 'MARKER: update markerdata'
        xy = self.locs
        if self.model is not None:
            xy = self.model(xy)

        self._marker.locs = xy
        
    def draw(self):
        if self.renderer is None or not self.visible: return
        Artist.draw(self)
        self.renderer.render_marker(self._markerid)
        

    def _renderer_changed(self, old, new):
        # we must make sure the contained artist gets the callback
        # first so we can update the path primitives properly
        self.path._renderer_changed(old, new)
        if old is not None:
            del old.markerd[self._markerid]
            
        if new is None: return 

        print 'MARKER renderer_changed; updating' 
        self._marker = renderer.new_marker_primitive()
        new.markerd[self._markerid] = self._marker
        self._update_marker()        
    
    def _model_changed(self, old, new):
        self._update_locs()

    def _locs_changed(self, old, new):
        if len(new.shape)!=2:
            raise ValueError('new must be nx2 array')
        self._update_locs()

class Line(Path):

    XY = Array('d')
    sequence        = 'lines'

    def _facecolor_default(self):
        return None
    
    def _XY_default(self):
        return npy.array([[0,1],[0,1]], npy.float_)
    
    def _XY_changed(self):
        #print 'LINE: XY changed'
        codes = LINETO*npy.ones(len(self.XY), npy.uint8)
        codes[0] = MOVETO
        #print 'LINE shapes', codes.shape, self.XY.shape
        self.pathdata = codes, self.XY



class Polygon(Path):
    zorder    = Float(2.0)
    vertices  = Array('d')
    sequence  = 'polygons'

    def _vertices_default(self):
        return npy.array([[-1,0], [0,1], [1,0], [0,0]], npy.float_)

    def _vertices_changed(self, old, new):
        #print 'POLY: verts changed'
        N = len(new)
        
        codes = LINETO*npy.ones(N, npy.uint8)
        codes[0] = MOVETO
        codes[-1] = CLOSEPOLY

        pathdata = codes, new
        self.pathdata = pathdata
        self._pathdata_changed(None, pathdata)
        
class Rectangle(Polygon, Box):
    sequence = 'rectangles'
    def __init__(self):
        Polygon.__init__(self)
        self._bounds_changed(None, self.bounds)
        
    def _bounds_changed(self, old, new):
        Box._bounds_changed(self, old, new)
        #print 'RECT: bounds changed'
        l,b,w,h = new
        t = b+h
        r = l+w
        self.vertices = npy.array([(l,b), (l,t), (r, t), (r, b), (0,0)], npy.float_)
        #XXX: do we need to otify traits change
        self._vertices_changed(None, self.vertices)

class RegularPolygon(Polygon):
    center = Tuple(Float, Float) 
    sides = Int(6)
    radius = Float(1.0)
    theta = Float(0.)  # orientation in radians
    sequence = 'rectangles'

    def __init__(self):
        self._update_vertices()
    
    def _sides_changed(self, old, new):
        self._update_verts()

    def _theta_changed(self, old, new):
        self._update_verts()

    def _radius_changed(self, old, new):
        self._update_verts()
        
    def _update_verts(self):

        theta = 2*npy.pi/self.sides*npy.arange(self.sides) + self.theta
        x, y = self.center

        xy = npy.zeros((self.sides,2))

        xy[:,0] = x + self.radius*npy.cos(theta)
        xy[:,1] = y + self.radius*npy.sin(theta)

        self.vertices = xy

class Figure(ArtistContainer):

    rectangle = Instance(Rectangle, ())
    sequence = None  # figure is top level container
    def __init__(self):
        ArtistContainer.__init__(self)
        self.rectangle.zorder = 0
        self.rectangle.facecolor = '0.75'
        self.rectangle.bounds = [0,0,1,1]
        self.add_artist(self.rectangle)

class Axis(ArtistContainer):
    zorder = Float(1.5)
    tickmarker  = Instance(Marker, ())
    line = Instance(Line, ())
    ticklocs = Array('d')
    ticksize = Float(7.0)

    
    loc  = Float(0.)          # the y location of the x-axis
    tickoffset = Float(0)  # -1 for outer, -0.5 for centered, 0 for inner
    sequence  = 'axes'

    def __init__(self):
        ArtistContainer.__init__(self)
        self.affine.on_trait_change(self._update_blended_affine, 'vec6')
        self.tickmarker.antialiased = False
        self.line.antialiased = False

        self.add_artist(self.line, followdata=False)
        self.add_artist(self.tickmarker, followdata=False)

        # XXX, do we have to manually call these or will they get
        # calle dautomagically in init
        self._update_tick_path()
        self._update_marker_locations()
        self._update_blended_affine()
        self._update_linepath()
    
    def _ticklocs_changed(self, old, new):
        self._update_marker_locations()

    def _loc_changed(self, old, new):
        self._update_blended_affine()
    
    def _ticksize_changed(self, old, new):
        self._update_tick_path()

    def _tickoffset_changed(self, old, new):
        self._update_tick_path()

    def _update_blended_affine(self):
        'blend of xdata and y axis affine'
        raise NotImplementedError
        
    def _update_marker_locations(self):
        raise NotImplementedError
    
    def _update_tick_path(self):
        raise NotImplementedError

    def _update_linepath(self):
        raise NotImplementedError

class XAxis(Axis):
    sequence = 'xaxes'
    def _update_blended_affine(self):
        'blend of xdata and y axis affine'
        sx, b, tx = self.adata.data[0]
        a = Affine()
        a.vec6 = sx, b, 0, 1, tx, self.loc
        self.tickmarker.affine.vec6 = (self.aview * a).vec6
        
        a = Affine()
        a.translate = 0, self.loc
        self.line.affine.vec6 = (self.aview * a).vec6        

    def _update_marker_locations(self):
        Nticks = len(self.ticklocs)
        locs = self.loc*npy.ones((Nticks,2))
        locs[:,0] = self.ticklocs
        self.tickmarker.locs = locs

    def _update_tick_path(self):
        codes = MOVETO, LINETO
        verts = npy.array([[0., self.tickoffset], [0, self.tickoffset-1]])*self.ticksize

        pathdata = codes, verts
        self.tickmarker.path.pathdata = pathdata    

    def _update_linepath(self):
        
        codes = MOVETO, LINETO
        X = npy.array([[0, 1], [0, 0]], npy.float_).T
        pathdata = codes, X
        self.line.pathdata = pathdata

class YAxis(Axis):
    sequence = 'yaxes'

    def _update_blended_affine(self):
        'blend of xdata and y axis affine'
        c, sy, ty = self.adata.data[1]
        a = Affine()
        a.vec6 = 1, 0, 0, sy, self.loc, ty
        self.tickmarker.affine.vec6 = (self.aview * a).vec6
        
        a = Affine()
        a.translate = self.loc, 0
        self.line.affine.vec6 = (self.aview * a).vec6        

    def _update_marker_locations(self):
        Nticks = len(self.ticklocs)
        locs = self.loc*npy.ones((Nticks,2))
        locs[:,1] = self.ticklocs
        self.tickmarker.locs = locs

    def _update_tick_path(self):
        codes = MOVETO, LINETO
        verts = npy.array([[self.tickoffset,0], [self.tickoffset+1,0]])*self.ticksize
        pathdata = codes, verts
        self.tickmarker.path.pathdata = pathdata

    def _update_linepath(self):
        codes = MOVETO, LINETO
        X = npy.array([[0, 0], [0, 1]], npy.float_).T

        pathdata = codes, X
        self.line.pathdata = pathdata

class FigurePane(ArtistContainer, Box):
    """
    The figure pane conceptually like the matplotlib Axes, but now
    almost all of it's functionality is modular into the Axis and
    Affine instances.  It is a shell of it's former self: it has a
    rectangle and a default x and y axis instance
    """
    rectangle = Instance(Rectangle, ())
    #gridabove = traits.false # TODO handle me
    xaxis     = Instance(XAxis, ())
    yaxis     = Instance(YAxis, ())
    sequence  = 'panes'
    
    def __init__(self):
        ArtistContainer.__init__(self)
        self.rectangle.zorder = 0
        self.rectangle.facecolor = 'white'
        self.rectangle.edgecolor = 'white'        
        self.rectangle.linewidth = 0
        
        self.rectangle.bounds = [0,0,1,1]
        self.add_artist(self.rectangle, followdata=False)
        self.add_artist(self.xaxis)
        self.add_artist(self.yaxis)                

    def _bounds_changed(self, old, new):
        Box._bounds_changed(self, old, new)
        l,b,w,h = self.bounds
        self.aview.scale = w, h
        self.aview.translate = l, b


## begin examples
        
def classic(fig):

    pane = FigurePane().set(bounds=[0.1, 0.1, 0.8, 0.8])
    fig.add_artist(pane, followdata=False, followview=False)
    # update the view limits, all the affines should be automagically updated
    x = npy.arange(0, 10., 0.01)
    y = npy.sin(2*npy.pi*x)
    y = npy.exp(-x/2.)
    line1 = Line().set(XY=npy.array([x,y]).T,
                       color='blue', linewidth=2.0, 
                       )


    pane.add_artist(line1)
    pane.adata.xlim = 0, 10
    pane.adata.ylim = -0.1, 1.1

    # axis placement is still broken
    xax, yax = pane.xaxis, pane.yaxis
    xax.ticklocs = npy.arange(0., 11., 1)
    xax.tickoffset = 0.5
    xax.loc = -0.05
    xax.line.color = 'black'
    xax.tickmarker.path.color = 'black'

    yax.ticklocs = npy.arange(-1.0, 1.1, 0.2)
    yax.tickoffset = -0.5
    yax.loc = -0.05
    yax.line.color = 'black'
    yax.tickmarker.path.color = 'black'

    if 0:
        x = npy.arange(0, 10., 0.1)
        y = npy.sin(2*npy.pi*x)

        marker = Marker().set(
            locs=npy.array([x,y]).T,  color='ref', linewidth=1.0,
            size=10)


        pane.add_artist(marker)


    if 0:
        xax, yax = pane.xaxis, pane.yaxis
        xax.ticklocs = npy.arange(0., 11., 1)
        xax.ticksize = 8
        xax.line.color = 'black'
        xax.line.linewidth = 2.0
        xax.tickoffset = .5
        xax.tickmarker.path.color = 'black'
        xax.tickmarker.path.linewidth = 2

        xax.loc = 0.5
        xax.zorder = 10  

        yax.ticklocs = npy.arange(-1.0, 1.1, 0.2)
        yax.line.color = 'black'
        yax.line.linewidth = 2.0
        yax.tickmarker.path.color = 'black'
        yax.ticksize = 8
        yax.tickoffset = -0.5
        yax.tickmarker.path.linewidth = 2
        yax.loc = 0.5
        yax.zorder = 10
    if 0:
        # add a right and top axis; the markers are getting the loc
        # but the line path isn't...  It appears all the line paths
        # are getting 0
        xaxis2 = XAxis()
        xaxis2.loc = 0.6
        xaxis2.tickoffset = -1
        xaxis2.ticklocs = npy.arange(0., 10.1, 0.5)
        yaxis2 = YAxis().set(loc=0.6, tickoffset=-1)    

        yaxis2.tickmarker.path.color = 'green'
        yaxis2.loc = 0.5
        yaxis2.ticksize = 10
        yaxis2.tickmarker.path.linewidth = 1
        yaxis2.line.color = 'green'
        yaxis2.tickmarker.path.color = 'green'
        yaxis2.ticklocs = npy.arange(-1.0, 1.1, 0.1)

        pane.add_artist(xaxis2)
        pane.add_artist(yaxis2)
        # uncomment to change Axes wwidth
        #pane.width = 0.8

    if 0:
        # XXX: axes lines and tick markes are placed in vastly
        # different locations depending on whether this is commented
        # or uncommented, suggesting that the problem is caused by
        # events not propogating unless lim are changed.  If we set
        # these lim to be the same as the lim above (note they are
        # almost identical) then the graphs are the same regardless of
        # whether the lim are set
        pane.adata.xlim = 0.01, 10.01
        pane.adata.ylim = -0.101, 1.101


def make_subplot_ll(fig):
    x1 = npy.arange(0, 10., 0.05)
    x2 = npy.arange(0, 10., 0.1)
    y1 = npy.cos(2*npy.pi*x1)
    y2 = 10*npy.exp(-x1)


    pane = FigurePane().set(bounds=[0.1, 0.1, 0.4, 0.4])
    fig.add_artist(pane, followdata=False, followview=False)


    line1 = Line().set(X=npy.array([x1,y1]).T,
                       color='blue', linewidth=2.0)


    pane.add_artist(line1)

    # update the view limits, all the affines should be automagically updated
    pane.adata.xlim = 0, 10
    pane.adata.ylim = -1.1, 1.1

    pane.xaxis.ticklocs = npy.arange(0., 11., 1.)
    pane.xaxis.loc = -0.1
    pane.xaxis.tickoffset = -0.5
    pane.xaxis.line.color = 'red'

    Pane.yaxis.ticklocs = npy.arange(-1.0, 1.1, 0.2)
    pane.yaxis.loc = -0.1
    pane.xaxis.tickoffset = -0.5    
    
    pane.yaxis.line.color = 'blue'
    pane.yaxis.tickmarker.color = 'blue'

    # uncomment to change Axes wwidth
    #pane.width = 0.8

def make_subplot_ur(fig):
    axes2 = Axes()

    axes2.aview.scale = 0.4, 0.4
    axes2.aview.translate = 0.55, 0.55
    
    fig.add_artist(axes2, followdata=False, followview=False)

    r = npy.arange(0.0, 1.0, 0.01)
    theta = r*4*npy.pi
    
    line2 = Line().set(X=npy.array([r, theta]).T, model=Polar(), color='#ee8d18', linewidth=2.0)
    axes2.add_artist(line2)

    rect2 = Rectangle().set(bounds=[0,0,1,1], facecolor='#d5de9c')
    axes2.add_artist(rect2, followdata=False)

    axes2.adata.xlim = -1.1, 1.1
    axes2.adata.ylim = -1.1, 1.1    



if __name__=='__main__':

    renderer = RendererAgg()
    fig = Figure()
    fig.renderer = renderer
    classic(fig)
    #make_subplot_ll(fig)
    #make_subplot_ur(fig)
    fig.draw()
    renderer.show()

