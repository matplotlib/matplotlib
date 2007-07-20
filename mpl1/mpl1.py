"""
Install instructions for traits 2.0

   # blow away old enthought
   rm -rf ~/dev/lib/python2.4/site-packages/enthought.*

   # get easy_install, if necessary
   wget  http://peak.telecommunity.com/dist/ez_setup.py
   sudo python sez_setup.py

   sudo easy_install -f http://code.enthought.com/enstaller/eggs/source/unstable "enthought.etsconfig < 3.0a" "enthought.util <3.0a" "enthought.debug <3.0a"

   svn co https://svn.enthought.com/svn/enthought/branches/enthought.traits_2.0 enthought_traits

   cd enthought_traits/
   sudo python setup.py install 

"""

# see install instructions for enthrought traits2 in mtraits
import enthought.traits.api as traits

from matplotlib import agg
from matplotlib import colors as mcolors
from matplotlib import cbook
import numpy as npy

is_string_like = cbook.is_string_like



class Affine(traits.HasTraits):
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
    data = traits.Array('d', (3,3))

    translate = traits.Property(traits.Array('d', (2,)))
    scale = traits.Property(traits.Array('d', (2,)))    
    vec6 = traits.Property(traits.Array('d', (6,)))

    xlim = traits.Property(traits.Array('d', (2,)))
    ylim = traits.Property(traits.Array('d', (2,)))            

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


    
class ColorHandler(traits.TraitHandler):
    """
    This is a clever little traits mechanism -- users can specify the
    color as any mpl color, and the traited object will keep the
    original color, but will add a new attribute with a '_' postfix
    which is the color rgba tuple.

    Eg

    class C(HasTraits):
        fillcolor = traits.Trait('black', ColorHandler())

    c = C()
    c.fillcolor = 'red'
    print c.fillcolor    # prints red
    print c.fillcolor_   # print (1,0,0,1)
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
    DPI         = traits.Float(72.)

    Alpha       = traits.Range(0., 1., 0.)
    Affine      = traits.Trait(Affine())
    AntiAliased = traits.true
    Codes       = traits.Array('b', value=npy.array([0,0], dtype=npy.uint8))
    Color       = traits.Trait('black', ColorHandler())
    DPI         = traits.Float(72.)
    Interval    = traits.Array('d', (2,), npy.array([0.0, 1.0], npy.float_))
    LineStyle   = traits.Trait('-', '--', '-.', ':', 'steps', None)
    LineWidth   = traits.Float(1.0)
    Marker      = traits.Trait(None, '.', ',', 'o', '^', 'v', '<', '>', 's',
                 '+', 'x', 'd', 'D', '|', '_', 'h', 'H',
                 'p', '1', '2', '3', '4')
    MarkerSize  = traits.Float(6)
    Verts       = traits.Array('d', value=npy.array([[0,0],[0,0]], npy.float_))
    PathData    = traits.Tuple(Codes(), Verts())
    Visible     = traits.true


mtraits = MTraitsNamespace()

class Renderer(traits.HasTraits):
    dpi = mtraits.DPI
    size = traits.Tuple(traits.Int(600), traits.Int(400))

    adisplay = traits.Instance(Affine, ()) 

    def __init__(self, size=(600,400)):

        self.pathd = dict()    # path id -> Path instance
        self.markersd = dict()      # path id -> Markers instance

        self._size_changed(None, size)
        
    def _size_changed(self, old, new):
        width, height = new

        # almost all renderers assume 0,0 is left, upper, so we'll flip y here by default
        self.adisplay.translate = 0, height
        self.adisplay.scale = width, -height
        
        

    def add_path(self, pathid, path):
        # todo, we could use a traits dict here
        if not isinstance(path, Path):
            raise TypeError('add_path takes an ID and a Path instance')
        self.pathd[pathid] = path

    def remove_path(self, pathid):
        if pathid in self.pathd:
            del self.pathd[pathid]

    def add_markers(self, markersid, markers):
        # todo, we could use a traits dict here
        if not isinstance(markers, Markers):
            raise TypeError('add_markers takes an ID and a Markers instance')
        self.markersd[markersid] = markers

    def remove_markers(self, markersid):
        if markersid in self.markersd:
            del self.markersd[markersid]

    def render_path(self, pathid):
        pass
            


class RendererAgg(Renderer):
    gray = agg.rgba8(128,128,128,255)    
    white = agg.rgba8(255,255,255,255)
    blue = agg.rgba8(0,0,255,255)
    black = agg.rgba8(0,0,0,0)


    def _size_changed(self, old, new):
        Renderer._size_changed(self, old, new)

        width, height = self.size
        stride = width*4
        self.buf = buf = agg.buffer(width, height, stride)

        self.rbuf = rbuf = agg.rendering_buffer()
        rbuf.attachb(buf)

        self.pf = pf = agg.pixel_format_rgba(rbuf)
        self.rbase = rbase = agg.renderer_base_rgba(pf)
        rbase.clear_rgba8(self.gray)

        # the antialiased renderers
        self.renderer =  agg.renderer_scanline_aa_solid_rgba(rbase);        
        self.rasterizer = agg.rasterizer_scanline_aa()
        self.scanline = agg.scanline_p8()
        self.trans = None

        # the aliased renderers
        self.rendererbin =  agg.renderer_scanline_bin_solid_rgba(rbase);
        self.scanlinebin = agg.scanline_bin()


    def add_path(self, pathid, path):
        self.pathd[pathid] = AggPath(path)


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

        if path.fillcolor is not None:
            self.rasterizer.add_path(transpath)
            renderer.color_rgba8( path.agg_fillcolor )
            render_scanlines(self.rasterizer, scanline, renderer);

        if path.strokecolor is not None:
            stroke = agg.conv_stroke_transpath(transpath)
            stroke.width(path.linewidth)
            self.rasterizer.add_path(stroke)
            renderer.color_rgba8( path.agg_strokecolor )        
            render_scanlines(self.rasterizer, scanline, renderer);

    def render_markers(self, markerid):
        markers = self.markersd[markerid]

        path = AggPath(markers.path)
        
        if path.antialiased:
            renderer = self.renderer
            scanline = self.scanline
            render_scanlines = agg.render_scanlines_rgba
        else:
            renderer = self.rendererbin
            scanline = self.scanlinebin
            render_scanlines = agg.render_scanlines_bin_rgba



        affineverts = self.adisplay * markers.affine

        Nmarkers = markers.verts.shape[0]
        Locs = npy.ones((3, Nmarkers))
        Locs[0] = markers.verts[:,0]
        Locs[1] = markers.verts[:,1]        

        Locs = affineverts * Locs
        

        dpiscale = self.dpi/72. # for some reason this is broken 
        # this will need to be highly optimized and hooked into some
        # extension code using cached marker rasters as we now do in
        # _backend_agg

        pathcodes, pathverts = markers.path.pathdata
        pathx = dpiscale*pathverts[:,0] 
        pathy = dpiscale*pathverts[:,1]

        Npath = len(pathcodes)
        XY = npy.ones((Npath, 2))


        for xv,yv,tmp in Locs.T:
            XY[:,0] = pathx + xv
            XY[:,1] = pathy + yv
            
            pathdata = pathcodes, XY
            aggpath = AggPath.make_agg_path(pathdata)

            if path.fillcolor is not None:
                self.rasterizer.add_path(aggpath)
                renderer.color_rgba8( path.agg_fillcolor )
                render_scanlines(self.rasterizer, scanline, renderer);

            if path.strokecolor is not None:
                stroke = agg.conv_stroke_path(aggpath)
                stroke.width(path.linewidth)
                self.rasterizer.add_path(stroke)
                renderer.color_rgba8( path.agg_strokecolor )        
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


class Func(traits.HasTraits):
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


mtraits.Model = traits.Instance(Func, ())


    
class Path(traits.HasTraits):
    """
    The path is an object that talks to the backends, and is an
    intermediary between the high level path artists like Line and
    Polygon, and the backend renderer
    """
    MOVETO, LINETO, CLOSEPOLY = range(3)
    
    strokecolor = mtraits.Color('black')
    fillcolor   = mtraits.Color('blue')
    alpha       = mtraits.Alpha(1.0)
    linewidth   = mtraits.LineWidth(1.0)
    antialiased = mtraits.AntiAliased
    pathdata    = mtraits.PathData
    affine      = traits.Instance(Affine, ())

    def __init__(self):

        # this is a quick workaround to deal with the problem that
        # traits inited at the class level are shared between
        # instances, which is not what I want
        self.pathdata  = (npy.array([0,0], npy.uint8),   # codes
                          npy.array([[0,0], [0,0]]))     # verts
    
class AggPath(Path):

    def __init__(self, path):
        self.strokecolor = path.strokecolor
        self.fillcolor = path.fillcolor
        self.alpha = path.alpha
        self.linewidth = path.linewidth
        self.antialiased = path.antialiased

        self.pathdata = path.pathdata
        self.affine = path.affine

        
        path.sync_trait('strokecolor', self, mutual=False)
        path.sync_trait('fillcolor', self, mutual=False)
        path.sync_trait('alpha', self, mutual=False)
        path.sync_trait('linewidth', self, mutual=False)
        path.sync_trait('antialiased', self, mutual=False)
        path.sync_trait('pathdata', self, mutual=False)

        path.on_trait_change(self.affine.follow, 'vec6')



        # hmm, I would have thought these would be called by the attr
        # setting above
        self._pathdata_changed(None, self.pathdata)
        self._fillcolor__changed(None, self.fillcolor_)
        self._strokecolor__changed(None, self.strokecolor_)                

        
    @staticmethod
    def make_agg_path(pathdata):
        MOVETO, LINETO, CLOSEPOLY = Path.MOVETO, Path.LINETO, Path.CLOSEPOLY
        agg_path = agg.path_storage()
        codes, verts = pathdata
        N = len(codes)
        for i in range(N):
            x, y = verts[i]
            code = codes[i]
            if code==MOVETO:
                agg_path.move_to(x, y)
            elif code==LINETO:
                agg_path.line_to(x, y)                
            elif code==CLOSEPOLY:
                agg_path.close_polygon()
        return agg_path

    def _pathdata_changed(self, olddata, newdata):        
        self.agg_path = AggPath.make_agg_path(newdata)

        
    def _fillcolor__changed(self, oldcolor, newcolor):        
        self.agg_fillcolor = self.color_to_rgba8(newcolor)

    def _strokecolor__changed(self, oldcolor, newcolor):                

        c = self.color_to_rgba8(newcolor)
        self.agg_strokecolor = c


    def color_to_rgba8(self, color):
        if color is None: return None
        rgba = [int(255*c) for c in color]
        return agg.rgba8(*rgba)


mtraits.Path = traits.Instance(Path, ())
class Markers(traits.HasTraits):
    verts  = mtraits.Verts     # locations to draw the markers at
    path   = mtraits.Path       # marker path in points
    affine = traits.Instance(Affine, ()) # transformation for the verts



    
mtraits.Markers = traits.Instance(Markers, ())
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


primitiveID = IDGenerator()
artistID = IDGenerator()

class Artist(traits.HasTraits):
    zorder  = traits.Float(1.0)
    alpha   = mtraits.Alpha()
    visible = mtraits.Visible()

    adata = traits.Instance(Affine, ())  # the data affine
    aview = traits.Instance(Affine, ())  # the view affine
    
    affine  = traits.Instance(Affine, ()) # the product of the data and view affine

    renderer = traits.Trait(None, Renderer)

    
    def __init__(self):
        self.artistid = artistID()
        self.artistd = dict()

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
    
    def add_artist(self, artist, followdata=True, followview=True):        
        # this is a very interesting change from matplotlib -- every
        # artist acts as a container that can hold other artists, and
        # respects zorder drawing internally.  This makes zordering
        # much more flexibel
        self.artistd[artist.artistid] = artist
        artist.renderer = self.renderer

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


        self.sync_trait('renderer', artist, mutual=False)

    def remove_artist(self, artist):

        if artist.followview:
            self.aview.on_trait_change(artist.aview.follow, 'vec6', remove=True)
            del artist.followview
            
        if artist.followdata:
            self.adata.on_trait_change(artist.adata.follow, 'vec6', remove=True)
            del artist.followdata

        self.sync_trait('renderer', artist, remove=True)
        del self.artistd[artist.artistid]        

    def draw(self):
        if self.renderer is None or not self.visible: return

        dsu = [(artist.zorder, artist.artistid, artist) for artist in self.artistd.values()]
        dsu.sort()
        for zorder, artistid, artist in dsu:
            artist.draw()

            
class Line(Artist):

    linestyle       = mtraits.LineStyle('-')
    antialiased     = mtraits.AntiAliased()
    color           = mtraits.Color('blue')
    linewidth       = mtraits.LineWidth(1.0) 
    marker          = mtraits.Marker(None)
    markerfacecolor = mtraits.Color('blue')
    markeredgecolor = mtraits.Color('black')
    markeredgewidth = mtraits.LineWidth(0.5)
    markersize      = mtraits.MarkerSize(6.0)
    path            = mtraits.Path
    markers         = mtraits.Markers
    X               = mtraits.Verts
    model           = mtraits.Model
    zorder          = traits.Float(2.0)
    
    def __init__(self):
        """
        The model is a function taking Nx2->Nx2.  This is where the
        nonlinear transformation can be used
        """
        Artist.__init__(self)

        # this is potentially a big problem because you have to know
        # which attrs may be shared and hence have to be initialized
        # and which ones don't.  Eg, if you comment out the self.path
        # init, the code breaks

                                        
        self.sync_trait('linewidth', self.path, 'linewidth', mutual=False)
        self.sync_trait('color', self.path, 'strokecolor', mutual=False)
        self.sync_trait('markerfacecolor', self.markers.path, 'fillcolor', mutual=False)
        self.sync_trait('markeredgecolor', self.markers.path, 'strokecolor', mutual=False)
        self.sync_trait('markeredgewidth', self.markers.path, 'linewidth', mutual=False)                        

        # sync up the markers affine
        self.markers.affine.follow(self.affine.vec6)
        self.affine.on_trait_change(self.markers.affine.follow, 'vec6')

        # sync up the path affine
        self.path.affine.follow(self.affine.vec6)
        self.affine.on_trait_change(self.path.affine.follow, 'vec6')
        

        self.path.fillcolor = None

        self.pathid = primitiveID()
        self.markerid = primitiveID()

        self.markerfuncd = {
            's': self._markers_square,
            }        

    def draw(self):

        if self.renderer is None or not self.visible: return
        Artist.draw(self)

        if self.linestyle is not None:
            self.renderer.render_path(self.pathid)
        if self.marker is not None:
            self.renderer.render_markers(self.markerid)        
        

    def _renderer_changed(self, old, new):
        if old is not None:
            old.remove_path(self.pathid)
            old.remove_markers(self.markerid)

        if new is not None:
            new.add_path(self.pathid, self.path)
            new.add_markers(self.markerid, self.markers)        


    def _model_changed(self, old, new):
        self._update_data()

    def _X_changed(self, old, new):
        self._update_data()
        
    def _update_data(self):
        N = self.X.shape[0]
        codes = Path.LINETO*npy.ones(N, dtype=npy.uint8)
        codes[0] = Path.MOVETO

        # todo, having touble setting Model to default to Identity so
        # allowing None as a hack workaround
        #print 'X changed', self.model
        if self.model is not None:
            modelx = self.model(self.X)
        else:
            modelx = self.X
        self.path.pathdata = codes, modelx
        self.markers.verts = modelx

    def _markersize_changed(self, oldX, newX):
        self._refresh_markers()

    def _marker_changed(self, oldX, newX):
        self._refresh_markers()


    def _refresh_markers(self):
        if self.marker is not None:
            markerfunc = self.markerfuncd.get(self.marker)
            if markerfunc is not None: markerfunc()
        

    def _markers_square(self):

        verts = self.markersize*npy.array([[-0.5,-0.5], [-0.5,0.5], [0.5,0.5], [0.5,-0.5], [0,0]])
        codes = Path.LINETO*npy.ones(len(verts), dtype=npy.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        
        self.markers.path.pathdata = codes, verts

mtraits.Line = traits.Instance(Line, ())

class Rectangle(Artist):
    facecolor = mtraits.Color('yellow')
    edgecolor = mtraits.Color('black')
    edgewidth = mtraits.LineWidth(1.0)
    lbwh      = traits.Array('d', (4,), [0,0,1,1])
    path      = mtraits.Path
    zorder    = traits.Float(1.0)

    def __init__(self):
        Artist.__init__(self)

        
        self.sync_trait('facecolor', self.path, 'fillcolor', mutual=False)
        self.sync_trait('edgecolor', self.path, 'strokecolor', mutual=False)
        self.sync_trait('edgewidth', self.path, 'linewidth', mutual=False)

        self.pathid = primitiveID()
        # sync up the path affine
        self.path.affine.follow(self.affine.vec6)
        self.affine.on_trait_change(self.path.affine.follow, 'vec6')




    def _lbwh_changed(self, old, new):
        l,b,w,h = new
        t = b+h
        r = l+w
        verts = npy.array([(l,b), (l,t), (r, t), (r, b), (0,0)], npy.float_)
        codes = Path.LINETO*npy.ones(5, npy.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY

        self.path.pathdata = codes, verts

    def _renderer_changed(self, old, new):
        if old is not None:
            old.remove_path(self.pathid)

        if new is not None:
            new.add_path(self.pathid, self.path)

    def draw(self):
        if self.renderer is None or not self.visible: return
        Artist.draw(self)
        self.renderer.render_path(self.pathid)


mtraits.Rectangle = traits.Instance(Rectangle, ())        

class Figure(Artist):
    pass



class Axes(Artist):
    zorder = traits.Float(0.5)
    
    xtickmarkers  = mtraits.Markers
    xaxisline = mtraits.Line
    xticklocs = traits.Array('d')
    xaxislocy  = traits.Float(0.)  # the y location of the x-axis

    ytickmarkers  = mtraits.Markers
    yaxisline = mtraits.Line
    yticklocs = traits.Array('d')
    yaxislocx  = traits.Float(0.)  # the x location of the y-axis

    def __init__(self):
        Artist.__init__(self)
        self.xtickmarkersid = primitiveID()

        self.xtickmarkers.affine.follow(self.adata.vec6)
        self.adata.on_trait_change(self.xtickmarkers.affine.follow, 'vec6')

    def _xticklocs_changed(self, old, new):
        self._update_xtick_markers()

    def _xaxislocy_changed(self, old, new):
        self._update_xtick_markers()
        
    def _update_xtick_markers(self):
        verts = self.xaxislocy*npy.ones(len(new))
        verts[:,1] = new
        self.xtickmarkers.verts = verts


    def _renderer_changed(self, old, new):
        if old is not None:
            old.remove_markers(self.xtickmarkersid)

        if new is not None:
            new.add_markers(self.xtickmarkersid, self.xtickmarkers)        

    def draw(self):
        if self.renderer is None or not self.visible: return
        #print 'Axes data affine:\n', self.adata
        #print 'Axes view affine :\n', self.aview
        #print 'Axes affine :\n', self.affine        
        Artist.draw(self)
        self.renderer.render_markers(self.xtickmarkersid)

x1 = npy.arange(0, 10., 0.05)
x2 = npy.arange(0, 10., 0.1)
y1 = npy.cos(2*npy.pi*x1)
y2 = 10*npy.exp(-x1)

fig = Figure()
axes = Axes()

axes.aview.scale = 0.4, 0.4
axes.aview.translate = 0.1, 0.1

fig.add_artist(axes, followdata=False, followview=False)

line1 = Line().set(X=npy.array([x1,y1]).T,
                   color='blue', linewidth=2.0, marker='s', markersize=5.0,
                   markerfacecolor='green', markeredgewidth=0.5)


axes.add_artist(line1)


rect1 = Rectangle().set(lbwh=[0,0,1,1], facecolor='white')
axes.add_artist(rect1, followdata=False)



# update the view limits, all the affines should be automagically updated
axes.adata.xlim = 0, 10
axes.adata.ylim = -1.1, 1.1


if 1:
    axes2 = Axes()

    axes2.aview.scale = 0.4, 0.4
    axes2.aview.translate = 0.55, 0.55
    
    fig.add_artist(axes2, followdata=False, followview=False)

    r = npy.arange(0.0, 1.0, 0.01)
    theta = r*4*npy.pi
    
    line2 = Line().set(X=npy.array([r, theta]).T, model=Polar(), color='#ee8d18', linewidth=2.0)
    axes2.add_artist(line2)

    rect2 = Rectangle().set(lbwh=[0,0,1,1], facecolor='#d5de9c')
    axes2.add_artist(rect2, followdata=False)

    axes2.adata.xlim = -1.1, 1.1
    axes2.adata.ylim = -1.1, 1.1    


if 1:
    renderer = RendererAgg()
    fig.renderer = renderer
    fig.draw()
    renderer.show()
