# see install instructions for enthrought traits2 in mtraits
import enthought.traits.api as traits

from matplotlib import agg
from matplotlib import colors as mcolors
import numpy as npy


class ColorHandler(traits.TraitHandler):
    is_mapped = True

    def post_setattr(self, object, name, value):
        object.__dict__[ name + '_' ] = self.mapped_value( value )

    def mapped_value(self, value ):
        if value is None: return None
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
    Affine      = traits.Array('d', (3,3), npy.array([[1,0,0],[0,1,0],[0,0,1]], npy.float_))
    Alpha       = traits.Range(0., 1., 0.)
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
    PathData    = traits.Tuple(Codes, Verts)
    Visible     = traits.true


mtraits = MTraitsNamespace()

        
def affine_axes(rect):
    'make an affine for a typical l,b,w,h axes rectangle'
    l,b,w,h = rect
    return npy.array([[w, 0, l], [0, h, b], [0, 0, 1]], dtype=npy.float_)

def affine_identity():
    return npy.array([[1,0,0],
                      [0,1,0],
                      [0,0,1]],
                     dtype=npy.float_)

def affine_translation(tx, ty):
    return npy.array([[1,0,tx],
                      [0,1,ty],
                      [0,0,1]],
                     dtype=npy.float_)

def affine_rotation(theta):
   a = npy.cos(theta)
   b = -npy.sin(theta)
   c = npy.sin(theta)
   d = npy.cos(theta)
    
   return npy.array([[a,b,0],
                     [c,d,0],
                     [0,0,1]],
                    dtype=npy.float_)


class Renderer(traits.HasTraits):
    dpi = traits.Float(72.)
    
    def __init__(self, width, height):
        self.width, self.height = width, height

        # almost all renderers assume 0,0 is left, upper, so we'll flip y here by default
        self.affinerenderer  = npy.array(
            [[width, 0, 0], [0, -height, height], [0, 0, 1]], dtype=npy.float_)
        self.pathd = dict() # dict mapping path id -> Path instance
        self.markersd = dict() # dict mapping path id -> Markers instance
        

    def add_path(self, pathid, path):
        self.pathd[pathid] = path

    def remove_path(self, pathid):
        if pathid in self.pathd:
            del self.pathd[pathid]

    def add_markers(self, markersid, markers):
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

    def __init__(self, width, height):
        Renderer.__init__(self, width, height)
        
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


        affine = npy.dot(self.affinerenderer, path.affine)
        #print 'display affine:\n', self.affinerenderer
        #print 'path affine:\n', path.affine
        #print 'product affine:\n', affine
        a, b, tx = affine[0]
        c, d, ty = affine[1]
        aggaffine = agg.trans_affine(a,b,c,d,tx,ty)
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



        affineverts = npy.dot(self.affinerenderer, markers.affine)

        Nmarkers = markers.verts.shape[0]
        Locs = npy.ones((3, Nmarkers))
        Locs[0] = markers.verts[:,0]
        Locs[1] = markers.verts[:,1]        
        Locs = npy.dot(affineverts, Locs)
        

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
        X.shape = self.height, self.width, 4
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
    pathdata    = mtraits.PathData()
    affine      = mtraits.Affine()

    def __init__(self):

        # this is a quick workaround to deal with the problem that
        # traits inited at the class level are shared between
        # instances, which is not what I want
        self.strokecolor = 'black'
        self.fillcolor = 'blue'
        self.affine = affine_identity()
        self.pathdata  = (npy.array([0,0], npy.uint8),   # codes
                          npy.array([[0,0], [0,0]]))     # verts
    
mtraits.Path = traits.Trait(Path())

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
        path.sync_trait('affine', self, mutual=False)

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

class Markers(traits.HasTraits):
    verts  = mtraits.Verts()     # locations to draw the markers at
    path   = mtraits.Path()       # marker path in points
    affine = mtraits.Affine()   # transformation for the verts
    x      = traits.Float(1.0)

    def __init__(self):
        # this is a quick workaround to prevent sharing obs; see Path
        self.verts = npy.array([[0,0], [0,0]], npy.float_)
        self.path = Path()
        self.affine = affine_identity()

    
mtraits.Markers = traits.Trait(Markers())
# coordinates:
#
#   artist model : a possibly nonlinear transformation (Func instance)
#     to a separable cartesian coordinate, eg for polar is takes r,
#     theta -> r*cos(theta), r*sin(theta)
#
#   AxesCoords.affineview : an affine 3x3 matrix that takes model output and
#     transforms it to axes 0,1.  We are kind of stuck with the
#     mpl/matlab convention that 0,0 is the bottom left of the axes,
#     even though it contradicts pretty much every GUI layout in the
#     world
#
#   AxesCoords.affineaxes: an affine 3x3 that transforms an axesview into figure
#     0,1 
#
#   Renderer.affinerenderer : takes an affine 3x3 and puts figure view into display.  0,
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
    affine  = mtraits.Affine()
    
    def __init__(self):
        self.artistid = artistID()
        self.renderer = None
        self.affine = affine_identity()
        
    def set_renderer(self, renderer):
        self.renderer = renderer

    def draw(self):
        pass

    def setp(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
            
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
    path            = mtraits.Path()
    markers         = mtraits.Markers()
    X               = mtraits.Verts()
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
        self.color      = 'blue'
        self.markerfacecolor = 'blue'
        self.markeredgecolor = 'black'
        self.path            = Path()
        self.markers         = Markers()
        self.X               = npy.array([[0,1], [0,1]], npy.float_)
        self.model           = Identity()
        #self.model           = None # switch comments with above to reveal bug        
                                        
        self.sync_trait('linewidth', self.path, 'linewidth', mutual=False)
        self.sync_trait('color', self.path, 'strokecolor', mutual=False)
        self.sync_trait('markerfacecolor', self.markers.path, 'fillcolor', mutual=False)
        self.sync_trait('markeredgecolor', self.markers.path, 'strokecolor', mutual=False)
        self.sync_trait('markeredgewidth', self.markers.path, 'linewidth', mutual=False)                        
        self.sync_trait('affine', self.markers, mutual=False)
        self.sync_trait('affine', self.path, mutual=False)

        self.path.fillcolor = None

        self.pathid = primitiveID()
        self.markerid = primitiveID()

        self.markerfuncd = {
            's': self._markers_square,
            }        

    def draw(self):
        if self.renderer is None:
            raise RuntimeError('First call set_renderer')

        if not self.visible: return


        if self.linestyle is not None:
            self.renderer.render_path(self.pathid)
        if self.marker is not None:
            self.renderer.render_markers(self.markerid)        
        

    def set_renderer(self, renderer):
        if self.renderer is not None:
            self.renderer.remove_path(self.pathid)
            self.renderer.remove_markers(self.markerid)

        renderer.add_path(self.pathid, self.path)
        renderer.add_markers(self.markerid, self.markers)        

        Artist.set_renderer(self, renderer)

    def _X_changed(self, old, newx):
        N = newx.shape[0]
        codes = Path.LINETO*npy.ones(N, dtype=npy.uint8)
        codes[0] = Path.MOVETO

        # todo, having touble setting Model to default to Identity so
        # allowing None as a hack workaround
        if self.model is not None:
            modelx = self.model(newx)
        else:
            modelx = newx
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


class Rectangle(Artist):
    facecolor = mtraits.Color('Yellow')
    edgecolor = mtraits.Color('Black')
    edgewidth = mtraits.LineWidth(1.0)
    lbwh      = traits.Array('d', (4,), [0,0,1,1])
    path      = mtraits.Path()
    zorder    = traits.Float(1.0)

    def __init__(self):
        Artist.__init__(self)
        self.facecolor = 'yellow'
        self.edgecolor = 'black'
        self.edgewidth = 1.0
        self.lbwh = 0,0,1,1
        self.path = Path()
        
        self.sync_trait('facecolor', self.path, 'fillcolor', mutual=False)
        self.sync_trait('edgecolor', self.path, 'strokecolor', mutual=False)
        self.sync_trait('edgewidth', self.path, 'linewidth', mutual=False)
        self.sync_trait('affine', self.path, mutual=False)

        self.pathid = primitiveID()


    def _lbwh_changed(self, old, new):
        l,b,w,h = new
        t = b+h
        r = l+w
        verts = npy.array([(l,b), (l,t), (r, t), (r, b), (0,0)], npy.float_)
        codes = Path.LINETO*npy.ones(5, npy.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY

        self.path.pathdata = codes, verts

    def set_renderer(self, renderer):
        if self.renderer is not None:
            self.renderer.remove_path(self.pathid)

        renderer.add_path(self.pathid, self.path)
        Artist.set_renderer(self, renderer)

    def draw(self):
        if self.renderer is None:
            raise RuntimeError('First call set_renderer')

        if not self.visible: return
        self.renderer.render_path(self.pathid)
        
class Figure:
    def __init__(self):
        self.renderer = None
        self.artistd = dict()

    def draw(self):
        if self.renderer is None:
            raise RuntimeError('call set_renderer renderer first')

        dsu = [(artist.zorder, artist.artistid, artist) for artist in self.artistd.values()]
        dsu.sort()
        for zorder, artistid, artist in dsu:
            artist.draw()
    
    def set_renderer(self, renderer):
        self.renderer = renderer

        for artist in self.artistd.values():
            artist.set_renderer(renderer)
    


class AxesCoords(traits.HasTraits):
    xviewlim   = mtraits.Interval()
    yviewlim   = mtraits.Interval()
    affineview = mtraits.Affine()
    affineaxes = mtraits.Affine()  
    affine     = mtraits.Affine()        

    def __init__(self):
        self.xviewlim = npy.array([0., 1.])
        self.yviewlim = npy.array([0., 1.])
        self.affineview = affine_identity()
        self.affineaxes = affine_identity()
        self.affine = affine_identity()   
        
    def _affineview_changed(self, old, new):
        self.affine = npy.dot(self.affineaxes, new)

    def _affineaxes_changed(self, old, new):
        self.affine = npy.dot(new, self.affineview)
        
    def _xviewlim_changed(self, old, new):

        xmin, xmax = new
        scale = 1./(xmax-xmin)
        tx = -xmin*scale
        self.affineview[0][0] = scale
        self.affineview[0][-1] = tx
        self.affine = npy.dot(self.affineaxes, self.affineview)
        
    def _yviewlim_changed(self, old, new):
        ymin, ymax = new
        scale = 1./(ymax-ymin)
        ty = -ymin*scale
        self.affineview[1][1] = scale
        self.affineview[1][-1] = ty
        self.affine = npy.dot(self.affineaxes, self.affineview)
                                       

x1 = npy.arange(0, 10., 0.05)
x2 = npy.arange(0, 10., 0.1)
y1 = npy.cos(2*npy.pi*x1)
y2 = 10*npy.exp(-x1)

# the axes rectangle
axrect1 = [0.1, 0.1, 0.4, 0.4]
coords1 = AxesCoords()
coords1.affineaxes = affine_axes(axrect1)

fig = Figure()

line1 = Line()
line1.X = npy.array([x1,y1]).T

line1.setp(color='blue', linewidth=2.0, marker='s', markersize=5.0,
          markerfacecolor='green', markeredgewidth=0.5)
coords1.sync_trait('affine', line1, mutual=False)

fig.artistd[line1.artistid] = line1


rect1 = Rectangle()
rect1.lbwh = [0,0,1,1]
rect1.facecolor = 'white'
fig.artistd[rect1.artistid] = rect1
coords1.sync_trait('affineaxes', rect1, 'affine', mutual=False)


# update the view limits, all the affines should be automagically updated
coords1.xviewlim = 0, 10
coords1.yviewlim = -1.1, 1.1



if 1:
    axrect2 = [0.55, 0.55, 0.4, 0.4]
    coords2 = AxesCoords()
    coords2.affineaxes = affine_axes(axrect2)


    r = npy.arange(0.0, 1.0, 0.01)
    theta = r*4*npy.pi
    
    line2 = Line()
    line2.model = Polar()
    line2.setp(color='#ee8d18', linewidth=2.0)
    line2.X = npy.array([r, theta]).T
    coords2.sync_trait('affine', line2, mutual=False)

    rect2 = Rectangle()
    rect2.lbwh = [0,0,1,1]
    rect2.facecolor = '#d5de9c'
    coords2.sync_trait('affineaxes', rect2, 'affine', mutual=False)

    fig.artistd[line2.artistid] = line2
    fig.artistd[rect2.artistid] = rect2    


    coords2.xviewlim = -1.1, 1.1
    coords2.yviewlim = -1.1, 1.1



if 1:
    renderer = RendererAgg(600,400)
    fig.set_renderer(renderer)
    fig.draw()
    renderer.show()
