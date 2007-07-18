import enthought.traits.api as traits

from matplotlib import agg
import numpy as npy

import mtraits  # some handy traits for mpl

class Func:
    def __call__(self, X):
        'transform the numpy array with shape N,2'
        raise NotImplementedError
    
    def invert(self, x, y):
        'invert the point x, y'
        raise NotImplementedError

    def point(self, x, y):
        'transform the point x, y'
        raise NotImplementedError

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

    
identity = Identity()


class Path(traits.HasTraits):
    """
    The path is an object that talks to the backends, and is an
    intermediary between the high level path artists like Line and
    Polygon, and the backend renderer
    """
    MOVETO, LINETO, CLOSEPOLY = range(3)
    
    strokecolor = mtraits.color('black')
    fillcolor = mtraits.color('blue')
    alpha = mtraits.alpha(1.0)
    linewidth = mtraits.linewidth(1.0)
    antialiased = mtraits.flexible_true_trait
    verts= mtraits.verts
    codes = mtraits.codes

mtraits.path = traits.Trait(Path())
        
class AggPath:
    def __init__(self, path):
        """
        Path stored with agg data structs        
        """
        MOVETO, LINETO, CLOSEPOLY = Path.MOVETO, Path.LINETO, Path.CLOSEPOLY
        aggpath = agg.path_storage()
        verts = path.verts
        codes = path.codes
        for i in range(len(verts)):
            x, y = verts[i]
            code = codes[i]
            if code==MOVETO:
                aggpath.move_to(x, y)
            elif code==LINETO:
                aggpath.line_to(x, y)                
            elif code==CLOSEPOLY:
                aggpath.close_polygon()

        self.fillcolor = self.color_to_rgba8(path.fillcolor)
        self.strokecolor = self.color_to_rgba8(path.strokecolor)

        self.aggpath = aggpath
        self.alpha = float(path.alpha)
        self.linewidth = float(path.linewidth)
        self.antialiased = bool(path.antialiased)

    def color_to_rgba8(self, color):
        if color is None: return None
        rgba = [int(255*c) for c in color.r, color.g, color.b, color.a]

        return agg.rgba8(*rgba)

# coordinates:
#
#   artist model : a possibly nonlinear transformation (Func instance)
#     to a separable cartesian coordinate, eg for polar is takes r,
#     theta -> r*cos(theta), r*sin(theta)
#
#   affineData : an affine 3x3 matrix that takes model output and
#     transforms it to axes 0,1.  We are kind of stuck with the
#     mpl/matlab convention that 0,0 is the bottom left of the axes,
#     even though it contradicts pretty much every GUI layout in the
#     world
#
#   affineFigure: an affine 3x3 that transforms an axes.view into figure
#     0,1 
#
#   affineDisplay : takes an affine 3x3 and puts figure view into display.  0,
#      0 is left, top, which is the typical coordinate system of most
#      graphics formats

class Renderer:
    def __init__(self, width, height):
        self.width, self.height = width, height

        # almost all renderers assume 0,0 is left, upper, so we'll flip y here by default
        self.displayview  = npy.array(
            [[width, 0, 0], [0, -height, height], [0, 0, 1]], dtype=npy.float_)
        self.pathd = dict() # dict mapping path id -> path instance
        
    def push_affine(self, affine):
        'set the current affine'
        self.affine = npy.dot(self.displayview, affine)

    def add_path(self, pathid, path):
        self.pathd[pathid] = path

    def remove_path(self, pathid):
        if pathid in self.pathd:
            del self.pathd[pathid]

    def render_path(self, pathid):
        pass
            


class RendererAgg(Renderer):
    gray = agg.rgba8(128,128,128,255)    
    white = agg.rgba8(255,255,255,255)
    blue = agg.rgba8(0,0,255,255)

    def __init__(self, width, height):
        Renderer.__init__(self, width, height)

        self.aggpathd = dict() # map path ids to AggPaths
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
        Renderer.add_path(self, pathid, path)
        self.aggpathd[pathid] = AggPath(path)

    def remove_path(self, pathid):
        Renderer.remove_path(self, pathid)
        if pathid in self.aggpathd:
            del self.aggpathd[pathid]

    def push_affine(self, affine):
        'set the current affine'
        Renderer.push_affine(self, affine)
        a, b, tx = self.affine[0]
        c, d, ty = self.affine[1]
        self.trans = agg.trans_affine(a,b,c,d,tx,ty)


    def render_path(self, pathid):
        if self.trans is None:
            raise RuntimeError('you must first push_affine')



        aggpath = self.aggpathd[pathid]

        if aggpath.antialiased:
            renderer = self.renderer
            scanline = self.scanline
            render_scanlines = agg.render_scanlines_rgba
        else:
            renderer = self.rendererbin
            scanline = self.scanlinebin
            render_scanlines = agg.render_scanlines_bin_rgba

        renderer.color_rgba8( aggpath.strokecolor )
        transpath = agg.conv_transform_path(aggpath.aggpath, self.trans)

        if aggpath.fillcolor is not None:
            self.rasterizer.add_path(transpath)
            renderer.color_rgba8( aggpath.fillcolor )
            render_scanlines(self.rasterizer, scanline, renderer);
            
        stroke = agg.conv_stroke_transpath(transpath)
        stroke.width(aggpath.linewidth)
        self.rasterizer.add_path(stroke)
        renderer.color_rgba8( aggpath.strokecolor )        
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





def rectangle(l, b, w, h, facecolor='yellow', edgecolor='black',
              edgewidth=1.0, alpha=1.0):

    t = b+h
    r = l+w
    verts = npy.array([(l,b), (l,t), (r, t), (r, b), (0,0)], npy.float_)
    codes = Path.LINETO*npy.ones(5, npy.uint8)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    path = Path()
    part.verts = verts
    path.codes = codes
    path.strokecolor = edgecolor
    path.fillcolor = facecolor
    path.linewidth = edgewidth
    path.alpha = alpha
    return path

def line(x, y, color='black', linewidth=1.0, alpha=1.0, antialiased=True,
         model=identity):
    X = npy.asarray([x,y]).T
    numrows, numcols = X.shape

    codes = Path.LINETO*npy.ones(numrows, npy.uint8)
    codes[0] = Path.MOVETO

    path = Path()
    path.verts = model(X)
    path.codes = codes        
    path.fillcolor = None
    path.strokecolor = color
    path.strokewidth = linewidth
    path.alpha = alpha
    path.antialiased = antialiased
    return path
    
        

class AxesCoords(traits.HasTraits):
    xviewlim = mtraits.interval
    yviewlim = mtraits.interval
    affineview = mtraits.affine
    affineaxes = mtraits.affine    
    affine = mtraits.affine        

        
    def _affineview_changed(self, old, new):
        print 'affine view changed'
        self.affine = npy.dot(self.affineaxes, new)

    def _affineaxes_changed(self, old, new):
        print 'affine axes changed'
        self.affine = npy.dot(new, self.affineview)

        
    def _xviewlim_changed(self, old, new):
        print 'xviewlim changed'
        xmin, xmax = new
        scale = 1./(xmax-xmin)
        tx = -xmin*scale
        self.affineview[0][0] = scale
        self.affineview[0][-1] = tx
        self.affine = npy.dot(self.affineaxes, self.affineview)
        print '\t', self.affine
        
    def _yviewlim_changed(self, old, new):
        print 'yviewlim changed'
        ymin, ymax = new
        scale = 1./(ymax-ymin)
        ty = -ymin*scale
        self.affineview[1][1] = scale
        self.affineview[1][-1] = ty
        self.affine = npy.dot(self.affineaxes, self.affineview)
        print '\t', self.affine
                                       

class Figure:
    def __init__(self):
        self.renderer = None
        self._pathid = 0
        self.pathd = dict()

    def add_path(self, path):
        id_ = self._pathid
        self.pathd[id_] = path
        self._pathid += 1
        return id_

    def remove_path(self, pathid):
        if pathid in self.pathd:        
            del self.pathd[pathid]
        if self.renderer is not None:
            self.renderer.remove_path(pathid)
            
    def draw(self):
        if self.renderer is None:
            raise RuntimeError('call set_renderer renderer first')

        for pathid, path in self.pathd.items():
            print 'path', pathid, path.affine
            renderer.push_affine(path.affine)
            renderer.render_path(pathid)

    
    def set_renderer(self, renderer):
        self.renderer = renderer
        for pathid, path in self.pathd.items():
            renderer.add_path(pathid, path)


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


coords1 = AxesCoords()
coords1.affineaxes = affine_axes([0.55, 0.55, 0.4, 0.4]) # upper right quadrant



fig = Figure()

x = npy.arange(0, 10, 0.01)
y1 = npy.cos(2*npy.pi*x)
y2 = 10*npy.exp(-x)

line1 = line(x, y1, color='blue', linewidth=2.0)
line1.affine = coords1.affine

fig.add_path(line1)

print 'before', line1.affine
# update the view limits, all the affines should be automagically updated
coords1.xviewlim = 0, 10
coords1.yviewlim = -1.1, 1.1

print 'after', line1.affine


if 0:
    coords2 = AxesCoords()
    coords2.xviewlim = coords1.xviewlim  # share the x axis
    coords2.affineaxes = affine_axes([0.1, 0.1, 0.4, 0.4]) # lower left quadrant


    line2 = line(x, y2, color='red', linewidth=2.0)
    line2.affine = coords2.affine
    coords2.yviewlim = 0, 10
    fig.add_path(line2)


if 0:
    renderer = RendererAgg(600,400)
    fig.set_renderer(renderer)
    fig.draw()
    renderer.show()
