# see install instructions for enthrought traits2 in mtraits
import enthought.traits.api as traits

from matplotlib import agg
import numpy as npy

import mtraits  # some handy traits for mpl
        

class Renderer:
    def __init__(self, width, height):
        self.width, self.height = width, height

        # almost all renderers assume 0,0 is left, upper, so we'll flip y here by default
        self.affinerenderer  = npy.array(
            [[width, 0, 0], [0, -height, height], [0, 0, 1]], dtype=npy.float_)
        self.pathd = dict() # dict mapping path id -> path instance
        

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

        renderer.color_rgba8( path.agg_strokecolor )
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
    
    strokecolor = mtraits.Color('black')
    fillcolor = mtraits.Color('blue')
    alpha = mtraits.Alpha(1.0)
    linewidth = mtraits.Linewidth(1.0)
    antialiased = mtraits.FlexibleTrueTrait
    pathdata = mtraits.PathData
    affine = mtraits.Affine
    
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

        
        path.sync_trait('strokecolor', self)
        path.sync_trait('fillcolor', self)
        path.sync_trait('alpha', self)
        path.sync_trait('linewidth', self)
        path.sync_trait('antialiased', self)
        path.sync_trait('pathdata', self)
        path.sync_trait('affine', self)

    def _pathdata_changed(self, olddata, newdata):
        MOVETO, LINETO, CLOSEPOLY = Path.MOVETO, Path.LINETO, Path.CLOSEPOLY
        agg_path = agg.path_storage()
        codes, verts = newdata
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
        
        self.agg_path = agg_path
        
    def _fillcolor_changed(self, oldcolor, newcolor):        
        self.agg_fillcolor = self.color_to_rgba8(newcolor)

    def _strokecolor_changed(self, oldcolor, newcolor):                

        c = self.color_to_rgba8(newcolor)
        #print 'stroke change: old=%s, new=%s, agg=%s, ret=%s'%(
        #    oldcolor, newcolor, self.agg_strokecolor, c)
        self.agg_strokecolor = c


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


class Rectangle(Path):
    facecolor = mtraits.Color('Yellow')
    edgecolor = mtraits.Color('Black')
    edgewidth = mtraits.Linewidth(1.0)
    
    def __init__(self, lbwh, **kwargs):

        # support some legacy names
        self.sync_trait('facecolor', self, 'fillcolor', True)
        self.sync_trait('edgecolor', self, 'strokecolor', True)
        self.sync_trait('edgewidth', self, 'strokewidth', True)

        for k,v in kwargs.items():
            setattr(self, k, v)

        l,b,w,h = lbwh
        t = b+h
        r = l+w
        verts = npy.array([(l,b), (l,t), (r, t), (r, b), (0,0)], npy.float_)
        codes = Path.LINETO*npy.ones(5, npy.uint8)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY

        self.pathdata = codes, verts



def Alias(name):
    return Property(lambda obj: getattr(obj, name),
                    lambda obj, val: setattr(obj, name, val))

class Line(Path):
    # aliases for matplotlib compat
    color = mtraits.Color('blue')
    linewidth = mtraits.Linewidth(1.0)

    
    def __init__(self, x, y, model=identity, **kwargs):
        """
        The model is a function taking Nx2->Nx2.  This is where the
        nonlinear transformation can be used
        """

        self.sync_trait('color', self, 'strokecolor', True)
        self.sync_trait('linewidth', self, 'strokewidth', True)

        # probably a better way to do this with traits
        for k,v in kwargs.items():
            setattr(self, k, v)

        X = npy.array([x,y]).T
        numrows, numcols = X.shape

        codes = Path.LINETO*npy.ones(numrows, npy.uint8)
        codes[0] = Path.MOVETO

        verts = model(X)

        self.pathdata = codes, verts    
        self.fillcolor = None

        


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

        for pathid in self.pathd:
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


class AxesCoords(traits.HasTraits):
    xviewlim = mtraits.Interval
    yviewlim = mtraits.Interval
    affineview = mtraits.Affine
    affineaxes = mtraits.Affine    
    affine = mtraits.Affine        

        
    def _affineview_changed(self, old, new):
        #print 'affineview changed before:\n', self.affine
        self.affine = npy.dot(self.affineaxes, new)
        #print 'affineview changed after:\n', self.affine

    def _affineaxes_changed(self, old, new):
        #print 'affineaxes changed before:\n', self.affine
        self.affine = npy.dot(new, self.affineview)
        #print 'affineaxes changed after:\n', self.affine
        
    def _xviewlim_changed(self, old, new):

        #print 'xviewlim changed before:\n', self.affine
        xmin, xmax = new
        scale = 1./(xmax-xmin)
        tx = -xmin*scale
        self.affineview[0][0] = scale
        self.affineview[0][-1] = tx
        self.affine = npy.dot(self.affineaxes, self.affineview)
        #print 'xviewlim changed after:\n', self.affine
        
    def _yviewlim_changed(self, old, new):
        #print 'yviewlim changed before:\n', self.affine
        ymin, ymax = new
        scale = 1./(ymax-ymin)
        ty = -ymin*scale
        self.affineview[1][1] = scale
        self.affineview[1][-1] = ty
        self.affine = npy.dot(self.affineaxes, self.affineview)
        #print 'yviewlim changed after:\n', self.affine
                                       

x = npy.arange(0, 10, 0.01)
y1 = npy.cos(2*npy.pi*x)
y2 = 10*npy.exp(-x)

# the axes rectangle
axrect1 = [0.1, 0.1, 0.4, 0.4]
coords1 = AxesCoords()
coords1.affineaxes = affine_axes(axrect1)

fig = Figure()

line1 = Line(x, y1, color='blue', linewidth=2.0)
rect1 = Rectangle([0,0,1,1], facecolor='white')
coords1.sync_trait('affine', line1)
coords1.sync_trait('affineaxes', rect1, 'affine')

fig.add_path(rect1)
fig.add_path(line1)

# update the view limits, all the affines should be automagically updated
coords1.xviewlim = 0, 10
coords1.yviewlim = -1.1, 1.1


# the axes rectangle
axrect2 = [0.55, 0.55, 0.4, 0.4]
coords2 = AxesCoords()
coords2.affineaxes = affine_axes(axrect2)


r = npy.arange(0.0, 1.0, 0.01)
theta = r*4*npy.pi

line2 = Line(r, theta, model=Polar(), color='#ee8d18', linewidth=2.0)
rect2 = Rectangle([0,0,1,1], facecolor='#d5de9c')
coords2.sync_trait('affine', line2)
coords2.sync_trait('affineaxes', rect2, 'affine')

fig.add_path(rect2)
fig.add_path(line2)

# update the view limits, all the affines should be automagically updated
coords2.xviewlim = -1.1, 1.1
coords2.yviewlim = -1.1, 1.1



if 1:
    renderer = RendererAgg(600,400)
    fig.set_renderer(renderer)
    fig.draw()
    renderer.show()
