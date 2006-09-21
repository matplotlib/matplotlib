"""
A gd backend http://newcenturycomputers.net/projects/gdmodule.html
"""


from __future__ import division
import sys, os, math, warnings

try: import gd
except ImportError:
    print >>sys.stderr, 'You must first install the gd module http://newcenturycomputers.net/projects/gdmodule.html'
    sys.exit()

        
    
from matplotlib.backend_bases import RendererBase, \
     GraphicsContextBase, FigureManagerBase, FigureCanvasBase
from matplotlib import verbose
from matplotlib._pylab_helpers import Gcf
from matplotlib.cbook import enumerate, pieces, is_string_like
from matplotlib.colors import colorConverter
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
from matplotlib.font_manager import fontManager
from matplotlib.numerix import ones, array, nx, asarray
# support old font names
if (os.environ.has_key('GDFONTPATH') and not
    os.environ.has_key('TTFPATH')):
    os.environ['TTFPATH'] = os.environ['GDFONTPATH']




PIXELS_PER_INCH = 96  # constant GD uses for screen DPI


def round(x):
    return int(math.floor(x+0.5))


class RendererGD(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """

    
    # todo: can gd support cap and join styles?
    def __init__(self, im, dpi):
        "Initialize the renderer with a gd image instance"
        self.im = im
        self._cached = {}  # a map from get_color args to colors

        self.width, self.height = im.size()
        self.dpi = dpi


    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return self.width, self.height

    def get_text_width_height(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with fontsize in points
        """

        size = prop.get_size_in_points()
        font = fontManager.findfont(prop)

        scale = self.get_text_scale()
        try:
            llx, lly, lrx, lry, urx, ury, ulx, uly = \
                 self.im.get_bounding_rect(
                font, scale*size, 0.0, (0,0), s)
        except ValueError:
            raise RuntimeError('Could not load font %s.  Try setting TTFFONTPATH to include this font' % fontname)
            
        w = abs(lrx - llx)
        h = abs(lly - uly)
        return w, h

                              
    def flipy(self):
        'return true if y small numbers are top for renderer'
        return True
    

    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2, rotation):
        """
        Draw an arc centered at x,y with width and height and angles
        from 0.0 to 360.0
        """

        center = int(x), self.height-int(y)
        wh = int(width), int(height)
        a1, a2 = int(angle1), int(angle2)
        if rgbFace is not None:
            color = self.get_gd_color( rgbFace )
            self.im.filledEllipse(
                center, wh, color)
        color = self.get_gd_color( gc.get_rgb() )
        self.im.arc(center, wh, a1, a2, color)
        self.flush_clip()

    def draw_line(self, gc, x1, y1, x2, y2):
        """
        Draw a single line from x1,y1 to x2,y2
        """
        self.draw_lines(gc, array([x1, x2]), array([y1, y2]))
        
    def draw_lines(self, gc, x, y):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """

        x = x.astype(nx.Int16)
        y = self.height*ones(y.shape, nx.Int16) - y.astype(nx.Int16)  
        style = self._set_gd_style(gc)
        self.im.lines( zip(x,y), style)
        self.flush_clip()

    def draw_point(self, gc, x, y):
        """
        Draw a single point at x,y
        """
        self.im.setPixel((int(x),self.height-int(y)),
                         self.get_gd_color( gc.get_rgb() ))
        self.flush_clip()


    
    def draw_polygon(self, gc, rgbFace, points):
        """
        Draw a polygon.  points is a len vertices tuple, each element
        giving the x,y coords a vertex
        """

        edgecolor = self.get_gd_color( gc.get_rgb() )

        points = [(int(x), self.height-int(y)) for x,y in points]


        if rgbFace is not None:
            facecolor = self.get_gd_color( rgbFace )
            self.im.filledPolygon(points, facecolor)
        else: facecolor = None
        if edgecolor != facecolor:
            self.im.polygon(points, edgecolor)
        self.flush_clip()
        
    def draw_rectangle(self, gc, rgbFace, x, y, width, height):
        """
        Draw a rectangle at lower left x,y with width and height
        If filled=True, fill the rectangle with the gc foreground
        gc is a GraphicsContext instance
        """

        lb = int(x), self.height-int(y)
        ur = int(x+width), self.height-int((y+height))
        edgecolor = self.get_gd_color( gc.get_rgb() )
        
        if rgbFace is not None:
            facecolor = self.get_gd_color( rgbFace )
            self.im.filledRectangle(ur, lb, facecolor)
        else: facecolor = None

        if edgecolor != facecolor:
            self.im.rectangle(ur, lb, edgecolor)
        self.flush_clip()

    def draw_text(self, gc, x, y, s, prop, angle, ismath):            
        """
        Render the text using the RendererGD instance
        """

        size = prop.get_size_in_points()
        font = fontManager.findfont(prop)

        x = int(x)
        y = int(y)

        color = self.get_gd_color( gc.get_rgb() )

        angle *= math.pi/180.0

        scale = self.get_text_scale()
        self.im.string_ft(font, scale*size, angle,
                          (x, y), s, color)
        self.flush_clip()

    def finish(self):
        pass
        #self.im.writePng( file('xx.png', 'w') )

    
    def flush_clip(self):
        imw, imh = self.im.size()
        lb = 0, 0
        ur = imw, imh
        self.im.setClip(ur, lb)        


    def get_gd_color(self, rgb):
        """
        RGB is a unit RGB tuple, return a gd color
        """

        r,g,b = rgb
        rgbi = (int(r*255),int(g*255),int(b*255))
        
        try: return self._cached[rgbi]
        except KeyError: pass

        color = self.im.colorAllocate(  rgbi )

        if color==-1:
            warnings.warn('Unable to allocate color %1.3f, %1.3f, %1.3f; using nearest neighbor' % rgb)
            color = self.im.colorClosest(rgbi)

        self._cached[rgbi] = color
        return color




    def get_text_scale(self):
        """
        Return the scale factor for fontsize taking screendpi and pixels per
        inch into account
        """
        return self.dpi.get()/PIXELS_PER_INCH
    
    def new_gc(self):
        """
        Return an instance of a GraphicsContextGD
        """
        return GraphicsContextGD( self.im, self )

    def _set_gd_style(self, gc):
        color = self.get_gd_color( gc.get_rgb() )
        offset, dashes = gc.get_dashes()

        if dashes is not None:
            pixels = self.points_to_pixels(dashes)
            style = []
            for on, off in pieces(pixels):
                if on<1: on = 1
                else: on = round(on)
                if off<1: off = 1
                else: off = round(off)

                style.extend([color]*on)
                style.extend([gd.gdTransparent]*off)
            self.im.setStyle(style)
            return gd.gdStyled
        else:
            if gc.get_antialiased():
                self.im.setAntiAliased(color)
                return gd.gdAntiAliased
            else:
                self.im.setStyle([color])
                return gd.gdStyled


    def points_to_pixels(self, points):
        """
        convert point measures to pixes using dpi and the pixels per
        inch of the display
        """
        return asarray(points)*(PIXELS_PER_INCH/72.0*self.dpi.get()/72.0)

        
class GraphicsContextGD(GraphicsContextBase):
    """
    The graphics context provides the color, line styles, etc...  See
    the gtk and postscript backends for examples of mapping the
    graphics context attributes (cap styles, join styles, line widths,
    colors) to a particular backend.      """
    def __init__(self, im, renderer):
        """
        Initialize with a gd image
        """
        GraphicsContextBase.__init__(self)
        self.im = im
        self.renderer = renderer
        

    def set_clip_rectangle(self, rectangle):
        GraphicsContextBase.set_clip_rectangle(self, rectangle)
        x,y,w,h = rectangle
        imw, imh = self.im.size()
        lb = int(x), imh-int(y)
        ur = int(x+w), imh-int(y+h)
        self.im.setClip(ur, lb)        

    def set_linestyle(self, style):
        GraphicsContextBase.set_linestyle(self, style)
        offset, dashes = self.dashd[style]
        self.set_dashes(offset, dashes)

    def set_linewidth(self, lw):
        GraphicsContextBase.set_linewidth(self, lw)
        pixels = self.renderer.points_to_pixels(lw)
        if pixels<1: pixels = 1
        else: pixels = round(pixels)
        self.im.setThickness(pixels)
              
########################################################################
#    
# The following functions and classes are for matlab compatibility
# mode (pylab) and implement figure managers, etc...
#
########################################################################


def new_figure_manager(num, *args, **kwargs):
    """
    Add a new figure num (default autoincrement).  For GUI
    backends, you'll need to instantiate a new window and embed
    the figure in it.
    """
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    canvas = FigureCanvasGD(thisFig)
    manager = FigureManagerGD(canvas, num)
    return manager


class FigureCanvasGD(FigureCanvasBase):
    
    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait', **kwargs):

        """
        Render the figure to hardcopy using self.renderer as the
        renderer if neccessary

        filename can be a string filename or writable file instance

        """

        origDPI = self.figure.dpi.get()
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()

        self.figure.dpi.set(dpi)
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        im = self.draw()

        if is_string_like(filename):
            basename, ext = os.path.splitext(filename)
            if not len(ext): filename += '.png'
 
        im.writePng( filename )

        self.figure.dpi.set(origDPI)
        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)

    def draw(self):
        """
        Draw to a gd image and return the image instance
        
        """

        left, bottom, width, height = self.figure.bbox.get_bounds()
        im = gd.image((int(width), int(height)))

        if not hasattr(im, 'setAntiAliased'):
            raise RuntimeError('gd_requirements_failed')
        renderer = RendererGD(im, self.figure.dpi)
        self.figure.draw(renderer)
        renderer.finish()

        return im

class FigureManagerGD(FigureManagerBase):
    """
    This class manages all the figures for matlab mode
    """
    pass



########################################################################
#    
# Now just provide the standard names that backend.__init__ is expecting
# 
########################################################################

FigureManager = FigureManagerGD

         
