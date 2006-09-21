"""
This ia a paint (libart) backend

You can select it as a backend with

  import matplotlib
  matplotlib.use('Paint')

REQUIREMENTS

  backend_paint requires pypaint-0.??, which in turn requires
  libart and freetype1
"""

from __future__ import division
import sys
import os
import paint
from matplotlib import verbose
from matplotlib.numerix import asarray

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase,\
     GraphicsContextBase, FigureCanvasBase, FigureManagerBase
from matplotlib.cbook import enumerate
from matplotlib.figure import Figure
from matplotlib.text import Text, _process_text_args

from matplotlib.font_manager import fontManager

"""

 * added dpi instance to renderer so drawing could scale with dpi
 
 * added dash path - JDH

 * reversed the order of fill and stroke for rectangle, arc and
   polygon so edge color would be visible

 * adjusted circle centers
 
"""





#paint/font.c defined dpi as 96
PIXELS_PER_INCH = 96 # a constant used to scale text with dpi


class RendererPaint(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """

    fontd = {} # cache paint font instances
    
    def __init__(self, width, height, dpi):
        """creates a new image"""
        w, h = int(width), int(height)
        self.image = paint.image(w, h)
        self.width, self.height = w, h
        self.dpi = dpi  # for scaling dashes, linewidths

    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return self.width, self.height

    def get_text_width_height(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with fontsize in points
        """
        font = self._get_paint_font(s, prop, 0.0)
        return font.textsize(s)
                              
    def flipy(self):
        'return true if y small numbers are top for renderer'
        return True


    def get_text_scale(self):
        """
        Return the scale factor for fontsize taking screendpi and pixels per
        inch into account
        """
        return self.dpi.get()/PIXELS_PER_INCH

    

    def draw_text(self, gc, x, y, s, prop, angle, ismath):            
        """
        Render the text using the RendererPaint instance
        """
        font = self._get_paint_font(s, prop, angle)

        text_color = self.get_paint_color(gc.get_rgb())
        self.image.text(font, x, y, text_color, s)

    def _get_paint_font(self, s, prop, angle):
        """
        Get the paint font for text instance t, cacheing for efficiency
        """

        fname = fontManager.findfont(prop)
        size = self.get_text_scale() * prop.get_size_in_points()

        props =  fname, size, angle

        font = self.fontd.get(props)
        if font is None:
            font = paint.font(*props)
            self.fontd[props] = font
        return font

    def get_paint_color(self, rgb):
        """returns a paint color object based on the given rgb tuple"""
        r,g,b = rgb
        return paint.rgb(int(r*255),int(g*255),int(b*255))

    def draw_arc(self, gcEdge, rgbFace, x, y, width, height, angle1, angle2, rotation):
        """
        Draw an arc centered at x,y with width and height and angles
        from 0.0 to 360.0.

        If rgbFace is not None, fill the rectangle with it.  gcEdge
        is a GraphicsContext instance
        """
        arc = paint.arc(x-0.5*width, self.height - (y-0.5*height),
                        x+0.5*width, self.height - (y+0.5*height),
                        angle1, angle2)
        if rgbFace:
            self.image.fill(arc, self.get_paint_color(rgbFace))
        self.image.stroke(arc, self.get_paint_color(gcEdge.get_rgb()),
                          self.points_to_pixels(gcEdge.get_linewidth()))

    
    def draw_line(self, gc, x1, y1, x2, y2):
        """
        Draw a single line from x1,y1 to x2,y2
        """
        path = paint.line(x1, self.height - y1, x2, self.height - y2)
        path = self.dash_path(gc, path)
        self.image.stroke(path, \
                          self.get_paint_color(gc.get_rgb()),
                          self.points_to_pixels(gc.get_linewidth()))
        

    def dash_path(self, gc, path):
        """
        Add dashes to the path and return it if dashes are set
        """
        offset, dashes = gc.get_dashes()
        if dashes is not None:

            dashes = tuple(self.points_to_pixels(asarray(dashes)))            
            return path.dash(offset, dashes)
        else:
            return path
        
    def draw_lines(self, gc, x, y):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        assert(len(x)==len(y))
        # faster as a list comp

        path = [(paint.MOVETO, x[0], self.height-y[0])]
        path.extend( [ (paint.LINETO, x[i], self.height-y[i]) for i in range(1, len(x))])

        path = self.dash_path(gc, paint.make_path(path))
        self.image.stroke(path,
                          self.get_paint_color(gc.get_rgb()),
                          self.points_to_pixels(gc.get_linewidth()))
            

    def draw_polygon(self, gcEdge, rgbFace, points):
        """
        Draw a polygon.  points is a len vertices tuple, each element
        giving the x,y coords a vertex.

        If rgbFace is not None, fill the rectangle with it.  gcEdge
        is a GraphicsContext instance
        """
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        path = [(paint.MOVETO, x[0], self.height - y[0])]
        for i in range(len(x)-1):
            path.append((paint.LINETO, x[i+1], self.height - y[i+1]))
        path.append((paint.LINETO, x[0], self.height - y[0]))
        path = paint.make_path(path)

        if rgbFace:
            self.image.fill(path, self.get_paint_color(rgbFace))
        self.image.stroke(path,
                          self.get_paint_color(gcEdge.get_rgb()),
                          self.points_to_pixels(gcEdge.get_linewidth()))
    def draw_rectangle(self, gcEdge, rgbFace, x, y, width, height):
        """
        Draw a rectangle at lower left x,y with width and height.

        If rgbFace is not None, fill the rectangle with it.  gcEdge
        is a GraphicsContext instance
        """
        path = paint.rect(x, self.height - y, x+width, self.height - (y+height))

        if rgbFace:
            self.image.fill(path, self.get_paint_color(rgbFace))
        self.image.stroke(path,
                          self.get_paint_color(gcEdge.get_rgb()),
                          self.points_to_pixels(gcEdge.get_linewidth()))

    def draw_point(self, gc, x, y):
        """
        Draw a single point at x,y
        """
        self.image.stroke(
            paint.line(x, self.height - y, x, self.height - y),
            self.get_paint_color(gc.get_rgb()),
            self.points_to_pixels(gc.get_linewidth()))

    def points_to_pixels(self, points):
        return points*(PIXELS_PER_INCH/72.0*self.dpi.get()/72.0)
        

              
class FigureCanvasPaint(FigureCanvasBase):
        
    def draw(self):
        """
        Render the figure using RendererPaint instance renderer
        """
        t1,t2,width,height = self.figure.bbox.get_bounds()
        renderer = RendererPaint(width,height, self.figure.dpi)
        self.figure.draw(renderer)
        return renderer
        
    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait, '**kwargs):

        """
        Render the figure to hardcopy using self.renderer as the
        renderer if neccessary
        """
        origDPI = self.figure.dpi.get()
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()

        self.figure.dpi.set(dpi)
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        renderer = self.draw()

        basename, ext = os.path.splitext(filename)
        if not len(ext): filename += '.png'
        renderer.image.write_png(filename)        

        # now restore the old figure params
        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)
        self.figure.dpi.set(origDPI)
        
########################################################################
#    
# The following functions and classes are for matlab compatibility
# mode (pylab) and implement window/figure managers,
# etc...
#
########################################################################


def new_figure_manager_paint(num, *args, **kwargs):
    """
    Add a new figure num (default autoincrement).  For GUI
    backends, you'll need to instantiate a new window and embed
    the figure in it.
    """
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    canvas = FigureCanvasPaint(thisFig)
    manager = FigureManagerBase(canvas, num)
    return manager


########################################################################
#    
# Now just provide the standard names that backend.__init__ is expecting
# 
########################################################################
new_figure_manager = new_figure_manager_paint

         
