"""
An agg http://antigrain.com/ backend

Features that are implemented

 * capstyles and join styles
 * dashes
 * linewidth 
 * lines, rectangles, ellipses
 * clipping to a rectangle
 * output to RGBA and PNG
 * alpha blending
 * DPI scaling properly - everything scales properly (dashes, linewidths, etc)
 * draw polygon 
 * freetype2 w/ ft2font

TODO:

  * allow save to file handle

  * allow load from png

  * integrate screen dpi w/ ppi and text
  
INSTALLING 

  REQUIREMENTs

    python2.2+
    Numeric 22+
    agg2 (see below)
    freetype 2
    libpng
    libz
    
  Install AGG2 (cut and paste below into xterm should work)

    wget http://www.antigrain.com/agg2.tar.gz
    tar xvfz agg2.tar.gz
    cd agg2
    make

    (Optional) if you want to make the examples:
    cd examples/X11
    make

  Installing backend_agg

     
   Edit setup.py: change aggsrc to point to the agg2 src tree and
   replace if 0: with if 1: in the backend_agg section

   Then just do the usual thing: python setup.py build

   Please let me know if you encounter build problems, and tell me
   platform, gcc version, etc...  Currently the paths in setupext.py
   assume as linux like filesystem (eg X11 include dir, location of
   libttf, etcc) so you may need to tweak these

  Using agg backend

    python somefile.py -dAgg   

  or

    import matplotlib
    matplotlib.use('Agg')
    

"""
from __future__ import division

import os, sys
from matplotlib import verbose
from matplotlib.numerix import array, Float

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase,\
     GraphicsContextBase, FigureManagerBase, FigureCanvasBase

from matplotlib.cbook import enumerate, is_string_like, exception_to_str
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.ft2font import FT2Font
from matplotlib.mathtext import math_parse_s_ft2font


import matplotlib.numerix

if matplotlib.numerix.which[0] == "numarray":
    from _na_backend_agg import RendererAgg as _RendererAgg
else:
    from _nc_backend_agg import RendererAgg as _RendererAgg

backend_version = 'v2.2'
_fontd = {}     # a map from fname to font instances


class RendererAgg(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """
    
    debug=1
    def __init__(self, width, height, dpi):
        if __debug__: verbose.report('RendererAgg.__init__', 'debug-annoying')
        self.dpi = dpi
        self.width = width
        self.height = height
        self._renderer = _RendererAgg(int(width), int(height), dpi.get(),
                                    debug=False)
        self.draw_polygon = self._renderer.draw_polygon
        self.draw_rectangle = self._renderer.draw_rectangle
        self.draw_lines = self._renderer.draw_lines
        self.draw_markers = self._renderer.draw_markers
        self.draw_image = self._renderer.draw_image
        self.draw_line_collection = self._renderer.draw_line_collection
        self.draw_poly_collection = self._renderer.draw_poly_collection
        self.draw_regpoly_collection = self._renderer.draw_regpoly_collection

    def draw_arc(self, gcEdge, rgbFace, x, y, width, height, angle1, angle2):
        """
        Draw an arc centered at x,y with width and height and angles
        from 0.0 to 360.0

        If rgbFace is not None, fill the rectangle with that color.  gcEdge
        is a GraphicsContext instance

        Currently, I'm only supporting ellipses, ie angle args are
        ignored
        """
        if __debug__: verbose.report('RendererAgg.draw_arc', 'debug-annoying')
        self._renderer.draw_ellipse(
            gcEdge, rgbFace, x, y, width/2, height/2)  # ellipse takes radius
        

    def _draw_image(self, x, y, im):
        """
        Draw the Image instance into the current axes; x, y is the
        upper left hand corner of the image
        """
        if __debug__: verbose.report('RendererAgg.draw_image', 'debug-annoying')
        #self._renderer.draw_image(int(x), int(self.height-y), im)
        self._renderer.draw_image(int(x), int(y), im)        

    def draw_line(self, gc, x1, y1, x2, y2):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        if __debug__: verbose.report('RendererAgg.draw_line', 'debug-annoying')
        x = array([x1,x2], typecode=Float)
        y = array([y1,y2], typecode=Float)
        self._renderer.draw_lines(gc, x, y)


    def draw_point(self, gc, x, y):
        """
        Draw a single point at x,y
        """
        if __debug__: verbose.report('RendererAgg.draw_point', 'debug-annoying')
        rgbFace = gc.get_rgb()
        self._renderer.draw_ellipse(
            gc, rgbFace, x, y, 0.5, 0.5)        

    def draw_mathtext(self, gc, x, y, s, prop, angle):    
        """
        Draw the math text using matplotlib.mathtext
        """
        if __debug__: verbose.report('RendererAgg.draw_mathtext', 'debug-annoying')
        size = prop.get_size_in_points()
        width, height, fonts = math_parse_s_ft2font(
            s, self.dpi.get(), size, angle)
        
        if angle == 90:
            width, height = height, width
        for font in fonts:
            if angle == 90:             
                font.horiz_image_to_vert_image() # <-- Rotate
                self._renderer.draw_text( font, int(x)-width, int(y)-height, gc)
            else:
                self._renderer.draw_text( font, int(x), int(y)-height, gc)                
        if 0:
            self._renderer.draw_rectangle(gc, None,
                                          int(x),
                                          self.height-int(y),
                                          width, height)


    
    def draw_text(self, gc, x, y, s, prop, angle, ismath):
        """
        Render the text
        """
        if __debug__: verbose.report('RendererAgg.draw_text', 'debug-annoying')

        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)

        font = self._get_agg_font(prop)
        if font is None: return None
        if len(s)==1 and ord(s)>127:

            font.load_char(ord(s))
        else:
            font.set_text(str(s), angle)
        font.draw_glyphs_to_bitmap()

        self._renderer.draw_text(font, int(x), int(y), gc)

        
    def get_text_width_height(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with FontPropertry prop
        """

        if ismath:
            width, height, fonts = math_parse_s_ft2font(
                s, self.dpi.get(), prop.get_size_in_points())
            return width, height
        font = self._get_agg_font(prop)
        font.set_text(str(s), 0.0)  # the width and height of unrotated string
        w, h = font.get_width_height()
        w /= 64.0  # convert from subpixels
        h /= 64.0
        return w+2, h+2

    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return self.width, self.height


    def _get_agg_font(self, prop):
        """
        Get the font for text instance t, cacheing for efficiency
        """
        if __debug__: verbose.report('RendererAgg._get_agg_font', 'debug-annoying')

        key = hash(prop)
        font = _fontd.get(key)
        
        if font is None:
            fname = fontManager.findfont(prop)
            font = FT2Font(str(fname))
            _fontd[key] = font

        font.clear()
        size = prop.get_size_in_points()
        font.set_size(size, self.dpi.get())

        return font


    def points_to_pixels(self, points):
        """
        convert point measures to pixes using dpi and the pixels per
        inch of the display
        """
        if __debug__: verbose.report('RendererAgg.points_to_pixels', 'debug-annoying')
        return points*self.dpi.get()/72.0

    def tostring_rgb(self):
        if __debug__: verbose.report('RendererAgg.tostring_rgb', 'debug-annoying')
        return self._renderer.tostring_rgb()

    def tostring_argb(self):
        if __debug__: verbose.report('RendererAgg.tostring_argb', 'debug-annoying')
        return self._renderer.tostring_argb()
        
    def buffer_rgba(self):
        if __debug__: verbose.report('RendererAgg.buffer_rgba', 'debug-annoying')
        return self._renderer.buffer_rgba()
        
    def clear(self):
        self._renderer.clear()
        

    
def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if __debug__: verbose.report('backend_agg.new_figure_manager', 'debug-annoying')

    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasAgg(thisFig)
    manager = FigureManagerBase(canvas, num)
    return manager


class FigureCanvasAgg(FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance
    """
    

    def draw(self):
        """
        Draw the figure using the renderer
        """
        if __debug__: verbose.report('FigureCanvasAgg.draw', 'debug-annoying')

        renderer = self.get_renderer()
        self.figure.draw(renderer)

    def get_renderer(self):
        l,b,w,h = self.figure.bbox.get_bounds()
        key = w, h, self.figure.dpi.get()
        try: self._lastKey, self.renderer
        except AttributeError: need_new_renderer = True
        else:  need_new_renderer = (self._lastKey != key)

        if need_new_renderer:
            self.renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key
        return self.renderer
    
    def tostring_rgb(self):
        if __debug__: verbose.report('FigureCanvasAgg.tostring_rgb', 'debug-annoying')
        return self.renderer.tostring_rgb()

    def tostring_argb(self):
        if __debug__: verbose.report('FigureCanvasAgg.tostring_argb', 'debug-annoying')
        return self.renderer.tostring_argb()
        
    def buffer_rgba(self):
        if __debug__: verbose.report('FigureCanvasAgg.buffer_rgba', 'debug-annoying')
        return self.renderer.buffer_rgba()
    
    def print_figure(self, filename, dpi=150,
                     facecolor='w', edgecolor='w',
                     orientation='portrait'):
        """
        Render the figure to hardcopy.  Set the figure patch face and
        edge colors.  This is useful because some of the GUIs have a
        gray figure face color background and you'll probably want to
        override this on hardcopy

        If the extension matches PNG, write a PNG file

        If the extension matches BMP or RAW, write an RGBA bitmap file

        If filename is a fileobject, write png to file object (thus
        you can, for example, write the png to stdout
        """
        if __debug__: verbose.report('FigureCanvasAgg.print_figure', 'debug-annoying')

            

        # store the orig figure dpi, color and size information so we
        # can restore them later.  For image creation alone, this is
        # not important since after the print the figure is done.  But
        # backend_agg may be used as a renderer for a GUI figure, and
        # restoring figure props will be important in that case.
        # TODO: move most of this functionality into backend_bases

        origDPI = self.figure.dpi.get()
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()


        self.figure.dpi.set(dpi)
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        # render the printed figure
        self.draw()

        if  isinstance(filename, file):
            # assume png and write to fileobject
            self.renderer._renderer.write_png(filename)
            #pass
        else:
            # take a look at the extension and choose the print handler
            basename, ext = os.path.splitext(filename)
            if not len(ext):
                ext = '.png'
                filename += ext

            ext = ext.lower()
            if (ext.find('rgb')>=0 or
                ext.find('raw')>=0 or
                ext.find('bmp')>=0 ):
                # agg doesn't handle unicode yet
                self.renderer._renderer.write_rgba(str(filename))
            elif ext.find('png')>=0:
                # agg doesn't handle unicode yet
                self.renderer._renderer.write_png(str(filename))
                #pass
            elif ext.find('svg')>=0:
                from backend_svg import FigureCanvasSVG
                svg = self.switch_backends(FigureCanvasSVG)
                svg.print_figure(filename, dpi, facecolor, edgecolor, orientation)
            elif ext.find('ps')>=0 or ext.find('ep')>=0:
                from backend_ps import FigureCanvasPS # lazy import
                ps = self.switch_backends(FigureCanvasPS)
                ps.print_figure(filename, dpi, facecolor, edgecolor, orientation)
            else:
                raise IOError('Do not know know to handle extension *%s' % ext)

        # restore the original figure properties
        self.figure.dpi.set(origDPI)
        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)
        self.figure.set_canvas(self)
