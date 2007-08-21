"""
An agg http://antigrain.com/ backend

"""
from __future__ import division

import os, sys
import matplotlib.agg as agg

from matplotlib import verbose

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase,\
     GraphicsContextBase, FigureManagerBase, FigureCanvasBase

from matplotlib.cbook import enumerate, is_string_like, exception_to_str
from matplotlib.figure import Figure
from matplotlib.ft2font import FT2Font
from matplotlib.mathtext import MathTextParser


from _backend_agg import RendererAgg as _RendererAgg

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
        self.width = int(width)
        self.height = int(height)

        stride = self.width*4
        self.buffer = agg.buffer(self.width, self.height, stride)

        self.rbuf = agg.rendering_buffer()
        self.rbuf.attachb(self.buffer)

        self.pf = agg.pixel_format(self.rbuf)
        self.rbase = agg.renderer_base(self.pf)

        self.rasterizer = agg.rasterizer_scanline_aa()
        self.scanline = agg.scanline_p8()
        self.renderer =  agg.renderer_scanline_aa_solid(self.rbase);


    def draw_lines(self, gc, x, y, trans):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """

        x, y = trans.numerix_x_y(x,y)
        if len(x)<2: return
        path = agg.path_storage()
        path.move_to(x[0],self.height-y[0])
        for i in xrange(1, len(x)):
            path.line_to(x[i],self.height-y[i])

        stroke = agg.conv_stroke(path)
        stroke.width(1.0)
        r,g,b = [int(255*val) for val in gc.get_rgb()]
        a = int(255*gc.get_alpha())

        color = agg.rgba8(r,g,b,a)
        self.renderer.color(  color )
        self.rasterizer.add_path(stroke)
        agg.render_scanlines(self.rasterizer, self.scanline, self.renderer);

    def draw_markers(self, gc, path, rgbFace, xt, yt, trans):
        pass

    def draw_arc(self, gcEdge, rgbFace, x, y, width, height, angle1, angle2, rotation):
        pass

    def draw_image(self, x, y, im, origin, bbox):
        pass

    def draw_line(self, gc, x1, y1, x2, y2):
        pass

    def draw_point(self, gc, x, y):
        pass

    def draw_polygon(self, gcEdge, rgbFace, points):
        pass

    def draw_rectangle(self, gcEdge, rgbFace, x, y, width, height):
        pass

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):
        pass

    def flipy(self):
        return True

    def get_canvas_width_height(self):
        return 100, 100

    def get_text_width_height(self, s, prop, ismath):
        return 1, 1

    def new_gc(self):
        return GraphicsContextBase()


    def points_to_pixels(self, points):
        """
        convert point measures to pixes using dpi and the pixels per
        inch of the display
        """
        if __debug__: verbose.report('RendererAgg.points_to_pixels', 'debug-annoying')
        return points*self.dpi.get()/72.0





def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if __debug__: verbose.report('backend_agg.new_figure_manager', 'debug-annoying')
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
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
        return renderer

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

    def buffer_rgba(self,x,y):
        if __debug__: verbose.report('FigureCanvasAgg.buffer_rgba', 'debug-annoying')
        return self.renderer.buffer_rgba(x,y)


    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait', **kwargs):
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

        r = self.draw()
        s = r.buffer.to_string()
        import Image
        im = Image.fromstring( "RGBA", (r.width, r.height), s)
        im.show()
