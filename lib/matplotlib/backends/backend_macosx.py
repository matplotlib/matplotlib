from __future__ import division

import os
import numpy

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase, NavigationToolbar2
from matplotlib.cbook import maxdict
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.mathtext import MathTextParser
from matplotlib.colors import colorConverter



from matplotlib.widgets import SubplotTool

import matplotlib
from matplotlib.backends import _macosx

def show():
    """Show all the figures and enter the Cocoa mainloop.
    This function will not return until all windows are closed or
    the interpreter exits."""
    # Having a Python-level function "show" wrapping the built-in
    # function "show" in the _macosx extension module allows us to
    # to add attributes to "show". This is something ipython does.
    _macosx.show()

class RendererMac(RendererBase):
    """
    The renderer handles drawing/rendering operations. Most of the renderer's
    methods forwards the command to the renderer's graphics context. The
    renderer does not wrap a C object and is written in pure Python.
    """

    texd = maxdict(50)  # a cache of tex image rasters

    def __init__(self, dpi, width, height):
        RendererBase.__init__(self)
        self.dpi = dpi
        self.width = width
        self.height = height
        self.gc = GraphicsContextMac()
        self.mathtext_parser = MathTextParser('MacOSX')

    def set_width_height (self, width, height):
        self.width, self.height = width, height

    def draw_path(self, gc, path, transform, rgbFace=None):
        if rgbFace is not None:
            rgbFace = tuple(rgbFace)
        if gc!=self.gc:
            n = self.gc.level() - gc.level()
            for i in range(n): self.gc.restore()
            self.gc = gc
        gc.draw_path(path, transform, rgbFace)

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        if rgbFace is not None:
            rgbFace = tuple(rgbFace)
        if gc!=self.gc:
            n = self.gc.level() - gc.level()
            for i in range(n): self.gc.restore()
            self.gc = gc
        gc.draw_markers(marker_path, marker_trans, path, trans, rgbFace)

    def draw_path_collection(self, *args):
        gc = self.gc
        args = args[:13]
        gc.draw_path_collection(*args)

    def draw_quad_mesh(self, *args):
        gc = self.gc
        gc.draw_quad_mesh(*args)

    def new_gc(self):
        self.gc.reset()
        return self.gc

    def draw_image(self, x, y, im, bbox, clippath=None, clippath_trans=None):
        im.flipud_out()
        nrows, ncols, data = im.as_rgba_str()
        self.gc.draw_image(x, y, nrows, ncols, data, bbox, clippath, clippath_trans)
        im.flipud_out()
    
    def draw_tex(self, gc, x, y, s, prop, angle):
        if gc!=self.gc:
            n = self.gc.level() - gc.level()
            for i in range(n): self.gc.restore()
            self.gc = gc
        # todo, handle props, angle, origins
        size = prop.get_size_in_points()
        texmanager = self.get_texmanager()
        key = s, size, self.dpi, angle, texmanager.get_font_config()
        im = self.texd.get(key) # Not sure what this does; just copied from backend_agg.py
        if im is None:
            Z = texmanager.get_grey(s, size, self.dpi)
            Z = numpy.array(255.0 - Z * 255.0, numpy.uint8)

        gc.draw_mathtext(x, y, angle, Z)

    def _draw_mathtext(self, gc, x, y, s, prop, angle):
        if gc!=self.gc:
            n = self.gc.level() - gc.level()
            for i in range(n): self.gc.restore()
            self.gc = gc
        size = prop.get_size_in_points()
        ox, oy, width, height, descent, image, used_characters = \
            self.mathtext_parser.parse(s, self.dpi, prop)
        gc.draw_mathtext(x, y, angle, 255 - image.as_array())

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):
        if gc!=self.gc:
            n = self.gc.level() - gc.level()
            for i in range(n): self.gc.restore()
            self.gc = gc
        if ismath:
           self._draw_mathtext(gc, x, y, s, prop, angle)
        else:
            family =  prop.get_family()
            size = prop.get_size_in_points()
            weight = prop.get_weight()
            style = prop.get_style()
            gc.draw_text(x, y, unicode(s), family, size, weight, style, angle)

    def get_text_width_height_descent(self, s, prop, ismath):
        if ismath=='TeX':
            # TODO: handle props
            size = prop.get_size_in_points()
            texmanager = self.get_texmanager()
            Z = texmanager.get_grey(s, size, self.dpi)
            m,n = Z.shape
            # TODO: handle descent; This is based on backend_agg.py
            return n, m, 0
        if ismath:
            ox, oy, width, height, descent, fonts, used_characters = \
                self.mathtext_parser.parse(s, self.dpi, prop)
            return width, height, descent
        family =  prop.get_family()
        size = prop.get_size_in_points()
        weight = prop.get_weight()
        style = prop.get_style()
        return self.gc.get_text_width_height_descent(unicode(s), family, size, weight, style)

    def flipy(self):
        return False
    
    def points_to_pixels(self, points):
        return points/72.0 * self.dpi

    def option_image_nocomposite(self):
        return True

class GraphicsContextMac(_macosx.GraphicsContext, GraphicsContextBase):
    """
    The GraphicsContext wraps a Quartz graphics context. All methods
    are implemented at the C-level in macosx.GraphicsContext. These
    methods set drawing properties such as the line style, fill color,
    etc. The actual drawing is done by the Renderer, which draws into
    the GraphicsContext.
    """
    def __init__(self):
        GraphicsContextBase.__init__(self)
        _macosx.GraphicsContext.__init__(self)

    def set_foreground(self, fg, isRGB=False):
        if not isRGB:
            fg = colorConverter.to_rgb(fg)
        _macosx.GraphicsContext.set_foreground(self, fg)

    def set_clip_rectangle(self, box):
        GraphicsContextBase.set_clip_rectangle(self, box)
        if not box: return
        _macosx.GraphicsContext.set_clip_rectangle(self, box.bounds)

    def set_clip_path(self, path):
        GraphicsContextBase.set_clip_path(self, path)
        if not path: return
        path = path.get_fully_transformed_path()
        _macosx.GraphicsContext.set_clip_path(self, path)

########################################################################
#    
# The following functions and classes are for pylab and implement
# window/figure managers, etc...
#
########################################################################

def draw_if_interactive():
    """
    For performance reasons, we don't want to redraw the figure after
    each draw command. Instead, we mark the figure as invalid, so that
    it will be redrawn as soon as the event loop resumes via PyOS_InputHook.
    This function should be called after each draw event, even if
    matplotlib is not running interactively.
    """
    figManager =  Gcf.get_active()
    if figManager is not None:
        figManager.canvas.invalidate()

def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    FigureClass = kwargs.pop('FigureClass', Figure)
    figure = FigureClass(*args, **kwargs)
    canvas = FigureCanvasMac(figure)
    manager = FigureManagerMac(canvas, num)
    return manager

class FigureCanvasMac(_macosx.FigureCanvas, FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance

    Events such as button presses, mouse movements, and key presses
    are handled in the C code and the base class methods
    button_press_event, button_release_event, motion_notify_event,
    key_press_event, and key_release_event are called from there.
    """

    def __init__(self, figure):
        FigureCanvasBase.__init__(self, figure)
        width, height = self.get_width_height()
        self.renderer = RendererMac(figure.dpi, width, height)
        _macosx.FigureCanvas.__init__(self, width, height)

    def resize(self, width, height):
        self.renderer.set_width_height(width, height)
        dpi = self.figure.dpi
        width /= dpi
        height /= dpi
        self.figure.set_size_inches(width, height)

    def print_figure(self, filename, dpi=None, facecolor='w', edgecolor='w',
                     orientation='portrait', **kwargs):
        if dpi is None: dpi = matplotlib.rcParams['savefig.dpi']
        filename = unicode(filename)
        root, ext = os.path.splitext(filename)
        ext = ext[1:].lower()
        if not ext:
             ext = "png"
             filename = root + "." + ext
        if ext=="jpg": ext = "jpeg"

        # save the figure settings
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()

        # set the new parameters
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        if ext in ('jpeg', 'png', 'tiff', 'gif', 'bmp'):
            width, height = self.figure.get_size_inches()
            width, height = width*dpi, height*dpi
            self.write_bitmap(filename, width, height)
        elif ext == 'pdf':
            self.write_pdf(filename)
        elif ext in ('ps', 'eps'):
            from backend_ps import FigureCanvasPS
            # Postscript backend changes figure.dpi, but doesn't change it back
            origDPI = self.figure.dpi
            fc = self.switch_backends(FigureCanvasPS)
            fc.print_figure(filename, dpi, facecolor, edgecolor,
                            orientation, **kwargs)
            self.figure.dpi = origDPI
            self.figure.set_canvas(self)
        elif ext=='svg':
            from backend_svg import FigureCanvasSVG
            fc = self.switch_backends(FigureCanvasSVG)
            fc.print_figure(filename, dpi, facecolor, edgecolor,
                            orientation, **kwargs)
            self.figure.set_canvas(self)
        else:
            raise ValueError("Figure format not available (extension %s)" % ext)

        # restore original figure settings
        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)



class FigureManagerMac(_macosx.FigureManager, FigureManagerBase):
    """
    Wrap everything up into a window for the pylab interface
    """
    def __init__(self, canvas, num):
        FigureManagerBase.__init__(self, canvas, num)
        title = "Figure %d" % num
        _macosx.FigureManager.__init__(self, canvas, title)
        if matplotlib.rcParams['toolbar']=='classic':
            self.toolbar = NavigationToolbarMac(canvas)
        elif matplotlib.rcParams['toolbar']=='toolbar2':
            self.toolbar = NavigationToolbar2Mac(canvas)
        else:
            self.toolbar = None
        if self.toolbar is not None: 
            self.toolbar.update()

        def notify_axes_change(fig):
            'this will be called whenever the current axes is changed'
            if self.toolbar != None: self.toolbar.update()
        self.canvas.figure.add_axobserver(notify_axes_change)

        # This is ugly, but this is what tkagg and gtk are doing.
        # It is needed to get ginput() working.
        self.canvas.figure.show = lambda *args: self.show()

    def show(self):
        self.canvas.draw()

    def close(self):
        Gcf.destroy(self.num)

class NavigationToolbarMac(_macosx.NavigationToolbar):
 
    def __init__(self, canvas):
        self.canvas = canvas
        basedir = os.path.join(matplotlib.rcParams['datapath'], "images")
        images = {}
        for imagename in ("stock_left",
                          "stock_right",
                          "stock_up",
                          "stock_down",
                          "stock_zoom-in",
                          "stock_zoom-out",
                          "stock_save_as"):
            filename = os.path.join(basedir, imagename+".ppm")
            images[imagename] = self._read_ppm_image(filename)
        _macosx.NavigationToolbar.__init__(self, images)
        self.message = None

    def _read_ppm_image(self, filename):
        data = ""
        imagefile = open(filename)
        for line in imagefile:
            if "#" in line:
                i = line.index("#")
                line = line[:i] + "\n"
            data += line
        imagefile.close()
        magic, width, height, maxcolor, imagedata = data.split(None, 4)
        width, height = int(width), int(height)
        assert magic=="P6"
        assert len(imagedata)==width*height*3 # 3 colors in RGB
        return (width, height, imagedata)
        
    def panx(self, direction):
        axes = self.canvas.figure.axes
        selected = self.get_active()
        for i in selected:
            axes[i].xaxis.pan(direction)
        self.canvas.invalidate()

    def pany(self, direction):
        axes = self.canvas.figure.axes
        selected = self.get_active()
        for i in selected:
            axes[i].yaxis.pan(direction)
        self.canvas.invalidate()

    def zoomx(self, direction):
        axes = self.canvas.figure.axes
        selected = self.get_active()
        for i in selected:
            axes[i].xaxis.zoom(direction)
        self.canvas.invalidate()

    def zoomy(self, direction):
        axes = self.canvas.figure.axes
        selected = self.get_active()
        for i in selected:
            axes[i].yaxis.zoom(direction)
        self.canvas.invalidate()

    def save_figure(self):
        filename = _macosx.choose_save_file('Save the figure')
        if filename is None: # Cancel
            return
        self.canvas.print_figure(filename)

class NavigationToolbar2Mac(_macosx.NavigationToolbar2, NavigationToolbar2):

    def __init__(self, canvas):
        NavigationToolbar2.__init__(self, canvas)

    def _init_toolbar(self):
        basedir = os.path.join(matplotlib.rcParams['datapath'], "images")
        _macosx.NavigationToolbar2.__init__(self, basedir)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.set_rubberband(x0, y0, x1, y1)

    def release(self, event):
        self.canvas.remove_rubberband()

    def set_cursor(self, cursor):
        _macosx.set_cursor(cursor)

    def save_figure(self):
        filename = _macosx.choose_save_file('Save the figure')
        if filename is None: # Cancel
            return
        self.canvas.print_figure(filename)

    def prepare_configure_subplots(self):
        toolfig = Figure(figsize=(6,3))
        canvas = FigureCanvasMac(toolfig)
        toolfig.subplots_adjust(top=0.9)
        tool = SubplotTool(self.canvas.figure, toolfig)
        return canvas

    def set_message(self, message):
        _macosx.NavigationToolbar2.set_message(self, message.encode('utf-8'))

########################################################################
#    
# Now just provide the standard names that backend.__init__ is expecting
# 
########################################################################


FigureManager = FigureManagerMac
