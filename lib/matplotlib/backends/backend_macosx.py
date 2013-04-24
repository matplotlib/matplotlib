from __future__ import division, print_function

import os
import numpy

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase, NavigationToolbar2, TimerBase
from matplotlib.backend_bases import ShowBase

from matplotlib.cbook import maxdict
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.mathtext import MathTextParser
from matplotlib.colors import colorConverter
from matplotlib import rcParams

from matplotlib.widgets import SubplotTool

import matplotlib
from matplotlib.backends import _macosx


class Show(ShowBase):
    def mainloop(self):
        _macosx.show()
show = Show()


class RendererMac(RendererBase):
    """
    The renderer handles drawing/rendering operations. Most of the renderer's
    methods forward the command to the renderer's graphics context. The
    renderer does not wrap a C object and is written in pure Python.
    """

    texd = maxdict(50)  # a cache of tex image rasters

    def __init__(self, dpi, width, height):
        RendererBase.__init__(self)
        self.dpi = dpi
        self.width = width
        self.height = height
        self.gc = GraphicsContextMac()
        self.gc.set_dpi(self.dpi)
        self.mathtext_parser = MathTextParser('MacOSX')

    def set_width_height (self, width, height):
        self.width, self.height = width, height

    def draw_path(self, gc, path, transform, rgbFace=None):
        if rgbFace is not None:
            rgbFace = tuple(rgbFace)
        linewidth = gc.get_linewidth()
        gc.draw_path(path, transform, linewidth, rgbFace)

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        if rgbFace is not None:
            rgbFace = tuple(rgbFace)
        linewidth = gc.get_linewidth()
        gc.draw_markers(marker_path, marker_trans, path, trans, linewidth, rgbFace)

    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offsetTrans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        if offset_position=='data':
            offset_position = True
        else:
            offset_position = False
        path_ids = []
        for path, transform in self._iter_collection_raw_paths(
            master_transform, paths, all_transforms):
            path_ids.append((path, transform))
        master_transform = master_transform.get_matrix()
        all_transforms = [t.get_matrix() for t in all_transforms]
        offsetTrans = offsetTrans.get_matrix()
        gc.draw_path_collection(master_transform, path_ids, all_transforms,
                             offsets, offsetTrans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds,
                             offset_position)

    def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
                       coordinates, offsets, offsetTrans, facecolors,
                       antialiased, edgecolors):
        gc.draw_quad_mesh(master_transform.get_matrix(),
                          meshWidth,
                          meshHeight,
                          coordinates,
                          offsets,
                          offsetTrans.get_matrix(),
                          facecolors,
                          antialiased,
                          edgecolors)

    def new_gc(self):
        self.gc.save()
        self.gc.set_hatch(None)
        self.gc._alpha = 1.0
        self.gc._forced_alpha = False # if True, _alpha overrides A from RGBA
        return self.gc

    def draw_gouraud_triangle(self, gc, points, colors, transform):
        points = transform.transform(points)
        gc.draw_gouraud_triangle(points, colors)

    def draw_image(self, gc, x, y, im):
        im.flipud_out()
        nrows, ncols, data = im.as_rgba_str()
        gc.draw_image(x, y, nrows, ncols, data)
        im.flipud_out()

    def draw_tex(self, gc, x, y, s, prop, angle, ismath='TeX!', mtext=None):
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
        ox, oy, width, height, descent, image, used_characters = \
            self.mathtext_parser.parse(s, self.dpi, prop)
        gc.draw_mathtext(x, y, angle, 255 - image.as_array())

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if ismath:
            self._draw_mathtext(gc, x, y, s, prop, angle)
        else:
            family =  prop.get_family()
            weight = prop.get_weight()
            style = prop.get_style()
            points = prop.get_size_in_points()
            size = self.points_to_pixels(points)
            gc.draw_text(x, y, unicode(s), family, size, weight, style, angle)

    def get_text_width_height_descent(self, s, prop, ismath):
        if ismath=='TeX':
            # todo: handle props
            texmanager = self.get_texmanager()
            fontsize = prop.get_size_in_points()
            w, h, d = texmanager.get_text_width_height_descent(s, fontsize,
                                                               renderer=self)
            return w, h, d
        if ismath:
            ox, oy, width, height, descent, fonts, used_characters = \
                self.mathtext_parser.parse(s, self.dpi, prop)
            return width, height, descent
        family =  prop.get_family()
        weight = prop.get_weight()
        style = prop.get_style()
        points = prop.get_size_in_points()
        size = self.points_to_pixels(points)
        width, height, descent = self.gc.get_text_width_height_descent(unicode(s), family, size, weight, style)
        return  width, height, 0.0*descent

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

    def set_alpha(self, alpha):
        GraphicsContextBase.set_alpha(self, alpha)
        _alpha = self.get_alpha()
        _macosx.GraphicsContext.set_alpha(self, _alpha, self.get_forced_alpha())
        rgb = self.get_rgb()
        _macosx.GraphicsContext.set_foreground(self, rgb)

    def set_foreground(self, fg, isRGBA=False):
        GraphicsContextBase.set_foreground(self, fg, isRGBA)
        rgb = self.get_rgb()
        _macosx.GraphicsContext.set_foreground(self, rgb)

    def set_graylevel(self, fg):
        GraphicsContextBase.set_graylevel(self, fg)
        _macosx.GraphicsContext.set_graylevel(self, fg)

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
    if matplotlib.is_interactive():
        figManager =  Gcf.get_active()
        if figManager is not None:
            figManager.canvas.invalidate()


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    FigureClass = kwargs.pop('FigureClass', Figure)
    figure = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, figure)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasMac(figure)
    manager = FigureManagerMac(canvas, num)
    return manager


class TimerMac(_macosx.Timer, TimerBase):
    '''
    Subclass of :class:`backend_bases.TimerBase` that uses CoreFoundation
    run loops for timer events.

    Attributes:
    * interval: The time between timer events in milliseconds. Default
        is 1000 ms.
    * single_shot: Boolean flag indicating whether this timer should
        operate as single shot (run once and then stop). Defaults to False.
    * callbacks: Stores list of (func, args) tuples that will be called
        upon timer events. This list can be manipulated directly, or the
        functions add_callback and remove_callback can be used.
    '''
    # completely implemented at the C-level (in _macosx.Timer)


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

    filetypes = FigureCanvasBase.filetypes.copy()
    filetypes['bmp'] = 'Windows bitmap'
    filetypes['jpeg'] = 'JPEG'
    filetypes['jpg'] = 'JPEG'
    filetypes['gif'] = 'Graphics Interchange Format'
    filetypes['tif'] = 'Tagged Image Format File'
    filetypes['tiff'] = 'Tagged Image Format File'

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

    def _print_bitmap(self, filename, *args, **kwargs):
        # In backend_bases.py, print_figure changes the dpi of the figure.
        # But since we are essentially redrawing the picture, we need the
        # original dpi. Pick it up from the renderer.
        dpi = kwargs['dpi']
        old_dpi = self.figure.dpi
        self.figure.dpi = self.renderer.dpi
        width, height = self.figure.get_size_inches()
        width, height = width*dpi, height*dpi
        filename = unicode(filename)
        self.write_bitmap(filename, width, height, dpi)
        self.figure.dpi = old_dpi

    def print_bmp(self, filename, *args, **kwargs):
        self._print_bitmap(filename, *args, **kwargs)

    def print_jpg(self, filename, *args, **kwargs):
        self._print_bitmap(filename, *args, **kwargs)

    def print_jpeg(self, filename, *args, **kwargs):
        self._print_bitmap(filename, *args, **kwargs)

    def print_tif(self, filename, *args, **kwargs):
        self._print_bitmap(filename, *args, **kwargs)

    def print_tiff(self, filename, *args, **kwargs):
        self._print_bitmap(filename, *args, **kwargs)

    def print_gif(self, filename, *args, **kwargs):
        self._print_bitmap(filename, *args, **kwargs)

    def new_timer(self, *args, **kwargs):
        """
        Creates a new backend-specific subclass of :class:`backend_bases.Timer`.
        This is useful for getting periodic events through the backend's native
        event loop. Implemented only for backends with GUIs.

        optional arguments:

        *interval*
          Timer interval in milliseconds
        *callbacks*
          Sequence of (func, args, kwargs) where func(*args, **kwargs) will
          be executed by the timer every *interval*.
        """
        return TimerMac(*args, **kwargs)


class FigureManagerMac(_macosx.FigureManager, FigureManagerBase):
    """
    Wrap everything up into a window for the pylab interface
    """
    def __init__(self, canvas, num):
        FigureManagerBase.__init__(self, canvas, num)
        title = "Figure %d" % num
        _macosx.FigureManager.__init__(self, canvas, title)
        if rcParams['toolbar']=='classic':
            self.toolbar = NavigationToolbarMac(canvas)
        elif rcParams['toolbar']=='toolbar2':
            self.toolbar = NavigationToolbar2Mac(canvas)
        else:
            self.toolbar = None
        if self.toolbar is not None:
            self.toolbar.update()

        def notify_axes_change(fig):
            'this will be called whenever the current axes is changed'
            if self.toolbar != None: self.toolbar.update()
        self.canvas.figure.add_axobserver(notify_axes_change)

        if matplotlib.is_interactive():
            self.show()

    def close(self):
        Gcf.destroy(self.num)


class NavigationToolbarMac(_macosx.NavigationToolbar):

    def __init__(self, canvas):
        self.canvas = canvas
        basedir = os.path.join(rcParams['datapath'], "images")
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

    def save_figure(self, *args):
        filename = _macosx.choose_save_file('Save the figure',
                                            self.canvas.get_default_filename())
        if filename is None: # Cancel
            return
        self.canvas.print_figure(filename)


class NavigationToolbar2Mac(_macosx.NavigationToolbar2, NavigationToolbar2):

    def __init__(self, canvas):
        NavigationToolbar2.__init__(self, canvas)

    def _init_toolbar(self):
        basedir = os.path.join(rcParams['datapath'], "images")
        _macosx.NavigationToolbar2.__init__(self, basedir)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.set_rubberband(int(x0), int(y0), int(x1), int(y1))

    def release(self, event):
        self.canvas.remove_rubberband()

    def set_cursor(self, cursor):
        _macosx.set_cursor(cursor)

    def save_figure(self, *args):
        filename = _macosx.choose_save_file('Save the figure',
                                            self.canvas.get_default_filename())
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

    def dynamic_update(self):
        self.canvas.draw_idle()

########################################################################
#
# Now just provide the standard names that backend.__init__ is expecting
#
########################################################################


FigureManager = FigureManagerMac
