"""
GTK+ Matplotlib interface using Cairo (not GDK) drawing operations.
Author: Steve Chaplin
"""
from __future__ import division

import os
import sys
def function_name(): return sys._getframe(1).f_code.co_name

from matplotlib import verbose
from matplotlib.cbook import is_string_like, enumerate, True, False
from matplotlib.figure import Figure
from backend_cairo import FigureCanvasCairo, RendererCairo
from backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK,\
     show, draw_if_interactive,\
     error_msg, NavigationToolbar, PIXELS_PER_INCH, backend_version

try:
    import cairo
    import cairo.gtk
    # version > x, check - later
except:
    verbose.report_error('PyCairo is required to run the Matplotlib Cairo backend')
    raise SystemExit()
backend_version = '0.1.23' # cairo does not report version, yet


Debug = False


# Image formats that this backend supports (currently the same as backend_gtk)
image_format_list         = ['eps', 'jpg', 'png', 'ps', 'svg']
image_format_default      = 'png'


# ref gtk+/gtk/gtkwidget.h
def GTK_WIDGET_DRAWABLE(w): flags = w.flags(); return flags & gtk.VISIBLE !=0 and flags & gtk.MAPPED != 0


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if Debug: print 'backend_gtkcairo.%s()' % function_name()
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasGTKCairo(thisFig)
    return FigureManagerGTK(canvas, num)


class FigureCanvasGTKCairo(FigureCanvasGTK, FigureCanvasCairo):
    """Use all of the FigureCanvasGTK functionality.
    Override expose_event() and print_figure() to use Cairo rather than GDK.

    """
    def expose_event(self, widget, event):
        if Debug: print 'backend_gtkcairo.%s()' % function_name()

        if GTK_WIDGET_DRAWABLE(self) and self._new_pixmap:
            width, height = self.allocation.width, self.allocation.height

            self._pixmap = gtk.gdk.Pixmap (self.window, width, height)
            # cant (yet) use lines below - when width,height shrinks renderer still draws to the old width, height
            
            #if width > self._pixmap_width or height > self._pixmap_height:
            #    if Debug: print 'backend_gtkcairo.%s: new pixmap allocated' % function_name()
            #    self._pixmap = gtk.gdk.Pixmap (self.window, width, height)
            #    self._pixmap_width, self._pixmap_height = width, height

            # create in __init__() once and save?
            # - but when pixmap changes (resized larger) render must update width, height and redirect all gc to new pixmap!
            surface  = cairo.gtk.surface_create_for_drawable (self._pixmap)
            renderer = RendererCairo (surface, width, height, self.figure.dpi)

            self.figure.draw(renderer) # matplotlib draw command
            
            # Test drawing primitives here:
            #renderer.draw_rectangle (gc, (1.0,1.0,1.0), 0, 0, width, height) # white screen
            #renderer.draw_lines (gc, [0,width,width/2,0], [0,height,0,height])

            #renderer.draw_line (gc, 0,0, width,height)


            self.window.set_back_pixmap (self._pixmap, False)
            self.window.clear()  # draws the pixmap onto the window bg
            self._new_pixmap = False

        return True


    # print_figure() copied from backend_gtk.py with 
    # changes:
    #  change RendererGTK() -> RendererCairo()
    #  change 'error_mgs_gtk' to 'error_msg'
    # possibly later - use native cairo to generate png, ps, (svg. pdf)
    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w'):
        # orientation='portrait'):

        if is_string_like(filename):
            isFileName = True
            root, ext = os.path.splitext(filename)        
            ext = ext.lower()[1:]
            if not len(ext):
                filename, ext = filename + '.png', 'png' # default format

            extensions = {'png':'png', 'jpg':'jpeg', 'jpeg':'jpeg', 'ps':'ps', 'eps':'ps',
                          'svg':'svg'}
            try:
                ftype = extensions[ext]
            except KeyError:
                error_msg('Extension "%s" is not supported. Available formats are SVG, PNG, JPEG, PS and EPS' % ext)
                return
        else:
            isFileName = False  # could be a file?
            ftype='png'
        
        if not self._isRealized:  # no longer required?
            self._printQued.append((filename, dpi, facecolor, edgecolor))
            return

        if ftype in ('ps', 'eps'):
            from backend_ps import FigureCanvasPS
            origDPI = self.figure.dpi.get()
            ps = self.switch_backends(FigureCanvasPS)
            ps.figure.dpi.set(72)
            ps.print_figure(filename, 72, facecolor, edgecolor)
            self.figure.dpi.set(origDPI)
            return
        elif ftype == 'svg':
            from backend_svg import FigureCanvasSVG
            origDPI = self.figure.dpi.get()
            svg = self.switch_backends(FigureCanvasSVG)
            svg.figure.dpi.set(72)
            svg.print_figure(filename, 72, facecolor, edgecolor)
            self.figure.dpi.set(origDPI)                        
            return

        origDPI = self.figure.dpi.get()
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()

        self.figure.dpi.set(dpi)        
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        l,b,width, height = self.figure.bbox.get_bounds()

        width, height = int(width), int(height)
        pixmap   = gtk.gdk.Pixmap (self.window, width, height)
        surface  = cairo.gtk.surface_create_for_drawable (pixmap)
        renderer = RendererCairo (surface, width, height, self.figure.dpi)

        self.figure.draw (renderer)
        
        pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, 0, 8, width, height)
        pixbuf.get_from_drawable(pixmap, self.window.get_colormap(),
                                 0, 0, 0, 0, width, height)
        
        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)
        self.figure.dpi.set(origDPI)

        self.configure_event(self, 'configure') # a widget event in a print function? - sets fig size
            
        try: pixbuf.save(filename, ftype)
        except gobject.GError, msg:
            msg = raise_msg_to_str(msg)
            # note the error must be displayed here because trapping
            # the error on a call or print_figure may not work because
            # printing can be qued and called from realize
            if isFileName:
                error_msg('Could not save figure to %s\n\n%s' % (
                    filename, msg))
            else:
                error_msg('Could not save figure\n%s' % msg)
