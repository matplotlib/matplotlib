"""
GTK+ Matplotlib interface using Cairo (not GDK) drawing operations.
Author: Steve Chaplin
"""
from __future__ import division

import os
import sys
def _fn_name(): return sys._getframe(1).f_code.co_name

from matplotlib import verbose
from matplotlib.cbook import True, False
from matplotlib.figure import Figure
from backend_cairo import FigureCanvasCairo, RendererCairo, IMAGE_FORMAT, \
     IMAGE_FORMAT_DEFAULT, print_figure_fn
from backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK, show,    \
     draw_if_interactive, error_msg, NavigationToolbar, backend_version, \
     gdk_pixmap_save

import cairo
import cairo.gtk

# add version checking, if cairo adds version number support
#version_required = (1,99,16)
#if gtk.pygtk_version < version_required:
#    raise SystemExit ("PyGTK %d.%d.%d is installed\n"
#                      "PyGTK %d.%d.%d or later is required"
#                      % (gtk.pygtk_version + version_required))
#backend_version = "%d.%d.%d" % gtk.pygtk_version
backend_version = 'GTK(%s) Cairo (unknown)' % backend_version

DEBUG = False

# Image formats that this backend supports, same as backend_cairo + jpg from gtk
IMAGE_FORMAT += ['jpg']


# ref gtk+/gtk/gtkwidget.h
def GTK_WIDGET_DRAWABLE(w): flags = w.flags(); return flags & gtk.VISIBLE !=0 and flags & gtk.MAPPED != 0


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if DEBUG: print 'backend_gtkcairo.%s()' % _fn_name()
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasGTKCairo(thisFig)
    return FigureManagerGTK(canvas, num)


class FigureCanvasGTKCairo(FigureCanvasGTK, FigureCanvasCairo):
    """Use all of the FigureCanvasGTK functionality.
    Override expose_event() and print_figure() to use Cairo rather than GDK.

    """
    def expose_event(self, widget, event):
        if DEBUG: print 'backend_gtkcairo.%s()' % _fn_name()

        if GTK_WIDGET_DRAWABLE(self) and self._new_pixmap:
            width, height = self.allocation.width, self.allocation.height

            self._pixmap = gtk.gdk.Pixmap (self.window, width, height)
            # cant (yet) use lines below - when width,height shrinks renderer still draws to the old width, height
            
            #if width > self._pixmap_width or height > self._pixmap_height:
            #    if DEBUG: print 'backend_gtkcairo.%s: new pixmap allocated' % _fn_name()
            #    self._pixmap = gtk.gdk.Pixmap (self.window, width, height)
            #    self._pixmap_width, self._pixmap_height = width, height

            # create in __init__() once and save?
            # - but when pixmap changes (resized larger) render must update width, height and redirect all gc to new pixmap!

            surface  = cairo.gtk.surface_create_for_drawable (self._pixmap)
            matrix   = cairo.Matrix ()
            renderer = RendererCairo (surface, matrix, width, height, self.figure.dpi)
            
            self.figure.draw (renderer)
            
            self.window.set_back_pixmap (self._pixmap, False)
            self.window.clear()  # draws the pixmap onto the window bg
            self._new_pixmap = False

        return True


    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait'):

        root, ext = os.path.splitext(filename)       
        ext = ext.lower()[1:]
        if ext == '':
            ext      = IMAGE_FORMAT_DEFAULT
            filename = filename + '.' + ext

        ext = ext.lower()
        if ext in ('jpg'): # backend_gtk / gdk
            if self.flags() & gtk.REALIZED == 0:
                gtk.DrawingArea.realize(self) # for self.window and figure sizing

            # save figure settings
            origDPI       = self.figure.dpi.get()
            origfacecolor = self.figure.get_facecolor()
            origedgecolor = self.figure.get_edgecolor()
            origWIn, origHIn = self.figure.get_size_inches()

            self.figure.dpi.set(dpi)        
            self.figure.set_facecolor(facecolor)
            self.figure.set_edgecolor(edgecolor)

            width, height = self.figure.get_width_height()
            width, height = int(width), int(height)

            # render using Cairo, save file using gtk.gdk
            pixmap = gtk.gdk.Pixmap (self.window, width, height)
            ctx = cairo.Context()
            surface  = cairo.gtk.surface_create_for_drawable (pixmap)
            renderer = RendererCairo (surface, ctx.matrix, width, height, self.figure.dpi)
            self.figure.draw (renderer)
            gdk_pixmap_save (pixmap, filename, ext, width, height)        

            # restore figure settings
            self.figure.set_facecolor(origfacecolor)
            self.figure.set_edgecolor(origedgecolor)
            self.figure.dpi.set(origDPI)
            self.figure.set_figsize_inches(origWIn, origHIn)
            
        elif ext in IMAGE_FORMAT:
            #FigureCanvasCairo.print_figure (self, filename, dpi, facecolor, edgecolor,
            #                                orientation)
            print_figure_fn (self.figure, filename, dpi, facecolor, edgecolor,
                             orientation)

        else:
            error_msg('Format "%s" is not supported.\nSupported formats are %s.' %
                      (ext, ', '.join(IMAGE_FORMAT)))
