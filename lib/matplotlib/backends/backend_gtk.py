from __future__ import division

import os, math
import sys
def fn_name(): return sys._getframe(1).f_code.co_name

import matplotlib
from matplotlib import verbose, MPLError
from matplotlib.numerix import asarray, fromstring, UInt8, zeros, \
     where, transpose, nonzero, indices, ones, nx

import matplotlib.numerix as numerix
from matplotlib.cbook import is_string_like, enumerate, True, False, onetrue
from matplotlib.font_manager import fontManager

from matplotlib.backend_bases import RendererBase, GraphicsContextBase, \
     FigureManagerBase, FigureCanvasBase, NavigationToolbar2, cursors, MplEvent
from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure

from backend_gdk import RendererGDK

try: from matplotlib.mathtext import math_parse_s_ft2font
except ImportError:
    verbose.report_error('backend_gtk could not import mathtext (build with ft2font)')
    useMathText = False
else: useMathText = True

pygtk_version_required = (1,99,16)
try:
    import pygtk
    if not matplotlib.FROZEN:
        pygtk.require('2.0')
except:
    print >> sys.stderr, sys.exc_info()[1]
    raise SystemExit('PyGTK version %d.%d.%d or greater is required to run the GTK Matplotlib backends'
                     % pygtk_version_required)

import gtk, gobject, pango
from gtk import gdk
if gtk.pygtk_version < pygtk_version_required:
    raise SystemExit ("PyGTK %d.%d.%d is installed\n"
                      "PyGTK %d.%d.%d or later is required"
                      % (gtk.pygtk_version + pygtk_version_required))
backend_version = "%d.%d.%d" % gtk.pygtk_version
del pygtk_version_required


DEBUG = False

# the true dots per inch on the screen; should be display dependent
# see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi
PIXELS_PER_INCH = 96

# Image formats that this backend supports - for FileChooser and print_figure()
IMAGE_FORMAT          = ['eps', 'jpg', 'png', 'ps', 'svg'] + ['bmp'] # , 'raw', 'rgb']
IMAGE_FORMAT.sort()
IMAGE_FORMAT_DEFAULT  = 'png'


cursord = {
    cursors.MOVE          : gtk.gdk.Cursor(gtk.gdk.FLEUR),
    cursors.HAND          : gtk.gdk.Cursor(gtk.gdk.HAND2),
    cursors.POINTER       : gtk.gdk.Cursor(gtk.gdk.LEFT_PTR),
    cursors.SELECT_REGION : gtk.gdk.Cursor(gtk.gdk.TCROSS),
    }

# ref gtk+/gtk/gtkwidget.h
def GTK_WIDGET_DRAWABLE(w): flags = w.flags(); return flags & gtk.VISIBLE !=0 and flags & gtk.MAPPED != 0


def draw_if_interactive():
    """
    Is called after every pylab drawing command
    """
    if matplotlib.is_interactive():
        figManager =  Gcf.get_active()
        if figManager is not None:
            figManager.canvas.draw()


def show(mainloop=True):
    """
    Show all the figures and enter the gtk main loop

    This should be the last line of your script
    """

    for manager in Gcf.get_all_fig_managers():
        manager.window.show()
        
    if mainloop and gtk.main_level() == 0:
        if gtk.pygtk_version >= (2,4,0): gtk.main()
        else:                            gtk.mainloop()


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasGTK(thisFig)
    manager = FigureManagerGTK(canvas, num)
    # equals:
    #manager = FigureManagerGTK(FigureCanvasGTK(Figure(*args, **kwargs), num)
    return manager


class FigureCanvasGTK(gtk.DrawingArea, FigureCanvasBase):
    keyvald = {65507 : 'control',
               65505 : 'shift',
               65513 : 'alt',
               65508 : 'control',
               65506 : 'shift',
               65514 : 'alt',
               }

                          
    def __init__(self, figure):
        FigureCanvasBase.__init__(self, figure)
        gtk.DrawingArea.__init__(self)
        
        self._idleID        = 0      # used in gtkAgg
        self._draw_pixmap   = True
        self._pixmap_width  = -1
        self._pixmap_height = -1

        self._lastCursor = None
        self._button     = None  # the button pressed
        self._key        = None  # the key pressed

        self.set_flags(gtk.CAN_FOCUS)
        self.grab_focus()
        self.set_size_request (int (figure.bbox.width()),
                               int (figure.bbox.height()))
        self.set_double_buffered(False)

        self.connect('key_press_event',      self.key_press_event)
        self.connect('key_release_event',    self.key_release_event)
        self.connect('expose_event',         self.expose_event)
        self.connect('configure_event',      self.configure_event)
        self.connect('motion_notify_event',  self.motion_notify_event)
        self.connect('button_press_event',   self.button_press_event)
        self.connect('button_release_event', self.button_release_event)

        self.set_events(
            #gdk.FOCUS_CHANGE_MASK  |
            gdk.KEY_PRESS_MASK      |
            gdk.KEY_RELEASE_MASK    |
            gdk.EXPOSURE_MASK       |
            gdk.LEAVE_NOTIFY_MASK   |
            gdk.BUTTON_PRESS_MASK   |
            gdk.BUTTON_RELEASE_MASK |
            gdk.POINTER_MOTION_MASK  )

        self._renderer_init()

    def button_press_event(self, widget, event):
        self._button = event.button

    def button_release_event(self, widget, event):
        self._button = None

    def motion_notify_event(self, widget, event):
        #print 'backend_gtk', event.x, event.y
        pass
    
    def key_press_event(self, widget, event):

        if self.keyvald.has_key(event.keyval):
            key = self.keyvald[event.keyval]
        elif event.keyval <256:
            key = chr(event.keyval)
        else:
            key = None
            
        ctrl  = event.state & gtk.gdk.CONTROL_MASK
        shift = event.state & gtk.gdk.SHIFT_MASK
        
        self._key = key

    def key_release_event(self, widget, event):        
        self._key = None
        
    def mpl_connect(self, s, func):
        
        if s not in self.events:
            raise MPLError('Can only connect events of type "%s"\nDo not know how to handle "%s"' %(', '.join(self.events), s))
            #error_msg('Can only connect events of type "%s"\nDo not know how to handle "%s"' %(', '.join(self.events), s),
            #          parent=self)            

        def wrapper(widget, event):
            thisEvent = MplEvent(s, self) 

            thisEvent.x = event.x
            # flipy so y=0 is bottom of canvas
            thisEvent.y = self.figure.bbox.height() - event.y
 
            thisEvent.button = self._button
            thisEvent.key = self._key

            thisEvent.inaxes = None
            for a in self.figure.get_axes():
                if a.in_axes(thisEvent.x, thisEvent.y):
                    thisEvent.inaxes = a
                    xdata, ydata = a.transData.inverse_xy_tup((thisEvent.x, thisEvent.y))
                    thisEvent.xdata  = xdata
                    thisEvent.ydata  = ydata
                    break
                
            
            func(thisEvent)
            return False  # return True blocks other connects
        cid =  self.connect(s, wrapper)
        return cid

    def mpl_disconnect(self, cid):
        self.disconnect(cid)
        return None


    def configure_event(self, widget, event):
        if widget.window is None:
            return

        w,h = widget.window.get_size()
        if w==1 or h==1:
            return # empty fig

        # resize the figure (in inches)
        dpi = self.figure.dpi.get()
        self.figure.set_figsize_inches (w/dpi, h/dpi)
        
        self._draw_pixmap = True
        return True
        

    def draw(self):
        self._draw_pixmap = True
        self.expose_event(self, None)


    def _renderer_init(self):
        """Override by GTK backends to select a different renderer
        Renderer should provide the methods:
            _set_pixmap ()
            _set_width_height ()
        that are used by
            _render_to_pixmap()        
        """
        self._renderer = RendererGDK (self, self.figure.dpi)


    def _render_to_pixmap(self, width, height):
        """Render the figure to a gdk.Pixmap, is used for
           - rendering the pixmap to display        (pylab.draw)
           - rendering the pixmap to save to a file (pylab.savefig)
        Should not be overridden
        """
        if DEBUG: print 'FigureCanvasGTK.%s' % fn_name()
        create_pixmap = False
        if width > self._pixmap_width:
            # increase the pixmap in 10%+ (rather than 1 pixel) steps
            self._pixmap_width  = max (int (self._pixmap_width  * 1.1), width)
            create_pixmap = True

        if height > self._pixmap_height:
            self._pixmap_height = max (int (self._pixmap_height * 1.1), height)
            create_pixmap = True

        if create_pixmap:
            if DEBUG: print 'FigureCanvasGTK.%s new pixmap' % fn_name()
            self._pixmap = gtk.gdk.Pixmap (self.window, self._pixmap_width,
                                           self._pixmap_height)
            self._renderer._set_pixmap (self._pixmap)

        self._renderer._set_width_height (width, height)
        self.figure.draw (self._renderer)


    def expose_event(self, widget, event):
        """Expose_event for all GTK backends
        Should not be overridden.
        """
        if DEBUG: print 'FigureCanvasGTK.%s' % fn_name()
        if self._draw_pixmap and GTK_WIDGET_DRAWABLE(self):
            width, height = self.allocation.width, self.allocation.height
            self._render_to_pixmap(width, height)
            self.window.set_back_pixmap (self._pixmap, False)
            self.window.clear()  # draw pixmap as the gdk.Window's bg
            self._draw_pixmap = False
        return True


    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait'):
        root, ext = os.path.splitext(filename)       
        ext = ext[1:]
        if ext == '':
            ext      = IMAGE_FORMAT_DEFAULT
            filename = filename + '.' + ext        

        # save figure settings
        origDPI       = self.figure.dpi.get()
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()
        origWIn, origHIn = self.figure.get_size_inches()

        if self.flags() & gtk.REALIZED == 0:
            # for self.window(for pixmap) and has a side effect of altering figure width,height (via configure-event?)
            gtk.DrawingArea.realize(self) 

        self.figure.dpi.set(dpi)        
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        ext = ext.lower()
        if ext in ('jpg', 'png'):          # native printing
            width, height = self.figure.get_width_height()
            width, height = int(width), int(height)
            self._render_to_pixmap(width, height)

            # jpg colors don't match the display very well, png colors match better
            pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, 0, 8,
                                    width, height)
            pixbuf.get_from_drawable(self._pixmap, self._pixmap.get_colormap(),
                                     0, 0, 0, 0, width, height)
        
            # pixbuf.save() recognises 'jpeg' not 'jpg'
            if ext == 'jpg': ext = 'jpeg' 
            pixbuf.save(filename, ext)
            try: pixbuf.save(filename, ext)
            except gobject.GError, exc:
                error_msg('Save figure failure:\n%s' % (exc,), parent=self)

        elif ext in ('eps', 'ps', 'svg',):
            if ext == 'svg':
                from backend_svg import FigureCanvasSVG as FigureCanvas
            else:
                from backend_ps  import FigureCanvasPS  as FigureCanvas

            fc = self.switch_backends(FigureCanvas)
            fc.print_figure(filename, dpi, facecolor, edgecolor, orientation)
            try:
                fc = self.switch_backends(FigureCanvas)
                fc.print_figure(filename, dpi, facecolor, edgecolor, orientation)
            except IOError, exc:
                error_msg("Save figure failure:\n%s: %s" %
                          (exc.filename, exc.strerror), parent=self)
            except Exception, exc:
                error_msg("Save figure failure:\n%s" % exc, parent=self)

        elif ext in ('bmp', 'raw', 'rgb',):
            try: 
                from backend_agg import FigureCanvasAgg  as FigureCanvas
            except:
                error_msg('Save figure failure:\n'
                          'Agg must be loaded to save as bmp, raw and rgb',
                          parent=self)                
            else:
                fc = self.switch_backends(FigureCanvas)
                fc.print_figure(filename, dpi, facecolor, edgecolor, orientation)

        else:
            error_msg('Format "%s" is not supported.\nSupported formats are %s.' %
                      (ext, ', '.join(IMAGE_FORMAT)),
                      parent=self)

        # restore figure settings
        self.figure.dpi.set(origDPI)
        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)
        self.figure.set_figsize_inches(origWIn, origHIn)


class FigureManagerGTK(FigureManagerBase):
    """
    Public attributes

    canvas      : The FigureCanvas instance
    num         : The Figure number
    toolbar     : The gtk.Toolbar  (gtk only)
    window      : The gtk.Window   (gtk only)
    """
    def __init__(self, canvas, num):
        if DEBUG: print 'FigureManagerGTK.%s' % fn_name()
        FigureManagerBase.__init__(self, canvas, num)
        
        self.window = gtk.Window()
        self.window.set_title("Figure %d" % num)

        vbox = gtk.VBox()
        self.window.add(vbox)
        vbox.show()

        self.canvas.show()
        vbox.pack_start(self.canvas, True, True)

        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams['toolbar']=='classic':
            self.toolbar = NavigationToolbar (canvas, self.window)
        elif matplotlib.rcParams['toolbar']=='toolbar2':
            self.toolbar = NavigationToolbar2GTK (canvas, self.window)
        else:
            self.toolbar = None

        if self.toolbar is not None:
            self.toolbar.show()
            vbox.pack_end(self.toolbar, False, False)

        def destroy(*args): Gcf.destroy(num)
        self.window.connect("destroy", destroy)
        self.window.connect("delete_event", destroy)        
        if matplotlib.is_interactive():
            self.window.show()


    def add_subplot(self, *args, **kwargs):
        if DEBUG: print 'FigureManagerGTK.%s' % fn_name()
        a = FigureManagerBase.add_subplot(self, *args, **kwargs)
        if self.toolbar is not None: self.toolbar.update()
        return a
    
    def add_axes(self, rect, **kwargs):
        if DEBUG: print 'FigureManagerGTK.%s' % fn_name()
        a = FigureManagerBase.add_axes(self, rect, **kwargs)
        if self.toolbar is not None: self.toolbar.update()
        return a
    
    def destroy(self, *args):
        if DEBUG: print 'FigureManagerGTK.%s' % fn_name()
        self.window.destroy()
        if Gcf.get_num_fig_managers()==0 and not matplotlib.is_interactive():
            gtk.main_quit()

        
class NavigationToolbar2GTK(NavigationToolbar2, gtk.Toolbar):
    # list of toolitems to add to the toolbar, format is:
    # text, tooltip_text, image_file, callback(str)
    toolitems = (
        ('Home', 'Reset original view', 'home.png', 'home'),
        ('Back', 'Back to  previous view','back.png', 'back'),
        ('Forward', 'Forward to next view','forward.png', 'forward'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move.png','pan'),
        ('Zoom', 'Zoom to rectangle','zoom_to_rect.png', 'zoom'),
        (None, None, None, None),
        ('Save', 'Save the figure','filesave.png', 'save_figure'),
        )
        
    def __init__(self, canvas, window):
        self.win = window
        gtk.Toolbar.__init__(self)
        NavigationToolbar2.__init__(self, canvas)
        self._idleId = 0

    def set_message(self, s):
        if self._idleId==0: self.message.set_label(s)

        
    def set_cursor(self, cursor):
        self.canvas.window.set_cursor(cursord[cursor])

    def release(self, event):
        try: self._imageBack
        except AttributeError: pass
        else: del self._imageBack

    def dynamic_update(self):
        def idle_draw(*args):
            self.canvas.draw()
            self._idleId = 0
            return False
        if self._idleId==0:
            self._idleId = gtk.idle_add(idle_draw)
        
    def draw_rubberband(self, event, x0, y0, x1, y1):
        'adapted from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/189744'
        drawable = self.canvas.window
        if drawable is None: return

        gc = drawable.new_gc()

        height = self.canvas.figure.bbox.height()
        y1 = height - y1
        y0 = height - y0
        
        w = abs(x1 - x0)
        h = abs(y1 - y0)

        rect = [int(val)for val in min(x0,x1), min(y0, y1), w, h]
        try: lastrect, imageBack = self._imageBack
        except AttributeError:
            #snap image back        
            if event.inaxes is None: return

            ax = event.inaxes
            l,b,w,h = [int(val) for val in ax.bbox.get_bounds()]
            b = int(height)-(b+h)            # b = y = int(height)-(b+h)  # y not used
            axrect = l,b,w,h
            self._imageBack = axrect, drawable.get_image(*axrect)            
            drawable.draw_rectangle(gc, False, *rect)
            self._idleId = 0
        else:
            def idle_draw(*args):

                drawable.draw_image(gc, imageBack, 0, 0, *lastrect)
                drawable.draw_rectangle(gc, False, *rect)
                self._idleId = 0
                return False
            if self._idleId==0:
                self._idleId = gtk.idle_add(idle_draw)
        

    def _init_toolbar(self):
        self.set_style(gtk.TOOLBAR_ICONS)

        if gtk.pygtk_version >= (2,4,0):
            self._init_toolbar2_4()
        else:
            self._init_toolbar2_2()


    def _init_toolbar2_2(self):
        basedir = matplotlib.rcParams['datapath']

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                 self.append_space()
                 continue
            
            fname = os.path.join(basedir, image_file)
            image = gtk.Image()
            image.set_from_file(fname)
            w = self.append_item(text,
                                 tooltip_text,
                                 'Private',
                                 image,
                                 getattr(self, callback)
                                 )

        self.append_space()

        self.message = gtk.Label()
        self.append_widget(self.message, None, None)
        self.message.show()

        self.fileselect = FileSelection(title='Save the figure',
                                        parent=self.win,)

        
        
    def _init_toolbar2_4(self):
        basedir = matplotlib.rcParams['datapath']
        self.tooltips = gtk.Tooltips()

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.insert( gtk.SeparatorToolItem(), -1 )
                continue
            fname = os.path.join(basedir, image_file)
            image = gtk.Image()
            image.set_from_file(fname)
            tbutton = gtk.ToolButton(image, text)
            self.insert(tbutton, -1)
            tbutton.connect('clicked', getattr(self, callback))
            tbutton.set_tooltip(self.tooltips, tooltip_text, 'Private')

        toolitem = gtk.SeparatorToolItem()
        self.insert(toolitem, -1)
        toolitem.set_draw(False)  # set_draw() not making separator invisible, bug #143692 fixed Jun 06 2004, will be in GTK+ 2.6
        toolitem.set_expand(True)

        toolitem = gtk.ToolItem()
        self.insert(toolitem, -1)
        self.message = gtk.Label()
        toolitem.add(self.message)

        self.show_all()

        self.fileselect = FileChooserDialog(title='Save the figure',
                                            parent=self.win,)
                                            
    
    def save_figure(self, button):
        fname = self.fileselect.get_filename_from_user()
        if fname:
            self.canvas.print_figure(fname)
                
            
class NavigationToolbar(gtk.Toolbar):
    """
    Public attributes

      canvas - the FigureCanvas  (gtk.DrawingArea)
      win    - the gtk.Window

    """
    # list of toolitems to add to the toolbar, format is:
    # text, tooltip_text, image, callback(str), callback_arg, scroll(bool)
    toolitems = (
        ('Left', 'Pan left with click or wheel mouse (bidirectional)',
         gtk.STOCK_GO_BACK, 'panx', -1, True),
        ('Right', 'Pan right with click or wheel mouse (bidirectional)',
         gtk.STOCK_GO_FORWARD, 'panx', 1, True),
        ('Zoom In X', 'Zoom In X (shrink the x axis limits) with click or wheel' 
         ' mouse (bidirectional)',
         gtk.STOCK_ZOOM_IN, 'zoomx', 1, True),
        ('Zoom Out X', 'Zoom Out X (expand the x axis limits) with click or wheel'
         ' mouse (bidirectional)',
         gtk.STOCK_ZOOM_OUT, 'zoomx', -1, True),
        (None, None, None, None, None, None,),   
        ('Up', 'Pan up with click or wheel mouse (bidirectional)',
         gtk.STOCK_GO_UP, 'pany', 1, True),
        ('Down', 'Pan down with click or wheel mouse (bidirectional)',
         gtk.STOCK_GO_DOWN, 'pany', -1, True),
        ('Zoom In Y', 'Zoom in Y (shrink the y axis limits) with click or wheel'
         ' mouse (bidirectional)',
         gtk.STOCK_ZOOM_IN, 'zoomy', 1, True),
        ('Zoom Out Y', 'Zoom Out Y (expand the y axis limits) with click or wheel'
         ' mouse (bidirectional)',
         gtk.STOCK_ZOOM_OUT, 'zoomy', -1, True),
        (None, None, None, None, None, None,),
        ('Save', 'Save the figure',
         gtk.STOCK_SAVE, 'save_figure', None, False),
        )
    
    def __init__(self, canvas, window):
        """
        figManager is the FigureManagerGTK instance that contains the
        toolbar, with attributes figure, window and drawingArea
        
        """
        gtk.Toolbar.__init__(self)

        self.canvas = canvas
        self.win    = window # Note: gtk.Toolbar already has a 'window' attribute
        
        self.set_style(gtk.TOOLBAR_ICONS)

        if gtk.pygtk_version >= (2,4,0):
            self._create_toolitems_2_4()
            self.update = self._update_2_4
            self.fileselect = FileChooserDialog(title='Save the figure',
                                                parent=self.win,) 
        else:
            self._create_toolitems_2_2()
            self.update = self._update_2_2
            self.fileselect = FileSelection(title='Save the figure',
                                            parent=self.win)
        self.show_all()            
        self.update()


    def _create_toolitems_2_4(self):
        # use the GTK+ 2.4 GtkToolbar API
        iconSize = gtk.ICON_SIZE_SMALL_TOOLBAR
        self.tooltips = gtk.Tooltips()

        for text, tooltip_text, image, callback, callback_arg, scroll \
                in self.toolitems:
            if text is None:
                self.insert( gtk.SeparatorToolItem(), -1 )
                continue
            tbutton = gtk.ToolButton(gtk.image_new_from_stock(image, iconSize),
                                     text)
            self.insert(tbutton, -1)
            if callback_arg:
                tbutton.connect('clicked', getattr(self, callback), callback_arg)
            else:
                tbutton.connect('clicked', getattr(self, callback))
            if scroll:
                tbutton.connect('scroll_event', getattr(self, callback))
            tbutton.set_tooltip(self.tooltips, tooltip_text, 'Private')

        # Axes toolitem, is empty at start, update() adds a menu if >=2 axes
        self.axes_toolitem = gtk.ToolItem()
        self.insert(self.axes_toolitem, 0)
        self.axes_toolitem.set_tooltip(self.tooltips,
                                       tip_text='Select axes that controls affect',
                                       tip_private = 'Private')

        align = gtk.Alignment (xalign=0.5, yalign=0.5, xscale=0.0, yscale=0.0)
        self.axes_toolitem.add(align)

        self.menubutton = gtk.Button ("Axes")
        align.add (self.menubutton)

        def position_menu (menu):
            """Function for positioning a popup menu.
            Place menu below the menu button, but ensure it does not go off
            the bottom of the screen.
            The default is to popup menu at current mouse position
            """
            x0, y0    = self.window.get_origin()      
            x1, y1, m = self.window.get_pointer()     
            x2, y2    = self.menubutton.get_pointer() 
            sc_h      = self.get_screen().get_height()  # requires GTK+ 2.2 +
            w, h      = menu.size_request()

            x = x0 + x1 - x2
            y = y0 + y1 - y2 + self.menubutton.allocation.height
            y = min(y, sc_h - h)
            return x, y, True
        
        def button_clicked (button, data=None):
            self.axismenu.popup (None, None, position_menu, 0, gtk.get_current_event_time())

        self.menubutton.connect ("clicked", button_clicked)

        
    def _update_2_4(self):
        # for GTK+ 2.4+
        # called by __init__() and FigureManagerGTK
        
        self._axes = self.canvas.figure.axes

        if len(self._axes) >= 2:
            self.axismenu = self._make_axis_menu()
            self.menubutton.show_all()
        else:
            self.menubutton.hide()
            
        self.set_active(range(len(self._axes)))


    def _create_toolitems_2_2(self):
        # use the GTK+ 2.2 (and lower) GtkToolbar API
        iconSize = gtk.ICON_SIZE_SMALL_TOOLBAR

        for text, tooltip_text, image, callback, callback_arg, scroll \
                in self.toolitems:
            if text == None:
                self.append_space()
                continue
            item = self.append_item(text, tooltip_text, 'Private',
                                    gtk.image_new_from_stock(image, iconSize),
                                    getattr(self, callback), callback_arg)
            if scroll:
                item.connect("scroll_event", getattr(self, callback))

        self.omenu = gtk.OptionMenu()
        self.omenu.set_border_width(3)
        self.insert_widget(
            self.omenu,
            'Select axes that controls affect',
            'Private', 0)


    def _update_2_2(self):
        # for GTK+ 2.2 and lower
        # called by __init__() and FigureManagerGTK
        
        self._axes = self.canvas.figure.axes
        
        if len(self._axes) >= 2:                
            # set up the axis menu
            self.omenu.set_menu( self._make_axis_menu() )
            self.omenu.show_all()
        else:
            self.omenu.hide()
            
        self.set_active(range(len(self._axes))) 


    def _make_axis_menu(self):
        # called by self._update*()

        def toggled(item, data=None):
            if item == self.itemAll:
                for item in items: item.set_active(True)
            elif item == self.itemInvert:
                for item in items:
                    item.set_active(not item.get_active())

            ind = [i for i,item in enumerate(items) if item.get_active()]
            self.set_active(ind)
            
        menu = gtk.Menu()

        self.itemAll = gtk.MenuItem("All")
        menu.append(self.itemAll)
        self.itemAll.connect("activate", toggled)

        self.itemInvert = gtk.MenuItem("Invert")
        menu.append(self.itemInvert)
        self.itemInvert.connect("activate", toggled)

        items = []
        for i in range(len(self._axes)):
            item = gtk.CheckMenuItem("Axis %d" % (i+1))
            menu.append(item)
            item.connect("toggled", toggled)
            item.set_active(True)
            items.append(item)

        menu.show_all()
        return menu
    

    def set_active(self, ind):
        self._ind = ind
        self._active = [ self._axes[i] for i in self._ind ]
        
    def panx(self, button, arg):
        """arg is either user callback data or a scroll event
        """
        try:
            if arg.direction == gdk.SCROLL_UP: direction=1
            else: direction=-1
        except AttributeError:
            direction = arg

        for a in self._active:
            a.panx(direction)
        self.canvas.draw()
        return True
    
    def pany(self, button, arg):
        try:
            if arg.direction == gdk.SCROLL_UP: direction=1
            else: direction=-1
        except AttributeError:
            direction = arg

        for a in self._active:
            a.pany(direction)
        self.canvas.draw()
        return True
    
    def zoomx(self, button, arg):
        try:
            if arg.direction == gdk.SCROLL_UP: direction=1
            else: direction=-1
        except AttributeError:
            direction = arg

        for a in self._active:
            a.zoomx(direction)
        self.canvas.draw()
        return True

    def zoomy(self, button, arg):
        try:
            if arg.direction == gdk.SCROLL_UP: direction=1
            else: direction=-1
        except AttributeError:
            direction = arg

        for a in self._active:
            a.zoomy(direction)
        self.canvas.draw()
        return True


    def save_figure(self, button):
        fname = self.fileselect.get_filename_from_user()
        if fname:
            self.canvas.print_figure(fname)
            

class FileSelection(gtk.FileSelection):
    """GTK+ 2.2 and lower file selector which remembers the last
    file/directory selected
    """
    def __init__(self, path=None, title='Select a file', parent=None):
        super(FileSelection, self).__init__(title)

        if path: self.path = path
        else:    self.path = os.getcwd() + os.sep

        if parent: self.set_transient_for(parent)
            
    def get_filename_from_user(self, path=None, title=None):
        if path:  self.path = path
        if title: self.set_title(title)
        self.set_filename(self.path)

        filename = None
        if self.run() == gtk.RESPONSE_OK:
            self.path = filename = self.get_filename()
        self.hide()
        return filename
    

if gtk.pygtk_version >= (2,4,0):
    class FileChooserDialog(gtk.FileChooserDialog):
        """GTK+ 2.4 file selector which remembers the last file/directory
        selected and presents the user with a menu of supported image formats
        """
        def __init__ (self,
                      title   = 'Save file',
                      parent  = None,
                      action  = gtk.FILE_CHOOSER_ACTION_SAVE,
                      buttons = (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                 gtk.STOCK_SAVE,   gtk.RESPONSE_OK),
                      backend = '',
                      path    = None,
                      ):
            super (FileChooserDialog, self).__init__ (title, parent, action,
                                                      buttons, backend)
            self.set_default_response (gtk.RESPONSE_OK)

            if path: self.path = path
            else:    self.path = os.getcwd() + os.sep

            # create an extra widget to list supported image formats
            self.set_current_folder (self.path)
            self.set_current_name ('image.' + IMAGE_FORMAT_DEFAULT)

            hbox = gtk.HBox (spacing=10)
            hbox.pack_start (gtk.Label ("Image Format:"), expand=False)
            
            self.cbox = gtk.combo_box_new_text()
            hbox.pack_start (self.cbox)

            for item in IMAGE_FORMAT:
                self.cbox.append_text (item)
            self.cbox.set_active (IMAGE_FORMAT.index (IMAGE_FORMAT_DEFAULT))

            def cb_cbox_changed (cbox, data=None):
                """File extension changed"""
                head, filename = os.path.split(self.get_filename())
                root, ext = os.path.splitext(filename)
                ext = ext[1:]
                new_ext = IMAGE_FORMAT[cbox.get_active()]

                if ext in IMAGE_FORMAT:
                    filename = filename.replace(ext, new_ext)
                elif ext == '':
                    filename = filename.rstrip('.') + '.' + new_ext
                    
                self.set_current_name (filename)
            self.cbox.connect ("changed", cb_cbox_changed)

            hbox.show_all()
            self.set_extra_widget(hbox)
            

        def get_filename_from_user (self):
            filename = None
            while True:
                if self.run() != gtk.RESPONSE_OK:
                    filename = None
                    break
                filename = self.get_filename()
                menu_ext  = IMAGE_FORMAT[self.cbox.get_active()]
                root, ext = os.path.splitext(filename)
                ext = ext[1:]
                if ext == '':
                    ext = menu_ext
                    filename += '.' + ext

                if ext in IMAGE_FORMAT:
                    self.path = filename
                    break
                else:
                    error_msg('Image format "%s" is not supported' % ext,
                              parent=self)
                    self.set_current_name(os.path.split(root)[1] + '.' + menu_ext)
                    
            self.hide()
            return filename


# set icon used when windows are minimized, it requires
# gtk.pygtk_version >= (2,2,0) with a GDK pixbuf loader for SVG installed
try:
    gtk.window_set_default_icon_from_file (
        os.path.join (matplotlib.rcParams['datapath'], 'matplotlib.svg'))
except:
    verbose.report('Could not load matplotlib icon: %s' % sys.exc_info()[1])


def error_msg_gtk(msg, parent=None):
    if parent: # find the toplevel gtk.Window
        parent = parent.get_toplevel()
        if not parent.flags() & gtk.TOPLEVEL:
            parent = None

    if not is_string_like(msg):
        msg = ','.join(map(str,msg))
                          
    dialog = gtk.MessageDialog(
        parent         = parent,
        type           = gtk.MESSAGE_ERROR,
        buttons        = gtk.BUTTONS_OK,
        message_format = msg)
    dialog.run()
    dialog.destroy()


def exception_handler(type, value, tb):
    """Handle uncaught exceptions
    It does not catch SystemExit
    """
    msg = ''
    # get the filename attribute if available (for IOError)
    if hasattr(value, 'filename') and value.filename != None:
        msg = value.filename + ': '
    if hasattr(value, 'strerror') and value.strerror != None:
        msg += value.strerror
    else:
        msg += str(value)

    if len(msg) :error_msg_gtk(msg)

# override excepthook only if it has not already been overridden
#if sys.__excepthook__ is sys.excepthook:
#    sys.excepthook = exception_handler

FigureManager = FigureManagerGTK
error_msg = error_msg_gtk
