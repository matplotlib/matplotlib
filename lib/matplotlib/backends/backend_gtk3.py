from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import os, sys
def fn_name(): return sys._getframe(1).f_code.co_name

try:
    import gi
except ImportError:
    raise ImportError("Gtk3 backend requires pygobject to be installed.")

try:
    gi.require_version("Gtk", "3.0")
except AttributeError:
    raise ImportError(
        "pygobject version too old -- it must have require_version")
except ValueError:
    raise ImportError(
        "Gtk3 backend requires the GObject introspection bindings for Gtk 3 "
        "to be installed.")

try:
    from gi.repository import Gtk, Gdk, GObject, GLib
except ImportError:
    raise ImportError("Gtk3 backend requires pygobject to be installed.")

import matplotlib
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase, \
     FigureManagerBase, FigureCanvasBase, NavigationToolbar2, cursors, TimerBase
from matplotlib.backend_bases import (ShowBase, ToolContainerBase,
                                      StatusbarBase)
from matplotlib.backend_managers import ToolManager
from matplotlib import backend_tools

from matplotlib.cbook import is_string_like, is_writable_file_like
from matplotlib.colors import colorConverter
from matplotlib.figure import Figure
from matplotlib.widgets import SubplotTool

from matplotlib import lines
from matplotlib import cbook
from matplotlib import verbose
from matplotlib import rcParams

backend_version = "%s.%s.%s" % (Gtk.get_major_version(), Gtk.get_micro_version(), Gtk.get_minor_version())

_debug = False
#_debug = True

# the true dots per inch on the screen; should be display dependent
# see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi
PIXELS_PER_INCH = 96

cursord = {
    cursors.MOVE          : Gdk.Cursor.new(Gdk.CursorType.FLEUR),
    cursors.HAND          : Gdk.Cursor.new(Gdk.CursorType.HAND2),
    cursors.POINTER       : Gdk.Cursor.new(Gdk.CursorType.LEFT_PTR),
    cursors.SELECT_REGION : Gdk.Cursor.new(Gdk.CursorType.TCROSS),
    }

def draw_if_interactive():
    """
    Is called after every pylab drawing command
    """
    if matplotlib.is_interactive():
        figManager =  Gcf.get_active()
        if figManager is not None:
            figManager.canvas.draw_idle()

class Show(ShowBase):
    def mainloop(self):
        if Gtk.main_level() == 0:
            Gtk.main()

show = Show()


class TimerGTK3(TimerBase):
    '''
    Subclass of :class:`backend_bases.TimerBase` that uses GTK3 for timer events.

    Attributes:
    * interval: The time between timer events in milliseconds. Default
        is 1000 ms.
    * single_shot: Boolean flag indicating whether this timer should
        operate as single shot (run once and then stop). Defaults to False.
    * callbacks: Stores list of (func, args) tuples that will be called
        upon timer events. This list can be manipulated directly, or the
        functions add_callback and remove_callback can be used.
    '''
    def _timer_start(self):
        # Need to stop it, otherwise we potentially leak a timer id that will
        # never be stopped.
        self._timer_stop()
        self._timer = GLib.timeout_add(self._interval, self._on_timer)

    def _timer_stop(self):
        if self._timer is not None:
            GLib.source_remove(self._timer)
            self._timer = None

    def _timer_set_interval(self):
        # Only stop and restart it if the timer has already been started
        if self._timer is not None:
            self._timer_stop()
            self._timer_start()

    def _on_timer(self):
        TimerBase._on_timer(self)

        # Gtk timeout_add() requires that the callback returns True if it
        # is to be called again.
        if len(self.callbacks) > 0 and not self._single:
            return True
        else:
            self._timer = None
            return False

class FigureCanvasGTK3 (Gtk.DrawingArea, FigureCanvasBase):
    keyvald = {65507 : 'control',
               65505 : 'shift',
               65513 : 'alt',
               65508 : 'control',
               65506 : 'shift',
               65514 : 'alt',
               65361 : 'left',
               65362 : 'up',
               65363 : 'right',
               65364 : 'down',
               65307 : 'escape',
               65470 : 'f1',
               65471 : 'f2',
               65472 : 'f3',
               65473 : 'f4',
               65474 : 'f5',
               65475 : 'f6',
               65476 : 'f7',
               65477 : 'f8',
               65478 : 'f9',
               65479 : 'f10',
               65480 : 'f11',
               65481 : 'f12',
               65300 : 'scroll_lock',
               65299 : 'break',
               65288 : 'backspace',
               65293 : 'enter',
               65379 : 'insert',
               65535 : 'delete',
               65360 : 'home',
               65367 : 'end',
               65365 : 'pageup',
               65366 : 'pagedown',
               65438 : '0',
               65436 : '1',
               65433 : '2',
               65435 : '3',
               65430 : '4',
               65437 : '5',
               65432 : '6',
               65429 : '7',
               65431 : '8',
               65434 : '9',
               65451 : '+',
               65453 : '-',
               65450 : '*',
               65455 : '/',
               65439 : 'dec',
               65421 : 'enter',
               }

    # Setting this as a static constant prevents
    # this resulting expression from leaking
    event_mask = (Gdk.EventMask.BUTTON_PRESS_MASK   |
                  Gdk.EventMask.BUTTON_RELEASE_MASK |
                  Gdk.EventMask.EXPOSURE_MASK       |
                  Gdk.EventMask.KEY_PRESS_MASK      |
                  Gdk.EventMask.KEY_RELEASE_MASK    |
                  Gdk.EventMask.ENTER_NOTIFY_MASK   |
                  Gdk.EventMask.LEAVE_NOTIFY_MASK   |
                  Gdk.EventMask.POINTER_MOTION_MASK |
                  Gdk.EventMask.POINTER_MOTION_HINT_MASK|
                  Gdk.EventMask.SCROLL_MASK)

    def __init__(self, figure):
        if _debug: print('FigureCanvasGTK3.%s' % fn_name())
        FigureCanvasBase.__init__(self, figure)
        GObject.GObject.__init__(self)

        self._idle_draw_id  = 0
        self._need_redraw   = True
        self._lastCursor    = None

        self.connect('scroll_event',         self.scroll_event)
        self.connect('button_press_event',   self.button_press_event)
        self.connect('button_release_event', self.button_release_event)
        self.connect('configure_event',      self.configure_event)
        self.connect('draw',                 self.on_draw_event)
        self.connect('key_press_event',      self.key_press_event)
        self.connect('key_release_event',    self.key_release_event)
        self.connect('motion_notify_event',  self.motion_notify_event)
        self.connect('leave_notify_event',   self.leave_notify_event)
        self.connect('enter_notify_event',   self.enter_notify_event)
        self.connect('size_allocate',        self.size_allocate)

        self.set_events(self.__class__.event_mask)

        self.set_double_buffered(True)
        self.set_can_focus(True)
        self._renderer_init()
        default_context = GLib.main_context_get_thread_default() or GLib.main_context_default()

    def destroy(self):
        #Gtk.DrawingArea.destroy(self)
        self.close_event()
        if self._idle_draw_id != 0:
            GLib.source_remove(self._idle_draw_id)

    def scroll_event(self, widget, event):
        if _debug: print('FigureCanvasGTK3.%s' % fn_name())
        x = event.x
        # flipy so y=0 is bottom of canvas
        y = self.get_allocation().height - event.y
        if event.direction==Gdk.ScrollDirection.UP:
            step = 1
        else:
            step = -1
        FigureCanvasBase.scroll_event(self, x, y, step, guiEvent=event)
        return False  # finish event propagation?

    def button_press_event(self, widget, event):
        if _debug: print('FigureCanvasGTK3.%s' % fn_name())
        x = event.x
        # flipy so y=0 is bottom of canvas
        y = self.get_allocation().height - event.y
        FigureCanvasBase.button_press_event(self, x, y, event.button, guiEvent=event)
        return False  # finish event propagation?

    def button_release_event(self, widget, event):
        if _debug: print('FigureCanvasGTK3.%s' % fn_name())
        x = event.x
        # flipy so y=0 is bottom of canvas
        y = self.get_allocation().height - event.y
        FigureCanvasBase.button_release_event(self, x, y, event.button, guiEvent=event)
        return False  # finish event propagation?

    def key_press_event(self, widget, event):
        if _debug: print('FigureCanvasGTK3.%s' % fn_name())
        key = self._get_key(event)
        if _debug: print("hit", key)
        FigureCanvasBase.key_press_event(self, key, guiEvent=event)
        return False  # finish event propagation?

    def key_release_event(self, widget, event):
        if _debug: print('FigureCanvasGTK3.%s' % fn_name())
        key = self._get_key(event)
        if _debug: print("release", key)
        FigureCanvasBase.key_release_event(self, key, guiEvent=event)
        return False  # finish event propagation?

    def motion_notify_event(self, widget, event):
        if _debug: print('FigureCanvasGTK3.%s' % fn_name())
        if event.is_hint:
            t, x, y, state = event.window.get_pointer()
        else:
            x, y, state = event.x, event.y, event.get_state()

        # flipy so y=0 is bottom of canvas
        y = self.get_allocation().height - y
        FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=event)
        return False  # finish event propagation?

    def leave_notify_event(self, widget, event):
        FigureCanvasBase.leave_notify_event(self, event)

    def enter_notify_event(self, widget, event):
        FigureCanvasBase.enter_notify_event(self, event)

    def size_allocate(self, widget, allocation):
        if _debug:
            print("FigureCanvasGTK3.%s" % fn_name())
            print("size_allocate (%d x %d)" % (allocation.width, allocation.height))
        dpival = self.figure.dpi
        winch = allocation.width / dpival
        hinch = allocation.height / dpival
        self.figure.set_size_inches(winch, hinch)
        FigureCanvasBase.resize_event(self)
        self.draw_idle()

    def _get_key(self, event):
        if event.keyval in self.keyvald:
            key = self.keyvald[event.keyval]
        elif event.keyval < 256:
            key = chr(event.keyval)
        else:
            key = None

        modifiers = [
                     (Gdk.ModifierType.MOD4_MASK, 'super'),
                     (Gdk.ModifierType.MOD1_MASK, 'alt'),
                     (Gdk.ModifierType.CONTROL_MASK, 'ctrl'),
                    ]
        for key_mask, prefix in modifiers:
            if event.state & key_mask:
                key = '{0}+{1}'.format(prefix, key)

        return key

    def configure_event(self, widget, event):
        if _debug: print('FigureCanvasGTK3.%s' % fn_name())
        if widget.get_property("window") is None:
            return
        w, h = event.width, event.height
        if w < 3 or h < 3:
            return # empty fig

        # resize the figure (in inches)
        dpi = self.figure.dpi
        self.figure.set_size_inches (w/dpi, h/dpi)
        self._need_redraw = True

        return False  # finish event propagation?

    def on_draw_event(self, widget, ctx):
        # to be overwritten by GTK3Agg or GTK3Cairo
        pass

    def draw(self):
        self._need_redraw = True
        if self.get_visible() and self.get_mapped():
            self.queue_draw()
            # do a synchronous draw (its less efficient than an async draw,
            # but is required if/when animation is used)
            self.get_property("window").process_updates (False)

    def draw_idle(self):
        if self._idle_draw_id != 0:
            return
        def idle_draw(*args):
            try:
                self.draw()
            finally:
                self._idle_draw_id = 0
            return False
        self._idle_draw_id = GLib.idle_add(idle_draw)

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
        return TimerGTK3(*args, **kwargs)

    def flush_events(self):
        Gdk.threads_enter()
        while Gtk.events_pending():
            Gtk.main_iteration()
        Gdk.flush()
        Gdk.threads_leave()

    def start_event_loop(self,timeout):
        FigureCanvasBase.start_event_loop_default(self,timeout)
    start_event_loop.__doc__=FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        FigureCanvasBase.stop_event_loop_default(self)
    stop_event_loop.__doc__=FigureCanvasBase.stop_event_loop_default.__doc__


class FigureManagerGTK3(FigureManagerBase):
    """
    Public attributes

    canvas      : The FigureCanvas instance
    num         : The Figure number
    toolbar     : The Gtk.Toolbar  (gtk only)
    vbox        : The Gtk.VBox containing the canvas and toolbar (gtk only)
    window      : The Gtk.Window   (gtk only)
    """
    def __init__(self, canvas, num):
        if _debug: print('FigureManagerGTK3.%s' % fn_name())
        FigureManagerBase.__init__(self, canvas, num)

        self.window = Gtk.Window()
        self.set_window_title("Figure %d" % num)
        try:
            self.window.set_icon_from_file(window_icon)
        except (SystemExit, KeyboardInterrupt):
            # re-raise exit type Exceptions
            raise
        except:
            # some versions of gtk throw a glib.GError but not
            # all, so I am not sure how to catch it.  I am unhappy
            # doing a blanket catch here, but am not sure what a
            # better way is - JDH
            verbose.report('Could not load matplotlib icon: %s' % sys.exc_info()[1])

        self.vbox = Gtk.Box()
        self.vbox.set_property("orientation", Gtk.Orientation.VERTICAL)
        self.window.add(self.vbox)
        self.vbox.show()

        self.canvas.show()

        self.vbox.pack_start(self.canvas, True, True, 0)
        # calculate size for window
        w = int (self.canvas.figure.bbox.width)
        h = int (self.canvas.figure.bbox.height)

        self.toolmanager = self._get_toolmanager()
        self.toolbar = self._get_toolbar()
        self.statusbar = None

        def add_widget(child, expand, fill, padding):
            child.show()
            self.vbox.pack_end(child, False, False, 0)
            size_request = child.size_request()
            return size_request.height

        if self.toolmanager:
            backend_tools.add_tools_to_manager(self.toolmanager)
            if self.toolbar:
                backend_tools.add_tools_to_container(self.toolbar)
                self.statusbar = StatusbarGTK3(self.toolmanager)
                h += add_widget(self.statusbar, False, False, 0)
                h += add_widget(Gtk.HSeparator(), False, False, 0)

        if self.toolbar is not None:
            self.toolbar.show()
            h += add_widget(self.toolbar, False, False, 0)

        self.window.set_default_size (w, h)

        def destroy(*args):
            Gcf.destroy(num)
        self.window.connect("destroy", destroy)
        self.window.connect("delete_event", destroy)
        if matplotlib.is_interactive():
            self.window.show()
            self.canvas.draw_idle()

        def notify_axes_change(fig):
            'this will be called whenever the current axes is changed'
            if self.toolmanager is not None:
                pass
            elif self.toolbar is not None:
                self.toolbar.update()
        self.canvas.figure.add_axobserver(notify_axes_change)

        self.canvas.grab_focus()

    def destroy(self, *args):
        if _debug: print('FigureManagerGTK3.%s' % fn_name())
        self.vbox.destroy()
        self.window.destroy()
        self.canvas.destroy()
        if self.toolbar:
            self.toolbar.destroy()

        if Gcf.get_num_fig_managers()==0 and \
               not matplotlib.is_interactive() and \
               Gtk.main_level() >= 1:
            Gtk.main_quit()

    def show(self):
        # show the figure window
        self.window.show()

    def full_screen_toggle (self):
        self._full_screen_flag = not self._full_screen_flag
        if self._full_screen_flag:
            self.window.fullscreen()
        else:
            self.window.unfullscreen()
    _full_screen_flag = False


    def _get_toolbar(self):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if rcParams['toolbar'] == 'toolbar2':
            toolbar = NavigationToolbar2GTK3 (self.canvas, self.window)
        elif rcParams['toolbar'] == 'toolmanager':
            toolbar = ToolbarGTK3(self.toolmanager)
        else:
            toolbar = None
        return toolbar

    def _get_toolmanager(self):
        # must be initialised after toolbar has been setted
        if rcParams['toolbar'] != 'toolbar2':
            toolmanager = ToolManager(self.canvas)
        else:
            toolmanager = None
        return toolmanager

    def get_window_title(self):
        return self.window.get_title()

    def set_window_title(self, title):
        self.window.set_title(title)

    def resize(self, width, height):
        'set the canvas size in pixels'
        #_, _, cw, ch = self.canvas.allocation
        #_, _, ww, wh = self.window.allocation
        #self.window.resize (width-cw+ww, height-ch+wh)
        self.window.resize(width, height)


class NavigationToolbar2GTK3(NavigationToolbar2, Gtk.Toolbar):
    def __init__(self, canvas, window):
        self.win = window
        GObject.GObject.__init__(self)
        NavigationToolbar2.__init__(self, canvas)
        self.ctx = None

    def set_message(self, s):
        self.message.set_label(s)

    def set_cursor(self, cursor):
        self.canvas.get_property("window").set_cursor(cursord[cursor])
        #self.canvas.set_cursor(cursord[cursor])

    def release(self, event):
        try: del self._pixmapBack
        except AttributeError: pass

    def dynamic_update(self):
        # legacy method; new method is canvas.draw_idle
        self.canvas.draw_idle()

    def draw_rubberband(self, event, x0, y0, x1, y1):
        'adapted from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/189744'
        self.ctx = self.canvas.get_property("window").cairo_create()

        # todo: instead of redrawing the entire figure, copy the part of
        # the figure that was covered by the previous rubberband rectangle
        self.canvas.draw()

        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        rect = [int(val) for val in (min(x0,x1), min(y0, y1), w, h)]

        self.ctx.new_path()
        self.ctx.set_line_width(0.5)
        self.ctx.rectangle(rect[0], rect[1], rect[2], rect[3])
        self.ctx.set_source_rgb(0, 0, 0)
        self.ctx.stroke()

    def _init_toolbar(self):
        self.set_style(Gtk.ToolbarStyle.ICONS)
        basedir = os.path.join(rcParams['datapath'],'images')

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.insert( Gtk.SeparatorToolItem(), -1 )
                continue
            fname = os.path.join(basedir, image_file + '.png')
            image = Gtk.Image()
            image.set_from_file(fname)
            tbutton = Gtk.ToolButton()
            tbutton.set_label(text)
            tbutton.set_icon_widget(image)
            self.insert(tbutton, -1)
            tbutton.connect('clicked', getattr(self, callback))
            tbutton.set_tooltip_text(tooltip_text)

        toolitem = Gtk.SeparatorToolItem()
        self.insert(toolitem, -1)
        toolitem.set_draw(False)
        toolitem.set_expand(True)

        toolitem = Gtk.ToolItem()
        self.insert(toolitem, -1)
        self.message = Gtk.Label()
        toolitem.add(self.message)

        self.show_all()

    def get_filechooser(self):
        fc = FileChooserDialog(
            title='Save the figure',
            parent=self.win,
            path=os.path.expanduser(rcParams.get('savefig.directory', '')),
            filetypes=self.canvas.get_supported_filetypes(),
            default_filetype=self.canvas.get_default_filetype())
        fc.set_current_name(self.canvas.get_default_filename())
        return fc

    def save_figure(self, *args):
        chooser = self.get_filechooser()
        fname, format = chooser.get_filename_from_user()
        chooser.destroy()
        if fname:
            startpath = os.path.expanduser(rcParams.get('savefig.directory', ''))
            if startpath == '':
                # explicitly missing key or empty str signals to use cwd
                rcParams['savefig.directory'] = startpath
            else:
                # save dir for next time
                rcParams['savefig.directory'] = os.path.dirname(six.text_type(fname))
            try:
                self.canvas.print_figure(fname, format=format)
            except Exception as e:
                error_msg_gtk(str(e), parent=self)

    def configure_subplots(self, button):
        toolfig = Figure(figsize=(6,3))
        canvas = self._get_canvas(toolfig)
        toolfig.subplots_adjust(top=0.9)
        tool =  SubplotTool(self.canvas.figure, toolfig)

        w = int (toolfig.bbox.width)
        h = int (toolfig.bbox.height)


        window = Gtk.Window()
        try:
            window.set_icon_from_file(window_icon)
        except (SystemExit, KeyboardInterrupt):
            # re-raise exit type Exceptions
            raise
        except:
            # we presumably already logged a message on the
            # failure of the main plot, don't keep reporting
            pass
        window.set_title("Subplot Configuration Tool")
        window.set_default_size(w, h)
        vbox = Gtk.Box()
        vbox.set_property("orientation", Gtk.Orientation.VERTICAL)
        window.add(vbox)
        vbox.show()

        canvas.show()
        vbox.pack_start(canvas, True, True, 0)
        window.show()

    def _get_canvas(self, fig):
        return self.canvas.__class__(fig)


class FileChooserDialog(Gtk.FileChooserDialog):
    """GTK+ file selector which remembers the last file/directory
    selected and presents the user with a menu of supported image formats
    """
    def __init__ (self,
                  title   = 'Save file',
                  parent  = None,
                  action  = Gtk.FileChooserAction.SAVE,
                  buttons = (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                             Gtk.STOCK_SAVE,   Gtk.ResponseType.OK),
                  path    = None,
                  filetypes = [],
                  default_filetype = None
                  ):
        super (FileChooserDialog, self).__init__ (title, parent, action,
                                                  buttons)
        self.set_default_response (Gtk.ResponseType.OK)

        if not path: path = os.getcwd() + os.sep

        # create an extra widget to list supported image formats
        self.set_current_folder (path)
        self.set_current_name ('image.' + default_filetype)

        hbox = Gtk.Box(spacing=10)
        hbox.pack_start(Gtk.Label(label="File Format:"), False, False, 0)

        liststore = Gtk.ListStore(GObject.TYPE_STRING)
        cbox = Gtk.ComboBox() #liststore)
        cbox.set_model(liststore)
        cell = Gtk.CellRendererText()
        cbox.pack_start(cell, True)
        cbox.add_attribute(cell, 'text', 0)
        hbox.pack_start(cbox, False, False, 0)

        self.filetypes = filetypes
        self.sorted_filetypes = list(six.iteritems(filetypes))
        self.sorted_filetypes.sort()
        default = 0
        for i, (ext, name) in enumerate(self.sorted_filetypes):
            liststore.append(["%s (*.%s)" % (name, ext)])
            if ext == default_filetype:
                default = i
        cbox.set_active(default)
        self.ext = default_filetype

        def cb_cbox_changed (cbox, data=None):
            """File extension changed"""
            head, filename = os.path.split(self.get_filename())
            root, ext = os.path.splitext(filename)
            ext = ext[1:]
            new_ext = self.sorted_filetypes[cbox.get_active()][0]
            self.ext = new_ext

            if ext in self.filetypes:
                filename = root + '.' + new_ext
            elif ext == '':
                filename = filename.rstrip('.') + '.' + new_ext

            self.set_current_name (filename)
        cbox.connect ("changed", cb_cbox_changed)

        hbox.show_all()
        self.set_extra_widget(hbox)

    def get_filename_from_user (self):
        while True:
            filename = None
            if self.run() != int(Gtk.ResponseType.OK):
                break
            filename = self.get_filename()
            break

        return filename, self.ext


class RubberbandGTK3(backend_tools.RubberbandBase):
    def __init__(self, *args, **kwargs):
        backend_tools.RubberbandBase.__init__(self, *args, **kwargs)
        self.ctx = None

    def draw_rubberband(self, x0, y0, x1, y1):
        # 'adapted from http://aspn.activestate.com/ASPN/Cookbook/Python/
        # Recipe/189744'
        self.ctx = self.figure.canvas.get_property("window").cairo_create()

        # todo: instead of redrawing the entire figure, copy the part of
        # the figure that was covered by the previous rubberband rectangle
        self.figure.canvas.draw()

        height = self.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        rect = [int(val) for val in (min(x0, x1), min(y0, y1), w, h)]

        self.ctx.new_path()
        self.ctx.set_line_width(0.5)
        self.ctx.rectangle(rect[0], rect[1], rect[2], rect[3])
        self.ctx.set_source_rgb(0, 0, 0)
        self.ctx.stroke()


class ToolbarGTK3(ToolContainerBase, Gtk.Box):
    def __init__(self, toolmanager):
        ToolContainerBase.__init__(self, toolmanager)
        Gtk.Box.__init__(self)
        self.set_property("orientation", Gtk.Orientation.VERTICAL)

        self._toolarea = Gtk.Box()
        self._toolarea.set_property('orientation', Gtk.Orientation.HORIZONTAL)
        self.pack_start(self._toolarea, False, False, 0)
        self._toolarea.show_all()
        self._groups = {}
        self._toolitems = {}

    def add_toolitem(self, name, group, position, image_file, description,
                     toggle):
        if toggle:
            tbutton = Gtk.ToggleToolButton()
        else:
            tbutton = Gtk.ToolButton()
        tbutton.set_label(name)

        if image_file is not None:
            image = Gtk.Image()
            image.set_from_file(image_file)
            tbutton.set_icon_widget(image)

        if position is None:
            position = -1

        self._add_button(tbutton, group, position)
        signal = tbutton.connect('clicked', self._call_tool, name)
        tbutton.set_tooltip_text(description)
        tbutton.show_all()
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append((tbutton, signal))

    def _add_button(self, button, group, position):
        if group not in self._groups:
            if self._groups:
                self._add_separator()
            toolbar = Gtk.Toolbar()
            toolbar.set_style(Gtk.ToolbarStyle.ICONS)
            self._toolarea.pack_start(toolbar, False, False, 0)
            toolbar.show_all()
            self._groups[group] = toolbar
        self._groups[group].insert(button, position)

    def _call_tool(self, btn, name):
        self.trigger_tool(name)

    def toggle_toolitem(self, name, toggled):
        if name not in self._toolitems:
            return
        for toolitem, signal in self._toolitems[name]:
            toolitem.handler_block(signal)
            toolitem.set_active(toggled)
            toolitem.handler_unblock(signal)

    def remove_toolitem(self, name):
        if name not in self._toolitems:
            self.toolmanager.message_event('%s Not in toolbar' % name, self)
            return

        for group in self._groups:
            for toolitem, _signal in self._toolitems[name]:
                if toolitem in self._groups[group]:
                    self._groups[group].remove(toolitem)
        del self._toolitems[name]

    def _add_separator(self):
        sep = Gtk.Separator()
        sep.set_property("orientation", Gtk.Orientation.VERTICAL)
        self._toolarea.pack_start(sep, False, True, 0)
        sep.show_all()


class StatusbarGTK3(StatusbarBase, Gtk.Statusbar):
    def __init__(self, *args, **kwargs):
        StatusbarBase.__init__(self, *args, **kwargs)
        Gtk.Statusbar.__init__(self)
        self._context = self.get_context_id('message')

    def set_message(self, s):
        self.pop(self._context)
        self.push(self._context, s)


class SaveFigureGTK3(backend_tools.SaveFigureBase):

    def get_filechooser(self):
        fc = FileChooserDialog(
            title='Save the figure',
            parent=self.figure.canvas.manager.window,
            path=os.path.expanduser(rcParams.get('savefig.directory', '')),
            filetypes=self.figure.canvas.get_supported_filetypes(),
            default_filetype=self.figure.canvas.get_default_filetype())
        fc.set_current_name(self.figure.canvas.get_default_filename())
        return fc

    def trigger(self, *args, **kwargs):
        chooser = self.get_filechooser()
        fname, format_ = chooser.get_filename_from_user()
        chooser.destroy()
        if fname:
            startpath = os.path.expanduser(
                rcParams.get('savefig.directory', ''))
            if startpath == '':
                # explicitly missing key or empty str signals to use cwd
                rcParams['savefig.directory'] = startpath
            else:
                # save dir for next time
                rcParams['savefig.directory'] = os.path.dirname(
                    six.text_type(fname))
            try:
                self.figure.canvas.print_figure(fname, format=format_)
            except Exception as e:
                error_msg_gtk(str(e), parent=self)


class SetCursorGTK3(backend_tools.SetCursorBase):
    def set_cursor(self, cursor):
        self.figure.canvas.get_property("window").set_cursor(cursord[cursor])


class ConfigureSubplotsGTK3(backend_tools.ConfigureSubplotsBase, Gtk.Window):
    def __init__(self, *args, **kwargs):
        backend_tools.ConfigureSubplotsBase.__init__(self, *args, **kwargs)
        self.window = None

    def init_window(self):
        if self.window:
            return
        self.window = Gtk.Window(title="Subplot Configuration Tool")

        try:
            self.window.window.set_icon_from_file(window_icon)
        except (SystemExit, KeyboardInterrupt):
            # re-raise exit type Exceptions
            raise
        except:
            # we presumably already logged a message on the
            # failure of the main plot, don't keep reporting
            pass

        self.vbox = Gtk.Box()
        self.vbox.set_property("orientation", Gtk.Orientation.VERTICAL)
        self.window.add(self.vbox)
        self.vbox.show()
        self.window.connect('destroy', self.destroy)

        toolfig = Figure(figsize=(6, 3))
        canvas = self.figure.canvas.__class__(toolfig)

        toolfig.subplots_adjust(top=0.9)
        SubplotTool(self.figure, toolfig)

        w = int(toolfig.bbox.width)
        h = int(toolfig.bbox.height)

        self.window.set_default_size(w, h)

        canvas.show()
        self.vbox.pack_start(canvas, True, True, 0)
        self.window.show()

    def destroy(self, *args):
        self.window.destroy()
        self.window = None

    def _get_canvas(self, fig):
        return self.canvas.__class__(fig)

    def trigger(self, sender, event, data=None):
        self.init_window()
        self.window.present()


class DialogLineprops(object):
    """
    A GUI dialog for controlling lineprops
    """
    signals = (
        'on_combobox_lineprops_changed',
        'on_combobox_linestyle_changed',
        'on_combobox_marker_changed',
        'on_colorbutton_linestyle_color_set',
        'on_colorbutton_markerface_color_set',
        'on_dialog_lineprops_okbutton_clicked',
        'on_dialog_lineprops_cancelbutton_clicked',
        )

    linestyles = [ls for ls in lines.Line2D.lineStyles if ls.strip()]
    linestyled = dict([ (s,i) for i,s in enumerate(linestyles)])


    markers =  [m for m in lines.Line2D.markers if cbook.is_string_like(m)]

    markerd = dict([(s,i) for i,s in enumerate(markers)])

    def __init__(self, lines):
        import Gtk.glade

        datadir = matplotlib.get_data_path()
        gladefile = os.path.join(datadir, 'lineprops.glade')
        if not os.path.exists(gladefile):
            raise IOError('Could not find gladefile lineprops.glade in %s'%datadir)

        self._inited = False
        self._updateson = True # suppress updates when setting widgets manually
        self.wtree = Gtk.glade.XML(gladefile, 'dialog_lineprops')
        self.wtree.signal_autoconnect(dict([(s, getattr(self, s)) for s in self.signals]))

        self.dlg = self.wtree.get_widget('dialog_lineprops')

        self.lines = lines

        cbox = self.wtree.get_widget('combobox_lineprops')
        cbox.set_active(0)
        self.cbox_lineprops = cbox

        cbox = self.wtree.get_widget('combobox_linestyles')
        for ls in self.linestyles:
            cbox.append_text(ls)
        cbox.set_active(0)
        self.cbox_linestyles = cbox

        cbox = self.wtree.get_widget('combobox_markers')
        for m in self.markers:
            cbox.append_text(m)
        cbox.set_active(0)
        self.cbox_markers = cbox
        self._lastcnt = 0
        self._inited = True


    def show(self):
        'populate the combo box'
        self._updateson = False
        # flush the old
        cbox = self.cbox_lineprops
        for i in range(self._lastcnt-1,-1,-1):
            cbox.remove_text(i)

        # add the new
        for line in self.lines:
            cbox.append_text(line.get_label())
        cbox.set_active(0)

        self._updateson = True
        self._lastcnt = len(self.lines)
        self.dlg.show()

    def get_active_line(self):
        'get the active line'
        ind = self.cbox_lineprops.get_active()
        line = self.lines[ind]
        return line

    def get_active_linestyle(self):
        'get the active lineinestyle'
        ind = self.cbox_linestyles.get_active()
        ls = self.linestyles[ind]
        return ls

    def get_active_marker(self):
        'get the active lineinestyle'
        ind = self.cbox_markers.get_active()
        m = self.markers[ind]
        return m

    def _update(self):
        'update the active line props from the widgets'
        if not self._inited or not self._updateson: return
        line = self.get_active_line()
        ls = self.get_active_linestyle()
        marker = self.get_active_marker()
        line.set_linestyle(ls)
        line.set_marker(marker)

        button = self.wtree.get_widget('colorbutton_linestyle')
        color = button.get_color()
        r, g, b = [val/65535. for val in (color.red, color.green, color.blue)]
        line.set_color((r,g,b))

        button = self.wtree.get_widget('colorbutton_markerface')
        color = button.get_color()
        r, g, b = [val/65535. for val in (color.red, color.green, color.blue)]
        line.set_markerfacecolor((r,g,b))

        line.figure.canvas.draw()

    def on_combobox_lineprops_changed(self, item):
        'update the widgets from the active line'
        if not self._inited: return
        self._updateson = False
        line = self.get_active_line()

        ls = line.get_linestyle()
        if ls is None: ls = 'None'
        self.cbox_linestyles.set_active(self.linestyled[ls])

        marker = line.get_marker()
        if marker is None: marker = 'None'
        self.cbox_markers.set_active(self.markerd[marker])

        r,g,b = colorConverter.to_rgb(line.get_color())
        color = Gdk.Color(*[int(val*65535) for val in (r,g,b)])
        button = self.wtree.get_widget('colorbutton_linestyle')
        button.set_color(color)

        r,g,b = colorConverter.to_rgb(line.get_markerfacecolor())
        color = Gdk.Color(*[int(val*65535) for val in (r,g,b)])
        button = self.wtree.get_widget('colorbutton_markerface')
        button.set_color(color)
        self._updateson = True

    def on_combobox_linestyle_changed(self, item):
        self._update()

    def on_combobox_marker_changed(self, item):
        self._update()

    def on_colorbutton_linestyle_color_set(self, button):
        self._update()

    def on_colorbutton_markerface_color_set(self, button):
        'called colorbutton marker clicked'
        self._update()

    def on_dialog_lineprops_okbutton_clicked(self, button):
        self._update()
        self.dlg.hide()

    def on_dialog_lineprops_cancelbutton_clicked(self, button):
        self.dlg.hide()


# Define the file to use as the GTk icon
if sys.platform == 'win32':
    icon_filename = 'matplotlib.png'
else:
    icon_filename = 'matplotlib.svg'
window_icon = os.path.join(matplotlib.rcParams['datapath'], 'images', icon_filename)


def error_msg_gtk(msg, parent=None):
    if parent is not None: # find the toplevel Gtk.Window
        parent = parent.get_toplevel()
        if not parent.is_toplevel():
            parent = None

    if not is_string_like(msg):
        msg = ','.join(map(str,msg))

    dialog = Gtk.MessageDialog(
        parent         = parent,
        type           = Gtk.MessageType.ERROR,
        buttons        = Gtk.ButtonsType.OK,
        message_format = msg)
    dialog.run()
    dialog.destroy()


backend_tools.ToolSaveFigure = SaveFigureGTK3
backend_tools.ToolConfigureSubplots = ConfigureSubplotsGTK3
backend_tools.ToolSetCursor = SetCursorGTK3
backend_tools.ToolRubberband = RubberbandGTK3

Toolbar = ToolbarGTK3
FigureCanvas = FigureCanvasGTK3
FigureManager = FigureManagerGTK3
