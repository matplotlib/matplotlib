from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

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
     FigureManagerBase, FigureCanvasBase, NavigationToolbar2, cursors, TimerBase, \
     MultiFigureManagerBase, MultiFigureToolbarBase, ToolBase, ChildFigureManager
from matplotlib.backend_bases import ShowBase

from matplotlib.cbook import is_string_like, is_writable_file_like
from matplotlib.colors import colorConverter
from matplotlib.figure import Figure
from matplotlib.widgets import SubplotTool

from matplotlib import lines
from matplotlib import markers
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
        self._idle_event_id = GLib.idle_add(self.idle_event)

    def destroy(self):
        #Gtk.DrawingArea.destroy(self)
        self.close_event()
        GLib.source_remove(self._idle_event_id)
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
        def idle_draw(*args):
            try:
                self.draw()
            finally:
                self._idle_draw_id = 0
            return False
        if self._idle_draw_id == 0:
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
            Gtk.main_iteration(True)
        Gdk.flush()
        Gdk.threads_leave()

    def start_event_loop(self,timeout):
        FigureCanvasBase.start_event_loop_default(self,timeout)
    start_event_loop.__doc__=FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        FigureCanvasBase.stop_event_loop_default(self)
    stop_event_loop.__doc__=FigureCanvasBase.stop_event_loop_default.__doc__



class MultiFigureManagerGTK3(MultiFigureManagerBase):
    #to acces from figure instance
    #figure.canvas.manager.parent!!!!!

    def __init__(self, *args):
        self._children = []
        self._labels = {}
        self._w_min = 0
        self._h_min = 0

        if _debug: print('%s.%s' % (self.__class__.__name__, fn_name()))
        self.window = Gtk.Window()
        self.window.set_title("MultiFiguremanager")
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

        self.notebook = Gtk.Notebook()

        self.notebook.set_scrollable(True)

        self.notebook.connect('switch-page', self._on_switch_page)
        self.notebook.set_show_tabs(False)

        self.vbox.pack_start(self.notebook, True, True, 0)
        self.window.add(self.vbox)

        self.toolbar = self._get_toolbar()

        if self.toolbar is not None:
            self.toolbar.show_all()
            self.vbox.pack_end(self.toolbar, False, False, 0)

        def destroy_window(*args):
            nums = [manager.num for manager in self._children]
            for num in nums:
                Gcf.destroy(num)
        self.window.connect("destroy", destroy_window)
        self.window.connect("delete_event", destroy_window)

        self.vbox.show_all()

        if matplotlib.is_interactive():
            self.window.show()

    def _on_switch_page(self, notebook, pointer, num):
        canvas = self.notebook.get_nth_page(num)
        self.switch_child(canvas.manager)

    def destroy(self):
        if _debug: print('%s.%s' % (self.__class__.__name__, fn_name()))

        self.vbox.destroy()
        self.window.destroy()
        if self.toolbar:
            self.toolbar.destroy()

        if Gcf.get_num_fig_managers() == 0 and \
                not matplotlib.is_interactive() and \
                Gtk.main_level() >= 1:
            Gtk.main_quit()

    def remove_child(self, child):
        '''Remove the child from the multi figure, if it was the last one, destroy itself'''
        if _debug: print('%s.%s' % (self.__class__.__name__, fn_name()))
        if child not in self._children:
            raise AttributeError('This container does not control the given figure child')
        canvas = child.canvas
        id_ = self.notebook.page_num(canvas)
        if id_ > -1:
            del self._labels[child.num]
            self.notebook.remove_page(id_)
            self._children.remove(child)

        if self.notebook.get_n_pages() == 0:
            self.destroy()

        self._tabs_changed()

    def _tabs_changed(self):
        #Everytime we change the tabs configuration (add/remove)
        #we have to check to hide tabs and saveall button(if less than two)
        #we have to resize because the space used by tabs is not 0

        #hide tabs and saveall button if only one tab
        if self.notebook.get_n_pages() < 2:
            self.notebook.set_show_tabs(False)
            notebook_w = 0
            notebook_h = 0
        else:
            self.notebook.set_show_tabs(True)
            size_request = self.notebook.size_request()
            notebook_h = size_request.height
            notebook_w = size_request.width

        #if there are no children max will fail, so try/except
        try:
            canvas_w = max([int(manager.canvas.figure.bbox.width) for manager in self._children])
            canvas_h = max([int(manager.canvas.figure.bbox.height) for manager in self._children])
        except ValueError:
            canvas_w = 0
            canvas_h = 0

        if self.toolbar is not None:
            size_request = self.toolbar.size_request()
            toolbar_h = size_request.height
            toolbar_w = size_request.width
        else:
            toolbar_h = 0
            toolbar_w = 0

        w = max(canvas_w, notebook_w, toolbar_w)
        h = canvas_h + notebook_h + toolbar_h
        if w and h:
            self.window.resize(w, h)

    def set_child_title(self, child, title):
        self._labels[child.num].set_text(title)

    def get_child_title(self, child):
        return self._labels[child.num].get_text()

    def set_window_title(self, title):
        self.window.set_title(title)

    def get_window_title(self):
        return self.window.get_title()

    def _get_toolbar(self):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if rcParams['toolbar'] == 'toolbar2':
            toolbar = MultiFigureNavigationToolbar2GTK3(self.window)
        else:
            toolbar = None
        return toolbar

    def add_child(self, child):
        if _debug: print('%s.%s' % (self.__class__.__name__, fn_name()))
        if child in self._children:
            raise AttributeError('Impossible to add two times the same child')
        canvas = child.canvas
        num = child.num

        title = 'Fig %d' % num
        box = Gtk.Box()
        box.set_orientation(Gtk.Orientation.HORIZONTAL)
        box.set_spacing(5)

        label = Gtk.Label(title)
        self._labels[num] = label
        self._children.append(child)

        box.pack_start(label, True, True, 0)

        # close button
        button = Gtk.Button()
        button.set_tooltip_text('Close')
        button.set_relief(Gtk.ReliefStyle.NONE)
        button.set_focus_on_click(False)
        button.add(Gtk.Image.new_from_stock(Gtk.STOCK_CLOSE, Gtk.IconSize.MENU))
        box.pack_end(button, False, False, 0)

        def _remove(btn):
            Gcf.destroy(num)

        button.connect("clicked", _remove)

        # Detach button
        button = Gtk.Button()
        button.set_tooltip_text('Detach')
        button.set_relief(Gtk.ReliefStyle.NONE)
        button.set_focus_on_click(False)
        button.add(Gtk.Image.new_from_stock(Gtk.STOCK_JUMP_TO, Gtk.IconSize.MENU))
        box.pack_end(button, False, False, 0)

        def _detach(btn):
            child.detach()
        button.connect("clicked", _detach)

        box.show_all()
        canvas.show()

        self.notebook.append_page(canvas, box)
        self._tabs_changed()
        self.show_child(child)

    def show_child(self, child):
        if _debug: print('%s.%s' % (self.__class__.__name__, fn_name()))
        self.show()
        canvas = child.canvas
        id_ = self.notebook.page_num(canvas)
        self.notebook.set_current_page(id_)

    def show(self):
        if _debug: print('%s.%s' % (self.__class__.__name__, fn_name()))
#        self.window.show_all()
        self.window.show()


class FigureManagerGTK3(ChildFigureManager):
    parent_class = MultiFigureManagerGTK3


class MultiFigureNavigationToolbar2GTK3(Gtk.Box, MultiFigureToolbarBase):
    external_toolitems = ({'text': 'Subplots',
                           'tooltip_text': 'Configure subplots',
                           'image': 'subplots',
                           'callback': 'ConfigureSubplotsGTK3'},
                          {'callback': 'LinesProperties'},
                          {'callback': 'AxesProperties'}
                          )

    def __init__(self, window):
        self.win = window
        MultiFigureToolbarBase.__init__(self)

    def set_visible_tool(self, toolitem, visible):
        toolitem.set_visible(visible)

    def connect_toolitem(self, button, callback, *args, **kwargs):
        def mcallback(btn, cb, args, kwargs):
            getattr(self, cb)(*args, **kwargs)

        button.connect('clicked', mcallback, callback, args, kwargs)

    def add_toolitem(self, text='_', pos=-1,
                    tooltip_text='', image=None):
        timage = None
        if image:
            timage = Gtk.Image()

            if os.path.isfile(image):
                timage.set_from_file(image)
            else:
                basedir = os.path.join(rcParams['datapath'], 'images')
                fname = os.path.join(basedir, image + '.png')
                if os.path.isfile(fname):
                    timage.set_from_file(fname)
                else:
                    #TODO: Add the right mechanics to pass the image from string
#                    from gi.repository import GdkPixbuf
#                    pixbuf = GdkPixbuf.Pixbuf.new_from_inline(image, False)
                    timage = False

        tbutton = Gtk.ToolButton()

        tbutton.set_label(text)
        if timage:
            tbutton.set_icon_widget(timage)
        tbutton.set_tooltip_text(tooltip_text)
        self._toolbar.insert(tbutton, pos)
        tbutton.show()
        return tbutton

    def remove_tool(self, pos):
        widget = self._toolbar.get_nth_item(pos)
        if not widget:
            self.set_message('Impossible to remove tool %d' % pos)
            return
        self._toolbar.remove(widget)

    def move_tool(self, pos_ini, pos_fin):
        widget = self._toolbar.get_nth_item(pos_ini)
        if not widget:
            self.set_message('Impossible to remove tool %d' % pos_ini)
            return
        self._toolbar.remove(widget)
        self._toolbar.insert(widget, pos_fin)

    def add_separator(self, pos=-1):
        toolitem = Gtk.SeparatorToolItem()
        self._toolbar.insert(toolitem, pos)
        return toolitem

    def init_toolbar(self):
        Gtk.Box.__init__(self)
        self.set_property("orientation", Gtk.Orientation.VERTICAL)
        self._toolbar = Gtk.Toolbar()
        self._toolbar.set_style(Gtk.ToolbarStyle.ICONS)
        self.pack_start(self._toolbar, False, False, 0)

        self.show_all()

    def add_message(self):
        box = Gtk.Box()
        box.set_property("orientation", Gtk.Orientation.HORIZONTAL)
        sep = Gtk.Separator()
        sep.set_property("orientation", Gtk.Orientation.VERTICAL)
        box.pack_start(sep, False, True, 0)
        self.message = Gtk.Label()
        box.pack_end(self.message, False, False, 0)
        box.show_all()
        self.pack_end(box, False, False, 5)

        sep = Gtk.Separator()
        sep.set_property("orientation", Gtk.Orientation.HORIZONTAL)
        self.pack_end(sep, False, True, 0)
        self.show_all()

    def save_figure(self, *args):
        SaveFiguresDialogGTK3(self.get_figures()[0])
        
    def save_all_figures(self, *args):
        SaveFiguresDialogGTK3(*self.get_figures())

    def set_message(self, text):
        self.message.set_label(text)

    def set_navigation_cursor(self, navigation, cursor):
        navigation.canvas.get_property("window").set_cursor(cursord[cursor])


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


class ConfigureSubplotsGTK3(ToolBase):
    register = True
    
    def init_tool(self):
        self.window = Gtk.Window()
        
        try:
            self.window.set_icon_from_file(window_icon)
        except (SystemExit, KeyboardInterrupt):
            # re-raise exit type Exceptions
            raise
        except:
            # we presumably already logged a message on the
            # failure of the main plot, don't keep reporting
            pass
        self.window.set_title("Subplot Configuration Tool")
        self.vbox = Gtk.Box()
        self.vbox.set_property("orientation", Gtk.Orientation.VERTICAL)
        self.window.add(self.vbox)
        self.vbox.show()
        self.window.connect('destroy', self.destroy)

    def reset(self, *args):
        children = self.vbox.get_children()
        for child in children:
            self.vbox.remove(child)
        del children

    def set_figures(self, *figures):
        self.reset()
        figure = figures[0]
        toolfig = Figure(figsize=(6, 3))
        canvas = figure.canvas.__class__(toolfig)

        toolfig.subplots_adjust(top=0.9)
        SubplotTool(figure, toolfig)

        w = int(toolfig.bbox.width)
        h = int(toolfig.bbox.height)

        self.window.set_default_size(w, h)

        canvas.show()
        self.vbox.pack_start(canvas, True, True, 0)
        self.window.show()

    def show(self):
        self.window.present()


class SaveFiguresDialogGTK3(ToolBase):
    def set_figures(self, *figs):
        ref_figure = figs[0]
        self.figures = figs
        
        self.ref_canvas = ref_figure.canvas
        self.current_name = self.ref_canvas.get_default_filename()
        self.title = 'Save %d Figures' % len(figs)

        if len(figs) > 1:
            fname_end = '.' + self.ref_canvas.get_default_filetype()
            self.current_name = self.current_name[:-len(fname_end)]
            
        chooser = self._get_filechooser()
        fname, format_ = chooser.get_filename_from_user()
        chooser.destroy()
        if not fname:
            return
        self._save_figures(fname, format_)
       
    def _save_figures(self, basename, format_):
        figs = self.figures
        startpath = os.path.expanduser(rcParams.get('savefig.directory', ''))
        if startpath == '':
            # explicitly missing key or empty str signals to use cwd
            rcParams['savefig.directory'] = startpath
        else:
            # save dir for next time
            rcParams['savefig.directory'] = os.path.dirname(six.text_type(basename))

        # Get rid of the extension including the point
        extension = '.' + format_
        if basename.endswith(extension):
            basename = basename[:-len(extension)]

        # In the case of multiple figures, we have to insert a
        # "figure identifier" in the filename name
        n = len(figs)
        if n == 1:
            figure_identifier = ('',)
        else:
            figure_identifier = [str('_%.3d' % figs[i].canvas.manager.num) for i in range(n)]

        for i in range(n):
            canvas = figs[i].canvas
            fname = str('%s%s%s' % (basename, figure_identifier[i], extension))
            try:
                canvas.print_figure(fname, format=format_)
            except Exception as e:
                error_msg_gtk(str(e), parent=canvas.manager.window)

    def _get_filechooser(self):
        fc = FileChooserDialog(
            title=self.title,
            parent=self.ref_canvas.manager.window,
            path=os.path.expanduser(rcParams.get('savefig.directory', '')),
            filetypes=self.ref_canvas.get_supported_filetypes(),
            default_filetype=self.ref_canvas.get_default_filetype())
        fc.set_current_name(self.current_name)
        return fc
    
    
class LinesProperties(ToolBase):
    text = 'Lines'
    tooltip_text = 'Change line properties'
    register = True
    image = 'line_editor'
    
    _linestyle = [(k, ' '.join(v.split('_')[2:])) for k, v in lines.Line2D.lineStyles.items() if k.strip()]
    _drawstyle = [(k, ' '.join(v.split('_')[2:])) for k, v in lines.Line2D.drawStyles.items()]
    _marker = [(k, v) for k, v in markers.MarkerStyle.markers.items() if (k not in (None, '', ' '))]
    
    _pick_event = None
    
    def show(self):
        self.window.show_all()
        self.window.present()  
        
    def init_tool(self, pick=True):
        self._line = None
        self._pick = pick
        
        self.window = Gtk.Window(title='Line properties handler')
        
        try:
            self.window.set_icon_from_file(window_icon)
        except (SystemExit, KeyboardInterrupt):
            # re-raise exit type Exceptions
            raise
        except:
            pass
        
        self.window.connect('destroy', self.destroy)
        
        vbox = Gtk.Grid(orientation=Gtk.Orientation.VERTICAL,
                        column_spacing=5, row_spacing=10, border_width=10)
        
        self._lines_store = Gtk.ListStore(int, str)
        self.line_combo = Gtk.ComboBox.new_with_model(self._lines_store)
        renderer_text = Gtk.CellRendererText()
        self.line_combo.pack_start(renderer_text, True)
        self.line_combo.add_attribute(renderer_text, "text", 1)
        self.line_combo.connect("changed", self._on_line_changed)
        vbox.attach(self.line_combo, 0, 0, 2, 1)
        
        vbox.attach_next_to(Gtk.HSeparator(), self.line_combo, Gtk.PositionType.BOTTOM, 2, 1)
           
        self._visible = Gtk.CheckButton()
        self._visible.connect("toggled", self._on_visible_toggled)
        
        visible = Gtk.Label('Visible ')
        vbox.add(visible)
        vbox.attach_next_to(self._visible, visible, Gtk.PositionType.RIGHT, 1, 1)
        
        self.label = Gtk.Entry()
        self.label.connect('activate', self._on_label_activate)
        
        label = Gtk.Label('Label')
        vbox.add(label)
        vbox.attach_next_to(self.label, label, Gtk.PositionType.RIGHT, 1, 1)
   
        vbox.attach_next_to(Gtk.HSeparator(), label, Gtk.PositionType.BOTTOM, 2, 1)
        vbox.add(Gtk.Label('<b>Line</b>', use_markup=True))
   
        style = Gtk.Label('Style')
        vbox.add(style)

        drawstyle = Gtk.Label('Draw Style')
        vbox.add(drawstyle)
        
        linewidth = Gtk.Label('Width')
        vbox.add(linewidth)
        
        color = Gtk.Label('Color')
        vbox.add(color)

        vbox.attach_next_to(Gtk.HSeparator(), color, Gtk.PositionType.BOTTOM, 2, 1)
        vbox.add(Gtk.Label('<b>Marker</b>', use_markup=True))
        
        marker = Gtk.Label('Style')
        vbox.add(marker)

        markersize = Gtk.Label('Size')
        vbox.add(markersize)
        
        markerfacecolor = Gtk.Label('Face Color')
        vbox.add(markerfacecolor)
        
        markeredgecolor = Gtk.Label('Edge Color')
        vbox.add(markeredgecolor)
        
        for attr, pos in (('linewidth', linewidth), ('markersize', markersize)):
            button = Gtk.SpinButton(numeric=True, digits=1)
            adjustment = Gtk.Adjustment(0, 0, 100, 0.1, 10, 0)
            button.set_adjustment(adjustment)
            button.connect('value-changed', self._on_size_changed, attr)
            vbox.attach_next_to(button, pos, Gtk.PositionType.RIGHT, 1, 1)
            setattr(self, attr, button)
        
        for attr, pos in (('color', color),
                          ('markerfacecolor', markerfacecolor),
                          ('markeredgecolor', markeredgecolor)):
            button = Gtk.ColorButton()
            button.connect('color-set', self._on_color_set, attr)
            vbox.attach_next_to(button, pos, Gtk.PositionType.RIGHT, 1, 1)
            setattr(self, attr, button)
        
        for attr, pos in (('linestyle', style),
                          ('marker', marker),
                          ('drawstyle', drawstyle)):
            store = Gtk.ListStore(int, str)
            for i, v in enumerate(getattr(self, '_' + attr)):
                store.append([i, v[1]])
            combo = Gtk.ComboBox.new_with_model(store)
            renderer_text = Gtk.CellRendererText()
            combo.pack_start(renderer_text, True)
            combo.add_attribute(renderer_text, "text", 1) 
            combo.connect("changed", self._on_combo_changed, attr)
            vbox.attach_next_to(combo, pos, Gtk.PositionType.RIGHT, 1, 1)
            setattr(self, attr + '_combo', combo)
        
        self.window.add(vbox)
        self.window.show_all()
    
    def _on_combo_changed(self, combo, attr):
        if not self._line:
            return
        
        tree_iter = combo.get_active_iter()
        if tree_iter is None:
            return
        store = combo.get_model()
        id_ = store[tree_iter][0]
        getattr(self._line, 'set_' + attr)(getattr(self, '_' + attr)[id_][0])
        self._redraw()
        
    def _on_size_changed(self, button, attr):
        if not self._line:
            return
        
        getattr(self._line, 'set_' + attr)(getattr(self, attr).get_value())
        self._redraw()
        
    def _on_color_set(self, button, attr):
        if not self._line:
            return
        
        color = button.get_color()
        r, g, b = [val / 65535. for val in (color.red, color.green, color.blue)]
        getattr(self._line, 'set_' + attr)((r, g, b))
        self._redraw()
    
    def _on_label_activate(self, *args):
        if not self._line:
            return
        self._line.set_label(self.label.get_text())
        self._redraw()
        
    def _on_line_changed(self, combo):
        tree_iter = combo.get_active_iter()
        if tree_iter is None:
            self.line = None
            return
        
        id_ = self._lines_store[tree_iter][0]
        line = self.lines[id_]
        self._fill(line)
        
    def _on_visible_toggled(self, *args):
        if self._line:
            self._line.set_visible(self._visible.get_active())
            self._redraw()
    
    def set_figures(self, *figures):
        self._line = None
        self.figure = figures[0]
        self.lines = self._get_lines()
        
        self._lines_store.clear()
        
        for i, l in enumerate(self.lines):
            self._lines_store.append([i, l.get_label()])
            
        if self._pick:
            if self._pick_event:
                self.figure.canvas.mpl_disconnect(self._pick_event)
            self._pick_event = self.figure.canvas.mpl_connect('pick_event', self._on_pick)
        
    def _on_pick(self, event):
        artist = event.artist
        if not isinstance(artist, matplotlib.lines.Line2D):
            return
        
        try:
            i = self.lines.index(artist)
        except ValueError:
            return
        self.line_combo.set_active(i)
        
    def _get_lines(self):
        lines = set()
        for ax in self.figure.get_axes():
            for line in ax.get_lines():
                lines.add(line)

        #It is easier to find the lines if they are ordered by label
        lines = list(lines)
        labels = [line.get_label() for line in lines]
        a = [line for (_label, line) in sorted(zip(labels, lines))]
        return a

    def _fill(self, line=None):
        self._line = line
        if line is None:
            return
        
        self._visible.set_active(line.get_visible())
        self.label.set_text(line.get_label())
        
        for attr in ('linewidth', 'markersize'):
            getattr(self, attr).set_value(getattr(line, 'get_' + attr)())
            
        for attr in ('linestyle', 'marker', 'drawstyle'):
            v = getattr(line, 'get_' + attr)()
            for i, val in enumerate(getattr(self, '_' + attr)):
                if val[0] == v:
                    getattr(self, attr + '_combo').set_active(i)
                    break
        
        for attr in ('color', 'markerfacecolor', 'markeredgecolor'):
            r, g, b = colorConverter.to_rgb(getattr(line, 'get_' + attr)())
            color = Gdk.Color(*[int(val * 65535) for val in (r, g, b)])
            getattr(self, attr).set_color(color)
 
    def _redraw(self):
        if self._line:
            self._line.figure.canvas.draw()
    
    def destroy(self, *args):
        if self._pick_event:
            self.figure.canvas.mpl_disconnect(self._pick_event)
        
        self.unregister()
    
    
class AxesProperties(ToolBase):
    """Manage the axes properties
    
    Subclass of `ToolBase` for axes management
    """
    
    
    text = 'Axes'
    tooltip_text = 'Change axes properties'
    register = True
    image = 'axes_editor'

    _release_event = None
    
    def show(self):
        self.window.show_all()
        self.window.present()  
        
    def init_tool(self, release=True):
        self._line = None
        self._release = release
        
        self.window = Gtk.Window(title='Line properties handler')
        
        try:
            self.window.set_icon_from_file(window_icon)
        except (SystemExit, KeyboardInterrupt):
            # re-raise exit type Exceptions
            raise
        except:
            pass
        
        self.window.connect('destroy', self.destroy)
        
        vbox = Gtk.Grid(orientation=Gtk.Orientation.VERTICAL,
                        column_spacing=5, row_spacing=10, border_width=10)
        
        l = Gtk.Label('<b>Subplots</b>', use_markup=True)
        vbox.add(l)
        
        self._subplot_store = Gtk.ListStore(int, str)
        self._subplot_combo = Gtk.ComboBox.new_with_model(self._subplot_store)
        renderer_text = Gtk.CellRendererText()
        self._subplot_combo.pack_start(renderer_text, True)
        self._subplot_combo.add_attribute(renderer_text, "text", 1)
        self._subplot_combo.connect("changed", self._on_subplot_changed)
        vbox.attach_next_to(self._subplot_combo, l, Gtk.PositionType.BOTTOM, 2, 1)
        
        vbox.attach_next_to(Gtk.HSeparator(), self._subplot_combo, Gtk.PositionType.BOTTOM, 2, 1)
        l = Gtk.Label('<b>Axes</b>', use_markup=True)
        vbox.add(l)
#        vbox.attach_next_to(Gtk.HSeparator(), l, Gtk.PositionType.TOP, 2, 1)
        
        self._axes_store = Gtk.ListStore(int, str)
        self._axes_combo = Gtk.ComboBox.new_with_model(self._axes_store)
        renderer_text = Gtk.CellRendererText()
        self._axes_combo.pack_start(renderer_text, True)
        self._axes_combo.add_attribute(renderer_text, "text", 1)
        self._axes_combo.connect("changed", self._on_axes_changed)
        vbox.attach_next_to(self._axes_combo, l, Gtk.PositionType.BOTTOM, 2, 1)

        self._title = Gtk.Entry()
        self._title.connect('activate', self._on_title_activate)
        title = Gtk.Label('Title')
        vbox.add(title)
        vbox.attach_next_to(self._title, title, Gtk.PositionType.RIGHT, 1, 1)
        
        self._legend = Gtk.CheckButton()
        self._legend.connect("toggled", self._on_legend_toggled)
        
        legend = Gtk.Label('Legend')
        vbox.add(legend)
        vbox.attach_next_to(self._legend, legend, Gtk.PositionType.RIGHT, 1, 1) 
        
        vbox.attach_next_to(Gtk.HSeparator(), legend, Gtk.PositionType.BOTTOM, 2, 1)
        l = Gtk.Label('<b>X</b>', use_markup=True)
        vbox.add(l)
        
        xaxis = Gtk.Label('Visible')
        vbox.add(xaxis)   
        
        xlabel = Gtk.Label('Label')
        vbox.add(xlabel)    
        
        xmin = Gtk.Label('Min')
        vbox.add(xmin)    
        
        xmax = Gtk.Label('Max')
        vbox.add(xmax)    
        
        xscale = Gtk.Label('Scale')
        vbox.add(xscale) 
        
        xgrid = Gtk.Label('Grid')
        vbox.add(xgrid) 
        
        vbox.attach_next_to(Gtk.HSeparator(), xgrid, Gtk.PositionType.BOTTOM, 2, 1)
        l = Gtk.Label('<b>Y</b>', use_markup=True)
        vbox.add(l)
        
        yaxis = Gtk.Label('Visible')
        vbox.add(yaxis)  
        
        ylabel = Gtk.Label('Label')
        vbox.add(ylabel)
        
        ymin = Gtk.Label('Min')
        vbox.add(ymin)    
        
        ymax = Gtk.Label('Max')
        vbox.add(ymax)
        
        yscale = Gtk.Label('Scale')
        vbox.add(yscale) 
        
        ygrid = Gtk.Label('Grid')
        vbox.add(ygrid) 
        
        for attr, pos in (('xaxis', xaxis), ('yaxis', yaxis)):
            checkbox = Gtk.CheckButton()
            checkbox.connect("toggled", self._on_axis_visible, attr)
            vbox.attach_next_to(checkbox, pos, Gtk.PositionType.RIGHT, 1, 1)
            setattr(self, '_' + attr, checkbox)
        
        for attr, pos in (('xlabel', xlabel), ('ylabel', ylabel)):
            entry = Gtk.Entry()
            entry.connect('activate', self._on_label_activate, attr)
            vbox.attach_next_to(entry, pos, Gtk.PositionType.RIGHT, 1, 1)
            setattr(self, '_' + attr, entry)
                
        for attr, pos in (('x_min', xmin,), ('x_max', xmax), ('y_min', ymin), ('y_max', ymax)):
            entry = Gtk.Entry()
            entry.connect('activate', self._on_limit_activate, attr)
            vbox.attach_next_to(entry, pos, Gtk.PositionType.RIGHT, 1, 1)
            setattr(self, '_' + attr, entry)
        
        for attr, pos in (('xscale', xscale), ('yscale', yscale)):
            hbox = Gtk.Box(spacing=6)
            log_ = Gtk.RadioButton.new_with_label_from_widget(None, "Log")
            lin_ = Gtk.RadioButton.new_with_label_from_widget(log_, "Linear")
            log_.connect("toggled", self._on_scale_toggled, attr, "log")
            lin_.connect("toggled", self._on_scale_toggled, attr, "linear")

            hbox.pack_start(log_, False, False, 0)
            hbox.pack_start(lin_, False, False, 0)
            vbox.attach_next_to(hbox, pos, Gtk.PositionType.RIGHT, 1, 1)
            setattr(self, '_' + attr, {'log': log_, 'linear': lin_})
            
        for attr, pos in (('x', xgrid), ('y', ygrid)):
            combo = Gtk.ComboBoxText()
            for k in ('None', 'Major', 'Minor', 'Both'):
                combo.append_text(k)
            vbox.attach_next_to(combo, pos, Gtk.PositionType.RIGHT, 1, 1)
            combo.connect("changed", self._on_grid_changed, attr)
            setattr(self, '_' + attr + 'grid', combo)
                
        self.window.add(vbox)
        self.window.show_all()
    
    def _on_grid_changed(self, combo, attr):
        if self._ax is None:
            return
        
        marker = combo.get_active_text()
        self._ax.grid(False, axis=attr, which='both')

        if marker != 'None':
            self._ax.grid(False, axis=attr, which='both')
            self._ax.grid(True, axis=attr, which=marker)
        
        self._redraw()
    
    def _on_scale_toggled(self, button, attr, scale):
        if self._ax is None:
            return
        
        getattr(self._ax, 'set_' + attr)(scale)
        self._redraw()
    
    def _on_limit_activate(self, entry, attr):
        if self._ax is None:
            return
        
        direction = attr.split('_')[0]
        min_ = getattr(self, '_' + direction + '_min').get_text()
        max_ = getattr(self, '_' + direction + '_max').get_text()
        
        try:
            min_ = float(min_)
            max_ = float(max_)
        except:
            min_, max_ = getattr(self._ax, 'get_' + direction + 'lim')()
            getattr(self, '_' + direction + '_min').set_text(str(min_))
            getattr(self, '_' + direction + '_max').set_text(str(max_))
            return
        
        getattr(self._ax, 'set_' + direction + 'lim')(min_, max_)
        self._redraw()
    
    def _on_axis_visible(self, button, attr):
        if self._ax is None:
            return
        
        axis = getattr(self._ax, 'get_' + attr)()
        axis.set_visible(getattr(self, '_' + attr).get_active())
        self._redraw()
        
    def _on_label_activate(self, entry, attr):
        if self._ax is None:
            return
        
        getattr(self._ax, 'set_' + attr)(getattr(self, '_' + attr).get_text())
        self._redraw()
    
    def _on_legend_toggled(self, *args):
        if self._ax is None:
            return
        
        legend = self._ax.get_legend()
        if not legend:
            legend = self._ax.legend(loc='best', shadow=True)
            
        if legend:
            legend.set_visible(self._legend.get_active())
        #Put the legend always draggable, 
        #Maybe a bad idea, but fix the problem of possition
        try:
            legend.draggable(True)
        except:
            pass
        
        self._redraw()
        
    def _on_title_activate(self, *args):
        if self._ax is None:
            return
        self._ax.set_title(self._title.get_text())
        self._redraw()
        
    def _on_axes_changed(self, combo):
        self._ax = None
        if self._axes is None:
            return

        tree_iter = combo.get_active_iter()
        if tree_iter is None:
            return
        
        id_ = self._axes_store[tree_iter][0]
        ax = self._axes[id_]
        
        self._fill(ax)
        
    def _fill(self, ax=None):
        if ax is None:
            self._ax = None
            return
        
        self._title.set_text(ax.get_title())
        
        self._legend.set_active(bool(ax.get_legend()) and ax.get_legend().get_visible())
        
        for attr in ('xlabel', 'ylabel'):
            t = getattr(ax, 'get_' + attr)()
            getattr(self, '_' + attr).set_text(t)
            
        for attr in ('xaxis', 'yaxis'):
            axis = getattr(ax, 'get_' + attr)()
            getattr(self, '_' + attr).set_active(axis.get_visible())
            
        for attr in ('x', 'y'):    
            min_, max_ = getattr(ax, 'get_' + attr + 'lim')()
            getattr(self, '_' + attr + '_min').set_text(str(min_))
            getattr(self, '_' + attr + '_max').set_text(str(max_))      
            
        for attr in ('xscale', 'yscale'):
            scale = getattr(ax, 'get_' + attr)()
            getattr(self, '_' + attr)[scale].set_active(True) 
        
        for attr in ('x', 'y'):
            axis = getattr(ax, 'get_' + attr + 'axis')()
            if axis._gridOnMajor and not axis._gridOnMinor:
                gridon = 'Major'
            elif not axis._gridOnMajor and axis._gridOnMinor:
                gridon = 'Minor'
            elif axis._gridOnMajor and axis._gridOnMinor:
                gridon = 'Both'
            else:
                gridon = 'None'
        
            combo = getattr(self, '_' + attr + 'grid')
            model = combo.get_model()
            for i in range(len(model)):
                if model[i][0] == gridon:
                    combo.set_active(i)
                    break
            self._ax = ax
        
    def _on_subplot_changed(self, combo):
        self._axes = None
        self._ax = None
        self._axes_store.clear()
        
        tree_iter = combo.get_active_iter()
        if tree_iter is None:
            return
        
        id_ = self._subplot_store[tree_iter][0]
        self._axes = self._subplots[id_][1]
        
        for i in range(len(self._axes)):
            self._axes_store.append([i, 'Axes %d' % i])
        
        self._axes_combo.set_active(0)
        
    def set_figures(self, *figures):
        self._ax = None
        self.figure = figures[0]
        self._subplots = self._get_subplots()
        
        self._subplot_store.clear()
        
        for i, l in enumerate(self._subplots):
            self._subplot_store.append([i, str(l[0])])
        
        self._subplot_combo.set_active(0)
       
        if self._release:
            if self._release_event:
                self.figure.canvas.mpl_disconnect(self._release_event)
            self._release_event = self.figure.canvas.mpl_connect('button_release_event', self._on_release)
       
    def _on_release(self, event):
        try:
            ax = event.inaxes.axes
        except:
            return
        
        ax_subplot = [subplot[0] for subplot in self._subplots if ax in subplot[1]][0]
        current_subplot = [subplot[0] for subplot in self._subplots if self._ax in subplot[1]][0]
        if ax_subplot == current_subplot:
            return
        
        for i, subplot in enumerate(self._subplots):
            if subplot[0] == ax_subplot:
                self._subplot_combo.set_active(i)
                break
       
    def _get_subplots(self): 
        axes = {}
        alone = []
        rem = []
        for ax in self.figure.get_axes():
            try:
                axes.setdefault(ax.get_geometry(), []).append(ax)
            except AttributeError:
                alone.append(ax)
        
        #try to find if share something with one of the axes with geometry
        for ax in alone:
            for ax2 in [i for sl in axes.values() for i in sl]:
                if ((ax in ax2.get_shared_x_axes().get_siblings(ax2)) or 
                    (ax in ax2.get_shared_y_axes().get_siblings(ax2))):
                    axes[ax2.get_geometry()].append(ax)
                    rem.append(ax)
        
        for ax in rem:
            alone.remove(ax)
            
        for i, ax in enumerate(alone):
            axes[i] = [ax, ]

        return [(k, axes[k]) for k in sorted(axes.keys())]
#        return axes
        
    def destroy(self, *args):
        if self._release_event:
            self.figure.canvas.mpl_disconnect(self._release_event)
        
        self.unregister()

    def _redraw(self):
        if self._ax:
            self._ax.figure.canvas.draw()

    
    
    

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


FigureCanvas = FigureCanvasGTK3
FigureManager = FigureManagerGTK3
