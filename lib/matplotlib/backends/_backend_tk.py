from contextlib import contextmanager
import logging
import math
import os.path
import sys
import tkinter as tk
from tkinter.simpledialog import SimpleDialog
import tkinter.filedialog
import tkinter.messagebox

import numpy as np

import matplotlib as mpl
from matplotlib import backend_tools, cbook
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
    StatusbarBase, TimerBase, ToolContainerBase, cursors)
from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure
from matplotlib.widgets import SubplotTool
from . import _tkagg

try:
    from ._tkagg import Win32_GetForegroundWindow, Win32_SetForegroundWindow
except ImportError:
    @contextmanager
    def _restore_foreground_window_at_end():
        yield
else:
    @contextmanager
    def _restore_foreground_window_at_end():
        foreground = Win32_GetForegroundWindow()
        try:
            yield
        finally:
            if mpl.rcParams['tk.window_focus']:
                Win32_SetForegroundWindow(foreground)


_log = logging.getLogger(__name__)

backend_version = tk.TkVersion

cursord = {
    cursors.MOVE: "fleur",
    cursors.HAND: "hand2",
    cursors.POINTER: "arrow",
    cursors.SELECT_REGION: "tcross",
    cursors.WAIT: "watch",
    }


def blit(photoimage, aggimage, offsets, bbox=None):
    """
    Blit *aggimage* to *photoimage*.

    *offsets* is a tuple describing how to fill the ``offset`` field of the
    ``Tk_PhotoImageBlock`` struct: it should be (0, 1, 2, 3) for RGBA8888 data,
    (2, 1, 0, 3) for little-endian ARBG32 (i.e. GBRA8888) data and (1, 2, 3, 0)
    for big-endian ARGB32 (i.e. ARGB8888) data.

    If *bbox* is passed, it defines the region that gets blitted.
    """
    data = np.asarray(aggimage)
    height, width = data.shape[:2]
    dataptr = (height, width, data.ctypes.data)
    if bbox is not None:
        (x1, y1), (x2, y2) = bbox.__array__()
        x1 = max(math.floor(x1), 0)
        x2 = min(math.ceil(x2), width)
        y1 = max(math.floor(y1), 0)
        y2 = min(math.ceil(y2), height)
        bboxptr = (x1, x2, y1, y2)
    else:
        photoimage.blank()
        bboxptr = (0, width, 0, height)
    _tkagg.blit(
        photoimage.tk.interpaddr(), str(photoimage), dataptr, offsets, bboxptr)


class TimerTk(TimerBase):
    """Subclass of `backend_bases.TimerBase` using Tk timer events."""

    def __init__(self, parent, *args, **kwargs):
        self._timer = None
        TimerBase.__init__(self, *args, **kwargs)
        self.parent = parent

    def _timer_start(self):
        self._timer_stop()
        self._timer = self.parent.after(self._interval, self._on_timer)

    def _timer_stop(self):
        if self._timer is not None:
            self.parent.after_cancel(self._timer)
        self._timer = None

    def _on_timer(self):
        TimerBase._on_timer(self)
        # Tk after() is only a single shot, so we need to add code here to
        # reset the timer if we're not operating in single shot mode.  However,
        # if _timer is None, this means that _timer_stop has been called; so
        # don't recreate the timer in that case.
        if not self._single and self._timer:
            self._timer = self.parent.after(self._interval, self._on_timer)
        else:
            self._timer = None


class FigureCanvasTk(FigureCanvasBase):
    required_interactive_framework = "tk"

    keyvald = {65507: 'control',
               65505: 'shift',
               65513: 'alt',
               65515: 'super',
               65508: 'control',
               65506: 'shift',
               65514: 'alt',
               65361: 'left',
               65362: 'up',
               65363: 'right',
               65364: 'down',
               65307: 'escape',
               65470: 'f1',
               65471: 'f2',
               65472: 'f3',
               65473: 'f4',
               65474: 'f5',
               65475: 'f6',
               65476: 'f7',
               65477: 'f8',
               65478: 'f9',
               65479: 'f10',
               65480: 'f11',
               65481: 'f12',
               65300: 'scroll_lock',
               65299: 'break',
               65288: 'backspace',
               65293: 'enter',
               65379: 'insert',
               65535: 'delete',
               65360: 'home',
               65367: 'end',
               65365: 'pageup',
               65366: 'pagedown',
               65438: '0',
               65436: '1',
               65433: '2',
               65435: '3',
               65430: '4',
               65437: '5',
               65432: '6',
               65429: '7',
               65431: '8',
               65434: '9',
               65451: '+',
               65453: '-',
               65450: '*',
               65455: '/',
               65439: 'dec',
               65421: 'enter',
               }

    _keycode_lookup = {
                       262145: 'control',
                       524320: 'alt',
                       524352: 'alt',
                       1048584: 'super',
                       1048592: 'super',
                       131074: 'shift',
                       131076: 'shift',
                       }
    """_keycode_lookup is used for badly mapped (i.e. no event.key_sym set)
       keys on apple keyboards."""

    def __init__(self, figure, master=None, resize_callback=None):
        super(FigureCanvasTk, self).__init__(figure)
        self._idle = True
        self._idle_callback = None
        w, h = self.figure.bbox.size.astype(int)
        self._tkcanvas = tk.Canvas(
            master=master, background="white",
            width=w, height=h, borderwidth=0, highlightthickness=0)
        self._tkphoto = tk.PhotoImage(
            master=self._tkcanvas, width=w, height=h)
        self._tkcanvas.create_image(w//2, h//2, image=self._tkphoto)
        self._resize_callback = resize_callback
        self._tkcanvas.bind("<Configure>", self.resize)
        self._tkcanvas.bind("<Key>", self.key_press)
        self._tkcanvas.bind("<Motion>", self.motion_notify_event)
        self._tkcanvas.bind("<Enter>", self.enter_notify_event)
        self._tkcanvas.bind("<Leave>", self.leave_notify_event)
        self._tkcanvas.bind("<KeyRelease>", self.key_release)
        for name in ["<Button-1>", "<Button-2>", "<Button-3>"]:
            self._tkcanvas.bind(name, self.button_press_event)
        for name in [
                "<Double-Button-1>", "<Double-Button-2>", "<Double-Button-3>"]:
            self._tkcanvas.bind(name, self.button_dblclick_event)
        for name in [
                "<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>"]:
            self._tkcanvas.bind(name, self.button_release_event)

        # Mouse wheel on Linux generates button 4/5 events
        for name in "<Button-4>", "<Button-5>":
            self._tkcanvas.bind(name, self.scroll_event)
        # Mouse wheel for windows goes to the window with the focus.
        # Since the canvas won't usually have the focus, bind the
        # event to the window containing the canvas instead.
        # See http://wiki.tcl.tk/3893 (mousewheel) for details
        root = self._tkcanvas.winfo_toplevel()
        root.bind("<MouseWheel>", self.scroll_event_windows, "+")

        # Can't get destroy events by binding to _tkcanvas. Therefore, bind
        # to the window and filter.
        def filter_destroy(event):
            if event.widget is self._tkcanvas:
                self._master.update_idletasks()
                self.close_event()
        root.bind("<Destroy>", filter_destroy, "+")

        self._master = master
        self._tkcanvas.focus_set()

    def resize(self, event):
        width, height = event.width, event.height
        if self._resize_callback is not None:
            self._resize_callback(event)

        # compute desired figure size in inches
        dpival = self.figure.dpi
        winch = width / dpival
        hinch = height / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)

        self._tkcanvas.delete(self._tkphoto)
        self._tkphoto = tk.PhotoImage(
            master=self._tkcanvas, width=int(width), height=int(height))
        self._tkcanvas.create_image(
            int(width / 2), int(height / 2), image=self._tkphoto)
        self.resize_event()
        self.draw()

    def draw_idle(self):
        # docstring inherited
        if not self._idle:
            return

        self._idle = False

        def idle_draw(*args):
            try:
                self.draw()
            finally:
                self._idle = True

        self._idle_callback = self._tkcanvas.after_idle(idle_draw)

    def get_tk_widget(self):
        """
        Return the Tk widget used to implement FigureCanvasTkAgg.

        Although the initial implementation uses a Tk canvas,  this routine
        is intended to hide that fact.
        """
        return self._tkcanvas

    def motion_notify_event(self, event):
        x = event.x
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y
        FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=event)

    def enter_notify_event(self, event):
        x = event.x
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y
        FigureCanvasBase.enter_notify_event(self, guiEvent=event, xy=(x, y))

    def button_press_event(self, event, dblclick=False):
        x = event.x
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y
        num = getattr(event, 'num', None)

        if sys.platform == 'darwin':
            # 2 and 3 were reversed on the OSX platform I tested under tkagg.
            if num == 2:
                num = 3
            elif num == 3:
                num = 2

        FigureCanvasBase.button_press_event(
            self, x, y, num, dblclick=dblclick, guiEvent=event)

    def button_dblclick_event(self, event):
        self.button_press_event(event, dblclick=True)

    def button_release_event(self, event):
        x = event.x
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y

        num = getattr(event, 'num', None)

        if sys.platform == 'darwin':
            # 2 and 3 were reversed on the OSX platform I tested under tkagg.
            if num == 2:
                num = 3
            elif num == 3:
                num = 2

        FigureCanvasBase.button_release_event(self, x, y, num, guiEvent=event)

    def scroll_event(self, event):
        x = event.x
        y = self.figure.bbox.height - event.y
        num = getattr(event, 'num', None)
        step = 1 if num == 4 else -1 if num == 5 else 0
        FigureCanvasBase.scroll_event(self, x, y, step, guiEvent=event)

    def scroll_event_windows(self, event):
        """MouseWheel event processor"""
        # need to find the window that contains the mouse
        w = event.widget.winfo_containing(event.x_root, event.y_root)
        if w == self._tkcanvas:
            x = event.x_root - w.winfo_rootx()
            y = event.y_root - w.winfo_rooty()
            y = self.figure.bbox.height - y
            step = event.delta/120.
            FigureCanvasBase.scroll_event(self, x, y, step, guiEvent=event)

    def _get_key(self, event):
        val = event.keysym_num
        if val in self.keyvald:
            key = self.keyvald[val]
        elif (val == 0 and sys.platform == 'darwin'
              and event.keycode in self._keycode_lookup):
            key = self._keycode_lookup[event.keycode]
        elif val < 256:
            key = chr(val)
        else:
            key = None

        # add modifier keys to the key string. Bit details originate from
        # http://effbot.org/tkinterbook/tkinter-events-and-bindings.htm
        # BIT_SHIFT = 0x001; BIT_CAPSLOCK = 0x002; BIT_CONTROL = 0x004;
        # BIT_LEFT_ALT = 0x008; BIT_NUMLOCK = 0x010; BIT_RIGHT_ALT = 0x080;
        # BIT_MB_1 = 0x100; BIT_MB_2 = 0x200; BIT_MB_3 = 0x400;
        # In general, the modifier key is excluded from the modifier flag,
        # however this is not the case on "darwin", so double check that
        # we aren't adding repeat modifier flags to a modifier key.
        if sys.platform == 'win32':
            modifiers = [(17, 'alt', 'alt'),
                         (2, 'ctrl', 'control'),
                         ]
        elif sys.platform == 'darwin':
            modifiers = [(3, 'super', 'super'),
                         (4, 'alt', 'alt'),
                         (2, 'ctrl', 'control'),
                         ]
        else:
            modifiers = [(6, 'super', 'super'),
                         (3, 'alt', 'alt'),
                         (2, 'ctrl', 'control'),
                         ]

        if key is not None:
            # shift is not added to the keys as this is already accounted for
            for bitmask, prefix, key_name in modifiers:
                if event.state & (1 << bitmask) and key_name not in key:
                    key = '{0}+{1}'.format(prefix, key)

        return key

    def key_press(self, event):
        key = self._get_key(event)
        FigureCanvasBase.key_press_event(self, key, guiEvent=event)

    def key_release(self, event):
        key = self._get_key(event)
        FigureCanvasBase.key_release_event(self, key, guiEvent=event)

    def new_timer(self, *args, **kwargs):
        # docstring inherited
        return TimerTk(self._tkcanvas, *args, **kwargs)

    def flush_events(self):
        # docstring inherited
        self._master.update()


class FigureManagerTk(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : tk.Toolbar
        The tk.Toolbar
    window : tk.Window
        The tk.Window
    """

    def __init__(self, canvas, num, window):
        FigureManagerBase.__init__(self, canvas, num)
        self.window = window
        self.window.withdraw()
        self.set_window_title("Figure %d" % num)
        # packing toolbar first, because if space is getting low, last packed
        # widget is getting shrunk first (-> the canvas)
        self.toolbar = self._get_toolbar()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        if self.toolmanager:
            backend_tools.add_tools_to_manager(self.toolmanager)
            if self.toolbar:
                backend_tools.add_tools_to_container(self.toolbar)

        self._shown = False

    def _get_toolbar(self):
        if mpl.rcParams['toolbar'] == 'toolbar2':
            toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        elif mpl.rcParams['toolbar'] == 'toolmanager':
            toolbar = ToolbarTk(self.toolmanager, self.window)
        else:
            toolbar = None
        return toolbar

    def resize(self, width, height):
        max_size = 1_400_000  # the measured max on xorg 1.20.8 was 1_409_023

        if (width > max_size or height > max_size) and sys.platform == 'linux':
            raise ValueError(
                'You have requested to resize the '
                f'Tk window to ({width}, {height}), one of which '
                f'is bigger than {max_size}.  At larger sizes xorg will '
                'either exit with an error on newer versions (~1.20) or '
                'cause corruption on older version (~1.19).  We '
                'do not expect a window over a million pixel wide or tall '
                'to be intended behavior.')
        self.canvas._tkcanvas.configure(width=width, height=height)

    def show(self):
        with _restore_foreground_window_at_end():
            if not self._shown:
                def destroy(*args):
                    self.window = None
                    Gcf.destroy(self)
                self.canvas._tkcanvas.bind("<Destroy>", destroy)
                self.window.deiconify()
            else:
                self.canvas.draw_idle()
            if mpl.rcParams['figure.raise_window']:
                self.canvas.manager.window.attributes('-topmost', 1)
                self.canvas.manager.window.attributes('-topmost', 0)
            self._shown = True

    def destroy(self, *args):
        if self.window is not None:
            #self.toolbar.destroy()
            if self.canvas._idle_callback:
                self.canvas._tkcanvas.after_cancel(self.canvas._idle_callback)
            self.window.destroy()
        if Gcf.get_num_fig_managers() == 0:
            if self.window is not None:
                self.window.quit()
        self.window = None

    def get_window_title(self):
        return self.window.wm_title()

    def set_window_title(self, title):
        self.window.wm_title(title)

    def full_screen_toggle(self):
        is_fullscreen = bool(self.window.attributes('-fullscreen'))
        self.window.attributes('-fullscreen', not is_fullscreen)


class NavigationToolbar2Tk(NavigationToolbar2, tk.Frame):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The figure canvas on which to operate.
    win : tk.Window
        The tk.Window which owns this toolbar.
    pack_toolbar : bool, default: True
        If True, add the toolbar to the parent's pack manager's packing list
        during initialization with ``side='bottom'`` and ``fill='x'``.
        If you want to use the toolbar with a different layout manager, use
        ``pack_toolbar=False``.
    """
    def __init__(self, canvas, window, *, pack_toolbar=True):
        # Avoid using self.window (prefer self.canvas.get_tk_widget().master),
        # so that Tool implementations can reuse the methods.
        self.window = window

        tk.Frame.__init__(self, master=window, borderwidth=2,
                          width=int(canvas.figure.bbox.width), height=50)

        self._buttons = {}
        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                # Add a spacer; return value is unused.
                self._Spacer()
            else:
                self._buttons[text] = button = self._Button(
                    text,
                    str(cbook._get_data_path(f"images/{image_file}.gif")),
                    toggle=callback in ["zoom", "pan"],
                    command=getattr(self, callback),
                )
                if tooltip_text is not None:
                    ToolTip.createToolTip(button, tooltip_text)

        self.message = tk.StringVar(master=self)
        self._message_label = tk.Label(master=self, textvariable=self.message)
        self._message_label.pack(side=tk.RIGHT)

        NavigationToolbar2.__init__(self, canvas)
        if pack_toolbar:
            self.pack(side=tk.BOTTOM, fill=tk.X)

    def destroy(self, *args):
        del self.message
        tk.Frame.destroy(self, *args)

    def set_message(self, s):
        self.message.set(s)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        y0 = height - y0
        y1 = height - y1
        if hasattr(self, "lastrect"):
            self.canvas._tkcanvas.delete(self.lastrect)
        self.lastrect = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1)

    def release_zoom(self, event):
        super().release_zoom(event)
        if hasattr(self, "lastrect"):
            self.canvas._tkcanvas.delete(self.lastrect)
            del self.lastrect

    def set_cursor(self, cursor):
        window = self.canvas.get_tk_widget().master
        try:
            window.configure(cursor=cursord[cursor])
        except tkinter.TclError:
            pass
        else:
            window.update_idletasks()

    def _Button(self, text, image_file, toggle, command):
        image = (tk.PhotoImage(master=self, file=image_file)
                 if image_file is not None else None)
        if not toggle:
            b = tk.Button(master=self, text=text, image=image, command=command)
        else:
            # There is a bug in tkinter included in some python 3.6 versions
            # that without this variable, produces a "visual" toggling of
            # other near checkbuttons
            # https://bugs.python.org/issue29402
            # https://bugs.python.org/issue25684
            var = tk.IntVar()
            b = tk.Checkbutton(
                master=self, text=text, image=image, command=command,
                indicatoron=False, variable=var)
            b.var = var
        b._ntimage = image
        b.pack(side=tk.LEFT)
        return b

    def _Spacer(self):
        # Buttons are 30px high. Make this 26px tall +2px padding to center it.
        s = tk.Frame(
            master=self, height=26, relief=tk.RIDGE, pady=2, bg="DarkGray")
        s.pack(side=tk.LEFT, padx=5)
        return s

    def configure_subplots(self):
        toolfig = Figure(figsize=(6, 3))
        window = tk.Toplevel()
        canvas = type(self.canvas)(toolfig, master=window)
        toolfig.subplots_adjust(top=0.9)
        canvas.tool = SubplotTool(self.canvas.figure, toolfig)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        window.grab_set()

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes().copy()
        default_filetype = self.canvas.get_default_filetype()

        # Tk doesn't provide a way to choose a default filetype,
        # so we just have to put it first
        default_filetype_name = filetypes.pop(default_filetype)
        sorted_filetypes = ([(default_filetype, default_filetype_name)]
                            + sorted(filetypes.items()))
        tk_filetypes = [(name, '*.%s' % ext) for ext, name in sorted_filetypes]

        # adding a default extension seems to break the
        # asksaveasfilename dialog when you choose various save types
        # from the dropdown.  Passing in the empty string seems to
        # work - JDH!
        #defaultextension = self.canvas.get_default_filetype()
        defaultextension = ''
        initialdir = os.path.expanduser(mpl.rcParams['savefig.directory'])
        initialfile = self.canvas.get_default_filename()
        fname = tkinter.filedialog.asksaveasfilename(
            master=self.canvas.get_tk_widget().master,
            title='Save the figure',
            filetypes=tk_filetypes,
            defaultextension=defaultextension,
            initialdir=initialdir,
            initialfile=initialfile,
            )

        if fname in ["", ()]:
            return
        # Save dir for next time, unless empty str (i.e., use cwd).
        if initialdir != "":
            mpl.rcParams['savefig.directory'] = (
                os.path.dirname(str(fname)))
        try:
            # This method will handle the delegation to the correct type
            self.canvas.figure.savefig(fname)
        except Exception as e:
            tkinter.messagebox.showerror("Error saving file", str(e))

    def set_history_buttons(self):
        if self._nav_stack._pos > 0:
            self._buttons['Back']['state'] = tk.NORMAL
        else:
            self._buttons['Back']['state'] = tk.DISABLED
        if self._nav_stack._pos < len(self._nav_stack._elements) - 1:
            self._buttons['Forward']['state'] = tk.NORMAL
        else:
            self._buttons['Forward']['state'] = tk.DISABLED


class ToolTip:
    """
    Tooltip recipe from
    http://www.voidspace.org.uk/python/weblog/arch_d7_2006_07_01.shtml#e387
    """
    @staticmethod
    def createToolTip(widget, text):
        toolTip = ToolTip(widget)
        def enter(event):
            toolTip.showtip(text)
        def leave(event):
            toolTip.hidetip()
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        """Display text in tooltip window."""
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 27
        y = y + self.widget.winfo_rooty()
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        try:
            # For Mac OS
            tw.tk.call("::tk::unsupported::MacWindowStyle",
                       "style", tw._w,
                       "help", "noActivates")
        except tk.TclError:
            pass
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         relief=tk.SOLID, borderwidth=1)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class RubberbandTk(backend_tools.RubberbandBase):
    def draw_rubberband(self, x0, y0, x1, y1):
        height = self.figure.canvas.figure.bbox.height
        y0 = height - y0
        y1 = height - y1
        if hasattr(self, "lastrect"):
            self.figure.canvas._tkcanvas.delete(self.lastrect)
        self.lastrect = self.figure.canvas._tkcanvas.create_rectangle(
            x0, y0, x1, y1)

    def remove_rubberband(self):
        if hasattr(self, "lastrect"):
            self.figure.canvas._tkcanvas.delete(self.lastrect)
            del self.lastrect


class SetCursorTk(backend_tools.SetCursorBase):
    def set_cursor(self, cursor):
        NavigationToolbar2Tk.set_cursor(
            self._make_classic_style_pseudo_toolbar(), cursor)


class ToolbarTk(ToolContainerBase, tk.Frame):
    _icon_extension = '.gif'

    def __init__(self, toolmanager, window):
        ToolContainerBase.__init__(self, toolmanager)
        xmin, xmax = self.toolmanager.canvas.figure.bbox.intervalx
        height, width = 50, xmax - xmin
        tk.Frame.__init__(self, master=window,
                          width=int(width), height=int(height),
                          borderwidth=2)
        self._message = tk.StringVar(master=self)
        self._message_label = tk.Label(master=self, textvariable=self._message)
        self._message_label.pack(side=tk.RIGHT)
        self._toolitems = {}
        self.pack(side=tk.TOP, fill=tk.X)
        self._groups = {}

    def add_toolitem(
            self, name, group, position, image_file, description, toggle):
        frame = self._get_groupframe(group)
        button = NavigationToolbar2Tk._Button(self, name, image_file, toggle,
                                              lambda: self._button_click(name))
        if description is not None:
            ToolTip.createToolTip(button, description)
        self._toolitems.setdefault(name, [])
        self._toolitems[name].append(button)

    def _get_groupframe(self, group):
        if group not in self._groups:
            if self._groups:
                self._add_separator()
            frame = tk.Frame(master=self, borderwidth=0)
            frame.pack(side=tk.LEFT, fill=tk.Y)
            self._groups[group] = frame
        return self._groups[group]

    def _add_separator(self):
        separator = tk.Frame(master=self, bd=5, width=1, bg='black')
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=2)

    def _button_click(self, name):
        self.trigger_tool(name)

    def toggle_toolitem(self, name, toggled):
        if name not in self._toolitems:
            return
        for toolitem in self._toolitems[name]:
            if toggled:
                toolitem.select()
            else:
                toolitem.deselect()

    def remove_toolitem(self, name):
        for toolitem in self._toolitems[name]:
            toolitem.pack_forget()
        del self._toolitems[name]

    def set_message(self, s):
        self._message.set(s)


@cbook.deprecated("3.3")
class StatusbarTk(StatusbarBase, tk.Frame):
    def __init__(self, window, *args, **kwargs):
        StatusbarBase.__init__(self, *args, **kwargs)
        xmin, xmax = self.toolmanager.canvas.figure.bbox.intervalx
        height, width = 50, xmax - xmin
        tk.Frame.__init__(self, master=window,
                          width=int(width), height=int(height),
                          borderwidth=2)
        self._message = tk.StringVar(master=self)
        self._message_label = tk.Label(master=self, textvariable=self._message)
        self._message_label.pack(side=tk.RIGHT)
        self.pack(side=tk.TOP, fill=tk.X)

    def set_message(self, s):
        self._message.set(s)


class SaveFigureTk(backend_tools.SaveFigureBase):
    def trigger(self, *args):
        NavigationToolbar2Tk.save_figure(
            self._make_classic_style_pseudo_toolbar())


class ConfigureSubplotsTk(backend_tools.ConfigureSubplotsBase):
    def __init__(self, *args, **kwargs):
        backend_tools.ConfigureSubplotsBase.__init__(self, *args, **kwargs)
        self.window = None

    def trigger(self, *args):
        self.init_window()
        self.window.lift()

    def init_window(self):
        if self.window:
            return

        toolfig = Figure(figsize=(6, 3))
        self.window = tk.Tk()

        canvas = type(self.canvas)(toolfig, master=self.window)
        toolfig.subplots_adjust(top=0.9)
        SubplotTool(self.figure, toolfig)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.window.protocol("WM_DELETE_WINDOW", self.destroy)

    def destroy(self, *args, **kwargs):
        if self.window is not None:
            self.window.destroy()
            self.window = None


class HelpTk(backend_tools.ToolHelpBase):
    def trigger(self, *args):
        dialog = SimpleDialog(
            self.figure.canvas._tkcanvas, self._get_help_text(), ["OK"])
        dialog.done = lambda num: dialog.frame.master.withdraw()


backend_tools.ToolSaveFigure = SaveFigureTk
backend_tools.ToolConfigureSubplots = ConfigureSubplotsTk
backend_tools.ToolSetCursor = SetCursorTk
backend_tools.ToolRubberband = RubberbandTk
backend_tools.ToolHelp = HelpTk
backend_tools.ToolCopyToClipboard = backend_tools.ToolCopyToClipboardBase
Toolbar = ToolbarTk


@_Backend.export
class _BackendTk(_Backend):
    FigureManager = FigureManagerTk

    @classmethod
    def new_figure_manager_given_figure(cls, num, figure):
        """
        Create a new figure manager instance for the given figure.
        """
        with _restore_foreground_window_at_end():
            window = tk.Tk(className="matplotlib")
            window.withdraw()

            # Put a Matplotlib icon on the window rather than the default tk
            # icon.  Tkinter doesn't allow colour icons on linux systems, but
            # tk>=8.5 has a iconphoto command which we call directly.  See
            # http://mail.python.org/pipermail/tkinter-discuss/2006-November/000954.html
            icon_fname = str(cbook._get_data_path(
                'images/matplotlib_128.ppm'))
            icon_img = tk.PhotoImage(file=icon_fname, master=window)
            try:
                window.iconphoto(False, icon_img)
            except Exception as exc:
                # log the failure (due e.g. to Tk version), but carry on
                _log.info('Could not load matplotlib icon: %s', exc)

            canvas = cls.FigureCanvas(figure, master=window)
            manager = cls.FigureManager(canvas, num, window)
            if mpl.is_interactive():
                manager.show()
                canvas.draw_idle()
            return manager

    @staticmethod
    def trigger_manager_draw(manager):
        manager.show()

    @staticmethod
    def mainloop():
        managers = Gcf.get_all_fig_managers()
        if managers:
            managers[0].window.mainloop()
