# Todd Miller   jmiller@stsci.edu

from __future__ import division

import os, sys, math
import os.path

import Tkinter as Tk, FileDialog

# Paint image to Tk photo blitter extension
import matplotlib.backends.tkagg as tkagg

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.backends.windowing as windowing

import matplotlib
from matplotlib.cbook import is_string_like
from matplotlib.backend_bases import RendererBase, GraphicsContextBase
from matplotlib.backend_bases import FigureManagerBase, FigureCanvasBase
from matplotlib.backend_bases import NavigationToolbar2, cursors, TimerBase
from matplotlib.backend_bases import ShowBase
from matplotlib._pylab_helpers import Gcf

from matplotlib.figure import Figure

from matplotlib.widgets import SubplotTool

import matplotlib.cbook as cbook

rcParams = matplotlib.rcParams
verbose = matplotlib.verbose


backend_version = Tk.TkVersion

# the true dots per inch on the screen; should be display dependent
# see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi
PIXELS_PER_INCH = 75

cursord = {
    cursors.MOVE: "fleur",
    cursors.HAND: "hand2",
    cursors.POINTER: "arrow",
    cursors.SELECT_REGION: "tcross",
    }


def round(x):
    return int(math.floor(x+0.5))

def raise_msg_to_str(msg):
    """msg is a return arg from a raise.  Join with new lines"""
    if not is_string_like(msg):
        msg = '\n'.join(map(str, msg))
    return msg

def error_msg_tkpaint(msg, parent=None):
    import tkMessageBox
    tkMessageBox.showerror("matplotlib", msg)

def draw_if_interactive():
    if matplotlib.is_interactive():
        figManager =  Gcf.get_active()
        if figManager is not None:
            figManager.show()

class Show(ShowBase):
    def mainloop(self):
        Tk.mainloop()

show = Show()

def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    _focus = windowing.FocusManager()
    FigureClass = kwargs.pop('FigureClass', Figure)
    figure = FigureClass(*args, **kwargs)
    window = Tk.Tk()
    canvas = FigureCanvasTkAgg(figure, master=window)
    figManager = FigureManagerTkAgg(canvas, num, window)
    if matplotlib.is_interactive():
        figManager.show()
    return figManager


class TimerTk(TimerBase):
    '''
    Subclass of :class:`backend_bases.TimerBase` that uses Tk's timer events.

    Attributes:
    * interval: The time between timer events in milliseconds. Default
        is 1000 ms.
    * single_shot: Boolean flag indicating whether this timer should
        operate as single shot (run once and then stop). Defaults to False.
    * callbacks: Stores list of (func, args) tuples that will be called
        upon timer events. This list can be manipulated directly, or the
        functions add_callback and remove_callback can be used.
    '''
    def __init__(self, parent, *args, **kwargs):
        TimerBase.__init__(self, *args, **kwargs)
        self.parent = parent
        self._timer = None

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
        # reset the timer if we're not operating in single shot mode.
        if not self._single and len(self.callbacks) > 0:
            self._timer = self.parent.after(self._interval, self._on_timer)
        else:
            self._timer = None


class FigureCanvasTkAgg(FigureCanvasAgg):
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

    def __init__(self, figure, master=None, resize_callback=None):
        FigureCanvasAgg.__init__(self, figure)
        self._idle = True
        self._idle_callback = None
        t1,t2,w,h = self.figure.bbox.bounds
        w, h = int(w), int(h)
        self._tkcanvas = Tk.Canvas(
            master=master, width=w, height=h, borderwidth=4)
        self._tkphoto = Tk.PhotoImage(
            master=self._tkcanvas, width=w, height=h)
        self._tkcanvas.create_image(w//2, h//2, image=self._tkphoto)
        self._resize_callback = resize_callback
        self._tkcanvas.bind("<Configure>", self.resize)
        self._tkcanvas.bind("<Key>", self.key_press)
        self._tkcanvas.bind("<Motion>", self.motion_notify_event)
        self._tkcanvas.bind("<KeyRelease>", self.key_release)
        for name in "<Button-1>", "<Button-2>", "<Button-3>":
            self._tkcanvas.bind(name, self.button_press_event)
        for name in "<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>":
            self._tkcanvas.bind(name, self.button_release_event)

        # Mouse wheel on Linux generates button 4/5 events
        for name in "<Button-4>", "<Button-5>":
            self._tkcanvas.bind(name, self.scroll_event)
        # Mouse wheel for windows goes to the window with the focus.
        # Since the canvas won't usually have the focus, bind the
        # event to the window containing the canvas instead.
        # See http://wiki.tcl.tk/3893 (mousewheel) for details
        root = self._tkcanvas.winfo_toplevel()
        root.bind("<MouseWheel>", self.scroll_event_windows)

        # Can't get destroy events by binding to _tkcanvas. Therefore, bind
        # to the window and filter.
        def filter_destroy(evt):
            if evt.widget is self._tkcanvas:
                self.close_event()
        root.bind("<Destroy>", filter_destroy)

        self._master = master
        self._tkcanvas.focus_set()

    def resize(self, event):
        width, height = event.width, event.height
        if self._resize_callback is not None:
            self._resize_callback(event)

        # compute desired figure size in inches
        dpival = self.figure.dpi
        winch = width/dpival
        hinch = height/dpival
        self.figure.set_size_inches(winch, hinch)


        self._tkcanvas.delete(self._tkphoto)
        self._tkphoto = Tk.PhotoImage(
            master=self._tkcanvas, width=int(width), height=int(height))
        self._tkcanvas.create_image(int(width/2),int(height/2),image=self._tkphoto)
        self.resize_event()
        self.show()

    def draw(self):
        FigureCanvasAgg.draw(self)
        tkagg.blit(self._tkphoto, self.renderer._renderer, colormode=2)
        self._master.update_idletasks()

    def blit(self, bbox=None):
        tkagg.blit(self._tkphoto, self.renderer._renderer, bbox=bbox, colormode=2)
        self._master.update_idletasks()

    show = draw

    def draw_idle(self):
        'update drawing area only if idle'
        d = self._idle
        self._idle = False
        def idle_draw(*args):
            self.draw()
            self._idle = True

        if d:
            self._idle_callback = self._tkcanvas.after_idle(idle_draw)

    def get_tk_widget(self):
        """returns the Tk widget used to implement FigureCanvasTkAgg.
        Although the initial implementation uses a Tk canvas,  this routine
        is intended to hide that fact.
        """
        return self._tkcanvas

    def motion_notify_event(self, event):
        x = event.x
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y
        FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=event)


    def button_press_event(self, event):
        x = event.x
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y
        num = getattr(event, 'num', None)

        if sys.platform=='darwin':
            # 2 and 3 were reversed on the OSX platform I
            # tested under tkagg
            if   num==2: num=3
            elif num==3: num=2

        FigureCanvasBase.button_press_event(self, x, y, num, guiEvent=event)

    def button_release_event(self, event):
        x = event.x
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y

        num = getattr(event, 'num', None)

        if sys.platform=='darwin':
            # 2 and 3 were reversed on the OSX platform I
            # tested under tkagg
            if   num==2: num=3
            elif num==3: num=2

        FigureCanvasBase.button_release_event(self, x, y, num, guiEvent=event)

    def scroll_event(self, event):
        x = event.x
        y = self.figure.bbox.height - event.y
        num = getattr(event, 'num', None)
        if   num==4: step = +1
        elif num==5: step = -1
        else:        step =  0

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
        elif val<256:
            key = chr(val)
        else:
            key = None
        return key


    def key_press(self, event):
        key = self._get_key(event)
        FigureCanvasBase.key_press_event(self, key, guiEvent=event)

    def key_release(self, event):
        key = self._get_key(event)
        FigureCanvasBase.key_release_event(self, key, guiEvent=event)

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
        return TimerTk(self._tkcanvas, *args, **kwargs)

    def flush_events(self):
        self._master.update()

    def start_event_loop(self,timeout):
        FigureCanvasBase.start_event_loop_default(self,timeout)
    start_event_loop.__doc__=FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        FigureCanvasBase.stop_event_loop_default(self)
    stop_event_loop.__doc__=FigureCanvasBase.stop_event_loop_default.__doc__

class FigureManagerTkAgg(FigureManagerBase):
    """
    Public attributes

    canvas      : The FigureCanvas instance
    num         : The Figure number
    toolbar     : The tk.Toolbar
    window      : The tk.Window
    """
    def __init__(self, canvas, num, window):
        FigureManagerBase.__init__(self, canvas, num)
        self.window = window
        self.window.withdraw()
        self.window.wm_title("Figure %d" % num)
        self.canvas = canvas
        self._num =  num
        t1,t2,w,h = canvas.figure.bbox.bounds
        w, h = int(w), int(h)
        self.window.minsize(int(w*3/4),int(h*3/4))
        if matplotlib.rcParams['toolbar']=='classic':
            self.toolbar = NavigationToolbar( canvas, self.window )
        elif matplotlib.rcParams['toolbar']=='toolbar2':
            self.toolbar = NavigationToolbar2TkAgg( canvas, self.window )
        else:
            self.toolbar = None
        if self.toolbar is not None:
            self.toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self._shown = False

        def notify_axes_change(fig):
            'this will be called whenever the current axes is changed'
            if self.toolbar != None: self.toolbar.update()
        self.canvas.figure.add_axobserver(notify_axes_change)



        # attach a show method to the figure for pylab ease of use
        self.canvas.figure.show = lambda *args: self.show()


    def resize(self, width, height=None):
        # before 09-12-22, the resize method takes a single *event*
        # parameter. On the other hand, the resize method of other
        # FigureManager class takes *width* and *height* parameter,
        # which is used to change the size of the window. For the
        # Figure.set_size_inches with forward=True work with Tk
        # backend, I changed the function signature but tried to keep
        # it backward compatible. -JJL

        # when a single parameter is given, consider it as a event
        if height is None:
            width = width.width
        else:
            self.canvas._tkcanvas.master.geometry("%dx%d" % (width, height))

        self.toolbar.configure(width=width)


    def show(self):
        """
        this function doesn't segfault but causes the
        PyEval_RestoreThread: NULL state bug on win32
        """
        _focus = windowing.FocusManager()
        if not self._shown:
            def destroy(*args):
                self.window = None
                Gcf.destroy(self._num)
            self.canvas._tkcanvas.bind("<Destroy>", destroy)
            self.window.deiconify()
            # anim.py requires this
            self.window.update()
        else:
            self.canvas.draw_idle()
        self._shown = True


    def destroy(self, *args):
        if self.window is not None:
            #self.toolbar.destroy()
            if self.canvas._idle_callback:
                self.canvas._tkcanvas.after_cancel(self.canvas._idle_callback)
            self.window.destroy()
        if Gcf.get_num_fig_managers()==0:
            if self.window is not None:
                self.window.quit()
        self.window = None

    def set_window_title(self, title):
        self.window.wm_title(title)

class AxisMenu:
    def __init__(self, master, naxes):
        self._master = master
        self._naxes = naxes
        self._mbar = Tk.Frame(master=master, relief=Tk.RAISED, borderwidth=2)
        self._mbar.pack(side=Tk.LEFT)
        self._mbutton = Tk.Menubutton(
            master=self._mbar, text="Axes", underline=0)
        self._mbutton.pack(side=Tk.LEFT, padx="2m")
        self._mbutton.menu = Tk.Menu(self._mbutton)
        self._mbutton.menu.add_command(
            label="Select All", command=self.select_all)
        self._mbutton.menu.add_command(
            label="Invert All", command=self.invert_all)
        self._axis_var = []
        self._checkbutton = []
        for i in range(naxes):
            self._axis_var.append(Tk.IntVar())
            self._axis_var[i].set(1)
            self._checkbutton.append(self._mbutton.menu.add_checkbutton(
                label = "Axis %d" % (i+1),
                variable=self._axis_var[i],
                command=self.set_active))
            self._mbutton.menu.invoke(self._mbutton.menu.index("Select All"))
        self._mbutton['menu'] = self._mbutton.menu
        self._mbar.tk_menuBar(self._mbutton)
        self.set_active()

    def adjust(self, naxes):
        if self._naxes < naxes:
            for i in range(self._naxes, naxes):
                self._axis_var.append(Tk.IntVar())
                self._axis_var[i].set(1)
                self._checkbutton.append( self._mbutton.menu.add_checkbutton(
                    label = "Axis %d" % (i+1),
                    variable=self._axis_var[i],
                    command=self.set_active))
        elif self._naxes > naxes:
            for i in range(self._naxes-1, naxes-1, -1):
                del self._axis_var[i]
                self._mbutton.menu.forget(self._checkbutton[i])
                del self._checkbutton[i]
        self._naxes = naxes
        self.set_active()

    def get_indices(self):
        a = [i for i in range(len(self._axis_var)) if self._axis_var[i].get()]
        return a

    def set_active(self):
        self._master.set_active(self.get_indices())

    def invert_all(self):
        for a in self._axis_var:
            a.set(not a.get())
        self.set_active()

    def select_all(self):
        for a in self._axis_var:
            a.set(1)
        self.set_active()

class NavigationToolbar(Tk.Frame):
    """
    Public attriubutes

      canvas   - the FigureCanvas  (gtk.DrawingArea)
      win   - the gtk.Window

    """
    def _Button(self, text, file, command):
        file = os.path.join(rcParams['datapath'], 'images', file)
        im = Tk.PhotoImage(master=self, file=file)
        b = Tk.Button(
            master=self, text=text, padx=2, pady=2, image=im, command=command)
        b._ntimage = im
        b.pack(side=Tk.LEFT)
        return b

    def __init__(self, canvas, window):
        self.canvas = canvas
        self.window = window

        xmin, xmax = canvas.figure.bbox.intervalx
        height, width = 50, xmax-xmin
        Tk.Frame.__init__(self, master=self.window,
                          width=int(width), height=int(height),
                          borderwidth=2)

        self.update()  # Make axes menu

        self.bLeft = self._Button(
            text="Left", file="stock_left.ppm",
            command=lambda x=-1: self.panx(x))

        self.bRight = self._Button(
            text="Right", file="stock_right.ppm",
            command=lambda x=1: self.panx(x))

        self.bZoomInX = self._Button(
            text="ZoomInX",file="stock_zoom-in.ppm",
            command=lambda x=1: self.zoomx(x))

        self.bZoomOutX = self._Button(
            text="ZoomOutX", file="stock_zoom-out.ppm",
            command=lambda x=-1: self.zoomx(x))

        self.bUp = self._Button(
            text="Up", file="stock_up.ppm",
            command=lambda y=1: self.pany(y))

        self.bDown = self._Button(
            text="Down", file="stock_down.ppm",
            command=lambda y=-1: self.pany(y))

        self.bZoomInY = self._Button(
            text="ZoomInY", file="stock_zoom-in.ppm",
            command=lambda y=1: self.zoomy(y))

        self.bZoomOutY = self._Button(
            text="ZoomOutY",file="stock_zoom-out.ppm",
            command=lambda y=-1: self.zoomy(y))

        self.bSave = self._Button(
            text="Save", file="stock_save_as.ppm",
            command=self.save_figure)

        self.pack(side=Tk.BOTTOM, fill=Tk.X)


    def set_active(self, ind):
        self._ind = ind
        self._active = [ self._axes[i] for i in self._ind ]

    def panx(self, direction):
        for a in self._active:
            a.xaxis.pan(direction)
        self.canvas.draw()

    def pany(self, direction):
        for a in self._active:
            a.yaxis.pan(direction)
        self.canvas.draw()

    def zoomx(self, direction):

        for a in self._active:
            a.xaxis.zoom(direction)
        self.canvas.draw()

    def zoomy(self, direction):

        for a in self._active:
            a.yaxis.zoom(direction)
        self.canvas.draw()

    def save_figure(self, *args):
        fs = FileDialog.SaveFileDialog(master=self.window,
                                       title='Save the figure')
        try:
            self.lastDir
        except AttributeError:
            self.lastDir = os.curdir

        fname = fs.go(dir_or_file=self.lastDir) # , pattern="*.png")
        if fname is None: # Cancel
            return

        self.lastDir = os.path.dirname(fname)
        try:
            self.canvas.print_figure(fname)
        except IOError, msg:
            err = '\n'.join(map(str, msg))
            msg = 'Failed to save %s: Error msg was\n\n%s' % (
                fname, err)
            error_msg_tkpaint(msg)

    def update(self):
        _focus = windowing.FocusManager()
        self._axes = self.canvas.figure.axes
        naxes = len(self._axes)
        if not hasattr(self, "omenu"):
            self.set_active(range(naxes))
            self.omenu = AxisMenu(master=self, naxes=naxes)
        else:
            self.omenu.adjust(naxes)

class NavigationToolbar2TkAgg(NavigationToolbar2, Tk.Frame):
    """
    Public attriubutes

      canvas   - the FigureCanvas  (gtk.DrawingArea)
      win   - the gtk.Window
    """
    def __init__(self, canvas, window):
        self.canvas = canvas
        self.window = window
        self._idle = True
        #Tk.Frame.__init__(self, master=self.canvas._tkcanvas)
        NavigationToolbar2.__init__(self, canvas)

    def destroy(self, *args):
        del self.message
        Tk.Frame.destroy(self, *args)

    def set_message(self, s):
        self.message.set(s)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        y0 =  height-y0
        y1 =  height-y1
        try: self.lastrect
        except AttributeError: pass
        else: self.canvas._tkcanvas.delete(self.lastrect)
        self.lastrect = self.canvas._tkcanvas.create_rectangle(x0, y0, x1, y1)

        #self.canvas.draw()

    def release(self, event):
        try: self.lastrect
        except AttributeError: pass
        else:
            self.canvas._tkcanvas.delete(self.lastrect)
            del self.lastrect

    def set_cursor(self, cursor):
        self.window.configure(cursor=cursord[cursor])

    def _Button(self, text, file, command):
        file = os.path.join(rcParams['datapath'], 'images', file)
        im = Tk.PhotoImage(master=self, file=file)
        b = Tk.Button(
            master=self, text=text, padx=2, pady=2, image=im, command=command)
        b._ntimage = im
        b.pack(side=Tk.LEFT)
        return b

    def _init_toolbar(self):
        xmin, xmax = self.canvas.figure.bbox.intervalx
        height, width = 50, xmax-xmin
        Tk.Frame.__init__(self, master=self.window,
                          width=int(width), height=int(height),
                          borderwidth=2)

        self.update()  # Make axes menu

        self.bHome = self._Button( text="Home", file="home.ppm",
                                   command=self.home)

        self.bBack = self._Button( text="Back", file="back.ppm",
                                   command = self.back)

        self.bForward = self._Button(text="Forward", file="forward.ppm",
                                     command = self.forward)

        self.bPan = self._Button( text="Pan", file="move.ppm",
                                  command = self.pan)

        self.bZoom = self._Button( text="Zoom",
                                   file="zoom_to_rect.ppm",
                                   command = self.zoom)

        self.bsubplot = self._Button( text="Configure Subplots", file="subplots.ppm",
                                   command = self.configure_subplots)

        self.bsave = self._Button( text="Save", file="filesave.ppm",
                                   command = self.save_figure)
        self.message = Tk.StringVar(master=self)
        self._message_label = Tk.Label(master=self, textvariable=self.message)
        self._message_label.pack(side=Tk.RIGHT)
        self.pack(side=Tk.BOTTOM, fill=Tk.X)


    def configure_subplots(self):
        toolfig = Figure(figsize=(6,3))
        window = Tk.Tk()
        canvas = FigureCanvasTkAgg(toolfig, master=window)
        toolfig.subplots_adjust(top=0.9)
        tool =  SubplotTool(self.canvas.figure, toolfig)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    def save_figure(self, *args):
        from tkFileDialog import asksaveasfilename
        from tkMessageBox import showerror
        filetypes = self.canvas.get_supported_filetypes().copy()
        default_filetype = self.canvas.get_default_filetype()

        # Tk doesn't provide a way to choose a default filetype,
        # so we just have to put it first
        default_filetype_name = filetypes[default_filetype]
        del filetypes[default_filetype]

        sorted_filetypes = filetypes.items()
        sorted_filetypes.sort()
        sorted_filetypes.insert(0, (default_filetype, default_filetype_name))

        tk_filetypes = [
            (name, '*.%s' % ext) for (ext, name) in sorted_filetypes]

        # adding a default extension seems to break the
        # asksaveasfilename dialog when you choose various save types
        # from the dropdown.  Passing in the empty string seems to
        # work - JDH
        #defaultextension = self.canvas.get_default_filetype()
        defaultextension = ''
        fname = asksaveasfilename(
            master=self.window,
            title='Save the figure',
            filetypes = tk_filetypes,
            defaultextension = defaultextension
            )

        if fname == "" or fname == ():
            return
        else:
            try:
                # This method will handle the delegation to the correct type
                self.canvas.print_figure(fname)
            except Exception, e:
                showerror("Error saving file", str(e))

    def set_active(self, ind):
        self._ind = ind
        self._active = [ self._axes[i] for i in self._ind ]

    def update(self):
        _focus = windowing.FocusManager()
        self._axes = self.canvas.figure.axes
        naxes = len(self._axes)
        #if not hasattr(self, "omenu"):
        #    self.set_active(range(naxes))
        #    self.omenu = AxisMenu(master=self, naxes=naxes)
        #else:
        #    self.omenu.adjust(naxes)
        NavigationToolbar2.update(self)

    def dynamic_update(self):
        'update drawing area only if idle'
        # legacy method; new method is canvas.draw_idle
        self.canvas.draw_idle()


FigureManager = FigureManagerTkAgg
