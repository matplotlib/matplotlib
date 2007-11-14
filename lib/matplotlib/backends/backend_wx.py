from __future__ import division
"""

 backend_wx.py

 A wxPython backend for matplotlib, based (very heavily) on
 backend_template.py and backend_gtk.py

 Author: Jeremy O'Donoghue (jeremy@o-donoghue.com)

 Derived from original copyright work by John Hunter
 (jdhunter@ace.bsd.uchicago.edu)

 Copyright (C) Jeremy O'Donoghue & John Hunter, 2003-4

 License: This work is licensed under a PSF compatible license. A copy
 should be included with this source code.

"""
"""
KNOWN BUGS -
 - Mousewheel (on Windows) only works after menu button has been pressed
   at least once
 - Mousewheel on Linux (wxGTK linked against GTK 1.2) does not work at all
 - Vertical text renders horizontally if you use a non TrueType font
   on Windows. This is a known wxPython issue. Work-around is to ensure
   that you use a TrueType font.
 - Pcolor demo puts chart slightly outside bounding box (approx 1-2 pixels
   to the bottom left)
 - Outputting to bitmap more than 300dpi results in some text being incorrectly
   scaled. Seems to be a wxPython bug on Windows or font point sizes > 60, as
   font size is correctly calculated.
 - Performance poorer than for previous direct rendering version
 - TIFF output not supported on wxGTK. This is a wxGTK issue
 - Text is not anti-aliased on wxGTK. This is probably a platform
   configuration issue.
 - If a second call is made to show(), no figure is generated (#866965)

Not implemented:
 - Printing

Fixed this release:
 - Bug #866967: Interactive operation issues fixed [JDH]
 - Bug #866969: Dynamic update does not function with backend_wx [JOD]

Examples which work on this release:
 ---------------------------------------------------------------
                         | Windows 2000    |  Linux            |
                         | wxPython 2.3.3  |  wxPython 2.4.2.4 |
 --------------------------------------------------------------|
 - alignment_test.py     |     TBE         |     OK            |
 - arctest.py            |     TBE         |     (3)           |
 - axes_demo.py          |     OK          |     OK            |
 - axes_props.py         |     OK          |     OK            |
 - bar_stacked.py        |     TBE         |     OK            |
 - barchart_demo.py      |     OK          |     OK            |
 - color_demo.py         |     OK          |     OK            |
 - csd_demo.py           |     OK          |     OK            |
 - dynamic_demo.py       |     N/A         |     N/A           |
 - dynamic_demo_wx.py    |     TBE         |     OK            |
 - embedding_in_gtk.py   |     N/A         |     N/A           |
 - embedding_in_wx.py    |     OK          |     OK            |
 - errorbar_demo.py      |     OK          |     OK            |
 - figtext.py            |     OK          |     OK            |
 - histogram_demo.py     |     OK          |     OK            |
 - interactive.py        |     N/A (2)     |     N/A (2)       |
 - interactive2.py       |     N/A (2)     |     N/A (2)       |
 - legend_demo.py        |     OK          |     OK            |
 - legend_demo2.py       |     OK          |     OK            |
 - line_styles.py        |     OK          |     OK            |
 - log_demo.py           |     OK          |     OK            |
 - logo.py               |     OK          |     OK            |
 - mpl_with_glade.py     |     N/A (2)     |     N/A (2)       |
 - mri_demo.py           |     OK          |     OK            |
 - mri_demo_with_eeg.py  |     OK          |     OK            |
 - multiple_figs_demo.py |     OK          |     OK            |
 - pcolor_demo.py        |     OK          |     OK            |
 - psd_demo.py           |     OK          |     OK            |
 - scatter_demo.py       |     OK          |     OK            |
 - scatter_demo2.py      |     OK          |     OK            |
 - simple_plot.py        |     OK          |     OK            |
 - stock_demo.py         |     OK          |     OK            |
 - subplot_demo.py       |     OK          |     OK            |
 - system_monitor.py     |     N/A (2)     |     N/A (2)       |
 - text_handles.py       |     OK          |     OK            |
 - text_themes.py        |     OK          |     OK            |
 - vline_demo.py         |     OK          |     OK            |
 ---------------------------------------------------------------

 (2) - Script uses GTK-specific features - cannot not run,
       but wxPython equivalent should be written.
 (3) - Clipping seems to be broken.
"""

cvs_id = '$Id$'

import sys, os, os.path, math, StringIO

# Debugging settings here...
# Debug level set here. If the debug level is less than 5, information
# messages (progressively more info for lower value) are printed. In addition,
# traceback is performed, and pdb activated, for all uncaught exceptions in
# this case
_DEBUG = 5
if _DEBUG < 5:
    import traceback, pdb
_DEBUG_lvls = {1 : 'Low ', 2 : 'Med ', 3 : 'High', 4 : 'Error' }


try:
    import wx
    backend_version = wx.VERSION_STRING
except:
    print >>sys.stderr, "Matplotlib backend_wx requires wxPython be installed"
    sys.exit()

#!!! this is the call that is causing the exception swallowing !!!
#wx.InitAllImageHandlers()

def DEBUG_MSG(string, lvl=3, o=None):
    if lvl >= _DEBUG:
        cls = o.__class__
        # Jeremy, often times the commented line won't print but the
        # one below does.  I think WX is redefining stderr, damned
        # beast
        #print >>sys.stderr, "%s- %s in %s" % (_DEBUG_lvls[lvl], string, cls)
        print "%s- %s in %s" % (_DEBUG_lvls[lvl], string, cls)

def debug_on_error(type, value, tb):
    """Code due to Thomas Heller - published in Python Cookbook (O'Reilley)"""
    traceback.print_exc(type, value, tb)
    print
    pdb.pm()  # jdh uncomment

class fake_stderr:
    """Wx does strange things with stderr, as it makes the assumption that there
    is probably no console. This redirects stderr to the console, since we know
    that there is one!"""
    def write(self, msg):
        print "Stderr: %s\n\r" % msg

#if _DEBUG < 5:
#    sys.excepthook = debug_on_error
#    WxLogger =wx.LogStderr()
#    sys.stderr = fake_stderr


import matplotlib
from matplotlib import verbose
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureCanvasBase, FigureManagerBase, NavigationToolbar2, \
     cursors
from matplotlib._pylab_helpers import Gcf
from matplotlib.artist import Artist
from matplotlib.cbook import exception_to_str, is_string_like, is_writable_file_like
from matplotlib.figure import Figure
from matplotlib.text import _process_text_args, Text
from matplotlib.widgets import SubplotTool
from matplotlib import rcParams

##import wx
##backend_version = wx.VERSION_STRING


# the True dots per inch on the screen; should be display dependent
# see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi
PIXELS_PER_INCH = 75

def error_msg_wx(msg, parent=None):
    """
    Signal an error condition -- in a GUI, popup a error dialog
    """
    dialog =wx.MessageDialog(parent  = parent,
                             message = msg,
                             caption = 'Matplotlib backend_wx error',
                             style=wx.OK | wx.CENTRE)
    dialog.ShowModal()
    dialog.Destroy()
    return None

def raise_msg_to_str(msg):
    """msg is a return arg from a raise.  Join with new lines"""
    if not is_string_like(msg):
        msg = '\n'.join(map(str, msg))
    return msg

class RendererWx(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles. It acts as the
    'renderer' instance used by many classes in the hierarchy.
    """
    #In wxPython, drawing is performed on a wxDC instance, which will
    #generally be mapped to the client aread of the window displaying
    #the plot. Under wxPython, the wxDC instance has a wx.Pen which
    #describes the colour and weight of any lines drawn, and a wxBrush
    #which describes the fill colour of any closed polygon.


    fontweights = {
        100          : wx.LIGHT,
        200          : wx.LIGHT,
        300          : wx.LIGHT,
        400          : wx.NORMAL,
        500          : wx.NORMAL,
        600          : wx.NORMAL,
        700          : wx.BOLD,
        800          : wx.BOLD,
        900          : wx.BOLD,
        'ultralight' : wx.LIGHT,
        'light'      : wx.LIGHT,
        'normal'     : wx.NORMAL,
        'medium'     : wx.NORMAL,
        'semibold'   : wx.NORMAL,
        'bold'       : wx.BOLD,
        'heavy'      : wx.BOLD,
        'ultrabold'  : wx.BOLD,
        'black'      : wx.BOLD
        }
    fontangles = {
        'italic'  : wx.ITALIC,
        'normal'  : wx.NORMAL,
        'oblique' : wx.SLANT }

    # wxPython allows for portable font styles, choosing them appropriately
    # for the target platform. Map some standard font names to the portable
    # styles
    # QUESTION: Is it be wise to agree standard fontnames across all backends?
    fontnames = { 'Sans'       : wx.SWISS,
                  'Roman'      : wx.ROMAN,
                  'Script'     : wx.SCRIPT,
                  'Decorative' : wx.DECORATIVE,
                  'Modern'     : wx.MODERN,
                  'Courier'    : wx.MODERN,
                  'courier'    : wx.MODERN }


    def __init__(self, bitmap, dpi):
        """
        Initialise a wxWindows renderer instance.
        """
        DEBUG_MSG("__init__()", 1, self)
        self.width  = bitmap.GetWidth()
        self.height = bitmap.GetHeight()
        self.bitmap = bitmap
        self.fontd = {}
        self.dpi = dpi
        self.gc = None

    def flipy(self):
        return True

    def offset_text_height(self):
        return True

    def get_text_width_height_descent(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with FontPropertry prop
        """
        #return 1, 1
        if ismath: s = self.strip_math(s)

        if self.gc is None: gc = self.new_gc()
        font = self.get_wx_font(s, prop)
        self.gc.SetFont(font)
        w, h, descent, leading = self.gc.GetFullTextExtent(s)

        return w, h, descent

    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return self.width, self.height


    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2, rotation):
        """
        Draw an arc centered at x,y with width and height and angles
        from 0.0 to 360.0.
        If rgbFace is present, fill the figure in this colour, otherwise
        it is not filled.
        """
        gc.select()
        assert gc.Ok(), "wxMemoryDC not OK to use"
        # wxPython requires upper left corner of bounding rectange for ellipse
        # Theoretically you don't need the int() below, but it seems to make
        # rounding of arc centre point more accurate in screen co-ordinates
        ulX = x - int(width/2)
        ulY = self.height - int(y + (height/2))
        if rgbFace is not None:
            r,g,b = self._to_wx_rgb(rgbFace)
            new_brush =wx.Brush(wx.Colour(r,g,b), wx.SOLID)
            gc.SetBrush(new_brush)
        else:
            gc.SetBrush(wx.TRANSPARENT_BRUSH)
        gc.DrawEllipticArc(int(ulX), int(ulY), int(width)+1, int(height)+1,
                           int(angle1), int(angle2))
        gc.unselect()

    def draw_line(self, gc, x1, y1, x2, y2):
        """
        Draw a single line from x1,y1 to x2,y2
        """
        DEBUG_MSG("draw_line()", 1, self)
        gc.select()
        gc.DrawLine(int(x1), self.height - int(y1),
                    int(x2), self.height - int(y2))
        gc.unselect()

    def draw_lines(self, gc, x, y):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        gc.select()
        assert gc.Ok(), "wxMemoryDC not OK to use"
        assert len(x) == len(y), "draw_lines() x and y must be of equal length"
        gc.DrawLines([wx.Point(int(x[i]), self.height - int(y[i])) for i in range(len(x))])
        gc.unselect()

    def draw_polygon(self, gc, rgbFace, points):
        """
        Draw a polygon.  points is a len vertices tuple, each element
        giving the x,y coords a vertex
        """
        gc.select()
        assert gc.Ok(), "wxMemoryDC not OK to use"
        points = [(int(x), self.height - int(y)) for x,y in points]
        if rgbFace is not None:
            r,g,b = self._to_wx_rgb(rgbFace)
            new_brush =wx.Brush(wx.Colour(r,g,b), wx.SOLID)
            gc.SetBrush(new_brush)
        else:
            gc.SetBrush(wx.TRANSPARENT_BRUSH)
        gc.DrawPolygon(points)
        gc.unselect()

    def draw_rectangle(self, gc, rgbFace, x, y, width, height):
        """
        Draw a rectangle at lower left x,y with width and height
        If filled=True, fill the rectangle with the gc foreground
        gc is a GraphicsContext instance
        """
        # wxPython uses rectangle from TOP left!
        gc.select()
        assert gc.Ok(), "wxMemoryDC not OK to use"
        if rgbFace is not None:
            r,g,b = self._to_wx_rgb(rgbFace)
            new_brush =wx.Brush(wx.Colour(r,g,b), wx.SOLID)
            gc.SetBrush(new_brush)
        else:
            gc.SetBrush(wx.TRANSPARENT_BRUSH)
        gc.DrawRectangle(int(x), self.height - int(height + y),
                                 int(math.ceil(width)), int(math.ceil(height)))
        gc.unselect()

    def draw_point(self, gc, x, y):
        """
        Draw a single point at x,y
        """
        gc.select()
        assert gc.Ok(), "wxMemoryDC not OK to use"
        gc.DrawPoint(int(x), self.height - int(y))
        gc.unselect()

    def draw_text(self, gc, x, y, s, prop, angle, ismath):
        """
        Render the matplotlib.text.Text instance
        None)
        """
        if ismath: s = self.strip_math(s)
        DEBUG_MSG("draw_text()", 1, self)
        gc.select()

        font = self.get_wx_font(s, prop)
        gc.SetFont(font)
        assert gc.Ok(), "wxMemoryDC not OK to use"

        w, h, d = self.get_text_width_height_descent(s, prop, ismath)
        x = int(x)
        y = int(y-h)

        if angle!=0:
            # Correct for the fact that text if rotated around the upper-left corner,
            # rather than the lower-left corner as we expect.
            rads = angle / 180.0 * math.pi
            xo = h * math.sin(rads)
            yo = h * math.cos(rads)
            try: gc.DrawRotatedText(s, x - xo, y - yo, angle)
            except:
                verbose.print_error(exception_to_str('WX rotated text failed'))
        else:
            gc.DrawText(s, x, y)
        gc.unselect()

    def new_gc(self):
        """
        Return an instance of a GraphicsContextWx, and sets the current gc copy
        """
        DEBUG_MSG('new_gc()', 2, self)
        self.gc = GraphicsContextWx(self.bitmap, self)
        self.gc.select()
        assert self.gc.Ok(), "wxMemoryDC not OK to use"
        self.gc.unselect()
        return self.gc

    def get_gc(self):
        """
        Fetch the locally cached gc.
        """
        # This is a dirty hack to allow anything with access to a renderer to
        # access the current graphics context
        assert self.gc != None, "gc must be defined"
        return self.gc


    def get_wx_font(self, s, prop):
        """
        Return a wx font.  Cache instances in a font dictionary for
        efficiency
        """
        DEBUG_MSG("get_wx_font()", 1, self)


        key = hash(prop)
        fontprop = prop
        fontname = fontprop.get_name()

        font = self.fontd.get(key)
        if font is not None:
            return font

        # Allow use of platform independent and dependent font names
        wxFontname = self.fontnames.get(fontname, wx.ROMAN)
        wxFacename = '' # Empty => wxPython chooses based on wx_fontname

        # Font colour is determined by the active wx.Pen
        # TODO: It may be wise to cache font information
        size = self.points_to_pixels(fontprop.get_size_in_points())


        font =wx.Font(int(size+0.5),             # Size
                      wxFontname,                # 'Generic' name
                      self.fontangles[fontprop.get_style()],   # Angle
                      self.fontweights[fontprop.get_weight()], # Weight
                      False,                     # Underline
                      wxFacename)                # Platform font name

        # cache the font and gc and return it
        self.fontd[key] = font

        return font

    def _to_wx_rgb(self, rgb):
        """Takes a colour value and returns a tuple (r,g,b) suitable
        for instantiating a wx.Colour."""
        r, g, b = rgb
        return (int(r * 255), int(g * 255), int(b * 255))



    def points_to_pixels(self, points):
        """
        convert point measures to pixes using dpi and the pixels per
        inch of the display
        """
        return points*(PIXELS_PER_INCH/72.0*self.dpi.get()/72.0)

class GraphicsContextWx(GraphicsContextBase, wx.MemoryDC):
    """
    The graphics context provides the color, line styles, etc...

    In wxPython this is done by wrapping a wxDC object and forwarding the
    appropriate calls to it. Notice also that colour and line styles are
    mapped on the wx.Pen() member of the wxDC. This means that we have some
    rudimentary pen management here.

    The base GraphicsContext stores colors as a RGB tuple on the unit
    interval, eg, (0.5, 0.0, 1.0).  wxPython uses an int interval, but
    since wxPython colour management is rather simple, I have not chosen
    to implement a separate colour manager class.
    """
    _capd = { 'butt':       wx.CAP_BUTT,
              'projecting': wx.CAP_PROJECTING,
              'round':      wx.CAP_ROUND }

    _joind = { 'bevel':     wx.JOIN_BEVEL,
               'miter':     wx.JOIN_MITER,
               'round':     wx.JOIN_ROUND }

    _dashd_wx = { 'solid':     wx.SOLID,
                  'dashed':    wx.SHORT_DASH,
                  'dashdot':   wx.DOT_DASH,
                  'dotted':    wx.DOT }
    _lastWxDC = None

    def __init__(self, bitmap, renderer):
        GraphicsContextBase.__init__(self)
        wx.MemoryDC.__init__(self)
        #assert self.Ok(), "wxMemoryDC not OK to use"
        DEBUG_MSG("__init__()", 1, self)
        # Make sure (belt and braces!) that existing wxDC is not selected to
        # to a bitmap.
        if GraphicsContextWx._lastWxDC != None:

            GraphicsContextWx._lastWxDC.SelectObject(wx.NullBitmap)

        self.SelectObject(bitmap)
        self.bitmap = bitmap
        self.SetPen(wx.Pen('BLACK', 1, wx.SOLID))
        self._style=wx.SOLID
        self.renderer = renderer
        GraphicsContextWx._lastWxDC = self

    def select(self):
        """
        Select the current bitmap into this wxDC instance
        """

        if sys.platform=='win32':
            self.SelectObject(self.bitmap)
            self.IsSelected = True

    def unselect(self):
        """
        Select a Null bitmasp into this wxDC instance
        """
        if sys.platform=='win32':
            self.SelectObject(wx.NullBitmap)
            self.IsSelected = False

    def set_clip_rectangle(self, rect):
        """
        Destroys previous clipping region and defines a new one.
        """
        DEBUG_MSG("set_clip_rectangle()", 1, self)
        self.select()
        l,b,w,h = rect
        # this appears to be version dependent'
        if hasattr(self, 'SetClippingRegionXY'):
            clipfunc = getattr(self, 'SetClippingRegionXY')
        else:
            clipfunc = getattr(self, 'SetClippingRegion')

        clipfunc(int(l), self.renderer.height - int(b+h),
                 int(w), int(h))
        self.unselect()

    def set_foreground(self, fg, isRGB=None):
        """
        Set the foreground color.  fg can be a matlab format string, a
        html hex color string, an rgb unit tuple, or a float between 0
        and 1.  In the latter case, grayscale is used.
        """
        # Implementation note: wxPython has a separate concept of pen and
        # brush - the brush fills any outline trace left by the pen.
        # Here we set both to the same colour - if a figure is not to be
        # filled, the renderer will set the brush to be transparent
        # Same goes for text foreground...
        DEBUG_MSG("set_foreground()", 1, self)
        self.select()
        GraphicsContextBase.set_foreground(self, fg, isRGB)

        pen = self.GetPen()
        pen.SetColour(self.get_wxcolour())
        self.SetPen(pen)
        brush =wx.Brush(self.get_wxcolour(), wx.SOLID)
        self.SetBrush(brush)
        self.SetTextForeground(self.get_wxcolour())
        self.unselect()

    def set_graylevel(self, frac):
        """
        Set the foreground color.  fg can be a matlab format string, a
        html hex color string, an rgb unit tuple, or a float between 0
        and 1.  In the latter case, grayscale is used.
        """
        DEBUG_MSG("set_graylevel()", 1, self)
        self.select()
        GraphicsContextBase.set_graylevel(self, frac)
        pen = self.GetPen()
        pen.SetColour(self.get_wxcolour())
        self.SetPen(pen)
        brush =wx.Brush(self.get_wxcolour(), wx.SOLID)
        self.SetBrush(brush)
        self.unselect()

    def set_linewidth(self, w):
        """
        Set the line width.
        """
        DEBUG_MSG("set_linewidth()", 1, self)
        self.select()
        if w>0 and w<1: w = 1
        GraphicsContextBase.set_linewidth(self, w)
        pen = self.GetPen()
        lw = int(self.renderer.points_to_pixels(self._linewidth))
        if lw==0: lw = 1
        pen.SetWidth(lw)
        self.SetPen(pen)
        self.unselect()

    def set_capstyle(self, cs):
        """
        Set the capstyle as a string in ('butt', 'round', 'projecting')
        """
        DEBUG_MSG("set_capstyle()", 1, self)
        self.select()
        GraphicsContextBase.set_capstyle(self, cs)
        pen = self.GetPen()
        pen.SetCap(GraphicsContextWx._capd[self._capstyle])
        self.SetPen(pen)
        self.unselect()

    def set_joinstyle(self, js):
        """
        Set the join style to be one of ('miter', 'round', 'bevel')
        """
        DEBUG_MSG("set_joinstyle()", 1, self)
        self.select()
        GraphicsContextBase.set_joinstyle(self, js)
        pen = self.GetPen()
        pen.SetJoin(GraphicsContextWx._joind[self._joinstyle])
        self.SetPen(pen)
        self.unselect()

    def set_linestyle(self, ls):
        """
        Set the line style to be one of
        """
        DEBUG_MSG("set_linestyle()", 1, self)
        self.select()
        GraphicsContextBase.set_linestyle(self, ls)
        try:
            self._style = GraphicsContextWx._dashd_wx[ls]
        except KeyError:
            self._style=wx.LONG_DASH# Style not used elsewhere...

        # On MS Windows platform, only line width of 1 allowed for dash lines
        if wx.Platform == '__WXMSW__':
            self.set_linewidth(1)

        pen = self.GetPen()
        pen.SetStyle(self._style)
        self.SetPen(pen)
        self.unselect()

    def get_wxcolour(self):
        """return a wx.Colour from RGB format"""
        DEBUG_MSG("get_wx_color()", 1, self)
        r, g, b = self.get_rgb()
        r *= 255
        g *= 255
        b *= 255
        return wx.Colour(red=int(r), green=int(g), blue=int(b))

class FigureCanvasWx(FigureCanvasBase, wx.Panel):
    """
    The FigureCanvas contains the figure and does event handling.

    In the wxPython backend, it is derived from wxPanel, and (usually) lives
    inside a frame instantiated by a FigureManagerWx. The parent window probably
    implements a wx.Sizer to control the displayed control size - but we give a
    hint as to our preferred minimum size.
    """

    keyvald = {
        wx.WXK_CONTROL         : 'control',
        wx.WXK_SHIFT           : 'shift',
        wx.WXK_ALT             : 'alt',
        wx.WXK_LEFT            : 'left',
        wx.WXK_UP              : 'up',
        wx.WXK_RIGHT           : 'right',
        wx.WXK_DOWN            : 'down',
        wx.WXK_ESCAPE          : 'escape',
        wx.WXK_F1              : 'f1',
        wx.WXK_F2              : 'f2',
        wx.WXK_F3              : 'f3',
        wx.WXK_F4              : 'f4',
        wx.WXK_F5              : 'f5',
        wx.WXK_F6              : 'f6',
        wx.WXK_F7              : 'f7',
        wx.WXK_F8              : 'f8',
        wx.WXK_F9              : 'f9',
        wx.WXK_F10             : 'f10',
        wx.WXK_F11             : 'f11',
        wx.WXK_F12             : 'f12',
        wx.WXK_SCROLL          : 'scroll_lock',
        wx.WXK_PAUSE           : 'break',
        wx.WXK_BACK            : 'backspace',
        wx.WXK_RETURN          : 'enter',
        wx.WXK_INSERT          : 'insert',
        wx.WXK_DELETE          : 'delete',
        wx.WXK_HOME            : 'home',
        wx.WXK_END             : 'end',
        wx.WXK_PRIOR           : 'pageup',
        wx.WXK_NEXT            : 'pagedown',
        wx.WXK_PAGEUP          : 'pageup',
        wx.WXK_PAGEDOWN        : 'pagedown',
        wx.WXK_NUMPAD0         : '0',
        wx.WXK_NUMPAD1         : '1',
        wx.WXK_NUMPAD2         : '2',
        wx.WXK_NUMPAD3         : '3',
        wx.WXK_NUMPAD4         : '4',
        wx.WXK_NUMPAD5         : '5',
        wx.WXK_NUMPAD6         : '6',
        wx.WXK_NUMPAD7         : '7',
        wx.WXK_NUMPAD8         : '8',
        wx.WXK_NUMPAD9         : '9',
        wx.WXK_NUMPAD_ADD      : '+',
        wx.WXK_NUMPAD_SUBTRACT : '-',
        wx.WXK_NUMPAD_MULTIPLY : '*',
        wx.WXK_NUMPAD_DIVIDE   : '/',
        wx.WXK_NUMPAD_DECIMAL  : 'dec',
        wx.WXK_NUMPAD_ENTER    : 'enter',
        wx.WXK_NUMPAD_UP       : 'up',
        wx.WXK_NUMPAD_RIGHT    : 'right',
        wx.WXK_NUMPAD_DOWN     : 'down',
        wx.WXK_NUMPAD_LEFT     : 'left',
        wx.WXK_NUMPAD_PRIOR    : 'pageup',
        wx.WXK_NUMPAD_NEXT     : 'pagedown',
        wx.WXK_NUMPAD_PAGEUP   : 'pageup',
        wx.WXK_NUMPAD_PAGEDOWN : 'pagedown',
        wx.WXK_NUMPAD_HOME     : 'home',
        wx.WXK_NUMPAD_END      : 'end',
        wx.WXK_NUMPAD_INSERT   : 'insert',
        wx.WXK_NUMPAD_DELETE   : 'delete',
        }

    def __init__(self, parent, id, figure):
        """
        Initialise a FigureWx instance.

        - Initialise the FigureCanvasBase and wxPanel parents.
        - Set event handlers for:
          EVT_SIZE  (Resize event)
          EVT_PAINT (Paint event)
        """

        FigureCanvasBase.__init__(self, figure)
        # Set preferred window size hint - helps the sizer (if one is
        # connected)
        l,b,w,h = figure.bbox.get_bounds()
        w = int(math.ceil(w))
        h = int(math.ceil(h))

        wx.Panel.__init__(self, parent, id, size=wx.Size(w, h))
        # Create the drawing bitmap
        self.bitmap =wx.EmptyBitmap(w, h)
        DEBUG_MSG("__init__() - bitmap w:%d h:%d" % (w,h), 2, self)
        # TODO: Add support for 'point' inspection and plot navigation.
        self._isRealized = False
        self._isConfigured = False
        self._printQued = []

        if wx.VERSION_STRING >= '2.5':
            # Event handlers 2.5
            self.Bind(wx.EVT_SIZE, self._onSize)
            self.Bind(wx.EVT_PAINT, self._onPaint)
            self.Bind(wx.EVT_KEY_DOWN, self._onKeyDown)
            self.Bind(wx.EVT_KEY_UP, self._onKeyUp)
            self.Bind(wx.EVT_RIGHT_DOWN, self._onRightButtonDown)
            self.Bind(wx.EVT_RIGHT_DCLICK, self._onRightButtonDown)
            self.Bind(wx.EVT_RIGHT_UP, self._onRightButtonUp)
            self.Bind(wx.EVT_MOUSEWHEEL, self._onMouseWheel)
            self.Bind(wx.EVT_LEFT_DOWN, self._onLeftButtonDown)
            self.Bind(wx.EVT_LEFT_DCLICK, self._onLeftButtonDown)
            self.Bind(wx.EVT_LEFT_UP, self._onLeftButtonUp)
            self.Bind(wx.EVT_MOTION, self._onMotion)
            self.Bind(wx.EVT_LEAVE_WINDOW, self._onLeave)
        else:
            # Event handlers 2.4
            wx.EVT_SIZE(self, self._onSize)
            wx.EVT_PAINT(self, self._onPaint)
            wx.EVT_KEY_DOWN(self, self._onKeyDown)
            wx.EVT_KEY_UP(self, self._onKeyUp)
            wx.EVT_RIGHT_DOWN(self, self._onRightButtonDown)
            wx.EVT_RIGHT_DCLICK(self, self._onRightButtonDown)
            wx.EVT_RIGHT_UP(self, self._onRightButtonUp)
            wx.EVT_MOUSEWHEEL(self, self._onMouseWheel)
            wx.EVT_LEFT_DOWN(self, self._onLeftButtonDown)
            wx.EVT_LEFT_DCLICK(self, self._onLeftButtonDown)
            wx.EVT_LEFT_UP(self, self._onLeftButtonUp)
            wx.EVT_MOTION(self, self._onMotion)
            wx.EVT_LEAVE_WINDOW(self, self._onLeave)


        self.macros = {} # dict from wx id to seq of macros

        self.Printer_Init()

    def Destroy(self, *args, **kwargs):
        wx.Panel.Destroy(self, *args, **kwargs)


    def Copy_to_Clipboard(self, event=None):
        "copy bitmap of canvas to system clipboard"
        bmp_obj = wx.BitmapDataObject()
        bmp_obj.SetBitmap(self.bitmap)
        wx.TheClipboard.Open()
        wx.TheClipboard.SetData(bmp_obj)
        wx.TheClipboard.Close()

    def Printer_Init(self):
        """initialize printer settings using wx methods"""
        self.printerData = wx.PrintData()
        self.printerData.SetPaperId(wx.PAPER_LETTER)
        self.printerData.SetPrintMode(wx.PRINT_MODE_PRINTER)
        self.printerPageData= wx.PageSetupDialogData()
        self.printerPageData.SetMarginBottomRight((25,25))
        self.printerPageData.SetMarginTopLeft((25,25))
        self.printerPageData.SetPrintData(self.printerData)

        self.printer_width = 5.5
        self.printer_margin= 0.5

    def Printer_Setup(self, event=None):
        """set up figure for printing.  The standard wx Printer
        Setup Dialog seems to die easily. Therefore, this setup
        simply asks for image width and margin for printing. """

        dmsg = """Width of output figure in inches.
The current aspect ration will be kept."""

        dlg = wx.Dialog(self, -1, 'Page Setup for Printing' , (-1,-1))
        df = dlg.GetFont()
        df.SetWeight(wx.NORMAL)
        df.SetPointSize(11)
        dlg.SetFont(df)

        x_wid = wx.TextCtrl(dlg,-1,value="%.2f" % self.printer_width, size=(70,-1))
        x_mrg = wx.TextCtrl(dlg,-1,value="%.2f" % self.printer_margin,size=(70,-1))

        sizerAll = wx.BoxSizer(wx.VERTICAL)
        sizerAll.Add(wx.StaticText(dlg,-1,dmsg),
                    0, wx.ALL | wx.EXPAND, 5)

        sizer = wx.FlexGridSizer(0,3)
        sizerAll.Add(sizer, 0, wx.ALL | wx.EXPAND, 5)

        sizer.Add(wx.StaticText(dlg,-1,'Figure Width'),
                    1, wx.ALIGN_LEFT|wx.ALL, 2)
        sizer.Add(x_wid,
                    1, wx.ALIGN_LEFT|wx.ALL, 2)
        sizer.Add(wx.StaticText(dlg,-1,'in'),
                    1, wx.ALIGN_LEFT|wx.ALL, 2)

        sizer.Add(wx.StaticText(dlg,-1,'Margin'),
                    1, wx.ALIGN_LEFT|wx.ALL, 2)
        sizer.Add(x_mrg,
                    1, wx.ALIGN_LEFT|wx.ALL, 2)
        sizer.Add(wx.StaticText(dlg,-1,'in'),
                    1, wx.ALIGN_LEFT|wx.ALL, 2)

        btn = wx.Button(dlg,wx.ID_OK, " OK ")
        btn.SetDefault()
        sizer.Add(btn, 1, wx.ALIGN_LEFT, 5)
        btn = wx.Button(dlg,wx.ID_CANCEL, " CANCEL ")
        sizer.Add(btn, 1, wx.ALIGN_LEFT, 5)

        dlg.SetSizer(sizerAll)
        dlg.SetAutoLayout(True)
        sizerAll.Fit(dlg)

        if dlg.ShowModal() == wx.ID_OK:
            try:
                self.printer_width  = float(x_wid.GetValue())
                self.printer_margin = float(x_mrg.GetValue())
            except:
                pass

        if ((self.printer_width + self.printer_margin) > 7.5):
            self.printerData.SetOrientation(wx.LANDSCAPE)
        else:
            self.printerData.SetOrientation(wx.PORTRAIT)
        dlg.Destroy()
        return

    def Printer_Setup2(self, event=None):
        """set up figure for printing.  Using the standard wx Printer
        Setup Dialog. """

        if hasattr(self, 'printerData'):
            data = wx.PageSetupDialogData()
            data.SetPrintData(self.printerData)
        else:
            data = wx.PageSetupDialogData()
        data.SetMarginTopLeft( (15, 15) )
        data.SetMarginBottomRight( (15, 15) )

        dlg = wx.PageSetupDialog(self, data)

        if dlg.ShowModal() == wx.ID_OK:
            data = dlg.GetPageSetupData()
            tl = data.GetMarginTopLeft()
            br = data.GetMarginBottomRight()
        self.printerData = wx.PrintData(data.GetPrintData())
        dlg.Destroy()

    def Printer_Preview(self, event=None):
        """ generate Print Preview with wx Print mechanism"""
        po1  = PrintoutWx(self, width=self.printer_width,
                          margin=self.printer_margin)
        po2  = PrintoutWx(self, width=self.printer_width,
                          margin=self.printer_margin)
        self.preview = wx.PrintPreview(po1,po2,self.printerData)
        if not self.preview.Ok():  print "error with preview"

        self.preview.SetZoom(50)
        frameInst= self
        while not isinstance(frameInst, wx.Frame):
            frameInst= frameInst.GetParent()
        frame = wx.PreviewFrame(self.preview, frameInst, "Preview")
        frame.Initialize()
        frame.SetPosition(self.GetPosition())
        frame.SetSize((850,650))
        frame.Centre(wx.BOTH)
        frame.Show(True)
        self.gui_repaint()

    def Printer_Print(self, event=None):
        """ Print figure using wx Print mechanism"""
        pdd = wx.PrintDialogData()
        # SetPrintData for 2.4 combatibility
        pdd.SetPrintData(self.printerData)
        pdd.SetToPage(1)
        printer  = wx.Printer(pdd)
        printout  = PrintoutWx(self, width=int(self.printer_width),
                               margin=int(self.printer_margin))
        print_ok = printer.Print(self, printout, True)

        if wx.VERSION_STRING >= '2.5':
            if not print_ok and not printer.GetLastError() == wx.PRINTER_CANCELLED:
                wx.MessageBox("""There was a problem printing.
                Perhaps your current printer is not set correctly?""",
                              "Printing", wx.OK)
        else:
            if not print_ok:
                wx.MessageBox("""There was a problem printing.
                Perhaps your current printer is not set correctly?""",
                              "Printing", wx.OK)
        printout.Destroy()
        self.gui_repaint()


    def draw(self, repaint=True):
        """
        Render the figure using RendererWx instance renderer, or using a
        previously defined renderer if none is specified.
        """
        DEBUG_MSG("draw()", 1, self)
        self.renderer = RendererWx(self.bitmap, self.figure.dpi)
        self.figure.draw(self.renderer)
        if repaint:
            self.gui_repaint()

    def _get_imagesave_wildcards(self):
        'return the wildcard string for the filesave dialog'
        default_filetype = self.get_default_filetype()
        filetypes = self.get_supported_filetypes_grouped()
        sorted_filetypes = filetypes.items()
        sorted_filetypes.sort()
        wildcards = []
        extensions = []
        filter_index = 0
        for i, (name, exts) in enumerate(sorted_filetypes):
            ext_list = ';'.join(['*.%s' % ext for ext in exts])
            extensions.append(exts[0])
            wildcard = '%s (%s)|%s' % (name, ext_list, ext_list)
            if default_filetype in exts:
                filter_index = i
            wildcards.append(wildcard)
        wildcards = '|'.join(wildcards)
        return wildcards, extensions, filter_index

    def gui_repaint(self, drawDC=None):
        """
        Performs update of the displayed image on the GUI canvas, using the
        supplied device context.  If drawDC is None, a ClientDC will be used to
        redraw the image.
        """
        DEBUG_MSG("gui_repaint()", 1, self)
        if drawDC is None:
            drawDC=wx.ClientDC(self)

        drawDC.BeginDrawing()
        drawDC.DrawBitmap(self.bitmap, 0, 0)
        drawDC.EndDrawing()

    filetypes = FigureCanvasBase.filetypes.copy()
    filetypes['bmp'] = 'Windows bitmap'
    filetypes['jpeg'] = 'JPEG'
    filetypes['jpg'] = 'JPEG'
    filetypes['pcx'] = 'PCX'
    filetypes['png'] = 'Portable Network Graphics'
    filetypes['tif'] = 'Tagged Image Format File'
    filetypes['tiff'] = 'Tagged Image Format File'
    filetypes['xpm'] = 'X pixmap'

    def print_bmp(self, filename, *args, **kwargs):
        return self._print_image(filename, wx.BITMAP_TYPE_BMP, *args, **kwargs)
    
    def print_jpeg(self, filename, *args, **kwargs):
        return self._print_image(filename, wx.BITMAP_TYPE_JPEG, *args, **kwargs)
    print_jpg = print_jpeg

    def print_pcx(self, filename, *args, **kwargs):
        return self._print_image(filename, wx.BITMAP_TYPE_PCX, *args, **kwargs)

    def print_png(self, filename, *args, **kwargs):
        return self._print_image(filename, wx.BITMAP_TYPE_PNG, *args, **kwargs)
    
    def print_tiff(self, filename, *args, **kwargs):
        return self._print_image(filename, wx.BITMAP_TYPE_TIF, *args, **kwargs)
    print_tif = print_tiff

    def print_xpm(self, filename, *args, **kwargs):
        return self._print_image(filename, wx.BITMAP_TYPE_XPM, *args, **kwargs)
    
    def _print_image(self, filename, filetype, *args, **kwargs):
        origBitmap   = self.bitmap

        l,b,width,height = self.figure.bbox.get_bounds()
        width = int(math.ceil(width))
        height = int(math.ceil(height))

        # Following performs the same function as realize(), but without
        # setting GUI attributes - so GUI draw() will render correctly
        self.bitmap = wx.EmptyBitmap(width, height)
        renderer = RendererWx(self.bitmap, self.figure.dpi)

        gc = renderer.new_gc()

        self.figure.draw(renderer)

        # Now that we have rendered into the bitmap, save it
        # to the appropriate file type and clean up
        if is_string_like(filename):
            if not self.bitmap.SaveFile(filename, filetype):
                DEBUG_MSG('print_figure() file save error', 4, self)
                # note the error must be displayed here because trapping
                # the error on a call or print_figure may not work because
                # printing can be qued and called from realize
                raise RuntimeError('Could not save figure to %s\n' % (filename))
        elif is_writable_file_like(filename):
            if not self.bitmap.ConvertToImage().SaveStream(filename, filetype):
                DEBUG_MSG('print_figure() file save error', 4, self)
                raise RuntimeError('Could not save figure to %s\n' % (filename))

        # Restore everything to normal
        self.bitmap = origBitmap

        self.draw()
        self.Refresh()

    def get_default_filetype(self):
        return 'png'
        
    def realize(self):
        """
        This method will be called when the system is ready to draw,
        eg when a GUI window is realized
        """
        DEBUG_MSG("realize()", 1, self)
        self._isRealized = True
        for fname, dpi, facecolor, edgecolor in self._printQued:
            self.print_figure(fname, dpi, facecolor, edgecolor)
        self._printQued = []



    def _onPaint(self, evt):
        """
        Called when wxPaintEvt is generated
        """

        DEBUG_MSG("_onPaint()", 1, self)
        if not self._isRealized:
            self.realize()
        # Render to the bitmap
        self.draw(repaint=False)
        # Update the display using a PaintDC
        self.gui_repaint(drawDC=wx.PaintDC(self))
        evt.Skip()

    def _onSize(self, evt):
        """
        Called when wxEventSize is generated.

        In this application we attempt to resize to fit the window, so it
        is better to take the performance hit and redraw the whole window.
        """

        DEBUG_MSG("_onSize()", 2, self)
        # Create a new, correctly sized bitmap
        self._width, self._height = self.GetClientSize()
        self.bitmap =wx.EmptyBitmap(self._width, self._height)

        if self._width <= 1 or self._height <= 1: return # Empty figure

        # Scale the displayed image (but don't update self.figsize)
        if not self._isConfigured:
            self._isConfigured = True

        dpival = self.figure.dpi.get()
        winch = self._width/dpival
        hinch = self._height/dpival
        self.figure.set_size_inches(winch, hinch)

        if self._isRealized:
            self.draw()
        evt.Skip()

    def _get_key(self, evt):

        keyval = evt.m_keyCode
        if self.keyvald.has_key(keyval):
            key = self.keyvald[keyval]
        elif keyval <256:
            key = chr(keyval)
        else:
            key = None

        # why is wx upcasing this?
        if key is not None: key = key.lower()

        return key

    def _onKeyDown(self, evt):
        """Capture key press."""
        key = self._get_key(evt)
        evt.Skip()
        FigureCanvasBase.key_press_event(self, key, guiEvent=evt)

    def _onKeyUp(self, evt):
        """Release key."""
        key = self._get_key(evt)
        #print 'release key', key
        evt.Skip()
        FigureCanvasBase.key_release_event(self, key, guiEvent=evt)

    def _onRightButtonDown(self, evt):
        """Start measuring on an axis."""
        x = evt.GetX()
        y = self.figure.bbox.height() - evt.GetY()
        evt.Skip()
        self.CaptureMouse()
        FigureCanvasBase.button_press_event(self, x, y, 3, guiEvent=evt)


    def _onRightButtonUp(self, evt):
        """End measuring on an axis."""
        x = evt.GetX()
        y = self.figure.bbox.height() - evt.GetY()
        evt.Skip()
        if self.HasCapture(): self.ReleaseMouse()
        FigureCanvasBase.button_release_event(self, x, y, 3, guiEvent=evt)

    def _onLeftButtonDown(self, evt):
        """Start measuring on an axis."""
        x = evt.GetX()
        y = self.figure.bbox.height() - evt.GetY()
        evt.Skip()
        self.CaptureMouse()
        FigureCanvasBase.button_press_event(self, x, y, 1, guiEvent=evt)

    def _onLeftButtonUp(self, evt):
        """End measuring on an axis."""
        x = evt.GetX()
        y = self.figure.bbox.height() - evt.GetY()
        #print 'release button', 1
        evt.Skip()
        if self.HasCapture(): self.ReleaseMouse()
        FigureCanvasBase.button_release_event(self, x, y, 1, guiEvent=evt)

    def _onMouseWheel(self, evt):
        # TODO: implement mouse wheel handler
        pass

    def _onMotion(self, evt):
        """Start measuring on an axis."""

        x = evt.GetX()
        y = self.figure.bbox.height() - evt.GetY()
        evt.Skip()
        FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=evt)

    def _onLeave(self, evt):
        """Mouse has left the window; fake a motion event."""

        evt.Skip()
        FigureCanvasBase.motion_notify_event(self, -1, -1, guiEvent=evt)


########################################################################
#
# The following functions and classes are for pylab compatibility
# mode (matplotlib.pylab) and implement figure managers, etc...
#
########################################################################


def _create_wx_app():
    """
    Creates a wx.PySimpleApp instance if a wx.App has not been created.
    """
    wxapp = wx.GetApp()
    if wxapp is None:
        wxapp = wx.PySimpleApp()
        wxapp.SetExitOnFrameDelete(True)
        # retain a reference to the app object so it does not get garbage
        # collected and cause segmentation faults
        _create_wx_app.theWxApp = wxapp


def draw_if_interactive():
    """
    This should be overriden in a windowing environment if drawing
    should be done in interactive python mode
    """
    DEBUG_MSG("draw_if_interactive()", 1, None)

    if matplotlib.is_interactive():

        figManager = Gcf.get_active()
        if figManager is not None:
            figManager.canvas.draw()


def show():
    """
    Current implementation assumes that matplotlib is executed in a PyCrust
    shell. It appears to be possible to execute wxPython applications from
    within a PyCrust without having to ensure that wxPython has been created
    in a secondary thread (e.g. SciPy gui_thread).

    Unfortunately, gui_thread seems to introduce a number of further
    dependencies on SciPy modules, which I do not wish to introduce
    into the backend at this point. If there is a need I will look
    into this in a later release.
    """
    DEBUG_MSG("show()", 3, None)

    for figwin in Gcf.get_all_fig_managers():
        figwin.frame.Show()
        figwin.canvas.realize()
        figwin.canvas.draw()

    if show._needmain and not matplotlib.is_interactive():
        # start the wxPython gui event if there is not already one running
        wxapp = wx.GetApp()
        if wxapp is not None:
            # wxPython 2.4 has no wx.App.IsMainLoopRunning() method
            imlr = getattr(wxapp, 'IsMainLoopRunning', lambda: False)
            if not imlr():
                wxapp.MainLoop()
        show._needmain = False
show._needmain = True

def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    # in order to expose the Figure constructor to the pylab
    # interface we need to create the figure here
    DEBUG_MSG("new_figure_manager()", 3, None)
    _create_wx_app()

    FigureClass = kwargs.pop('FigureClass', Figure)
    fig = FigureClass(*args, **kwargs)
    frame = FigureFrameWx(num, fig)
    figmgr = frame.get_figure_manager()
    if matplotlib.is_interactive():
        figmgr.canvas.realize()
        figmgr.frame.Show()

    return figmgr

class FigureFrameWx(wx.Frame):
    def __init__(self, num, fig):
        # On non-Windows platform, explicitly set the position - fix
        # positioning bug on some Linux platforms
        if wx.Platform == '__WXMSW__':
            pos = wx.DefaultPosition
        else:
            pos =wx.Point(20,20)
        l,b,w,h = fig.bbox.get_bounds()
        wx.Frame.__init__(self, parent=None, id=-1, pos=pos,
                          title="Figure %d" % num,
                          size=(w,h))
        DEBUG_MSG("__init__()", 1, self)
        self.num = num

        self.canvas = self.get_canvas(fig)
        statbar = StatusBarWx(self)
        self.SetStatusBar(statbar)
        self.sizer =wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.TOP | wx.LEFT | wx.EXPAND)
        # By adding toolbar in sizer, we are able to put it at the bottom
        # of the frame - so appearance is closer to GTK version

        self.toolbar = self._get_toolbar(statbar)

        if self.toolbar is not None:
            self.toolbar.Realize()
            if wx.Platform == '__WXMAC__':
                # Mac platform (OSX 10.3, MacPython) does not seem to cope with
                # having a toolbar in a sizer. This work-around gets the buttons
                # back, but at the expense of having the toolbar at the top
                self.SetToolBar(self.toolbar)
            else:
                # On Windows platform, default window size is incorrect, so set
                # toolbar width to figure width.
                tw, th = self.toolbar.GetSizeTuple()
                fw, fh = self.canvas.GetSizeTuple()
                # By adding toolbar in sizer, we are able to put it at the bottom
                # of the frame - so appearance is closer to GTK version.
                # As noted above, doesn't work for Mac.
                self.toolbar.SetSize(wx.Size(fw, th))
                self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(self.sizer)
        self.Fit()

        self.figmgr = FigureManagerWx(self.canvas, num, self)

        if wx.VERSION_STRING >= '2.5':
            # Event handlers 2.5
            self.Bind(wx.EVT_CLOSE, self._onClose)
        else:
            # Event handlers 2.4
            wx.EVT_CLOSE(self, self._onClose)

    def _get_toolbar(self, statbar):
        if matplotlib.rcParams['toolbar']=='classic':
            toolbar = NavigationToolbarWx(self.canvas, True)
        elif matplotlib.rcParams['toolbar']=='toolbar2':
            toolbar = NavigationToolbar2Wx(self.canvas)
            toolbar.set_status_bar(statbar)
        else:
            toolbar = None
        return toolbar

    def get_canvas(self, fig):
        return FigureCanvasWx(self, -1, fig)

    def get_figure_manager(self):
        DEBUG_MSG("get_figure_manager()", 1, self)
        return self.figmgr

    def _onClose(self, evt):
        DEBUG_MSG("onClose()", 1, self)
        Gcf.destroy(self.num)
        #self.Destroy()

    def GetToolBar(self):
        """Override wxFrame::GetToolBar as we don't have managed toolbar"""
        return self.toolbar

    def Destroy(self, *args, **kwargs):
        wx.Frame.Destroy(self, *args, **kwargs)
        if self.toolbar is not None:
            self.toolbar.Destroy()
        wxapp = wx.GetApp()
        if wxapp:
            wxapp.Yield()
        return True

class FigureManagerWx(FigureManagerBase):
    """
    This class contains the FigureCanvas and GUI frame

    It is instantiated by GcfWx whenever a new figure is created. GcfWx is
    responsible for managing multiple instances of FigureManagerWx.

    NB: FigureManagerBase is found in _pylab_helpers

    public attrs

    canvas - a FigureCanvasWx(wx.Panel) instance
    window - a wxFrame instance - http://www.lpthe.jussieu.fr/~zeitlin/wxWindows/docs/wxwin_wxframe.html#wxframe
    """
    def __init__(self, canvas, num, frame):
        DEBUG_MSG("__init__()", 1, self)
        FigureManagerBase.__init__(self, canvas, num)
        self.frame = frame
        self.window = frame

        self.tb = frame.GetToolBar()
        self.toolbar = self.tb  # consistent with other backends
        def notify_axes_change(fig):
            'this will be called whenever the current axes is changed'
            if self.tb != None: self.tb.update()
        self.canvas.figure.add_axobserver(notify_axes_change)

        def showfig(*args):
            figwin.frame.Show()
            figwin.canvas.realize()
            figwin.canvas.draw()

        # attach a show method to the figure
        self.canvas.figure.show = showfig


    def destroy(self, *args):
        DEBUG_MSG("destroy()", 1, self)
        self.frame.Destroy()
        #if self.tb is not None: self.tb.Destroy()
        import wx
        #wx.GetApp().ProcessIdle()
        wx.WakeUpIdle()

    def set_window_title(self, title):
        self.window.SetTitle(title)

# Identifiers for toolbar controls - images_wx contains bitmaps for the images
# used in the controls. wxWindows does not provide any stock images, so I've
# 'stolen' those from GTK2, and transformed them into the appropriate format.
#import images_wx

_NTB_AXISMENU        =wx.NewId()
_NTB_AXISMENU_BUTTON =wx.NewId()
_NTB_X_PAN_LEFT      =wx.NewId()
_NTB_X_PAN_RIGHT     =wx.NewId()
_NTB_X_ZOOMIN        =wx.NewId()
_NTB_X_ZOOMOUT       =wx.NewId()
_NTB_Y_PAN_UP        =wx.NewId()
_NTB_Y_PAN_DOWN      =wx.NewId()
_NTB_Y_ZOOMIN        =wx.NewId()
_NTB_Y_ZOOMOUT       =wx.NewId()
#_NTB_SUBPLOT            =wx.NewId()
_NTB_SAVE            =wx.NewId()
_NTB_CLOSE           =wx.NewId()

def _load_bitmap(filename):
    """
    Load a bitmap file from the backends/images subdirectory in which the
    matplotlib library is installed. The filename parameter should not
    contain any path information as this is determined automatically.

    Bitmaps should be in XPM format, and of size 16x16 (unless you change
    the code!). I have converted the stock GTK2 16x16 icons to XPM format.

    Returns a wx.Bitmap object
    """

    basedir = os.path.join(rcParams['datapath'],'images')

    bmpFilename = os.path.normpath(os.path.join(basedir, filename))
    if not os.path.exists(bmpFilename):
        raise IOError('Could not find bitmap file "%s"; dying'%bmpFilename)

    bmp =wx.Bitmap(bmpFilename, wx.BITMAP_TYPE_XPM)
    return bmp

class MenuButtonWx(wx.Button):
    """
    wxPython does not permit a menu to be incorporated directly into a toolbar.
    This class simulates the effect by associating a pop-up menu with a button
    in the toolbar, and managing this as though it were a menu.
    """
    def __init__(self, parent):

        wx.Button.__init__(self, parent, _NTB_AXISMENU_BUTTON, "Axes:        ",
                          style=wx.BU_EXACTFIT)
        self._toolbar = parent
        self._menu =wx.Menu()
        self._axisId = []
        # First two menu items never change...
        self._allId =wx.NewId()
        self._invertId =wx.NewId()
        self._menu.Append(self._allId, "All", "Select all axes", False)
        self._menu.Append(self._invertId, "Invert", "Invert axes selected", False)
        self._menu.AppendSeparator()

        if wx.VERSION_STRING >= '2.5':
            self.Bind(wx.EVT_BUTTON, self._onMenuButton, id=_NTB_AXISMENU_BUTTON)
            self.Bind(wx.EVT_MENU, self._handleSelectAllAxes, id=self._allId)
            self.Bind(wx.EVT_MENU, self._handleInvertAxesSelected, id=self._invertId)
        else:
            wx.EVT_BUTTON(self, _NTB_AXISMENU_BUTTON, self._onMenuButton)
            wx.EVT_MENU(self, self._allId, self._handleSelectAllAxes)
            wx.EVT_MENU(self, self._invertId, self._handleInvertAxesSelected)

    def Destroy(self):
        self._menu.Destroy()
        self.Destroy()

    def _onMenuButton(self, evt):
        """Handle menu button pressed."""
        x, y = self.GetPositionTuple()
        w, h = self.GetSizeTuple()
        self.PopupMenuXY(self._menu, x, y+h-4)
                # When menu returned, indicate selection in button
        evt.Skip()

    def _handleSelectAllAxes(self, evt):
        """Called when the 'select all axes' menu item is selected."""
        if len(self._axisId) == 0:
            return
        for i in range(len(self._axisId)):
            self._menu.Check(self._axisId[i], True)
        self._toolbar.set_active(self.getActiveAxes())
        evt.Skip()

    def _handleInvertAxesSelected(self, evt):
        """Called when the invert all menu item is selected"""
        if len(self._axisId) == 0: return
        for i in range(len(self._axisId)):
            if self._menu.IsChecked(self._axisId[i]):
                self._menu.Check(self._axisId[i], False)
            else:
                self._menu.Check(self._axisId[i], True)
        self._toolbar.set_active(self.getActiveAxes())
        evt.Skip()

    def _onMenuItemSelected(self, evt):
        """Called whenever one of the specific axis menu items is selected"""
        current = self._menu.IsChecked(evt.GetId())
        if current:
            new = False
        else:
            new = True
        self._menu.Check(evt.GetId(), new)
        self._toolbar.set_active(self.getActiveAxes())
        evt.Skip()

    def updateAxes(self, maxAxis):
        """Ensures that there are entries for max_axis axes in the menu
        (selected by default)."""
        if maxAxis > len(self._axisId):
            for i in range(len(self._axisId) + 1, maxAxis + 1, 1):
                menuId =wx.NewId()
                self._axisId.append(menuId)
                self._menu.Append(menuId, "Axis %d" % i, "Select axis %d" % i, True)
                self._menu.Check(menuId, True)

                if wx.VERSION_STRING >= '2.5':
                    self.Bind(wx.EVT_MENU, self._onMenuItemSelected, id=menuId)
                else:
                    wx.EVT_MENU(self, menuId, self._onMenuItemSelected)
        self._toolbar.set_active(range(len(self._axisId)))

    def getActiveAxes(self):
        """Return a list of the selected axes."""
        active = []
        for i in range(len(self._axisId)):
            if self._menu.IsChecked(self._axisId[i]):
                active.append(i)
        return active

    def updateButtonText(self, lst):
        """Update the list of selected axes in the menu button"""
        axis_txt = ''
        for e in lst:
            axis_txt += '%d,' % (e+1)
        # remove trailing ',' and add to button string
        self.SetLabel("Axes: %s" % axis_txt[:-1])




cursord = {
    cursors.MOVE : wx.CURSOR_HAND,
    cursors.HAND : wx.CURSOR_HAND,
    cursors.POINTER : wx.CURSOR_ARROW,
    cursors.SELECT_REGION : wx.CURSOR_CROSS,
    }


class SubplotToolWX(wx.Frame):
    def __init__(self, targetfig):
        wx.Frame.__init__(self, None, -1, "Configure subplots")

        toolfig = Figure((6,3))
        canvas = FigureCanvasWx(self, -1, toolfig)

        # Create a figure manager to manage things
        figmgr = FigureManager(canvas, 1, self)

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(canvas, 1, wx.LEFT|wx.TOP|wx.GROW)
        self.SetSizer(sizer)
        self.Fit()
        tool = SubplotTool(targetfig, toolfig)


class NavigationToolbar2Wx(NavigationToolbar2, wx.ToolBar):

    def __init__(self, canvas):
        wx.ToolBar.__init__(self, canvas.GetParent(), -1)
        NavigationToolbar2.__init__(self, canvas)
        self.canvas = canvas
        self._idle = True
        self.statbar = None

    def get_canvas(self, frame, fig):
        return FigureCanvasWx(frame, -1, fig)

    def _init_toolbar(self):
        DEBUG_MSG("_init_toolbar", 1, self)

        self._parent = self.canvas.GetParent()
        _NTB2_HOME    =wx.NewId()
        self._NTB2_BACK    =wx.NewId()
        self._NTB2_FORWARD =wx.NewId()
        self._NTB2_PAN     =wx.NewId()
        self._NTB2_ZOOM    =wx.NewId()
        _NTB2_SAVE    = wx.NewId()
        _NTB2_SUBPLOT    =wx.NewId()

        self.SetToolBitmapSize(wx.Size(24,24))

        self.AddSimpleTool(_NTB2_HOME, _load_bitmap('home.xpm'),
                           'Home', 'Reset original view')
        self.AddSimpleTool(self._NTB2_BACK, _load_bitmap('back.xpm'),
                           'Back', 'Back navigation view')
        self.AddSimpleTool(self._NTB2_FORWARD, _load_bitmap('forward.xpm'),
                           'Forward', 'Forward navigation view')
        # todo: get new bitmap
        self.AddCheckTool(self._NTB2_PAN, _load_bitmap('move.xpm'),
                           shortHelp='Pan',
                           longHelp='Pan with left, zoom with right')
        self.AddCheckTool(self._NTB2_ZOOM, _load_bitmap('zoom_to_rect.xpm'),
                           shortHelp='Zoom', longHelp='Zoom to rectangle')

        self.AddSeparator()
        self.AddSimpleTool(_NTB2_SUBPLOT, _load_bitmap('subplots.xpm'),
                           'Configure subplots', 'Configure subplot parameters')

        self.AddSimpleTool(_NTB2_SAVE, _load_bitmap('filesave.xpm'),
                           'Save', 'Save plot contents to file')

        if wx.VERSION_STRING >= '2.5':
            self.Bind(wx.EVT_TOOL, self.home, id=_NTB2_HOME)
            self.Bind(wx.EVT_TOOL, self.forward, id=self._NTB2_FORWARD)
            self.Bind(wx.EVT_TOOL, self.back, id=self._NTB2_BACK)
            self.Bind(wx.EVT_TOOL, self.zoom, id=self._NTB2_ZOOM)
            self.Bind(wx.EVT_TOOL, self.pan, id=self._NTB2_PAN)
            self.Bind(wx.EVT_TOOL, self.configure_subplot, id=_NTB2_SUBPLOT)
            self.Bind(wx.EVT_TOOL, self.save, id=_NTB2_SAVE)
        else:
            wx.EVT_TOOL(self, _NTB2_HOME, self.home)
            wx.EVT_TOOL(self, self._NTB2_FORWARD, self.forward)
            wx.EVT_TOOL(self, self._NTB2_BACK, self.back)
            wx.EVT_TOOL(self, self._NTB2_ZOOM, self.zoom)
            wx.EVT_TOOL(self, self._NTB2_PAN, self.pan)
            wx.EVT_TOOL(self, _NTB2_SUBPLOT, self.configure_subplot)
            wx.EVT_TOOL(self, _NTB2_SAVE, self.save)

        self.Realize()


    def zoom(self, *args):
        self.ToggleTool(self._NTB2_PAN, False)
        NavigationToolbar2.zoom(self, *args)

    def pan(self, *args):
        self.ToggleTool(self._NTB2_ZOOM, False)
        NavigationToolbar2.pan(self, *args)


    def configure_subplot(self, evt):
        frame = wx.Frame(None, -1, "Configure subplots")

        toolfig = Figure((6,3))
        canvas = self.get_canvas(frame, toolfig)

        # Create a figure manager to manage things
        figmgr = FigureManager(canvas, 1, frame)

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(canvas, 1, wx.LEFT|wx.TOP|wx.GROW)
        frame.SetSizer(sizer)
        frame.Fit()
        tool = SubplotTool(self.canvas.figure, toolfig)
        frame.Show()

    def save(self, evt):
        # Fetch the required filename and file type.
        filetypes, exts, filter_index = self.canvas._get_imagesave_wildcards()
        default_file = "image." + self.canvas.get_default_filetype()
        dlg = wx.FileDialog(self._parent, "Save to file", "", default_file,
                            filetypes,
                            wx.SAVE|wx.OVERWRITE_PROMPT|wx.CHANGE_DIR)
        dlg.SetFilterIndex(filter_index)
        if dlg.ShowModal() == wx.ID_OK:
            dirname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            DEBUG_MSG('Save file dir:%s name:%s' % (dirname, filename), 3, self)
            format = exts[dlg.GetFilterIndex()]
            # Explicitly pass in the selected filetype to override the
            # actual extension if necessary
            try:
                self.canvas.print_figure(
                    os.path.join(dirname, filename), format=format)
            except Exception, e:
                error_msg_wx(str(e))


    def set_cursor(self, cursor):
        cursor =wx.StockCursor(cursord[cursor])
        self.canvas.SetCursor( cursor )

    def release(self, event):
        try: del self.lastrect
        except AttributeError: pass

    def dynamic_update(self):
        d = self._idle
        self._idle = False
        if d:
            self.canvas.draw()
            self._idle = True

    def draw_rubberband(self, event, x0, y0, x1, y1):
        'adapted from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/189744'
        canvas = self.canvas
        dc =wx.ClientDC(canvas)

        # Set logical function to XOR for rubberbanding
        dc.SetLogicalFunction(wx.XOR)

        # Set dc brush and pen
        # Here I set brush and pen to white and grey respectively
        # You can set it to your own choices

        # The brush setting is not really needed since we
        # dont do any filling of the dc. It is set just for
        # the sake of completion.

        wbrush =wx.Brush(wx.Colour(255,255,255), wx.TRANSPARENT)
        wpen =wx.Pen(wx.Colour(200, 200, 200), 1, wx.SOLID)
        dc.SetBrush(wbrush)
        dc.SetPen(wpen)


        dc.ResetBoundingBox()
        dc.BeginDrawing()
        height = self.canvas.figure.bbox.height()
        y1 = height - y1
        y0 = height - y0

        if y1<y0: y0, y1 = y1, y0
        if x1<y0: x0, x1 = x1, x0

        w = x1 - x0
        h = y1 - y0

        rect = int(x0), int(y0), int(w), int(h)
        try: lastrect = self.lastrect
        except AttributeError: pass
        else: dc.DrawRectangle(*lastrect)  #erase last
        self.lastrect = rect
        dc.DrawRectangle(*rect)
        dc.EndDrawing()

    def set_status_bar(self, statbar):
        self.statbar = statbar

    def set_message(self, s):
        if self.statbar is not None: self.statbar.set_function(s)

    def set_history_buttons(self):
        can_backward = (self._views._pos > 0)
        can_forward = (self._views._pos < len(self._views._elements) - 1)
        self.EnableTool(self._NTB2_BACK, can_backward)
        self.EnableTool(self._NTB2_FORWARD, can_forward)


class NavigationToolbarWx(wx.ToolBar):
    def __init__(self, canvas, can_kill=False):
        """
        figure is the Figure instance that the toolboar controls

        win, if not None, is the wxWindow the Figure is embedded in
        """
        wx.ToolBar.__init__(self, canvas.GetParent(), -1)
        DEBUG_MSG("__init__()", 1, self)
        self.canvas = canvas
        self._lastControl = None
        self._mouseOnButton = None

        self._parent = canvas.GetParent()
        self._NTB_BUTTON_HANDLER = {
            _NTB_X_PAN_LEFT  : self.panx,
            _NTB_X_PAN_RIGHT : self.panx,
            _NTB_X_ZOOMIN    : self.zoomx,
            _NTB_X_ZOOMOUT   : self.zoomy,
            _NTB_Y_PAN_UP    : self.pany,
            _NTB_Y_PAN_DOWN  : self.pany,
            _NTB_Y_ZOOMIN    : self.zoomy,
            _NTB_Y_ZOOMOUT   : self.zoomy }


        self._create_menu()
        self._create_controls(can_kill)
        self.Realize()

    def _create_menu(self):
        """
        Creates the 'menu' - implemented as a button which opens a
        pop-up menu since wxPython does not allow a menu as a control
        """
        DEBUG_MSG("_create_menu()", 1, self)
        self._menu = MenuButtonWx(self)
        self.AddControl(self._menu)
        self.AddSeparator()

    def _create_controls(self, can_kill):
        """
        Creates the button controls, and links them to event handlers
        """
        DEBUG_MSG("_create_controls()", 1, self)
        # Need the following line as Windows toolbars default to 15x16

        self.SetToolBitmapSize(wx.Size(16,16))
        self.AddSimpleTool(_NTB_X_PAN_LEFT, _load_bitmap('stock_left.xpm'),
                           'Left', 'Scroll left')
        self.AddSimpleTool(_NTB_X_PAN_RIGHT, _load_bitmap('stock_right.xpm'),
                           'Right', 'Scroll right')
        self.AddSimpleTool(_NTB_X_ZOOMIN, _load_bitmap('stock_zoom-in.xpm'),
                           'Zoom in', 'Increase X axis magnification')
        self.AddSimpleTool(_NTB_X_ZOOMOUT, _load_bitmap('stock_zoom-out.xpm'),
                           'Zoom out', 'Decrease X axis magnification')
        self.AddSeparator()
        self.AddSimpleTool(_NTB_Y_PAN_UP,_load_bitmap('stock_up.xpm'),
                           'Up', 'Scroll up')
        self.AddSimpleTool(_NTB_Y_PAN_DOWN, _load_bitmap('stock_down.xpm'),
                           'Down', 'Scroll down')
        self.AddSimpleTool(_NTB_Y_ZOOMIN, _load_bitmap('stock_zoom-in.xpm'),
                           'Zoom in', 'Increase Y axis magnification')
        self.AddSimpleTool(_NTB_Y_ZOOMOUT, _load_bitmap('stock_zoom-out.xpm'),
                           'Zoom out', 'Decrease Y axis magnification')
        self.AddSeparator()
        self.AddSimpleTool(_NTB_SAVE, _load_bitmap('stock_save_as.xpm'),
                           'Save', 'Save plot contents as images')
        self.AddSeparator()

        if wx.VERSION_STRING >= '2.5':
            self.Bind(wx.EVT_TOOL, self._onLeftScroll, id=_NTB_X_PAN_LEFT)
            self.Bind(wx.EVT_TOOL, self._onRightScroll, id=_NTB_X_PAN_RIGHT)
            self.Bind(wx.EVT_TOOL, self._onXZoomIn, id=_NTB_X_ZOOMIN)
            self.Bind(wx.EVT_TOOL, self._onXZoomOut, id=_NTB_X_ZOOMOUT)
            self.Bind(wx.EVT_TOOL, self._onUpScroll, id=_NTB_Y_PAN_UP)
            self.Bind(wx.EVT_TOOL, self._onDownScroll, id=_NTB_Y_PAN_DOWN)
            self.Bind(wx.EVT_TOOL, self._onYZoomIn, id=_NTB_Y_ZOOMIN)
            self.Bind(wx.EVT_TOOL, self._onYZoomOut, id=_NTB_Y_ZOOMOUT)
            self.Bind(wx.EVT_TOOL, self._onSave, id=_NTB_SAVE)
            self.Bind(wx.EVT_TOOL_ENTER, self._onEnterTool, id=self.GetId())
            if can_kill:
                self.Bind(wx.EVT_TOOL, self._onClose, id=_NTB_CLOSE)
            self.Bind(wx.EVT_MOUSEWHEEL, self._onMouseWheel)
        else:
            wx.EVT_TOOL(self, _NTB_X_PAN_LEFT, self._onLeftScroll)
            wx.EVT_TOOL(self, _NTB_X_PAN_RIGHT, self._onRightScroll)
            wx.EVT_TOOL(self, _NTB_X_ZOOMIN, self._onXZoomIn)
            wx.EVT_TOOL(self, _NTB_X_ZOOMOUT, self._onXZoomOut)
            wx.EVT_TOOL(self, _NTB_Y_PAN_UP, self._onUpScroll)
            wx.EVT_TOOL(self, _NTB_Y_PAN_DOWN, self._onDownScroll)
            wx.EVT_TOOL(self, _NTB_Y_ZOOMIN, self._onYZoomIn)
            wx.EVT_TOOL(self, _NTB_Y_ZOOMOUT, self._onYZoomOut)
            wx.EVT_TOOL(self, _NTB_SAVE, self._onSave)
            wx.EVT_TOOL_ENTER(self, self.GetId(), self._onEnterTool)
            if can_kill:
                wx.EVT_TOOL(self, _NTB_CLOSE, self._onClose)
            wx.EVT_MOUSEWHEEL(self, self._onMouseWheel)

    def set_active(self, ind):
        """
        ind is a list of index numbers for the axes which are to be made active
        """
        DEBUG_MSG("set_active()", 1, self)
        self._ind = ind
        if ind != None:
            self._active = [ self._axes[i] for i in self._ind ]
        else:
            self._active = []
        # Now update button text wit active axes
        self._menu.updateButtonText(ind)

    def get_last_control(self):
        """Returns the identity of the last toolbar button pressed."""
        return self._lastControl

    def panx(self, direction):

        DEBUG_MSG("panx()", 1, self)
        for a in self._active:
            a.panx(direction)
        self.canvas.draw()
        self.canvas.Refresh(eraseBackground=False)

    def pany(self, direction):
        DEBUG_MSG("pany()", 1, self)
        for a in self._active:
            a.pany(direction)
        self.canvas.draw()
        self.canvas.Refresh(eraseBackground=False)

    def zoomx(self, in_out):
        DEBUG_MSG("zoomx()", 1, self)
        for a in self._active:
            a.zoomx(in_out)
        self.canvas.draw()
        self.canvas.Refresh(eraseBackground=False)

    def zoomy(self, in_out):
        DEBUG_MSG("zoomy()", 1, self)
        for a in self._active:
            a.zoomy(in_out)
        self.canvas.draw()
        self.canvas.Refresh(eraseBackground=False)

    def update(self):
        """
        Update the toolbar menu - called when (e.g.) a new subplot or axes are added
        """
        DEBUG_MSG("update()", 1, self)
        self._axes = self.canvas.figure.get_axes()
        self._menu.updateAxes(len(self._axes))

    def _do_nothing(self, d):
        """A NULL event handler - does nothing whatsoever"""
        pass

    # Local event handlers - mainly supply parameters to pan/scroll functions
    def _onEnterTool(self, evt):
        toolId = evt.GetSelection()
        try:
            self.button_fn = self._NTB_BUTTON_HANDLER[toolId]
        except KeyError:
            self.button_fn = self._do_nothing
        evt.Skip()

    def _onLeftScroll(self, evt):
        self.panx(-1)
        evt.Skip()

    def _onRightScroll(self, evt):
        self.panx(1)
        evt.Skip()

    def _onXZoomIn(self, evt):
        self.zoomx(1)
        evt.Skip()

    def _onXZoomOut(self, evt):
        self.zoomx(-1)
        evt.Skip()

    def _onUpScroll(self, evt):
        self.pany(1)
        evt.Skip()

    def _onDownScroll(self, evt):
        self.pany(-1)
        evt.Skip()

    def _onYZoomIn(self, evt):
        self.zoomy(1)
        evt.Skip()

    def _onYZoomOut(self, evt):
        self.zoomy(-1)
        evt.Skip()

    def _onMouseEnterButton(self, button):
        self._mouseOnButton = button

    def _onMouseLeaveButton(self, button):
        if self._mouseOnButton == button:
            self._mouseOnButton = None

    def _onMouseWheel(self, evt):
        if evt.GetWheelRotation() > 0:
            direction = 1
        else:
            direction = -1
        self.button_fn(direction)

    def _onRedraw(self, evt):
        self.canvas.draw()

    _onSave = NavigationToolbar2Wx.save

    def _onClose(self, evt):
        self.GetParent().Destroy()



class StatusBarWx(wx.StatusBar):
    """
    A status bar is added to _FigureFrame to allow measurements and the
    previously selected scroll function to be displayed as a user
    convenience.
    """
    def __init__(self, parent):
        wx.StatusBar.__init__(self, parent, -1)
        self.SetFieldsCount(2)
        self.SetStatusText("None", 1)
        #self.SetStatusText("Measurement: None", 2)
        #self.Reposition()

    def set_function(self, string):
        self.SetStatusText("%s" % string, 1)

    #def set_measurement(self, string):
    #    self.SetStatusText("Measurement: %s" % string, 2)


#< Additions for printing support: Matt Newville

class PrintoutWx(wx.Printout):
    """Simple wrapper around wx Printout class -- all the real work
    here is scaling the matplotlib canvas bitmap to the current
    printer's definition.
    """
    def __init__(self, canvas, width=5.5,margin=0.5, title='matplotlib'):
        wx.Printout.__init__(self,title=title)
        self.canvas = canvas
        # width, in inches of output figure (approximate)
        self.width  = width
        self.margin = margin

    def HasPage(self, page):
        #current only supports 1 page print
        return page == 1

    def GetPageInfo(self):
        return (1, 1, 1, 1)

    def OnPrintPage(self, page):
        self.canvas.draw()

        dc        = self.GetDC()
        (ppw,pph) = self.GetPPIPrinter()      # printer's pixels per in
        (pgw,pgh) = self.GetPageSizePixels()  # page size in pixels
        (dcw,dch) = dc.GetSize()
        (grw,grh) = self.canvas.GetSizeTuple()

        # save current figure dpi resolution and bg color,
        # so that we can temporarily set them to the dpi of
        # the printer, and the bg color to white
        bgcolor   = self.canvas.figure.get_facecolor()
        fig_dpi   = self.canvas.figure.dpi.get()

        # draw the bitmap, scaled appropriately
        vscale    = float(ppw) / fig_dpi

        # set figure resolution,bg color for printer
        self.canvas.figure.dpi.set(ppw)
        self.canvas.figure.set_facecolor('#FFFFFF')

        renderer  = RendererWx(self.canvas.bitmap, self.canvas.figure.dpi)
        self.canvas.figure.draw(renderer)
        self.canvas.bitmap.SetWidth(  int(self.canvas.bitmap.GetWidth() * vscale))
        self.canvas.bitmap.SetHeight( int(self.canvas.bitmap.GetHeight()* vscale))
        self.canvas.draw()

        # page may need additional scaling on preview
        page_scale = 1.0
        if self.IsPreview():   page_scale = float(dcw)/pgw

        # get margin in pixels = (margin in in) * (pixels/in)
        top_margin  = int(self.margin * pph * page_scale)
        left_margin = int(self.margin * ppw * page_scale)

        # set scale so that width of output is self.width inches
        # (assuming grw is size of graph in inches....)
        user_scale = (self.width * fig_dpi * page_scale)/float(grw)

        dc.SetDeviceOrigin(left_margin,top_margin)
        dc.SetUserScale(user_scale,user_scale)

        # this cute little number avoid API inconsistencies in wx
        try:
            dc.DrawBitmap(self.canvas.bitmap, 0, 0)
        except:
            try:
                dc.DrawBitmap(self.canvas.bitmap, (0, 0))
            except:
                pass

        # restore original figure  resolution
        self.canvas.figure.set_facecolor(bgcolor)
        self.canvas.figure.dpi.set(fig_dpi)
        self.canvas.draw()
        return True
#>

########################################################################
#
# Now just provide the standard names that backend.__init__ is expecting
#
########################################################################

Toolbar = NavigationToolbarWx
FigureManager = FigureManagerWx
