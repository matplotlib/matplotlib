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
    from wxPython.wx import *
except:
    print >>sys.stderr, "Matplotlib backend_wx requires wxPython be installed"
    sys.exit()    

wxapp = wxPySimpleApp()
wxapp.SetExitOnFrameDelete(True)


#!!! this is the call that is causing the exception swallowing !!!
#wxInitAllImageHandlers()

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
#    WxLogger = wxLogStderr()
#    sys.stderr = fake_stderr


import matplotlib
from matplotlib import verbose
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureCanvasBase, FigureManagerBase, error_msg, NavigationToolbar2, \
     MplEvent, cursors
from matplotlib._matlab_helpers import Gcf
from matplotlib.artist import Artist
from matplotlib.cbook import exception_to_str
from matplotlib.figure import Figure
from matplotlib.text import _process_text_args, Text

from matplotlib import rcParams

import wx
backend_version = wx.VERSION_STRING


# the true dots per inch on the screen; should be display dependent
# see http://groups.google.com/groups?q=screen+dpi+x11&hl=en&lr=&ie=UTF-8&oe=UTF-8&safe=off&selm=7077.26e81ad5%40swift.cs.tcd.ie&rnum=5 for some info about screen dpi
PIXELS_PER_INCH = 75

def error_msg_wx(msg, parent=None):
    """
    Signal an error condition -- in a GUI, popup a error dialog
    """
    dialog = wxMessageDialog(parent  = parent,
                             message = msg,
                             caption = 'Matplotlib backend_wx error',
                             style   = wxOK | wxCENTRE)
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
    #the plot. Under wxPython, the wxDC instance has a wxPen which
    #describes the colour and weight of any lines drawn, and a wxBrush
    #which describes the fill colour of any closed polygon.


    fontweights = {
        100          : wxLIGHT,
        200          : wxLIGHT,
        300          : wxLIGHT,
        400          : wxNORMAL,
        500          : wxNORMAL,
        600          : wxNORMAL,
        700          : wxBOLD,
        800          : wxBOLD,
        900          : wxBOLD,
        'ultralight' : wxLIGHT,
        'light'      : wxLIGHT,
        'normal'     : wxNORMAL,
        'medium'     : wxNORMAL,
        'semibold'   : wxNORMAL,
        'bold'       : wxBOLD,
        'heavy'      : wxBOLD,
        'ultrabold'  : wxBOLD,
        'black'      : wxBOLD
        }
    fontangles = {
        'italic'  : wxITALIC,
        'normal'  : wxNORMAL,
        'oblique' : wxSLANT }

    # wxPython allows for portable font styles, choosing them appropriately
    # for the target platform. Map some standard font names to the portable
    # styles
    # QUESTION: Is it be wise to agree standard fontnames across all backends?
    fontnames = { 'Sans'       : wxSWISS,
                  'Roman'      : wxROMAN,
                  'Script'     : wxSCRIPT,
                  'Decorative' : wxDECORATIVE,
                  'Modern'     : wxMODERN,
                  'Courier'    : wxMODERN,
                  'courier'    : wxMODERN }

    
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

    def get_text_width_height(self, s, prop, ismath):
        """
        get the width and height in display coords of the string s
        with FontPropertry prop
        """
        #return 1, 1
        if ismath: s = self.strip_math(s)

        if self.gc is None: gc = self.new_gc()
        font = self.get_wx_font(s, prop)
        self.gc.SetFont(font)
        w, h = self.gc.GetTextExtent(s)

        return w, h

    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return self.width, self.height

    
    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2):
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
            new_brush = wxBrush(wxColour(r,g,b), wxSOLID)
            gc.SetBrush(new_brush)
        else:
            gc.SetBrush(wxTRANSPARENT_BRUSH)
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
        gc.DrawLines([wxPoint(int(x[i]), self.height - int(y[i])) for i in range(len(x))])
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
            new_brush = wxBrush(wxColour(r,g,b), wxSOLID)
            gc.SetBrush(new_brush)
        else:
            gc.SetBrush(wxTRANSPARENT_BRUSH)
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
            new_brush = wxBrush(wxColour(r,g,b), wxSOLID)
            gc.SetBrush(new_brush)
        else:
            gc.SetBrush(wxTRANSPARENT_BRUSH)
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

        w, h = self.get_text_width_height(s, prop, ismath)
        x = int(x)
        y = int(y-h)

        if angle!=0:
            try: gc.DrawRotatedText(s, x, y, angle)
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
        wxFontname = self.fontnames.get(fontname, wxROMAN)
        wxFacename = '' # Empty => wxPython chooses based on wx_fontname

        # Font colour is determined by the active wxPen
        # TODO: It may be wise to cache font information
        size = self.points_to_pixels(fontprop.get_size_in_points())

        
        font = wxFont(int(size+0.5),             # Size
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
        for instantiating a wxColour."""
        r, g, b = rgb
        return (int(r * 255), int(g * 255), int(b * 255))



    def points_to_pixels(self, points):
        """
        convert point measures to pixes using dpi and the pixels per
        inch of the display
        """
        return points*(PIXELS_PER_INCH/72.0*self.dpi.get()/72.0)

class GraphicsContextWx(GraphicsContextBase, wxMemoryDC):
    """
    The graphics context provides the color, line styles, etc...
    
    In wxPython this is done by wrapping a wxDC object and forwarding the
    appropriate calls to it. Notice also that colour and line styles are
    mapped on the wxPen() member of the wxDC. This means that we have some
    rudimentary pen management here.
    
    The base GraphicsContext stores colors as a RGB tuple on the unit
    interval, eg, (0.5, 0.0, 1.0).  wxPython uses an int interval, but
    since wxPython colour management is rather simple, I have not chosen
    to implement a separate colour manager class.
    """
    _capd = { 'butt':       wxCAP_BUTT,
              'projecting': wxCAP_PROJECTING,
              'round':      wxCAP_ROUND }
    
    _joind = { 'bevel':     wxJOIN_BEVEL,
               'miter':     wxJOIN_MITER,
               'round':     wxJOIN_ROUND }
			   
    _dashd_wx = { 'solid':     wxSOLID,
                  'dashed':    wxSHORT_DASH,
                  'dashdot':   wxDOT_DASH,
                  'dotted':    wxDOT }
    _lastWxDC = None
    
    def __init__(self, bitmap, renderer):
        GraphicsContextBase.__init__(self)
        wxMemoryDC.__init__(self)
        #assert self.Ok(), "wxMemoryDC not OK to use"
        DEBUG_MSG("__init__()", 1, self)
        # Make sure (belt and braces!) that existing wxDC is not selected to
        # to a bitmap.
        if GraphicsContextWx._lastWxDC != None:

            GraphicsContextWx._lastWxDC.SelectObject(wxNullBitmap)

        self.SelectObject(bitmap)
        self.bitmap = bitmap
        self.SetPen(wxPen('BLACK', 1, wxSOLID))
        self._style = wxSOLID
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
            self.SelectObject(wxNullBitmap)
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
        brush = wxBrush(self.get_wxcolour(), wxSOLID)
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
        brush = wxBrush(self.get_wxcolour(), wxSOLID)
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
            self._style = wxLONG_DASH # Style not used elsewhere...
        
        # On MS Windows platform, only line width of 1 allowed for dash lines
        if wxPlatform == '__WXMSW__':
            self.set_linewidth(1)
            
        pen = self.GetPen()
        pen.SetStyle(self._style)
        self.SetPen(pen)
        self.unselect()
        
    def get_wxcolour(self):
        """return a wxColour from RGB format"""
        DEBUG_MSG("get_wx_color()", 1, self)
        r, g, b = self.get_rgb()
        r *= 255
        g *= 255
        b *= 255
        return wxColour(red=int(r), green=int(g), blue=int(b))
        

# Filetypes supported for saving files
_FILETYPES = {'.bmp': wxBITMAP_TYPE_BMP,
              '.jpg': wxBITMAP_TYPE_JPEG,
              '.png': wxBITMAP_TYPE_PNG,
              '.pcx': wxBITMAP_TYPE_PCX,
              '.tif': wxBITMAP_TYPE_TIF,
              '.xpm': wxBITMAP_TYPE_XPM}
              
class FigureCanvasWx(FigureCanvasBase, wxPanel):
    """
    The FigureCanvas contains the figure and does event handling.
    
    In the wxPython backend, it is derived from wxPanel, and (usually) lives
    inside a frame instantiated by a FigureManagerWx. The parent window probably
    implements a wxSizer to control the displayed control size - but we give a
    hint as to our preferred minimum size.
    """
    keyvald = {308 : 'control',
               306 : 'shift',
               307 : 'alt',
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

        self._key = None

        
        wxPanel.__init__(self, parent, id, size=wxSize(w, h))
        # Create the drawing bitmap
        self.bitmap = wxEmptyBitmap(w, h)
        DEBUG_MSG("__init__() - bitmap w:%d h:%d" % (w,h), 2, self)
        # TODO: Add support for 'point' inspection and plot navigation.
        self._isRealized = False
        self._isConfigured = False
        self._printQued = []		
        # Event handlers
        EVT_SIZE(self, self._onSize)
        EVT_PAINT(self, self._onPaint)
        EVT_KEY_DOWN(self, self._onKeyDown)
        EVT_KEY_UP(self, self._onKeyUp)
        EVT_RIGHT_DOWN(self, self._onRightButtonDown)
        EVT_RIGHT_UP(self, self._onRightButtonUp)
        EVT_MOUSEWHEEL(self, self._onMouseWheel)
        EVT_LEFT_DOWN(self, self._onLeftButtonDown)
        EVT_LEFT_UP(self, self._onLeftButtonUp)

        self.macros = {} # dict from wx id to seq of macros

        
    def mpl_disconnect(self, cid):
        for macro in self.macros.get(cid, []):
            macro(self, None)
        return None

        
    def mpl_connect(self, s, func):

        if s not in self.events:
            error_msg('Can only connect events of type "%s"\nDo not know how to handle "%s"' %(', '.join(self.events), s))    
            
        cid = wxNewId()

        def wrapper(event):
            thisEvent = MplEvent(s, self) 

            thisEvent.x = event.GetX()
            # flipy so y=0 is bottom of canvas
            thisEvent.y = self.figure.bbox.height() - event.GetY()

            if event.LeftDown(): button = 1
            elif event.MiddleDown(): button = 2
            elif event.RightDown(): button = 3
            else: button = None
            thisEvent.button = button
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
            event.Skip()
            return False  # return True blocks other connects


        if s=='button_press_event':
            EVT_LEFT_DOWN(self,  wrapper)
            EVT_RIGHT_DOWN(self, wrapper)
            self.macros[cid] = (EVT_LEFT_DOWN, EVT_RIGHT_DOWN)
        if s=='button_release_event':
            EVT_LEFT_UP(self,  wrapper)            
            EVT_RIGHT_UP(self, wrapper)
            self.macros[cid] = (EVT_LEFT_UP, EVT_RIGHT_UP)
        elif s=='motion_notify_event':
            EVT_MOTION(self, wrapper)
            self.macros[cid] = (EVT_MOTION,)
        return cid
               
    def draw(self):
        """
        Render the figure using RendererWx instance renderer, or using a
        previously defined renderer if none is specified.
        """
        DEBUG_MSG("draw()", 1, self)
        self.renderer = RendererWx(self.bitmap, self.figure.dpi)
        self.figure.draw(self.renderer)
        self.gui_repaint()

    def _get_imagesave_wildcards(self):
        'return the wildcard string for the filesave dialog'
        return "JPEG (*.jpg)|*.jpg|" \
               "PS (*.ps)|*.ps|"     \
               "EPS (*.eps)|*.eps|"  \
               "SVG (*.svg)|*.svg|"  \
               "BMP (*.bmp)|*.bmp|"  \
               "PCX (*.pcx)|*.pcx|"  \
               "PNG (*.png)|*.png|"  \
               "XPM (*.xpm)|*.xpm" 

    def gui_repaint(self):
        """
        Performs update of the displayed image on the GUI canvas
        
        MUST NOT be called during a Paint event
        """
        DEBUG_MSG("gui_repaint()", 1, self)
        drawDC=wxClientDC(self)
        drawDC.BeginDrawing()
        #drawDC.Clear()
        drawDC.DrawBitmap(self.bitmap, 0, 0)
        drawDC.EndDrawing()        
        
    def print_figure(self, filename, dpi=150,
                     facecolor='w', edgecolor='w',
                     orientation='portrait'):

        """
        Render the figure to hardcopy
        """

        DEBUG_MSG("print_figure()", 1, self)
        # Save state information, and set save DPI

        root, ext = os.path.splitext(filename)

        if ext.find('ps')>=0:
            # enable ps save from WX backend only import this if we
            # need it since it parse afm files on import
            from backend_ps import FigureCanvasPS

            DEBUG_MSG("print_figure() saving PS", 1, self)
            origDPI = self.figure.dpi.get()
            ps = self.switch_backends(FigureCanvasPS)
            ps.figure.dpi.set(72)

            ps.print_figure(filename, 72, facecolor, edgecolor)
            self.figure.dpi.set(origDPI)
            return
        elif ext.find('svg')>=0:
            # enable svg save from WX backend only import this if we
            # need it since it parse afm files on import
            from backend_svg import FigureCanvasSVG

            DEBUG_MSG("print_figure() saving SVG", 1, self)
            origDPI = self.figure.dpi.get()
            svg = self.switch_backends(FigureCanvasSVG)
            svg.figure.dpi.set(72)
            svg.print_figure(filename, 72, facecolor, edgecolor)
            self.figure.dpi.set(origDPI)                                       
            return

        if not self._isRealized:
            self._printQued.append((filename, dpi, facecolor, edgecolor))
            return

        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()
        origDPI      = self.figure.dpi.get()

        self.figure.dpi.set(dpi)        
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)


        origBitmap   = self.bitmap
		
        l,b,width,height = self.figure.bbox.get_bounds()
        width = int(math.ceil(width))
        height = int(math.ceil(height))

        # Now create a bitmap and draw onto it
        DEBUG_MSG('print_figure(): bitmap w:%d h:%d' % (width, height), 2, self)
        
        # Following performs the same function as realize(), but without
        # setting GUI attributes - so GUI draw() will render correctly
        self.bitmap = wxEmptyBitmap(width, height)
        renderer = RendererWx(self.bitmap, self.figure.dpi)

        gc = renderer.new_gc()

        self.figure.draw(renderer)

        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)
        self.figure.dpi.set(origDPI)

        # Now that we have rendered into the bitmap, save it
        # to the appropriate file type and clean up
        try:
            filetype = _FILETYPES[os.path.splitext(filename)[1]]
        except KeyError:
            filetype = wxBITMAP_TYPE_JPEG
            filename = filename + '.jpg'
        if not self.bitmap.SaveFile(filename, filetype):
            DEBUG_MSG('print_figure() file save error', 4, self)
            # note the error must be displayed here because trapping
            # the error on a call or print_figure may not work because
            # printing can be qued and called from realize
            error_msg_wx('Could not save figure to %s\n' % (filename))

        # Restore everything to normal
        self.bitmap = origBitmap


        self.draw()
        self.Refresh()
            
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
        self.draw()
        # Must use wxPaintDC during paint event

        l,b,w,h = self.figure.bbox.get_bounds()
        w = int(math.ceil(w))
        h = int(math.ceil(h))

        
        # I decoupled this from GraphicsContextWx so it would play
        # nice with wxagg
        memDC = wxMemoryDC()
        memDC.SelectObject(self.bitmap)
        memDC.SetPen(wxPen('BLACK', 1, wxSOLID))

        drawDC=wxPaintDC(self)
        
        drawDC.BeginDrawing()
        drawDC.Clear()
        drawDC.Blit(0, 0, w, h, memDC, 0, 0)
        #drawDC.DrawBitmap(self.bitmap, 0, 0)
        drawDC.EndDrawing() 
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
        self.bitmap = wxEmptyBitmap(self._width, self._height)

        if self._width <= 1 or self._height <= 1: return # Empty figure
        
        # Scale the displayed image (but don't update self.figsize)
        if not self._isConfigured:
            self._isConfigured = True

        dpival = self.figure.dpi.get()
        winch = self._width/dpival
        hinch = self._height/dpival
        self.figure.set_figsize_inches(winch, hinch)

        if self._isRealized:
            self.draw()
        evt.Skip()
        

    def _onKeyDown(self, evt):
        """Capture key press."""
        evt.Skip()
        keyval = evt.m_keyCode
        if self.keyvald.has_key(keyval):
            key = self.keyvald[keyval]
        elif keyval <256:
            key = chr(keyval)
        else:
            key = None
            
        if key: self._key = key.lower()
        else:   self._key = key
        
    def _onKeyUp(self, evt):
        """Release key."""
        evt.Skip()
        self._key = None

    def _onRightButtonDown(self, evt):
        """Start measuring on an axis."""
        #print 'left down', evt, dir(evt)
        evt.Skip()
        
    def _onRightButtonUp(self, evt):
        """End measuring on an axis."""
        evt.Skip()


    def _onLeftButtonDown(self, evt):
        """Start measuring on an axis."""

        evt.Skip()
        
    def _onLeftButtonUp(self, evt):
        """End measuring on an axis."""
        evt.Skip()
        
    def _onMouseWheel(self, evt):
        # TODO: implement mouse wheel handler
        pass


    

########################################################################
#    
# The following functions and classes are for matlab compatibility
# mode (matplotlib.matlab) and implement figure managers, etc...
#
########################################################################



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
        wxapp.MainLoop()
        show._needmain = False        
show._needmain = True

def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    # in order to expose the Figure constructor to the matlab
    # interface we need to create the figure here
    DEBUG_MSG("new_figure_manager()", 3, None)
    fig = Figure(*args, **kwargs)
    frame = FigureFrameWx(num, fig)
    figmgr = frame.get_figure_manager()
    if matplotlib.is_interactive():
        figmgr.canvas.realize()
        figmgr.frame.Show()

    return figmgr

class FigureFrameWx(wxFrame):
    def __init__(self, num, fig):
        # On non-Windows platform, explicitly set the position - fix
        # positioning bug on some Linux platforms
        if wxPlatform == '__WXMSW__':
            pos = wxDefaultPosition
        else:
            pos = wxPoint(20,20)
        wxFrame.__init__(self, parent=None, id=-1, pos=pos,
                          title="Figure %d" % num)
        DEBUG_MSG("__init__()", 1, self)
        self.num = num

        self.canvas = self.get_canvas(fig)
        self.SetStatusBar(StatusBarWx(self))
        self.sizer = wxBoxSizer(wxVERTICAL)
        self.sizer.Add(self.canvas, 1, wxTOP | wxLEFT | wxEXPAND)
        # By adding toolbar in sizer, we are able to put it at the bottom
        # of the frame - so appearance is closer to GTK version


        if matplotlib.rcParams['toolbar']=='classic':
            self.toolbar = NavigationToolbarWx(self.canvas, True)
        elif matplotlib.rcParams['toolbar']=='toolbar2':
            self.toolbar = NavigationToolbar2Wx(self.canvas)            
        else:
            self.toolbar = None
        

        if self.toolbar is not None:
            self.toolbar.Realize()
            if wxPlatform == '__WXMAC__':
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
                self.toolbar.SetSize(wxSize(fw, th))
                self.sizer.Add(self.toolbar, 0, wxLEFT | wxEXPAND)
        self.SetSizer(self.sizer)
        self.Fit()

        self.figmgr = FigureManagerWx(self.canvas, num, self)

        # Event handlers
        EVT_CLOSE(self, self._onClose)

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
    
class FigureManagerWx(FigureManagerBase):
    """
    This class contains the FigureCanvas and GUI frame
    
    It is instantiated by GcfWx whenever a new figure is created. GcfWx is
    responsible for managing multiple instances of FigureManagerWx.
    
    NB: FigureManagerBase is found in _matlab_helpers
    """
    def __init__(self, canvas, num, frame):
        DEBUG_MSG("__init__()", 1, self)
        FigureManagerBase.__init__(self, canvas, num)
        self.frame = frame
        self.window = frame
        self.tb = frame.GetToolBar()

    def add_subplot(self, *args, **kwargs):
        DEBUG_MSG("add_subplot()", 1, self)
        a = FigureManagerBase.add_subplot(self, *args, **kwargs)
        if self.tb is not None: self.tb.update()
        return a
        
    def add_axes(self, rect, **kwargs):
        DEBUG_MSG("add_axes()", 1, self)
        a = FigureManagerBase.add_axes(self, rect, **kwargs)
        if self.tb is not None: self.tb.update()
        return a
    
    def set_current_axes(self, a):
        DEBUG_MSG("set_current_axes()", 1, self)
        if a not in self.axes.values():
            error_msg_wx('Axes is not in current figure')
        FigureManagerBase.set_current_axes(self, a)
        
    def destroy(self, *args):
        DEBUG_MSG("destroy()", 1, self)
        self.frame.Destroy()
        import wx
        #wx.GetApp().ProcessIdle()
        wx.WakeUpIdle()

# Identifiers for toolbar controls - images_wx contains bitmaps for the images
# used in the controls. wxWindows does not provide any stock images, so I've
# 'stolen' those from GTK2, and transformed them into the appropriate format.
#import images_wx

_NTB_AXISMENU        = wxNewId()
_NTB_AXISMENU_BUTTON = wxNewId()
_NTB_X_PAN_LEFT      = wxNewId()
_NTB_X_PAN_RIGHT     = wxNewId()
_NTB_X_ZOOMIN        = wxNewId()
_NTB_X_ZOOMOUT       = wxNewId()
_NTB_Y_PAN_UP        = wxNewId()
_NTB_Y_PAN_DOWN      = wxNewId()
_NTB_Y_ZOOMIN        = wxNewId()
_NTB_Y_ZOOMOUT       = wxNewId()
_NTB_SAVE            = wxNewId()
_NTB_CLOSE           = wxNewId()

def _load_bitmap(filename):
    """
    Load a bitmap file from the backends/images subdirectory in which the
    matplotlib library is installed. The filename parameter should not
    contain any path information as this is determined automatically.
    
    Bitmaps should be in XPM format, and of size 16x16 (unless you change
    the code!). I have converted the stock GTK2 16x16 icons to XPM format.
    
    Returns a wxBitmap object
    """

    basedir = rcParams['datapath']

    bmpFilename = os.path.normpath(os.path.join(basedir, filename))
    if not os.path.exists(bmpFilename):
        verbose.report_error('Could not find bitmap file "%s"; dying'%bmpFilename)
        sys.exit()
    bmp = wxBitmap(bmpFilename, wxBITMAP_TYPE_XPM)
    return bmp

class MenuButtonWx(wxButton):
    """
    wxPython does not permit a menu to be incorporated directly into a toolbar.
    This class simulates the effect by associating a pop-up menu with a button
    in the toolbar, and managing this as though it were a menu.
    """
    def __init__(self, parent):

        wxButton.__init__(self, parent, _NTB_AXISMENU_BUTTON, "Axes:        ",
                          style=wxBU_EXACTFIT)
        self._toolbar = parent
        self._menu = wxMenu()
        self._axisId = []
        # First two menu items never change...
        self._allId = wxNewId()
        self._invertId = wxNewId()
        self._menu.Append(self._allId, "All", "Select all axes", False)
        self._menu.Append(self._invertId, "Invert", "Invert axes selected", False)
        self._menu.AppendSeparator()
        EVT_BUTTON(self, _NTB_AXISMENU_BUTTON, self._onMenuButton)
        EVT_MENU(self, self._allId, self._handleSelectAllAxes)
        EVT_MENU(self, self._invertId, self._handleInvertAxesSelected)
        
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
                menuId = wxNewId()
                self._axisId.append(menuId)
                self._menu.Append(menuId, "Axis %d" % i, "Select axis %d" % i, True)
                self._menu.Check(menuId, True)
                EVT_MENU(self, menuId, self._onMenuItemSelected)
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
    cursors.MOVE : wxCURSOR_HAND,    
    cursors.HAND : wxCURSOR_HAND,
    cursors.POINTER : wxCURSOR_ARROW,
    cursors.SELECT_REGION : wxCURSOR_CROSS,
    }

class NavigationToolbar2Wx(NavigationToolbar2, wxToolBar):
    def __init__(self, canvas):
        wxToolBar.__init__(self, canvas.GetParent(), -1)
        NavigationToolbar2.__init__(self, canvas)
        self.canvas = canvas
        self._idle = True
        
    def _init_toolbar(self):
        DEBUG_MSG("_init_toolbar", 1, self)

        self._parent = self.canvas.GetParent()
        _NTB2_HOME    = wxNewId()
        _NTB2_BACK    = wxNewId()
        _NTB2_FORWARD = wxNewId()
        _NTB2_PAN     = wxNewId()
        _NTB2_ZOOM    = wxNewId()
        _NTB2_SAVE    = wxNewId()        
        
        self.SetToolBitmapSize(wxSize(24,24))

        self.AddSimpleTool(_NTB2_HOME, _load_bitmap('home.xpm'),
                           '', 'Reset original view')
        self.AddSimpleTool(_NTB2_BACK, _load_bitmap('back.xpm'),
                           '', 'Back navigation view')
        self.AddSimpleTool(_NTB2_FORWARD, _load_bitmap('forward.xpm'),
                           '', 'Forward navigation view')
        # todo: get new bitmap
        self.AddSimpleTool(_NTB2_PAN, _load_bitmap('move.xpm'),
                           '', 'Pan with left, zoom with right')
        self.AddSimpleTool(_NTB2_ZOOM, _load_bitmap('zoom_to_rect.xpm'),
                           '', 'Zoom to rectangle')
        
        self.AddSeparator()
        self.AddSimpleTool(_NTB2_SAVE, _load_bitmap('filesave.xpm'),
                           '', 'Save plot contents to file')

        
        EVT_TOOL(self, _NTB2_HOME, self.home)
        EVT_TOOL(self, _NTB2_FORWARD, self.forward)
        EVT_TOOL(self, _NTB2_BACK, self.back)
        EVT_TOOL(self, _NTB2_ZOOM, self.zoom)
        EVT_TOOL(self, _NTB2_PAN, self.pan)
        EVT_TOOL(self, _NTB2_SAVE, self.save)

        self.Realize()


    def save(self, evt):
        # Fetch the required filename and file type.
        filetypes = self.canvas._get_imagesave_wildcards()
        dlg = wxFileDialog(self._parent, "Save to file", "", "", filetypes,
                           wxSAVE|wxOVERWRITE_PROMPT|wxCHANGE_DIR)
        if dlg.ShowModal() == wxID_OK:
            dirname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            DEBUG_MSG('Save file dir:%s name:%s' % (dirname, filename), 3, self)
            self.canvas.print_figure(os.path.join(dirname, filename))


    def set_cursor(self, cursor):
        cursor = wxStockCursor(cursord[cursor])
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
        dc = wxClientDC(canvas)
            
        # Set logical function to XOR for rubberbanding
        dc.SetLogicalFunction(wxXOR)
            
        # Set dc brush and pen
        # Here I set brush and pen to white and grey respectively
        # You can set it to your own choices
            
        # The brush setting is not really needed since we
        # dont do any filling of the dc. It is set just for 
        # the sake of completion.

        wbrush = wxBrush(wxColour(255,255,255), wxTRANSPARENT)
        wpen = wxPen(wxColour(200, 200, 200), 1, wxSOLID)
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
               

class NavigationToolbarWx(wxToolBar):
    def __init__(self, canvas, can_kill=False):
        """
        figure is the Figure instance that the toolboar controls

        win, if not None, is the wxWindow the Figure is embedded in
        """
        wxToolBar.__init__(self, canvas.GetParent(), -1)
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

        self.SetToolBitmapSize(wxSize(16,16))
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
        EVT_TOOL(self, _NTB_X_PAN_LEFT, self._onLeftScroll)
        EVT_TOOL(self, _NTB_X_PAN_RIGHT, self._onRightScroll)
        EVT_TOOL(self, _NTB_X_ZOOMIN, self._onXZoomIn)
        EVT_TOOL(self, _NTB_X_ZOOMOUT, self._onXZoomOut)
        EVT_TOOL(self, _NTB_Y_PAN_UP, self._onUpScroll)
        EVT_TOOL(self, _NTB_Y_PAN_DOWN, self._onDownScroll)
        EVT_TOOL(self, _NTB_Y_ZOOMIN, self._onYZoomIn)
        EVT_TOOL(self, _NTB_Y_ZOOMOUT, self._onYZoomOut)
        EVT_TOOL(self, _NTB_SAVE, self._onSave)
        EVT_TOOL_ENTER(self, self.GetId(), self._onEnterTool)
        if can_kill:
            EVT_TOOL(self, _NTB_CLOSE, self._onClose)
        EVT_MOUSEWHEEL(self, self._onMouseWheel)
        
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

    def _onSave(self, evt):
        # Fetch the required filename and file type.
        filetypes = self.canvas._get_imagesave_wildcards()
        dlg = wxFileDialog(self._parent, "Save to file", "", "", filetypes,
                           wxSAVE|wxOVERWRITE_PROMPT|wxCHANGE_DIR)
        if dlg.ShowModal() == wxID_OK:
            dirname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            DEBUG_MSG('Save file dir:%s name:%s' % (dirname, filename), 3, self)
            self.canvas.print_figure(os.path.join(dirname, filename))
        
    def _onClose(self, evt):
        self.GetParent().Destroy()

        

class StatusBarWx(wxStatusBar):
    """
    A status bar is added to _FigureFrame to allow measurements and the
    previously selected scroll function to be displayed as a user
    convenience.
    """
    def __init__(self, parent):
        wxStatusBar.__init__(self, parent, -1)
        self.SetFieldsCount(3)
        self.SetStatusText("Function: None", 1)
        self.SetStatusText("Measurement: None", 2)
        #self.Reposition()
        
    def set_function(self, string):
        self.SetStatusText("Function: %s" % string, 1)
        
    def set_measurement(self, string):
        self.SetStatusText("Measurement: %s" % string, 2)

########################################################################
#    
# Now just provide the standard names that backend.__init__ is expecting
# 
########################################################################

Toolbar = NavigationToolbarWx
FigureManager = FigureManagerWx
error_msg = error_msg_wx

