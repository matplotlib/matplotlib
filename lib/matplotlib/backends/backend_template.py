"""
This is a fully functional do nothing backend to provide a template to
backend writers.  It is fully functional in that you can select it as
a backend with

  import matplotlib
  matplotlib.use('Template')

and your matplotlib scripts will (should!) run without error, though
no output is produced.  This provides a nice starting point for
backend writers because you can selectively implement methods
(draw_rectangle, draw_lines, etc...) and slowly see your figure come
to life w/o having to have a full blown implementation before getting
any results.

Copy this to backend_xxx.py and replace all instances of 'template'
with 'xxx'.  Then implement the class methods and functions below, and
add 'xxx' to the switchyard in matplotlib/backends/__init__.py and
'xxx' to the _knownBackends dict in matplotlib/__init__.py and you're
off.  You can use your backend with

  import matplotlib
  matplotlib.use('xxx')
  from matplotlib.matlab import *
  plot([1,2,3])
  show()

The files that are most relevant to backend_writers are

  matplotlib/backends/backend_your_backend.py
  matplotlib/backend_bases.py
  matplotlib/backends/__init__.py
  matplotlib/__init__.py
  matplotlib/_matlab_helpers.py
  
Naming Conventions

  * classes MixedUpperCase

  * varables lowerUpper

  * functions underscore_separated

REQUIREMENTS

  matplotlib requires python2.2 and Numeric, and I don't yet want to
  make python2.3 a requirement.  I provide the Python Cookbook version
  of enumerate in cbook.py and define the constants True and False if
  version <=2.3.  Of course as a backend writer, you are free to make
  additional requirements, but the less required the better.

"""

import sys
from matplotlib._matlab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase, error_msg

from matplotlib.cbook import enumerate, True, False
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox



def error_msg_template(msg, *args):
    """
    Signal an error condition -- in a GUI, popup a error dialog
    """
    print >>sys.stderr, 'Error:', msg
    sys.exit()
    
class RendererTemplate(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """


    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return 100, 100

    def get_text_width_height(self, s, fontsize, ismath=False):
        """
        get the width and height in display coords of the string s
        with fontsize in points
        """
        return 1,1

                              
    def flipy(self):
        'return true if y small numbers are top for renderer'
        return True
    
    def points_to_pixels(self, points):
        """
        convert points to display units; unless your backend doesn't
        have dpi, eg, postscript, you need to overrride this function
        """
        return points  

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):    
        """
        Render the matplotlib.text.Text instance at x, y in window
        coords using GraphicsContext gc
        """
        pass
         
    
    def draw_arc(self, gcEdge, rgbFace, x, y, width, height, angle1, angle2):
        """
        Draw an arc centered at x,y with width and height and angles
        from 0.0 to 360.0.

        If rgbFace is not None, fill the rectangle with it.  gcEdge
        is a GraphicsContext instance
        """
        pass
    
    def draw_line(self, gc, x1, y1, x2, y2):
        """
        Draw a single line from x1,y1 to x2,y2
        """
        pass
    
    def draw_lines(self, gc, x, y):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        pass

    def draw_image(self, x, y, im, origin, bbox):
        """
        Draw the Image instance into the current axes; x is the
        distance in pixels from the left hand side of the canvas. y is
        the distance from the origin.  That is, if origin is upper, y
        is the distance from top.  If origin is lower, y is the
        distance from bottom

        origin is 'upper' or 'lower'

        bbox is a matplotlib.transforms.BBox instance for clipping, or
        None
        """
        pass
    
    def draw_polygon(self, gcEdge, rgbFace, points):
        """
        Draw a polygon.  points is a len vertices tuple, each element
        giving the x,y coords a vertex.

        If rgbFace is not None, fill the rectangle with it.  gcEdge
        is a GraphicsContext instance
        """  
        pass

    def draw_rectangle(self, gcEdge, rgbFace, x, y, width, height):
        """
        Draw a rectangle at lower left x,y with width and height.

        If rgbFace is not None, fill the rectangle with it.  gcEdge
        is a GraphicsContext instance
        """
        pass

    def draw_point(self, gc, x, y):
        """
        Draw a single point at x,y
        """
        pass


    def new_gc(self):
        """
        Return an instance of a GraphicsContextTemplate
        """
        return GraphicsContextTemplate()


    def points_to_pixels(self, points):
        """
        convert points to display units.  Many imaging systems assume
        some value for pixels per inch.  Eg, suppose yours is 96 and
        dpi = 300.  Then points to pixels is
        """
        return 96/72.0 * 300/72.0 * points

class GraphicsContextTemplate(GraphicsContextBase):
    """
    The graphics context provides the color, line styles, etc...  See
    the gtk and postscript backends for examples of mapping the
    graphics context attributes (cap styles, join styles, line widths,
    colors) to a particular backend.  In GTK this is done by wrapping
    a gtk.gdk.GC object and forwarding the appropriate calls to it
    using a dictionary mapping styles to gdk constants.  In
    Postscript, all the work is done by the renderer, mapping line
    styles to postscript calls.

    The base GraphicsContext stores colors as a RGB tuple on the unit
    interval, eg, (0.5, 0.0, 1.0).  You will probably need to map this
    to colors appropriate for your backend.  Eg, see the ColorManager
    class for the GTK backend.  If it's more appropriate to do the
    mapping at the renderer level (as in the postscript backend), you
    don't need to override any of the GC methods.  If it's more
    approritate to wrap an instance (as in the GTK backend) and do the
    mapping here, you'll need to override several of the setter
    methods.
    """
    pass
              

        
        
########################################################################
#    
# The following functions and classes are for matlab compatibility
# mode (matplotlib.matlab) and implement window/figure managers,
# etc...
#
########################################################################

def draw_if_interactive():
    """
    This should be overriden in a windowing environment if drawing
    should be done in interactive python mode
    """
    pass

def show():
    """
    This is usually the last line of a matlab script and tells the
    backend that it is time to draw.  In interactive mode, this may be
    a do nothing func.  See the GTK backend for an example of how to
    handle interactive versus batch mode
    """
    for manager in Gcf.get_all_fig_managers():
        manager.canvas.realize()


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasTemplate(thisFig)
    manager = FigureManagerTemplate(canvas, num)
    return manager


class FigureCanvasTemplate(FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance
    """

    def draw(self):
        """
        Draw the figure using the renderer
        """
        renderer = RendererTemplate()
        self.figure.draw(renderer)
        
    def print_figure(self, filename, dpi=150,
                     facecolor='w', edgecolor='w',
                     orientation='portrait'):

        """
        Render the figure to hardcopy.  Set the figure patch face and
        edge colors.  This is useful because some of the GUIs have a
        gray figure face color background and you'll probably want to
        override this on hardcopy
        """
        # set the new parameters
        origDPI = self.figure.dpi.get()
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()
        self.figure.dpi.set(dpi)
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)        

        renderer = RendererTemplate()
        self.figure.draw(renderer)
        # do something to save to hardcopy

        # restore the new params and redraw the screen if necessary
        self.figure.dpi.set(origDPI)
        self.figure.set_facecolor(origfacecolor)
        self.figure.set_edgecolor(origedgecolor)
        self.draw()
        
    def realize(self, *args):
        """
        This method will be called when the system is ready to draw,
        eg when a GUI window is realized
        """
        self._isRealized = True  
        self.draw()
    
class FigureManagerTemplate(FigureManagerBase):
    """
    Wrap everything up into a window for the matlab interface

    For non interactive backends, the base class does all the work
    """
    pass

########################################################################
#    
# Now just provide the standard names that backend.__init__ is expecting
# 
########################################################################


FigureManager = FigureManagerTemplate
error_msg = error_msg_template
         
