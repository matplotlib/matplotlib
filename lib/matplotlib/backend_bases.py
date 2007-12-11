"""
Abstract base classes define the primitives that renderers and
graphics contexts must implement to serve as a matplotlib backend
"""

from __future__ import division
import os, sys, warnings

import numpy as npy
import matplotlib.numerix.npyma as ma
import matplotlib.cbook as cbook
import matplotlib.colors as colors
import matplotlib.transforms as transforms
import matplotlib.widgets as widgets
from matplotlib import rcParams

class RendererBase:
    """An abstract base class to handle drawing/rendering operations
    """

    def __init__(self):
        self._texmanager = None

    def open_group(self, s):
        """open a grouping element with label s
        Is only currently used by backend_svg
        """
        pass

    def close_group(self, s):
        """close a grouping element with label s
        Is only currently used by backend_svg
        """
        pass


    def draw_arc(self, gc, rgbFace, x, y, width, height, angle1, angle2,
                 rotation):
        """
        Draw an arc using GraphicsContext instance gcEdge, centered at x,y,
        with width and height and angles from 0.0 to 360.0
        0 degrees is at 3-o'clock
        positive angles are anti-clockwise
        draw rotated 'rotation' degrees anti-clockwise about x,y

        If the color rgbFace is not None, fill the arc with it.
        """
        raise NotImplementedError

    def get_image_magnification(self):
        """
        Get the factor by which to magnify images passed to draw_image.
        Allows a backend to have images at a different resolution to other
        artists.
        """
        return 1.0

    def draw_image(self, x, y, im, bbox):
        """
        Draw the Image instance into the current axes; x is the
        distance in pixels from the left hand side of the canvas. y is
        the distance from the origin.  That is, if origin is upper, y
        is the distance from top.  If origin is lower, y is the
        distance from bottom

        bbox is a matplotlib.transforms.BBox instance for clipping, or
        None
        """
        raise NotImplementedError

    def option_image_nocomposite(self):
        """
        overwrite this method for renderers that do not necessarily
        want to rescale and composite raster images. (like SVG)
        """
        return False

    def _draw_markers(self, bgc, path, rgbFace, x, y, trans):
        """
        This method is currently underscore hidden because the
        draw_markers method is being used as a sentinel for newstyle
        backend drawing

        path - a matplotlib.agg.path_storage instance

        Draw the marker specified in path with graphics context gc at
        each of the locations in arrays x and y.  trans is a
        matplotlib.transforms.Transformation instance used to
        transform x and y to display coords.  It consists of an
        optional nonlinear component and an affine.  You can access
        these two components as

        if transform.need_nonlinear():
          x,y = transform.nonlinear_only_numerix(x, y)
        # the a,b,c,d,tx,ty affine which transforms x and y
        vec6 = transform.as_vec6_val()
        ...backend dependent affine...
        """
        pass

    def draw_line_collection(self, segments, transform, clipbox,
                             colors, linewidths, linestyle, antialiaseds,
                             offsets, transOffset):
        """
        This is a function for optimized line drawing. If you need to draw
        many line segments with similar properties, it is faster to avoid the
        overhead of all the object creation etc. The lack of total
        configurability is compensated for with efficiency. Hence we don't use
        a GC and many of the line props it supports. See
        matplotlib.collections for more details.

        segments is a sequence of ( line0, line1, line2), where linen =
        is an Mx2 array with columns x, y.  Each line can be a
        different length

        transform is used to Transform the lines

        clipbox is a  xmin, ymin, width, height clip rect

        colors is a tuple of RGBA tuples

        linewidths is a tuple of linewidths
        *** really should be called 'dashes' not 'linestyle', since
        we call gc.set_dashes() not gc.set_linestyle() ***

        linestyle is an (offset, onoffseq) tuple or None,None for solid

        antialiseds is a tuple of ones or zeros indicating whether the
        segment should be aa or not

        offsets, if not None, is an Nx2 array of x,y offsets to
        translate the lines by after transform is used to transform
        the offset coords

        This function could be overridden in the backend to possibly implement
        faster drawing, but it is already much faster than using draw_lines()
        by itself.
        """

        newstyle = getattr(self, 'draw_markers', None) is not None
        identity = transforms.identity_transform()
        gc = self.new_gc()
        if clipbox is not None:
            gc.set_clip_rectangle(clipbox.get_bounds())
        gc.set_dashes(*linestyle)

        Nc        = len(colors)
        Nlw       = len(linewidths)
        Naa       = len(antialiaseds)
        Nsegments = len(segments)

        usingOffsets = offsets is not None
        Noffsets  = 0
        if usingOffsets:
            Noffsets = offsets.shape[0]
            offsets = transOffset.numerix_xy(offsets)

        for i in xrange(max(Noffsets, Nsegments)):
            color = colors[i % Nc]
            rgb   = color[0], color[1], color[2]
            alpha = color[-1]

            gc.set_foreground(rgb, isRGB=True)
            gc.set_alpha( alpha )
            gc.set_linewidth( linewidths[i % Nlw] )
            gc.set_antialiased( antialiaseds[i % Naa] )
            seg = segments[i % Nsegments]
            if not len(seg): continue
            xy = transform.numerix_xy(seg)
            if usingOffsets:
                xy = xy + offsets[i % Noffsets]

            if newstyle: self.draw_lines(gc, xy[:,0], xy[:,1], identity)
            else: self.draw_lines(gc, xy[:,0], xy[:,1])

    def draw_line(self, gc, x1, y1, x2, y2):
        """
        Draw a single line from x1,y1 to x2,y2
        """
        raise NotImplementedError

    def draw_lines(self, gc, x, y, transform=None):
        """
        x and y are equal length arrays, draw lines connecting each
        point in x, y
        """
        raise NotImplementedError

    def draw_point(self, gc, x, y):
        """
        Draw a single point at x,y
        Where 'point' is a device-unit point (or pixel), not a matplotlib point
        """
        raise NotImplementedError

    def draw_quad_mesh(self, meshWidth, meshHeight, colors,
                        xCoords, yCoords, clipbox,
                        transform, offsets, transOffset, showedges):
        """
        Draw a quadrilateral mesh
        See documentation in QuadMesh class in collections.py for details
        """
        # print "draw_quad_mesh not found, using function in backend_bases"
        verts = npy.zeros(((meshWidth * meshHeight), 4, 2), npy.float32)
        indices = npy.arange((meshWidth + 1) * (meshHeight + 1))
        indices = npy.compress((indices + 1) % (meshWidth + 1), indices)
        indices = indices[:(meshWidth * meshHeight)]
        verts[:, 0, 0] = npy.take(xCoords, indices)
        verts[:, 0, 1] = npy.take(yCoords, indices)
        verts[:, 1, 0] = npy.take(xCoords, (indices + 1))
        verts[:, 1, 1] = npy.take(yCoords, (indices + 1))
        verts[:, 2, 0] = npy.take(xCoords, (indices + meshWidth + 2))
        verts[:, 2, 1] = npy.take(yCoords, (indices + meshWidth + 2))
        verts[:, 3, 0] = npy.take(xCoords, (indices + meshWidth + 1))
        verts[:, 3, 1] = npy.take(yCoords, (indices + meshWidth + 1))
        if (showedges):
            edgecolors = colors
        else:
            edgecolors = (0, 0, 0, 0),
        self.draw_poly_collection(verts, transform,
                                clipbox, colors, edgecolors,
                                (0.25,), (0,), offsets, transOffset)

    def draw_poly_collection(
        self, verts, transform, clipbox, facecolors, edgecolors,
        linewidths, antialiaseds, offsets, transOffset):
        """
        Draw a polygon collection

        verts are a sequence of polygon vectors, where each polygon
        vector is a sequence of x,y tuples of vertices

        facecolors and edgecolors are a sequence of RGBA tuples
        linewidths are a sequence of linewidths
        antialiaseds are a sequence of 0,1 integers whether to use aa

        If a linewidth is zero or an edgecolor alpha is zero, the
        line will be omitted; similarly, the fill will be omitted
        if the facecolor alpha is zero.
        """
        ## line and/or fill OK
        Nface = len(facecolors)
        Nedge = len(edgecolors)
        Nlw = len(linewidths)
        Naa = len(antialiaseds)

        usingOffsets = offsets is not None
        Noffsets = 0
        Nverts = len(verts)
        if usingOffsets:
            Noffsets = len(offsets)

        N = max(Noffsets, Nverts)

        gc = self.new_gc()
        if clipbox is not None:
            gc.set_clip_rectangle(clipbox.get_bounds())


        for i in xrange(N):
            polyverts = ma.filled(verts[i % Nverts], npy.nan)
            if npy.any(npy.isnan(polyverts)):
                continue
            linewidth = linewidths[i % Nlw]
            rf,gf,bf,af = facecolors[i % Nface]
            re,ge,be,ae = edgecolors[i % Nedge]
            if af==0:
                if ae==0 or linewidth == 0:
                    continue
                rgbFace = None
                alpha = ae
            else:
                rgbFace = rf,gf,bf
            if ae==0:
                alpha = af
                gc.set_linewidth(0)
            else:
                # the draw_poly interface can't handle separate alphas for
                # edge and face so we'll just use the maximum
                alpha = max(af,ae)
                gc.set_foreground( (re,ge,be), isRGB=True)
                gc.set_linewidth( linewidths[i % Nlw] )
                #print 'verts', zip(thisxverts, thisyverts)

            gc.set_antialiased( antialiaseds[i % Naa] )  # Used for fill only?
            gc.set_alpha( alpha )
            tverts = transform.seq_xy_tups(polyverts)
            if usingOffsets:
                xo,yo = transOffset.xy_tup(offsets[i % Noffsets])
                tverts = [(x+xo,y+yo) for x,y in tverts]

            self.draw_polygon(gc, rgbFace, tverts)

    def draw_polygon(self, gc, rgbFace, points):
        """
        Draw a polygon using the GraphicsContext instance gc.
        points is a len vertices tuple, each element
        giving the x,y coords a vertex

        If the color rgbFace is not None, fill the polygon with it
        """
        raise NotImplementedError

    def draw_rectangle(self, gcEdge, rgbFace, x, y, width, height):
        """
        Draw a non-filled rectangle using the GraphicsContext instance gcEdge,
        with lower left at x,y with width and height.

        If rgbFace is not None, fill the rectangle with it.
        """
        raise NotImplementedError

    def draw_regpoly_collection(
        self, clipbox, offsets, transOffset, verts, sizes,
        facecolors, edgecolors, linewidths, antialiaseds):
        """
        Draw a regular poly collection

        offsets   - is a sequence is x,y tuples
        transOffset - maps this to display coords

        verts - are the vertices of the regular polygon at the origin

        sizes are the area of the circle that circumscribes the
        polygon in points^2

        facecolors and edgecolors are a sequence of RGBA tuples
        linewidths are a sequence of linewidths
        antialiaseds are a sequence of 0,1 integers whether to use aa
        """
        ## line and/or fill OK
        gc = self.new_gc()
        if clipbox is not None:
            gc.set_clip_rectangle(clipbox.get_bounds())

        xverts, yverts = zip(*verts)
        xverts = npy.asarray(xverts)
        yverts = npy.asarray(yverts)

        Nface  = len(facecolors)
        Nedge  = len(edgecolors)
        Nlw    = len(linewidths)
        Naa    = len(antialiaseds)
        Nsizes = len(sizes)

        for i, loc in enumerate(offsets):
            xo,yo = transOffset.xy_tup(loc)
            #print 'xo, yo', loc, (xo, yo)
            scale = sizes[i % Nsizes]

            thisxverts = scale*xverts + xo
            thisyverts = scale*yverts + yo
            #print 'xverts', xverts

            linewidth = linewidths[i % Nlw]
            rf,gf,bf,af = facecolors[i % Nface]
            re,ge,be,ae = edgecolors[i % Nedge]
            if af==0:
                if ae==0 or linewidth == 0:
                    continue
                rgbFace = None
                alpha = ae
            else:
                rgbFace = rf,gf,bf
            if ae==0:
                alpha = af
                gc.set_linewidth(0)
            else:
                # the draw_poly interface can't handle separate alphas for
                # edge and face so we'll just use the maximum
                alpha = max(af,ae)
                gc.set_foreground( (re,ge,be), isRGB=True)
                gc.set_linewidth( linewidths[i % Nlw] )
                #print 'verts', zip(thisxverts, thisyverts)

            gc.set_antialiased( antialiaseds[i % Naa] )  # Used for fill only?
            gc.set_alpha( alpha )
            #print 'verts', zip(thisxverts, thisyverts)
            self.draw_polygon(gc, rgbFace, zip(thisxverts, thisyverts))


    def draw_tex(self, gc, x, y, s, prop, angle, ismath='TeX!'):
        raise NotImplementedError

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):
        """
        Draw the text.Text instance s at x,y (display coords) with font
        properties instance prop at angle in degrees, using GraphicsContext gc

        **backend implementers note**

        When you are trying to determine if you have gotten your bounding box
        right (which is what enables the text layout/alignment to work
        properly), it helps to change the line in text.py

                  if 0: bbox_artist(self, renderer)

        to if 1, and then the actual bounding box will be blotted along with
        your text.
        """
        raise NotImplementedError

    def flipy(self):
        """return true if y small numbers are top for renderer
        Is used for drawing text (text.py) and images (image.py) only
        """
        return True

    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return 1, 1

    def get_texmanager(self):
        if self._texmanager is None:
            from matplotlib.texmanager import TexManager
            self._texmanager = TexManager()
        return self._texmanager

    def get_text_extent(self, text): # is not used, can be removed?
        """
        Get the text extent in window coords
        """
        return transforms.lbwh_to_bbox(0,0,1,1)  # your values here

    def get_text_width_height_descent(self, s, prop, ismath):
        """
        get the width and height, and the offset from the bottom to the
        baseline (descent), in display coords of the string s with
        FontPropertry prop
        """
        return 1,1,1

    def new_gc(self):
        """
        Return an instance of a GraphicsContextBase
        """
        return GraphicsContextBase()

    def points_to_pixels(self, points):
        """
        Convert points to display units
        points - a float or a numpy array of float
        return points converted to pixels

        You need to override this function (unless your backend doesn't have a
        dpi, eg, postscript or svg).
        Some imaging systems assume some value for pixels per inch.
        points to pixels = points * pixels_per_inch/72.0 * dpi/72.0
        """
        return points

    def strip_math(self, s):
        return cbook.strip_math(s)


class GraphicsContextBase:
    """An abstract base class that provides color, line styles, etc...
    """

    # a mapping from dash styles to suggested offset, dash pairs
    dashd = {
        'solid'   : (None, None),
        'dashed'  : (0, (6.0, 6.0)),
        'dashdot' : (0, (3.0, 5.0, 1.0, 5.0)),
        'dotted'  : (0, (1.0, 3.0)),
              }

    def __init__(self):
        self._alpha = 1.0
        self._antialiased = 1  # use 0,1 not True, False for extension code
        self._capstyle = 'butt'
        self._cliprect = None
        self._clippath = None
        self._dashes = None, None
        self._joinstyle = 'miter'
        self._linestyle = 'solid'
        self._linewidth = 1
        self._rgb = (0.0, 0.0, 0.0)
        self._hatch = None

    def copy_properties(self, gc):
        'Copy properties from gc to self'
        self._alpha = gc._alpha
        self._antialiased = gc._antialiased
        self._capstyle = gc._capstyle
        self._cliprect = gc._cliprect
        self._clippath = gc._clippath
        self._dashes = gc._dashes
        self._joinstyle = gc._joinstyle
        self._linestyle = gc._linestyle
        self._linewidth = gc._linewidth
        self._rgb = gc._rgb
        self._hatch = gc._hatch

    def get_alpha(self):
        """
        Return the alpha value used for blending - not supported on
        all backends
        """
        return self._alpha

    def get_antialiased(self):
        "Return true if the object should try to do antialiased rendering"
        return self._antialiased

    def get_capstyle(self):
        """
        Return the capstyle as a string in ('butt', 'round', 'projecting')
        """
        return self._capstyle

    def get_clip_rectangle(self):
        """
        Return the clip rectangle as (left, bottom, width, height)
        """
        return self._cliprect

    def get_clip_path(self):
        """
        Return the clip path
        """
        return self._clippath

    def get_dashes(self):
        """
        Return the dash information as an offset dashlist tuple The
        dash list is a even size list that gives the ink on, ink off
        in pixels.  See p107 of to postscript BLUEBOOK for more info

        Default value is None
        """
        return self._dashes

    def get_joinstyle(self):
        """
        Return the line join style as one of ('miter', 'round', 'bevel')
        """
        return self._joinstyle

    def get_linestyle(self, style):
        """
        Return the linestyle: one of ('solid', 'dashed', 'dashdot',
        'dotted').
        """
        return self._linestyle

    def get_linewidth(self):
        """
        Return the line width in points as a scalar
        """
        return self._linewidth

    def get_rgb(self):
        """
        returns a tuple of three floats from 0-1.  color can be a
        matlab format string, a html hex color string, or a rgb tuple
        """
        return self._rgb

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on
        all backends
        """
        self._alpha = alpha

    def set_antialiased(self, b):
        """
        True if object should be drawn with antialiased rendering
        """

        # use 0, 1 to make life easier on extension code trying to read the gc
        if b: self._antialiased = 1
        else: self._antialiased = 0

    def set_capstyle(self, cs):
        """
        Set the capstyle as a string in ('butt', 'round', 'projecting')
        """
        if cs in ('butt', 'round', 'projecting'):
            self._capstyle = cs
        else:
            raise ValueError('Unrecognized cap style.  Found %s' % cs)

    def set_clip_rectangle(self, rectangle):
        """
        Set the clip rectangle with sequence (left, bottom, width, height)
        """
        self._cliprect = rectangle

    def set_clip_path(self, path):
        """
        Set the clip path
        """
        self._clippath = path

    def set_dashes(self, dash_offset, dash_list):
        """
        Set the dash style for the gc.
        dash_offset is the offset (usually 0).
        dash_list specifies the on-off sequence as points
        (None, None) specifies a solid line
        """
        self._dashes = dash_offset, dash_list

    def set_foreground(self, fg, isRGB=False):
        """
        Set the foreground color.  fg can be a matlab format string, a
        html hex color string, an rgb unit tuple, or a float between 0
        and 1.  In the latter case, grayscale is used.

        The GraphicsContext converts colors to rgb internally.  If you
        know the color is rgb already, you can set isRGB to True to
        avoid the performace hit of the conversion
        """
        if isRGB:
            self._rgb = fg
        else:
            self._rgb = colors.colorConverter.to_rgb(fg)

    def set_graylevel(self, frac):
        """
        Set the foreground color to be a gray level with frac frac
        """
        self._rgb = (frac, frac, frac)

    def set_joinstyle(self, js):
        """
        Set the join style to be one of ('miter', 'round', 'bevel')
        """
        if js in ('miter', 'round', 'bevel'):
            self._joinstyle = js
        else:
            raise ValueError('Unrecognized join style.  Found %s' % js)

    def set_linewidth(self, w):
        """
        Set the linewidth in points
        """
        self._linewidth = w

    def set_linestyle(self, style):
        """
        Set the linestyle to be one of ('solid', 'dashed', 'dashdot',
        'dotted').
        """
        try:
            offset, dashes = self.dashd[style]
        except:
            raise ValueError('Unrecognized linestyle: %s' % style)
        self._linestyle = style
        self.set_dashes(offset, dashes)

    def set_hatch(self, hatch):
        """
        Sets the hatch style for filling
        """
        self._hatch = hatch

    def get_hatch(self):
        """
        Gets the current hatch style
        """
        return self._hatch

class Event:
    """
    A matplotlib event.  Attach additional attributes as defined in
    FigureCanvas.connect.  The following attributes are defined and
    shown with their default values
    name                # the event name
    canvas              # the FigureCanvas instance generating the event

    """
    def __init__(self, name, canvas,guiEvent=None):
        self.name = name
        self.canvas = canvas
        self.guiEvent = guiEvent


class DrawEvent(Event):
    """
    An event triggered by a draw operation on the canvas

    Attributes are
      name
      canvas
      renderer - the Renderer instance
    """
    def __init__(self, name, canvas, renderer):
        Event.__init__(self, name, canvas)
        self.renderer = renderer

class ResizeEvent(Event):
    """
    An event triggered by a canvas resize

    Attributes are
      name
      canvas
      width   # width of the canvas in pixels
      height  # height of the canvas in pixels
    """
    def __init__(self, name, canvas):
        Event.__init__(self, name, canvas)
        self.width, self.height = canvas.get_width_height()

class LocationEvent(Event):
    """
    A event that has a screen location

    The following additional attributes are defined and shown with
    their default values

    x      = None       # x position - pixels from left of canvas
    y      = None       # y position - pixels from bottom of canvas
    inaxes = None       # the Axes instance if mouse us over axes
    xdata  = None       # x coord of mouse in data coords
    ydata  = None       # y coord of mouse in data coords

    """
    x      = None       # x position - pixels from left of canvas
    y      = None       # y position - pixels from right of canvas
    button = None       # button pressed None, 1, 2, 3
    inaxes = None       # the Axes instance if mouse us over axes
    xdata  = None       # x coord of mouse in data coords
    ydata  = None       # y coord of mouse in data coords

    def __init__(self, name, canvas, x, y,guiEvent=None):
        """
        x, y in figure coords, 0,0 = bottom, left
        button pressed None, 1, 2, 3
        """
        Event.__init__(self, name, canvas,guiEvent=guiEvent)
        self.x = x
        self.y = y

        if x is None or y is None:
            # cannot check if event was in axes if no x,y info
            return

        # Find all axes containing the mouse
        axes_list = [a for a in self.canvas.figure.get_axes() if a.in_axes(x, y)]

        if len(axes_list) == 0: # None found
            self.inaxes = None
            return
        elif (len(axes_list) > 1): # Overlap, get the highest zorder
            axCmp = lambda _x,_y: cmp(_x.zorder, _y.zorder)
            axes_list.sort(axCmp)
            self.inaxes = axes_list[-1] # Use the highest zorder
        else: # Just found one hit
            self.inaxes = axes_list[0]

        try: xdata, ydata = self.inaxes.transData.inverse_xy_tup((x, y))
        except ValueError:
            self.xdata  = None
            self.ydata  = None
        else:
            self.xdata  = xdata
            self.ydata  = ydata

class MouseEvent(LocationEvent):
    """
    A mouse event (button_press_event, button_release_event, scroll_event,
    motion_notify_event).

    The following attributes are defined and shown with their default
    values

    x      = None       # x position - pixels from left of canvas
    y      = None       # y position - pixels from bottom of canvas
    button = None       # button pressed None, 1, 2, 3, 'up', 'down'
                          (up and down are used for scroll events)
    key    = None       # the key pressed: None, chr(range(255), shift, win, or control
    inaxes = None       # the Axes instance if mouse us over axes
    xdata  = None       # x coord of mouse in data coords
    ydata  = None       # y coord of mouse in data coords

    """
    x      = None       # x position - pixels from left of canvas
    y      = None       # y position - pixels from right of canvas
    button = None       # button pressed None, 1, 2, 3
    inaxes = None       # the Axes instance if mouse us over axes
    xdata  = None       # x coord of mouse in data coords
    ydata  = None       # y coord of mouse in data coords

    def __init__(self, name, canvas, x, y, button=None, key=None,
                 guiEvent=None):
        """
        x, y in figure coords, 0,0 = bottom, left
        button pressed None, 1, 2, 3
        """
        LocationEvent.__init__(self, name, canvas, x, y, guiEvent=guiEvent)
        self.button = button
        self.key = key

class PickEvent(Event):
    """
    a pick event, fired when the user picks a location on the canvas
    sufficiently close to an artist.

    Attrs: all the Event attrs plus
    mouseevent : the MouseEvent that generated the pick
    artist    : the artist picked

    extra class dependent attrs -- eg a Line2D pick may define
    different extra attributes than a PatchCollection pick event
    """
    def __init__(self, name, canvas, mouseevent, artist, guiEvent=None, **kwargs):
        Event.__init__(self, name, canvas, guiEvent)
        self.mouseevent = mouseevent
        self.artist = artist
        self.__dict__.update(kwargs)


class KeyEvent(LocationEvent):
    """
    A key event (key press, key release).

    Attach additional attributes as defined in
    FigureCanvas.connect.

    The following attributes are defined and shown with their default
    values

    x      = None       # x position - pixels from left of canvas
    y      = None       # y position - pixels from bottom of canvas
    key    = None       # the key pressed: None, chr(range(255), shift, win, or control
    inaxes = None       # the Axes instance if mouse us over axes
    xdata  = None       # x coord of mouse in data coords
    ydata  = None       # y coord of mouse in data coords

    This interface may change slightly when better support for
    modifier keys is included
    """
    def __init__(self, name, canvas, key, x=0, y=0, guiEvent=None):
        LocationEvent.__init__(self, name, canvas, x, y, guiEvent=guiEvent)
        self.key = key



class FigureCanvasBase:
    """
    The canvas the figure renders into.

    Public attribute

      figure - A Figure instance

      """
    events = (
        'resize_event',
        'draw_event',
        'key_press_event',
        'key_release_event',
        'button_press_event',
        'button_release_event',
        'scroll_event',
        'motion_notify_event',
        'pick_event',
        )


    def __init__(self, figure):
        figure.set_canvas(self)
        self.figure = figure
        # a dictionary from event name to a dictionary that maps cid->func
        self.callbacks = cbook.CallbackRegistry(self.events)
        self.widgetlock = widgets.LockDraw()
        self._button     = None  # the button pressed
        self._key        = None  # the key pressed
        self._lastx, self._lasty = None, None
        self.button_pick_id = self.mpl_connect('button_press_event',self.pick)
        self.scroll_pick_id = self.mpl_connect('scroll_event',self.pick)

        if False:
            ## highlight the artists that are hit
            self.mpl_connect('motion_notify_event',self.onHilite)
            ## delete the artists that are clicked on
            #self.mpl_disconnect(self.button_pick_id)
            #self.mpl_connect('button_press_event',self.onRemove)

    def onRemove(self, ev):
        """
        Mouse event processor which removes the top artist
        under the cursor.  Connect this to the mouse_press_event
        using canvas.mpl_connect('mouse_press_event',canvas.onRemove)
        """
        def sort_artists(artists):
            # This depends on stable sort and artists returned
            # from get_children in z order.
            L = [ (h.zorder, h) for h in artists ]
            L.sort()
            return [ h for zorder, h in L ]

        # Find the top artist under the cursor
        under = sort_artists(self.figure.hitlist(ev))
        h = None
        if under: h = under[-1]

        # Try deleting that artist, or its parent if you
        # can't delete the artist
        while h:
            print "Removing",h
            if h.remove(): 
                self.draw_idle()
                break
            parent = None
            for p in under:
                if hasattr(p,'getchildren') and h in p.get_children():
                    parent = p
                    break
            h = parent
        
    def onHilite(self, ev):
        """
        Mouse event processor which highlights the artists
        under the cursor.  Connect this to the motion_notify_event
        using canvas.mpl_connect('motion_notify_event',canvas.onHilite)
        """
        if not hasattr(self,'_active'): self._active = dict()

        under = self.figure.hitlist(ev)
        enter = [a for a in under if a not in self._active]
        leave = [a for a in self._active if a not in under]
        print "within:"," ".join([str(x) for x in under])
        #print "entering:",[str(a) for a in enter]
        #print "leaving:",[str(a) for a in leave]
        # On leave restore the captured colour
        for a in leave:
            if hasattr(a,'get_color'):
                a.set_color(self._active[a])
            elif hasattr(a,'get_edgecolor'):
                a.set_edgecolor(self._active[a][0])
                a.set_facecolor(self._active[a][1])
            del self._active[a]
        # On enter, capture the color and repaint the artist
        # with the highlight colour.  Capturing colour has to
        # be done first in case the parent recolouring affects
        # the child.
        for a in enter:
            if hasattr(a,'get_color'):
                self._active[a] = a.get_color()
            elif hasattr(a,'get_edgecolor'):
                self._active[a] = (a.get_edgecolor(),a.get_facecolor())
            else: self._active[a] = None
        for a in enter:
            if hasattr(a,'get_color'):
                a.set_color('red')
            elif hasattr(a,'get_edgecolor'):
                a.set_edgecolor('red')
                a.set_facecolor('lightblue')
            else: self._active[a] = None
        self.draw_idle()

    def pick(self, mouseevent):
        if not self.widgetlock.locked():
            self.figure.pick(mouseevent)

    def blit(self, bbox=None):
        """
        blit the canvas in bbox (default entire canvas)
        """
        pass

    def resize(self, w, h):
        """
        set the canvas size in pixels
        """
        pass

    def draw_event(self, renderer):
        s = 'draw_event'
        event = DrawEvent(s, self, renderer)
        self.callbacks.process(s, event)

    def resize_event(self):
        s = 'resize_event'
        event = ResizeEvent(s, self)
        self.callbacks.process(s, event)

    def key_press_event(self, key, guiEvent=None):
        self._key = key
        s = 'key_press_event'
        event = KeyEvent(s, self, key, self._lastx, self._lasty, guiEvent=guiEvent)
        self.callbacks.process(s, event)

    def key_release_event(self, key, guiEvent=None):
        s = 'key_release_event'
        event = KeyEvent(s, self, key, self._lastx, self._lasty, guiEvent=guiEvent)
        self.callbacks.process(s, event)
        self._key = None

    def pick_event(self, mouseevent, artist, **kwargs):
        """
        This method will be called by artists who are picked and will
        fire off PickEvent callbacks registered listeners
        """
        s = 'pick_event'
        event = PickEvent(s, self, mouseevent, artist, **kwargs)
        self.callbacks.process(s, event)

    def scroll_event(self, x, y, button, guiEvent=None):
        """
        Backend derived classes should call this function on any
        scroll wheel event.  x,y are the canvas coords: 0,0 is lower,
        left.  button and key are as defined in MouseEvent
        """
        self._button = button
        s = 'scroll_event'
        mouseevent = MouseEvent(s, self, x, y, button, self._key, guiEvent=guiEvent)
        self.callbacks.process(s, mouseevent)


    def button_press_event(self, x, y, button, guiEvent=None):
        """
        Backend derived classes should call this function on any mouse
        button press.  x,y are the canvas coords: 0,0 is lower, left.
        button and key are as defined in MouseEvent
        """
        self._button = button
        s = 'button_press_event'
        mouseevent = MouseEvent(s, self, x, y, button, self._key, guiEvent=guiEvent)
        self.callbacks.process(s, mouseevent)

    def button_release_event(self, x, y, button, guiEvent=None):
        """
        Backend derived classes should call this function on any mouse
        button release.  x,y are the canvas coords: 0,0 is lower, left.
        button and key are as defined in MouseEvent
        """
        s = 'button_release_event'
        event = MouseEvent(s, self, x, y, button, self._key, guiEvent=guiEvent)
        self.callbacks.process(s, event)
        self._button = None

    def motion_notify_event(self, x, y, guiEvent=None):
        """
        Backend derived classes should call this function on any
        motion-notify-event. x,y are the canvas coords: 0,0 is lower, left.
        button and key are as defined in MouseEvent
        """
        self._lastx, self._lasty = x, y
        s = 'motion_notify_event'
        event = MouseEvent(s, self, x, y, self._button, self._key,
                           guiEvent=guiEvent)
        self.callbacks.process(s, event)

    def draw(self, *args, **kwargs):
        """
        Render the figure
        """
        pass

    def draw_idle(self, *args, **kwargs):
        """
        draw only if idle; defaults to draw but backends can overrride
        """
        self.draw(*args, **kwargs)

    def draw_cursor(self, event):
        """
        Draw a cursor in the event.axes if inaxes is not None.  Use
        native GUI drawing for efficiency if possible
        """
        pass

    def get_width_height(self):
        """return the figure width and height in points or pixels
        (depending on the backend), truncated to integers"""
        return int(self.figure.bbox.width()), int(self.figure.bbox.height())

    filetypes = {
        'emf': 'Enhanced Metafile',
        'eps': 'Encapsulated Postscript',
        'pdf': 'Portable Document Format',
        'png': 'Portable Network Graphics',
        'ps' : 'Postscript',
        'raw': 'Raw RGBA bitmap',
        'rgba': 'Raw RGBA bitmap',
        'svg': 'Scalable Vector Graphics',
        'svgz': 'Scalable Vector Graphics'
        }

    # All of these print_* functions do a lazy import because
    #  a) otherwise we'd have cyclical imports, since all of these
    #     classes inherit from FigureCanvasBase
    #  b) so we don't import a bunch of stuff the user may never use
    
    def print_emf(self, *args, **kwargs):
        from backends.backend_emf import FigureCanvasEMF # lazy import
        emf = self.switch_backends(FigureCanvasEMF)
        return emf.print_emf(*args, **kwargs)

    def print_eps(self, *args, **kwargs):
        from backends.backend_ps import FigureCanvasPS # lazy import
        ps = self.switch_backends(FigureCanvasPS)
        return ps.print_eps(*args, **kwargs)
    
    def print_pdf(self, *args, **kwargs):
        from backends.backend_pdf import FigureCanvasPdf # lazy import
        pdf = self.switch_backends(FigureCanvasPdf)
        return pdf.print_pdf(*args, **kwargs)

    def print_png(self, *args, **kwargs):
        from backends.backend_agg import FigureCanvasAgg # lazy import
        agg = self.switch_backends(FigureCanvasAgg)
        return agg.print_png(*args, **kwargs)
    
    def print_ps(self, *args, **kwargs):
        from backends.backend_ps import FigureCanvasPS # lazy import
        ps = self.switch_backends(FigureCanvasPS)
        return ps.print_ps(*args, **kwargs)

    def print_raw(self, *args, **kwargs):
        from backends.backend_agg import FigureCanvasAgg # lazy import
        agg = self.switch_backends(FigureCanvasAgg)
        return agg.print_raw(*args, **kwargs)
    print_bmp = print_rgb = print_raw

    def print_svg(self, *args, **kwargs):
        from backends.backend_svg import FigureCanvasSVG # lazy import
        svg = self.switch_backends(FigureCanvasSVG)
        return svg.print_svg(*args, **kwargs)
    
    def print_svgz(self, *args, **kwargs):
        from backends.backend_svg import FigureCanvasSVG # lazy import
        svg = self.switch_backends(FigureCanvasSVG)
        return svg.print_svgz(*args, **kwargs)
    
    def get_supported_filetypes(self):
        return self.filetypes

    def get_supported_filetypes_grouped(self):
        groupings = {}
        for ext, name in self.filetypes.items():
            groupings.setdefault(name, []).append(ext)
            groupings[name].sort()
        return groupings
    
    def print_figure(self, filename, dpi=None, facecolor='w', edgecolor='w',
                     orientation='portrait', format=None, **kwargs):
        """
        Render the figure to hardcopy. Set the figure patch face and edge
        colors.  This is useful because some of the GUIs have a gray figure
        face color background and you'll probably want to override this on
        hardcopy.

        filename    - can also be a file object on image backends
        orientation - only currently applies to PostScript printing.
        dpi - the dots per inch to save the figure in; if None, use savefig.dpi
        facecolor - the facecolor of the figure
        edgecolor - the edgecolor of the figure
        orientation - 'landscape' | 'portrait' (not supported on all backends)
        format - when set, forcibly set the file format to save to
        """
        if format is None:
            if cbook.is_string_like(filename):
                format = os.path.splitext(filename)[1][1:]
            if format is None or format == '':
                format = self.get_default_filetype()
                if cbook.is_string_like(filename):
                    filename = filename.rstrip('.') + '.' + format
        format = format.lower()

        method_name = 'print_%s' % format
        if (format not in self.filetypes or
            not hasattr(self, method_name)):
            formats = self.filetypes.keys()
            formats.sort()
            raise ValueError(
                'Format "%s" is not supported.\n'
                'Supported formats: '
                '%s.' % (format, ', '.join(formats)))

        if dpi is None:
            dpi = rcParams['savefig.dpi']
            
        origDPI = self.figure.dpi.get()
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()

        self.figure.dpi.set(dpi)
        self.figure.set_facecolor(facecolor)
        self.figure.set_edgecolor(edgecolor)

        try:
            result = getattr(self, method_name)(
                filename,
                dpi=dpi,
                facecolor=facecolor,
                edgecolor=edgecolor,
                orientation=orientation,
                **kwargs)
        finally:
            self.figure.dpi.set(origDPI)
            self.figure.set_facecolor(origfacecolor)
            self.figure.set_edgecolor(origedgecolor)
            self.figure.set_canvas(self)
            self.figure.canvas.draw()
            
        return result

    def get_default_filetype(self):
        raise NotImplementedError
    
    def set_window_title(self, title):
        """
        Set the title text of the window containing the figure.  Note that
        this has no effect if there is no window (eg, a PS backend).
        """
        if hasattr(self, "manager"):
            self.manager.set_window_title(title)

    def switch_backends(self, FigureCanvasClass):
        """
        instantiate an instance of FigureCanvasClass

        This is used for backend switching, eg, to instantiate a
        FigureCanvasPS from a FigureCanvasGTK.  Note, deep copying is
        not done, so any changes to one of the instances (eg, setting
        figure size or line props), will be reflected in the other
        """
        newCanvas = FigureCanvasClass(self.figure)
        return newCanvas

    def mpl_connect(self, s, func):
        """
        Connect event with string s to func.  The signature of func is

          def func(event)

        where event is a MplEvent.  The following events are recognized

        'resize_event',
        'draw_event',
        'key_press_event',
        'key_release_event',
        'button_press_event',
        'button_release_event',
        'scroll_event',
        'motion_notify_event',
        'pick_event',


        For the three events above, if the mouse is over the axes,
        the variable event.inaxes will be set to the axes it is over,
        and additionally, the variables event.xdata and event.ydata
        will be defined.  This is the mouse location in data coords.
        See backend_bases.MplEvent.

        return value is a connection id that can be used with
        mpl_disconnect """

        return self.callbacks.connect(s, func)

    def mpl_disconnect(self, cid):
        """
        disconnect callback id cid
        """
        return self.callbacks.disconnect(cid)


class FigureManagerBase:
    """
    Helper class for matlab mode, wraps everything up into a neat bundle

    Public attibutes
    canvas - A FigureCanvas instance
    num    - The figure number
    """
    def __init__(self, canvas, num):
        self.canvas = canvas
        canvas.manager = self  # store a pointer to parent
        self.num = num

        self.canvas.mpl_connect('key_press_event', self.key_press)

    def destroy(self):
        pass

    def full_screen_toggle (self):
        pass

    def resize(self, w, h):
        'For gui backends: resize window in pixels'
        pass

    def key_press(self, event):

        # these bindings happen whether you are over an axes or not
        #if event.key == 'q':
        #    self.destroy() # how cruel to have to destroy oneself!
        #    return

        if event.key == 'f':
            self.full_screen_toggle()

        if event.inaxes is None:
            return

        # the mouse has to be over an axes to trigger these
        if event.key == 'g':
            event.inaxes.grid()
            self.canvas.draw()
        elif event.key == 'l':
            event.inaxes.toggle_log_lineary()
            self.canvas.draw()
        elif event.key is not None and (event.key.isdigit() and event.key!='0') or event.key=='a':
            # 'a' enables all axes
            if event.key!='a':
                n=int(event.key)-1
            for i, a in enumerate(self.canvas.figure.get_axes()):
                if event.x is not None and event.y is not None and a.in_axes(event.x, event.y):
                    if event.key=='a':
                        a.set_navigate(True)
                    else:
                        a.set_navigate(i==n)


    def show_popup(self, msg):
        """
        Display message in a popup -- GUI only
        """
        pass

    def set_window_title(self, title):
        """
        Set the title text of the window containing the figure.  Note that
        this has no effect if there is no window (eg, a PS backend).
        """
        pass

# cursors
class Cursors:  #namespace
    HAND, POINTER, SELECT_REGION, MOVE = range(4)
cursors = Cursors()



class NavigationToolbar2:
    """
    Base class for the navigation cursor, version 2

    backends must implement a canvas that handles connections for
    'button_press_event' and 'button_release_event'.  See
    FigureCanvas.connect for more information


    They must also define

     * save_figure - save the current figure

     * set_cursor - if you want the pointer icon to change

     * _init_toolbar - create your toolbar widget

     * draw_rubberband (optional) : draw the zoom to rect
       "rubberband" rectangle

    * press : (optional) whenever a mouse button is pressed, you'll be
       notified with the event

    * release : (optional) whenever a mouse button is released,
       you'll be notified with the event

    * dynamic_update (optional) dynamically update the window while
      navigating

    * set_message (optional) - display message

    * set_history_buttons (optional) - you can change the history
       back / forward buttons to indicate disabled / enabled state.

    That's it, we'll do the rest!
    """

    def __init__(self, canvas):
        self.canvas = canvas
        canvas.toolbar = self
        # a dict from axes index to a list of view limits
        self._views = cbook.Stack()
        self._positions = cbook.Stack()  # stack of subplot positions
        self._xypress = None  # the  location and axis info at the time of the press
        self._idPress = None
        self._idRelease = None
        self._active = None
        self._lastCursor = None
        self._init_toolbar()
        self._idDrag=self.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self._button_pressed = None # determined by the button pressed at start

        self.mode = ''  # a mode string for the status bar
        self.set_history_buttons()

    def set_message(self, s):
        'display a message on toolbar or in status bar'
        pass

    def back(self, *args):
        'move back up the view lim stack'
        self._views.back()
        self._positions.back()
        self.set_history_buttons()
        self._update_view()

    def dynamic_update(self):
        pass

    def draw_rubberband(self, event, x0, y0, x1, y1):
        'draw a rectangle rubberband to indicate zoom limits'
        pass

    def forward(self, *args):
        'move forward in the view lim stack'
        self._views.forward()
        self._positions.forward()
        self.set_history_buttons()
        self._update_view()

    def home(self, *args):
        'restore the original view'
        self._views.home()
        self._positions.home()
        self.set_history_buttons()
        self._update_view()

    def _init_toolbar(self):
        """
        This is where you actually build the GUI widgets (called by
        __init__).  The icons home.xpm, back.xpm, forward.xpm,
        hand.xpm, zoom_to_rect.xpm and filesave.xpm are standard
        across backends (there are ppm versions in CVS also).

        You just need to set the callbacks

        home         : self.home
        back         : self.back
        forward      : self.forward
        hand         : self.pan
        zoom_to_rect : self.zoom
        filesave     : self.save_figure

        You only need to define the last one - the others are in the base
        class implementation.

        """
        raise NotImplementedError

    def mouse_move(self, event):
        #print 'mouse_move', event.button

        if not event.inaxes or not self._active:
            if self._lastCursor != cursors.POINTER:
                self.set_cursor(cursors.POINTER)
                self._lastCursor = cursors.POINTER
        else:
            if self._active=='ZOOM':
                if self._lastCursor != cursors.SELECT_REGION:
                    self.set_cursor(cursors.SELECT_REGION)
                    self._lastCursor = cursors.SELECT_REGION
                if self._xypress:
                    x, y = event.x, event.y
                    lastx, lasty, a, ind, lim, trans= self._xypress[0]
                    self.draw_rubberband(event, x, y, lastx, lasty)
            elif (self._active=='PAN' and
                  self._lastCursor != cursors.MOVE):
                self.set_cursor(cursors.MOVE)

                self._lastCursor = cursors.MOVE

        if event.inaxes and event.inaxes.get_navigate():

            try: s = event.inaxes.format_coord(event.xdata, event.ydata)
            except ValueError: pass
            except OverflowError: pass
            else:
                if len(self.mode):
                    self.set_message('%s : %s' % (self.mode, s))
                else:
                    self.set_message(s)
        else: self.set_message(self.mode)

    def pan(self,*args):
        'Activate the pan/zoom tool. pan with left button, zoom with right'
        # set the pointer icon and button press funcs to the
        # appropriate callbacks

        if self._active == 'PAN':
            self._active = None
        else:
            self._active = 'PAN'
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''

        if self._active:
            self._idPress = self.canvas.mpl_connect(
                'button_press_event', self.press_pan)
            self._idRelease = self.canvas.mpl_connect(
                'button_release_event', self.release_pan)
            self.mode = 'pan/zoom mode'
            self.canvas.widgetlock(self)
        else:
            self.canvas.widgetlock.release(self)

        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self._active)

        self.set_message(self.mode)

    def press(self, event):
        'this will be called whenver a mouse button is pressed'
        pass

    def press_pan(self, event):
        'the press mouse button in pan/zoom mode callback'

        if event.button == 1:
            self._button_pressed=1
        elif  event.button == 3:
            self._button_pressed=3
        else:
            self._button_pressed=None
            return

        x, y = event.x, event.y

        # push the current view to define home if stack is empty
        if self._views.empty(): self.push_current()

        self._xypress=[]
        for i, a in enumerate(self.canvas.figure.get_axes()):
            if x is not None and y is not None and a.in_axes(x, y) and a.get_navigate():
                xmin, xmax = a.get_xlim()
                ymin, ymax = a.get_ylim()
                lim = xmin, xmax, ymin, ymax
                self._xypress.append((x, y, a, i, lim,a.transData.deepcopy()))
                self.canvas.mpl_disconnect(self._idDrag)
                self._idDrag=self.canvas.mpl_connect('motion_notify_event', self.drag_pan)

        self.press(event)

    def press_zoom(self, event):
        'the press mouse button in zoom to rect mode callback'
        if event.button == 1:
            self._button_pressed=1
        elif  event.button == 3:
            self._button_pressed=3
        else:
            self._button_pressed=None
            return

        x, y = event.x, event.y

        # push the current view to define home if stack is empty
        if self._views.empty(): self.push_current()

        self._xypress=[]
        for i, a in enumerate(self.canvas.figure.get_axes()):
            if x is not None and y is not None and a.in_axes(x, y) and a.get_navigate():
                xmin, xmax = a.get_xlim()
                ymin, ymax = a.get_ylim()
                lim = xmin, xmax, ymin, ymax
                self._xypress.append(( x, y, a, i, lim, a.transData.deepcopy() ))

        self.press(event)

    def push_current(self):
        'push the current view limits and position onto the stack'
        lims = []; pos = []
        for a in self.canvas.figure.get_axes():
            xmin, xmax = a.get_xlim()
            ymin, ymax = a.get_ylim()
            lims.append( (xmin, xmax, ymin, ymax) )
            # Store both the original and modified positions
            pos.append( (
                    tuple( a.get_position(True) ),
                    tuple( a.get_position() ) ) )
        self._views.push(lims)
        self._positions.push(pos)
        self.set_history_buttons()



    def release(self, event):
        'this will be called whenever mouse button is released'
        pass

    def release_pan(self, event):
        'the release mouse button callback in pan/zoom mode'
        self.canvas.mpl_disconnect(self._idDrag)
        self._idDrag=self.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        if not self._xypress: return
        self._xypress = None
        self._button_pressed=None
        self.push_current()
        self.release(event)
        self.draw()

    def drag_pan(self, event):
        'the drag callback in pan/zoom mode'

        def format_deltas(event,dx,dy):
            if event.key=='control':
                if(abs(dx)>abs(dy)):
                    dy = dx
                else:
                    dx = dy
            elif event.key=='x':
                dy = 0
            elif event.key=='y':
                dx = 0
            elif event.key=='shift':
                if 2*abs(dx) < abs(dy):
                    dx=0
                elif 2*abs(dy) < abs(dx):
                    dy=0
                elif(abs(dx)>abs(dy)):
                    dy=dy/abs(dy)*abs(dx)
                else:
                    dx=dx/abs(dx)*abs(dy)
            return (dx,dy)

        for cur_xypress in self._xypress:
            lastx, lasty, a, ind, lim, trans = cur_xypress
            xmin, xmax, ymin, ymax = lim
            #safer to use the recorded button at the press than current button:
            #multiple button can get pressed during motion...
            if self._button_pressed==1:
                lastx, lasty = trans.inverse_xy_tup( (lastx, lasty) )
                x, y = trans.inverse_xy_tup( (event.x, event.y) )
                if a.get_xscale()=='log':
                    dx=1-lastx/x
                else:
                    dx=x-lastx
                if a.get_yscale()=='log':
                    dy=1-lasty/y
                else:
                    dy=y-lasty

                dx,dy=format_deltas(event,dx,dy)

                if a.get_xscale()=='log':
                    xmin *= 1-dx
                    xmax *= 1-dx
                else:
                    xmin -= dx
                    xmax -= dx
                if a.get_yscale()=='log':
                    ymin *= 1-dy
                    ymax *= 1-dy
                else:
                    ymin -= dy
                    ymax -= dy
            elif self._button_pressed==3:
                try:
                    dx=(lastx-event.x)/float(a.bbox.width())
                    dy=(lasty-event.y)/float(a.bbox.height())
                    dx,dy=format_deltas(event,dx,dy)
                    if a.get_aspect() != 'auto':
                        dx = 0.5*(dx + dy)
                        dy = dx
                    alphax = pow(10.0,dx)
                    alphay = pow(10.0,dy)#use logscaling, avoid singularities and smother scaling...
                    lastx, lasty = trans.inverse_xy_tup( (lastx, lasty) )
                    if a.get_xscale()=='log':
                        xmin = lastx*(xmin/lastx)**alphax
                        xmax = lastx*(xmax/lastx)**alphax
                    else:
                        xmin = lastx+alphax*(xmin-lastx)
                        xmax = lastx+alphax*(xmax-lastx)
                    if a.get_yscale()=='log':
                        ymin = lasty*(ymin/lasty)**alphay
                        ymax = lasty*(ymax/lasty)**alphay
                    else:
                        ymin = lasty+alphay*(ymin-lasty)
                        ymax = lasty+alphay*(ymax-lasty)
                except OverflowError:
                    warnings.warn('Overflow while panning')
                    return
            a.set_xlim(xmin, xmax)
            a.set_ylim(ymin, ymax)

        self.dynamic_update()

    def release_zoom(self, event):
        'the release mouse button callback in zoom to rect mode'
        if not self._xypress: return

        for cur_xypress in self._xypress:
            x, y = event.x, event.y
            lastx, lasty, a, ind, lim, trans = cur_xypress
            # ignore singular clicks - 5 pixels is a threshold
            if abs(x-lastx)<5 or abs(y-lasty)<5:
                self._xypress = None
                self.release(event)
                self.draw()
                return

            xmin, ymin, xmax, ymax = lim

            # zoom to rect
            lastx, lasty = a.transData.inverse_xy_tup( (lastx, lasty) )
            x, y = a.transData.inverse_xy_tup( (x, y) )
            Xmin,Xmax=a.get_xlim()
            Ymin,Ymax=a.get_ylim()

            if Xmin < Xmax:
                if x<lastx:  xmin, xmax = x, lastx
                else: xmin, xmax = lastx, x
                if xmin < Xmin: xmin=Xmin
                if xmax > Xmax: xmax=Xmax
            else:
                if x>lastx:  xmin, xmax = x, lastx
                else: xmin, xmax = lastx, x
                if xmin > Xmin: xmin=Xmin
                if xmax < Xmax: xmax=Xmax

            if Ymin < Ymax:
                if y<lasty:  ymin, ymax = y, lasty
                else: ymin, ymax = lasty, y
                if ymin < Ymin: ymin=Ymin
                if ymax > Ymax: ymax=Ymax
            else:
                if y>lasty:  ymin, ymax = y, lasty
                else: ymin, ymax = lasty, y
                if ymin > Ymin: ymin=Ymin
                if ymax < Ymax: ymax=Ymax

            if self._button_pressed == 1:
                a.set_xlim((xmin, xmax))
                a.set_ylim((ymin, ymax))
            elif self._button_pressed == 3:
                if a.get_xscale()=='log':
                    alpha=npy.log(Xmax/Xmin)/npy.log(xmax/xmin)
                    x1=pow(Xmin/xmin,alpha)*Xmin
                    x2=pow(Xmax/xmin,alpha)*Xmin
                else:
                    alpha=(Xmax-Xmin)/(xmax-xmin)
                    x1=alpha*(Xmin-xmin)+Xmin
                    x2=alpha*(Xmax-xmin)+Xmin
                if a.get_yscale()=='log':
                    alpha=npy.log(Ymax/Ymin)/npy.log(ymax/ymin)
                    y1=pow(Ymin/ymin,alpha)*Ymin
                    y2=pow(Ymax/ymin,alpha)*Ymin
                else:
                    alpha=(Ymax-Ymin)/(ymax-ymin)
                    y1=alpha*(Ymin-ymin)+Ymin
                    y2=alpha*(Ymax-ymin)+Ymin
                a.set_xlim((x1, x2))
                a.set_ylim((y1, y2))

        self.draw()
        self._xypress = None
        self._button_pressed = None

        self.push_current()
        self.release(event)

    def draw(self):
        'redraw the canvases, update the locators'
        for a in self.canvas.figure.get_axes():
            xaxis = getattr(a, 'xaxis', None)
            yaxis = getattr(a, 'yaxis', None)
            locators = []
            if xaxis is not None:
                locators.append(xaxis.get_major_locator())
                locators.append(xaxis.get_minor_locator())
            if yaxis is not None:
                locators.append(yaxis.get_major_locator())
                locators.append(yaxis.get_minor_locator())

            for loc in locators:
                loc.refresh()
        self.canvas.draw()



    def _update_view(self):
        '''update the viewlim and position from the view and
        position stack for each axes
        '''

        lims = self._views()
        if lims is None:  return
        pos = self._positions()
        if pos is None: return
        for i, a in enumerate(self.canvas.figure.get_axes()):
            xmin, xmax, ymin, ymax = lims[i]
            a.set_xlim((xmin, xmax))
            a.set_ylim((ymin, ymax))
            # Restore both the original and modified positions
            a.set_position( pos[i][0], 'original' )
            a.set_position( pos[i][1], 'active' )

        self.draw()


    def save_figure(self, *args):
        'save the current figure'
        raise NotImplementedError

    def set_cursor(self, cursor):
        """
        Set the current cursor to one of the backend_bases.Cursors
        enums values
        """
        pass

    def update(self):
        'reset the axes stack'
        self._views.clear()
        self._positions.clear()
        self.set_history_buttons()

    def zoom(self, *args):
        'activate zoom to rect mode'
        if self._active == 'ZOOM':
            self._active = None
        else:
            self._active = 'ZOOM'

        if self._idPress is not None:
            self._idPress=self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease=self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''

        if  self._active:
            self._idPress = self.canvas.mpl_connect('button_press_event', self.press_zoom)
            self._idRelease = self.canvas.mpl_connect('button_release_event', self.release_zoom)
            self.mode = 'Zoom to rect mode'
            self.canvas.widgetlock(self)
        else:
            self.canvas.widgetlock.release(self)

        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self._active)

        self.set_message(self.mode)


    def set_history_buttons(self):
        'enable or disable back/forward button'
        pass


