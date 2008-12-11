"""
Abstract base classes define the primitives that renderers and
graphics contexts must implement to serve as a matplotlib backend

:class:`RendererBase`
    An abstract base class to handle drawing/rendering operations.

:class:`FigureCanvasBase`
    The abstraction layer that separates the
    :class:`matplotlib.figure.Figure` from the backend specific
    details like a user interface drawing area

:class:`GraphicsContextBase`
    An abstract base class that provides color, line styles, etc...

:class:`Event`
    The base class for all of the matplotlib event
    handling.  Derived classes suh as :class:`KeyEvent` and
    :class:`MouseEvent` store the meta data like keys and buttons
    pressed, x and y locations in pixel and
    :class:`~matplotlib.axes.Axes` coordinates.

"""

from __future__ import division
import os, warnings, time

import numpy as np
import matplotlib.cbook as cbook
import matplotlib.colors as colors
import matplotlib.transforms as transforms
import matplotlib.widgets as widgets
from matplotlib import rcParams

class RendererBase:
    """An abstract base class to handle drawing/rendering operations.

    The following methods *must* be implemented in the backend:

    * :meth:`draw_path`
    * :meth:`draw_image`
    * :meth:`draw_text`
    * :meth:`get_text_width_height_descent`

    The following methods *should* be implemented in the backend for
    optimization reasons:

    * :meth:`draw_markers`
    * :meth:`draw_path_collection`
    * :meth:`draw_quad_mesh`
    """
    def __init__(self):
        self._texmanager = None

    def open_group(self, s):
        """
        Open a grouping element with label *s*. Is only currently used by
        :mod:`~matplotlib.backends.backend_svg`
        """
        pass

    def close_group(self, s):
        """
        Close a grouping element with label *s*
        Is only currently used by :mod:`~matplotlib.backends.backend_svg`
        """
        pass

    def draw_path(self, gc, path, transform, rgbFace=None):
        """
        Draws a :class:`~matplotlib.path.Path` instance using the
        given affine transform.
        """
        raise NotImplementedError

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        """
        Draws a marker at each of the vertices in path.  This includes
        all vertices, including control points on curves.  To avoid
        that behavior, those vertices should be removed before calling
        this function.

        *gc*
            the :class:`GraphicsContextBase` instance

        *marker_trans*
            is an affine transform applied to the marker.

        *trans*
             is an affine transform applied to the path.

        This provides a fallback implementation of draw_markers that
        makes multiple calls to :meth:`draw_path`.  Some backends may
        want to override this method in order to draw the marker only
        once and reuse it multiple times.
        """
        tpath = trans.transform_path(path)
        for vertices, codes in tpath.iter_segments():
            if len(vertices):
                x,y = vertices[-2:]
                self.draw_path(gc, marker_path,
                               marker_trans + transforms.Affine2D().translate(x, y),
                               rgbFace)

    def draw_path_collection(self, master_transform, cliprect, clippath,
                             clippath_trans, paths, all_transforms, offsets,
                             offsetTrans, facecolors, edgecolors, linewidths,
                             linestyles, antialiaseds, urls):
        """
        Draws a collection of paths, selecting drawing properties from
        the lists *facecolors*, *edgecolors*, *linewidths*,
        *linestyles* and *antialiaseds*. *offsets* is a list of
        offsets to apply to each of the paths.  The offsets in
        *offsets* are first transformed by *offsetTrans* before
        being applied.

        This provides a fallback implementation of
        :meth:`draw_path_collection` that makes multiple calls to
        draw_path.  Some backends may want to override this in order
        to render each set of path data only once, and then reference
        that path multiple times with the different offsets, colors,
        styles etc.  The generator methods
        :meth:`_iter_collection_raw_paths` and
        :meth:`_iter_collection` are provided to help with (and
        standardize) the implementation across backends.  It is highly
        recommended to use those generators, so that changes to the
        behavior of :meth:`draw_path_collection` can be made globally.
        """
        path_ids = []
        for path, transform in self._iter_collection_raw_paths(
            master_transform, paths, all_transforms):
            path_ids.append((path, transform))

        for xo, yo, path_id, gc, rgbFace in self._iter_collection(
            path_ids, cliprect, clippath, clippath_trans,
            offsets, offsetTrans, facecolors, edgecolors,
            linewidths, linestyles, antialiaseds, urls):
            path, transform = path_id
            transform = transforms.Affine2D(transform.get_matrix()).translate(xo, yo)
            self.draw_path(gc, path, transform, rgbFace)

    def draw_quad_mesh(self, master_transform, cliprect, clippath,
                       clippath_trans, meshWidth, meshHeight, coordinates,
                       offsets, offsetTrans, facecolors, antialiased,
                       showedges):
        """
        This provides a fallback implementation of
        :meth:`draw_quad_mesh` that generates paths and then calls
        :meth:`draw_path_collection`.
        """
        from matplotlib.collections import QuadMesh
        paths = QuadMesh.convert_mesh_to_paths(
            meshWidth, meshHeight, coordinates)

        if showedges:
            edgecolors = np.array([[0.0, 0.0, 0.0, 1.0]], np.float_)
            linewidths = np.array([1.0], np.float_)
        else:
            edgecolors = facecolors
            linewidths = np.array([0.0], np.float_)

        return self.draw_path_collection(
            master_transform, cliprect, clippath, clippath_trans,
            paths, [], offsets, offsetTrans, facecolors, edgecolors,
            linewidths, [], [antialiased], [None])

    def _iter_collection_raw_paths(self, master_transform, paths, all_transforms):
        """
        This is a helper method (along with :meth:`_iter_collection`) to make
        it easier to write a space-efficent :meth:`draw_path_collection`
        implementation in a backend.

        This method yields all of the base path/transform
        combinations, given a master transform, a list of paths and
        list of transforms.

        The arguments should be exactly what is passed in to
        :meth:`draw_path_collection`.

        The backend should take each yielded path and transform and
        create an object that can be referenced (reused) later.
        """
        Npaths      = len(paths)
        Ntransforms = len(all_transforms)
        N           = max(Npaths, Ntransforms)

        if Npaths == 0:
            return

        transform = transforms.IdentityTransform()
        for i in xrange(N):
            path = paths[i % Npaths]
            if Ntransforms:
                transform = all_transforms[i % Ntransforms]
            yield path, transform + master_transform

    def _iter_collection(self, path_ids, cliprect, clippath, clippath_trans,
                         offsets, offsetTrans, facecolors, edgecolors,
                         linewidths, linestyles, antialiaseds, urls):
        """
        This is a helper method (along with
        :meth:`_iter_collection_raw_paths`) to make it easier to write
        a space-efficent :meth:`draw_path_collection` implementation in a
        backend.

        This method yields all of the path, offset and graphics
        context combinations to draw the path collection.  The caller
        should already have looped over the results of
        :meth:`_iter_collection_raw_paths` to draw this collection.

        The arguments should be the same as that passed into
        :meth:`draw_path_collection`, with the exception of
        *path_ids*, which is a list of arbitrary objects that the
        backend will use to reference one of the paths created in the
        :meth:`_iter_collection_raw_paths` stage.

        Each yielded result is of the form::

           xo, yo, path_id, gc, rgbFace

        where *xo*, *yo* is an offset; *path_id* is one of the elements of
        *path_ids*; *gc* is a graphics context and *rgbFace* is a color to
        use for filling the path.
        """
        Npaths      = len(path_ids)
        Noffsets    = len(offsets)
        N           = max(Npaths, Noffsets)
        Nfacecolors = len(facecolors)
        Nedgecolors = len(edgecolors)
        Nlinewidths = len(linewidths)
        Nlinestyles = len(linestyles)
        Naa         = len(antialiaseds)
        Nurls       = len(urls)

        if (Nfacecolors == 0 and Nedgecolors == 0) or Npaths == 0:
            return
        if Noffsets:
            toffsets = offsetTrans.transform(offsets)

        gc = self.new_gc()

        gc.set_clip_rectangle(cliprect)
        if clippath is not None:
            clippath = transforms.TransformedPath(clippath, clippath_trans)
            gc.set_clip_path(clippath)

        if Nfacecolors == 0:
            rgbFace = None

        if Nedgecolors == 0:
            gc.set_linewidth(0.0)

        xo, yo = 0, 0
        for i in xrange(N):
            path_id = path_ids[i % Npaths]
            if Noffsets:
                xo, yo = toffsets[i % Noffsets]
            if Nfacecolors:
                rgbFace = facecolors[i % Nfacecolors]
            if Nedgecolors:
                gc.set_foreground(edgecolors[i % Nedgecolors])
                if Nlinewidths:
                    gc.set_linewidth(linewidths[i % Nlinewidths])
                if Nlinestyles:
                    gc.set_dashes(*linestyles[i % Nlinestyles])
            if rgbFace is not None and len(rgbFace)==4:
                gc.set_alpha(rgbFace[-1])
                rgbFace = rgbFace[:3]
            gc.set_antialiased(antialiaseds[i % Naa])

            if Nurls:
                gc.set_url(urls[i % Nurls])

            yield xo, yo, path_id, gc, rgbFace

    def get_image_magnification(self):
        """
        Get the factor by which to magnify images passed to :meth:`draw_image`.
        Allows a backend to have images at a different resolution to other
        artists.
        """
        return 1.0

    def draw_image(self, x, y, im, bbox, clippath=None, clippath_trans=None):
        """
        Draw the image instance into the current axes;

        *x*
            is the distance in pixels from the left hand side of the canvas.

        *y*
            the distance from the origin.  That is, if origin is
            upper, y is the distance from top.  If origin is lower, y
            is the distance from bottom

        *im*
            the :class:`matplotlib._image.Image` instance

        *bbox*
            a :class:`matplotlib.transforms.Bbox` instance for clipping, or
            None

        """
        raise NotImplementedError

    def option_image_nocomposite(self):
        """
        overwrite this method for renderers that do not necessarily
        want to rescale and composite raster images. (like SVG)
        """
        return False

    def draw_tex(self, gc, x, y, s, prop, angle, ismath='TeX!'):
        raise NotImplementedError

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False):
        """
        Draw the text instance

        *gc*
            the :class:`GraphicsContextBase` instance

        *x*
            the x location of the text in display coords

        *y*
            the y location of the text in display coords

        *s*
             a :class:`matplotlib.text.Text` instance

        *prop*
          a :class:`matplotlib.font_manager.FontProperties` instance

        *angle*
            the rotation angle in degrees

        **backend implementers note**

        When you are trying to determine if you have gotten your bounding box
        right (which is what enables the text layout/alignment to work
        properly), it helps to change the line in text.py::

            if 0: bbox_artist(self, renderer)

        to if 1, and then the actual bounding box will be blotted along with
        your text.
        """
        raise NotImplementedError

    def flipy(self):
        """
        Return true if y small numbers are top for renderer Is used
        for drawing text (:mod:`matplotlib.text`) and images
        (:mod:`matplotlib.image`) only
        """
        return True

    def get_canvas_width_height(self):
        'return the canvas width and height in display coords'
        return 1, 1

    def get_texmanager(self):
        """
        return the :class:`matplotlib.texmanager.TexManager` instance
        """
        if self._texmanager is None:
            from matplotlib.texmanager import TexManager
            self._texmanager = TexManager()
        return self._texmanager

    def get_text_width_height_descent(self, s, prop, ismath):
        """
        get the width and height, and the offset from the bottom to the
        baseline (descent), in display coords of the string s with
        :class:`~matplotlib.font_manager.FontProperties` prop
        """
        raise NotImplementedError

    def new_gc(self):
        """
        Return an instance of a :class:`GraphicsContextBase`
        """
        return GraphicsContextBase()

    def points_to_pixels(self, points):
        """
        Convert points to display units

        *points*
            a float or a numpy array of float

        return points converted to pixels

        You need to override this function (unless your backend
        doesn't have a dpi, eg, postscript or svg).  Some imaging
        systems assume some value for pixels per inch::

            points to pixels = points * pixels_per_inch/72.0 * dpi/72.0
        """
        return points

    def strip_math(self, s):
        return cbook.strip_math(s)

    def start_rasterizing(self):
        pass

    def stop_rasterizing(self):
        pass


class GraphicsContextBase:
    """
    An abstract base class that provides color, line styles, etc...
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
        self._url = None
        self._snap = None

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
        self._url = gc._url
        self._snap = gc._snap

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
        Return the clip rectangle as a :class:`~matplotlib.transforms.Bbox` instance
        """
        return self._cliprect

    def get_clip_path(self):
        """
        Return the clip path in the form (path, transform), where path
        is a :class:`~matplotlib.path.Path` instance, and transform is
        an affine transform to apply to the path before clipping.
        """
        if self._clippath is not None:
            return self._clippath.get_transformed_path_and_affine()
        return None, None

    def get_dashes(self):
        """
        Return the dash information as an offset dashlist tuple.

        The dash list is a even size list that gives the ink on, ink
        off in pixels.

        See p107 of to PostScript `BLUEBOOK
        <http://www-cdf.fnal.gov/offline/PostScript/BLUEBOOK.PDF>`_
        for more info.

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

    def get_url(self):
        """
        returns a url if one is set, None otherwise
        """
        return self._url

    def get_snap(self):
        """
        returns the snap setting which may be:

          * True: snap vertices to the nearest pixel center

          * False: leave vertices as-is

          * None: (auto) If the path contains only rectilinear line
            segments, round to the nearest pixel center
        """
        return self._snap

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
        Set the clip path and transformation.  Path should be a
        :class:`~matplotlib.transforms.TransformedPath` instance.
        """
        assert path is None or isinstance(path, transforms.TransformedPath)
        self._clippath = path

    def set_dashes(self, dash_offset, dash_list):
        """
        Set the dash style for the gc.

        *dash_offset*
            is the offset (usually 0).

        *dash_list*
            specifies the on-off sequence as points.  ``(None, None)`` specifies a solid line

        """
        self._dashes = dash_offset, dash_list

    def set_foreground(self, fg, isRGB=False):
        """
        Set the foreground color.  fg can be a matlab format string, a
        html hex color string, an rgb unit tuple, or a float between 0
        and 1.  In the latter case, grayscale is used.

        The :class:`GraphicsContextBase` converts colors to rgb
        internally.  If you know the color is rgb already, you can set
        ``isRGB=True`` to avoid the performace hit of the conversion
        """
        if isRGB:
            self._rgb = fg
        else:
            self._rgb = colors.colorConverter.to_rgba(fg)

    def set_graylevel(self, frac):
        """
        Set the foreground color to be a gray level with *frac*
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

    def set_url(self, url):
        """
        Sets the url for links in compatible backends
        """
        self._url = url

    def set_snap(self, snap):
        """
        Sets the snap setting which may be:

          * True: snap vertices to the nearest pixel center

          * False: leave vertices as-is

          * None: (auto) If the path contains only rectilinear line
            segments, round to the nearest pixel center
        """
        self._snap = snap

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
    :meth:`FigureCanvasBase.mpl_connect`.  The following attributes
    are defined and shown with their default values

    *name*
        the event name

    *canvas*
        the FigureCanvas instance generating the event

    *guiEvent*
        the GUI event that triggered the matplotlib event


    """
    def __init__(self, name, canvas,guiEvent=None):
        self.name = name
        self.canvas = canvas
        self.guiEvent = guiEvent

class IdleEvent(Event):
    """
    An event triggered by the GUI backend when it is idle -- useful
    for passive animation
    """
    pass

class DrawEvent(Event):
    """
    An event triggered by a draw operation on the canvas

    In addition to the :class:`Event` attributes, the following event attributes are defined:

    *renderer*
        the :class:`RendererBase` instance for the draw event

    """
    def __init__(self, name, canvas, renderer):
        Event.__init__(self, name, canvas)
        self.renderer = renderer

class ResizeEvent(Event):
    """
    An event triggered by a canvas resize

    In addition to the :class:`Event` attributes, the following event attributes are defined:

    *width*
        width of the canvas in pixels

    *height*
        height of the canvas in pixels

    """
    def __init__(self, name, canvas):
        Event.__init__(self, name, canvas)
        self.width, self.height = canvas.get_width_height()

class LocationEvent(Event):
    """
    A event that has a screen location

    The following additional attributes are defined and shown with
    their default values

    In addition to the :class:`Event` attributes, the following event attributes are defined:

    *x*
        x position - pixels from left of canvas

    *y*
        y position - pixels from bottom of canvas

    *inaxes*
        the :class:`~matplotlib.axes.Axes` instance if mouse is over axes

    *xdata*
        x coord of mouse in data coords

    *ydata*
        y coord of mouse in data coords

    """
    x      = None       # x position - pixels from left of canvas
    y      = None       # y position - pixels from right of canvas
    inaxes = None       # the Axes instance if mouse us over axes
    xdata  = None       # x coord of mouse in data coords
    ydata  = None       # y coord of mouse in data coords

    # the last event that was triggered before this one
    lastevent = None

    def __init__(self, name, canvas, x, y,guiEvent=None):
        """
        *x*, *y* in figure coords, 0,0 = bottom, left
        """
        Event.__init__(self, name, canvas,guiEvent=guiEvent)
        self.x = x
        self.y = y



        if x is None or y is None:
            # cannot check if event was in axes if no x,y info
            self.inaxes = None
            self._update_enter_leave()
            return

        # Find all axes containing the mouse
        axes_list = [a for a in self.canvas.figure.get_axes() if a.in_axes(self)]

        if len(axes_list) == 0: # None found
            self.inaxes = None
            self._update_enter_leave()
            return
        elif (len(axes_list) > 1): # Overlap, get the highest zorder
            axCmp = lambda _x,_y: cmp(_x.zorder, _y.zorder)
            axes_list.sort(axCmp)
            self.inaxes = axes_list[-1] # Use the highest zorder
        else: # Just found one hit
            self.inaxes = axes_list[0]

        try:
            xdata, ydata = self.inaxes.transData.inverted().transform_point((x, y))
        except ValueError:
            self.xdata  = None
            self.ydata  = None
        else:
            self.xdata  = xdata
            self.ydata  = ydata

        self._update_enter_leave()

    def _update_enter_leave(self):
        'process the figure/axes enter leave events'
        if LocationEvent.lastevent is not None:
            last = LocationEvent.lastevent
            if last.inaxes!=self.inaxes:
                # process axes enter/leave events
                if last.inaxes is not None:
                    last.canvas.callbacks.process('axes_leave_event', last)
                if self.inaxes is not None:
                    self.canvas.callbacks.process('axes_enter_event', self)

        else:
            # process a figure enter event
            if self.inaxes is not None:
                self.canvas.callbacks.process('axes_enter_event', self)


        LocationEvent.lastevent = self




class MouseEvent(LocationEvent):
    """
    A mouse event ('button_press_event', 'button_release_event', 'scroll_event',
    'motion_notify_event').

    In addition to the :class:`Event` and :class:`LocationEvent`
    attributes, the following attributes are defined:

    *button*
        button pressed None, 1, 2, 3, 'up', 'down' (up and down are used for scroll events)

    *key*
        the key pressed: None, chr(range(255), 'shift', 'win', or 'control'

    *step*
        number of scroll steps (positive for 'up', negative for 'down')


    Example usage::

        def on_press(event):
            print 'you pressed', event.button, event.xdata, event.ydata

        cid = fig.canvas.mpl_connect('button_press_event', on_press)

    """
    x      = None       # x position - pixels from left of canvas
    y      = None       # y position - pixels from right of canvas
    button = None       # button pressed None, 1, 2, 3
    inaxes = None       # the Axes instance if mouse us over axes
    xdata  = None       # x coord of mouse in data coords
    ydata  = None       # y coord of mouse in data coords
    step   = None       # scroll steps for scroll events

    def __init__(self, name, canvas, x, y, button=None, key=None,
                 step=0, guiEvent=None):
        """
        x, y in figure coords, 0,0 = bottom, left
        button pressed None, 1, 2, 3, 'up', 'down'
        """
        LocationEvent.__init__(self, name, canvas, x, y, guiEvent=guiEvent)
        self.button = button
        self.key = key
        self.step = step

class PickEvent(Event):
    """
    a pick event, fired when the user picks a location on the canvas
    sufficiently close to an artist.

    Attrs: all the :class:`Event` attributes plus

    *mouseevent*
        the :class:`MouseEvent` that generated the pick

    *artist*
        the :class:`~matplotlib.artist.Artist` picked

    other
        extra class dependent attrs -- eg a
        :class:`~matplotlib.lines.Line2D` pick may define different
        extra attributes than a
        :class:`~matplotlib.collections.PatchCollection` pick event


    Example usage::

        line, = ax.plot(rand(100), 'o', picker=5)  # 5 points tolerance

        def on_pick(event):
            thisline = event.artist
            xdata, ydata = thisline.get_data()
            ind = event.ind
            print 'on pick line:', zip(xdata[ind], ydata[ind])

        cid = fig.canvas.mpl_connect('pick_event', on_pick)

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
    :meth:`FigureCanvasBase.mpl_connect`.

    In addition to the :class:`Event` and :class:`LocationEvent`
    attributes, the following attributes are defined:

    *key*
        the key pressed: None, chr(range(255), shift, win, or control

    This interface may change slightly when better support for
    modifier keys is included.


    Example usage::

        def on_key(event):
            print 'you pressed', event.key, event.xdata, event.ydata

        cid = fig.canvas.mpl_connect('key_press_event', on_key)

    """
    def __init__(self, name, canvas, key, x=0, y=0, guiEvent=None):
        LocationEvent.__init__(self, name, canvas, x, y, guiEvent=guiEvent)
        self.key = key



class FigureCanvasBase:
    """
    The canvas the figure renders into.

    Public attributes

        *figure*
            A :class:`matplotlib.figure.Figure` instance

      """
    events = [
        'resize_event',
        'draw_event',
        'key_press_event',
        'key_release_event',
        'button_press_event',
        'button_release_event',
        'scroll_event',
        'motion_notify_event',
        'pick_event',
        'idle_event',
        'figure_enter_event',
        'figure_leave_event',
        'axes_enter_event',
        'axes_leave_event'
        ]


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
        under the cursor.  Connect this to the 'mouse_press_event'
        using::

            canvas.mpl_connect('mouse_press_event',canvas.onRemove)
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
                if h in p.get_children():
                    parent = p
                    break
            h = parent

    def onHilite(self, ev):
        """
        Mouse event processor which highlights the artists
        under the cursor.  Connect this to the 'motion_notify_event'
        using::

            canvas.mpl_connect('motion_notify_event',canvas.onHilite)
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
        """
        This method will be call all functions connected to the
        'draw_event' with a :class:`DrawEvent`
        """

        s = 'draw_event'
        event = DrawEvent(s, self, renderer)
        self.callbacks.process(s, event)

    def resize_event(self):
        """
        This method will be call all functions connected to the
        'resize_event' with a :class:`ResizeEvent`
        """

        s = 'resize_event'
        event = ResizeEvent(s, self)
        self.callbacks.process(s, event)

    def key_press_event(self, key, guiEvent=None):
        """
        This method will be call all functions connected to the
        'key_press_event' with a :class:`KeyEvent`
        """
        self._key = key
        s = 'key_press_event'
        event = KeyEvent(s, self, key, self._lastx, self._lasty, guiEvent=guiEvent)
        self.callbacks.process(s, event)

    def key_release_event(self, key, guiEvent=None):
        """
        This method will be call all functions connected to the
        'key_release_event' with a :class:`KeyEvent`
        """
        s = 'key_release_event'
        event = KeyEvent(s, self, key, self._lastx, self._lasty, guiEvent=guiEvent)
        self.callbacks.process(s, event)
        self._key = None

    def pick_event(self, mouseevent, artist, **kwargs):
        """
        This method will be called by artists who are picked and will
        fire off :class:`PickEvent` callbacks registered listeners
        """
        s = 'pick_event'
        event = PickEvent(s, self, mouseevent, artist, **kwargs)
        self.callbacks.process(s, event)

    def scroll_event(self, x, y, step, guiEvent=None):
        """
        Backend derived classes should call this function on any
        scroll wheel event.  x,y are the canvas coords: 0,0 is lower,
        left.  button and key are as defined in MouseEvent.

        This method will be call all functions connected to the
        'scroll_event' with a :class:`MouseEvent` instance.
        """
        if step >= 0:
            self._button = 'up'
        else:
            self._button = 'down'
        s = 'scroll_event'
        mouseevent = MouseEvent(s, self, x, y, self._button, self._key,
                                step=step, guiEvent=guiEvent)
        self.callbacks.process(s, mouseevent)


    def button_press_event(self, x, y, button, guiEvent=None):
        """
        Backend derived classes should call this function on any mouse
        button press.  x,y are the canvas coords: 0,0 is lower, left.
        button and key are as defined in :class:`MouseEvent`.

        This method will be call all functions connected to the
        'button_press_event' with a :class:`MouseEvent` instance.

        """
        self._button = button
        s = 'button_press_event'
        mouseevent = MouseEvent(s, self, x, y, button, self._key, guiEvent=guiEvent)
        self.callbacks.process(s, mouseevent)

    def button_release_event(self, x, y, button, guiEvent=None):
        """
        Backend derived classes should call this function on any mouse
        button release.

        *x*
            the canvas coordinates where 0=left

        *y*
            the canvas coordinates where 0=bottom

        *guiEvent*
            the native UI event that generated the mpl event


        This method will be call all functions connected to the
        'button_release_event' with a :class:`MouseEvent` instance.

        """
        s = 'button_release_event'
        event = MouseEvent(s, self, x, y, button, self._key, guiEvent=guiEvent)
        self.callbacks.process(s, event)
        self._button = None

    def motion_notify_event(self, x, y, guiEvent=None):
        """
        Backend derived classes should call this function on any
        motion-notify-event.

        *x*
            the canvas coordinates where 0=left

        *y*
            the canvas coordinates where 0=bottom

        *guiEvent*
            the native UI event that generated the mpl event


        This method will be call all functions connected to the
        'motion_notify_event' with a :class:`MouseEvent` instance.

        """
        self._lastx, self._lasty = x, y
        s = 'motion_notify_event'
        event = MouseEvent(s, self, x, y, self._button, self._key,
                           guiEvent=guiEvent)
        self.callbacks.process(s, event)

    def leave_notify_event(self, guiEvent=None):
        """
        Backend derived classes should call this function when leaving
        canvas

        *guiEvent*
            the native UI event that generated the mpl event

        """
        self.callbacks.process('figure_leave_event', LocationEvent.lastevent)
        LocationEvent.lastevent = None

    def enter_notify_event(self, guiEvent=None):
        """
        Backend derived classes should call this function when entering
        canvas

        *guiEvent*
            the native UI event that generated the mpl event

        """
        event = Event('figure_enter_event', self, guiEvent)
        self.callbacks.process('figure_enter_event', event)

    def idle_event(self, guiEvent=None):
        'call when GUI is idle'
        s = 'idle_event'
        event = IdleEvent(s, self, guiEvent=guiEvent)
        self.callbacks.process(s, event)


    def draw(self, *args, **kwargs):
        """
        Render the :class:`~matplotlib.figure.Figure`
        """
        pass

    def draw_idle(self, *args, **kwargs):
        """
        :meth:`draw` only if idle; defaults to draw but backends can overrride
        """
        self.draw(*args, **kwargs)

    def draw_cursor(self, event):
        """
        Draw a cursor in the event.axes if inaxes is not None.  Use
        native GUI drawing for efficiency if possible
        """
        pass

    def get_width_height(self):
        """
        return the figure width and height in points or pixels
        (depending on the backend), truncated to integers
        """
        return int(self.figure.bbox.width), int(self.figure.bbox.height)

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

        Arguments are:

        *filename*
            can also be a file object on image backends

        *orientation*
            only currently applies to PostScript printing.

        *dpi*
            the dots per inch to save the figure in; if None, use savefig.dpi

        *facecolor*
            the facecolor of the figure

        *edgecolor*
            the edgecolor of the figure

        *orientation*  '
            landscape' | 'portrait' (not supported on all backends)

        *format*
            when set, forcibly set the file format to save to
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

        origDPI = self.figure.dpi
        origfacecolor = self.figure.get_facecolor()
        origedgecolor = self.figure.get_edgecolor()

        self.figure.dpi = dpi
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
            self.figure.dpi = origDPI
            self.figure.set_facecolor(origfacecolor)
            self.figure.set_edgecolor(origedgecolor)
            self.figure.set_canvas(self)
            #self.figure.canvas.draw() ## seems superfluous
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
        Connect event with string *s* to *func*.  The signature of *func* is::

          def func(event)

        where event is a :class:`matplotlib.backend_bases.Event`.  The
        following events are recognized

        - 'button_press_event'
        - 'button_release_event'
        - 'draw_event'
        - 'key_press_event'
        - 'key_release_event'
        - 'motion_notify_event'
        - 'pick_event'
        - 'resize_event'
        - 'scroll_event'

        For the location events (button and key press/release), if the
        mouse is over the axes, the variable ``event.inaxes`` will be
        set to the :class:`~matplotlib.axes.Axes` the event occurs is
        over, and additionally, the variables ``event.xdata`` and
        ``event.ydata`` will be defined.  This is the mouse location
        in data coords.  See
        :class:`~matplotlib.backend_bases.KeyEvent` and
        :class:`~matplotlib.backend_bases.MouseEvent` for more info.

        Return value is a connection id that can be used with
        :meth:`~matplotlib.backend_bases.Event.mpl_disconnect`.

        Example usage::

            def on_press(event):
                print 'you pressed', event.button, event.xdata, event.ydata

            cid = canvas.mpl_connect('button_press_event', on_press)

        """

        return self.callbacks.connect(s, func)

    def mpl_disconnect(self, cid):
        """
        disconnect callback id cid

        Example usage::

            cid = canvas.mpl_connect('button_press_event', on_press)
            #...later
            canvas.mpl_disconnect(cid)
        """
        return self.callbacks.disconnect(cid)

    def flush_events(self):
        """
        Flush the GUI events for the figure. Implemented only for
        backends with GUIs.
        """
        raise NotImplementedError

    def start_event_loop(self,timeout):
        """
        Start an event loop.  This is used to start a blocking event
        loop so that interactive functions, such as ginput and
        waitforbuttonpress, can wait for events.  This should not be
        confused with the main GUI event loop, which is always running
        and has nothing to do with this.

        This is implemented only for backends with GUIs.
        """
        raise NotImplementedError

    def stop_event_loop(self):
        """
        Stop an event loop.  This is used to stop a blocking event
        loop so that interactive functions, such as ginput and
        waitforbuttonpress, can wait for events.

        This is implemented only for backends with GUIs.
        """
        raise NotImplementedError

    def start_event_loop_default(self,timeout=0):
        """
        Start an event loop.  This is used to start a blocking event
        loop so that interactive functions, such as ginput and
        waitforbuttonpress, can wait for events.  This should not be
        confused with the main GUI event loop, which is always running
        and has nothing to do with this.

        This function provides default event loop functionality based
        on time.sleep that is meant to be used until event loop
        functions for each of the GUI backends can be written.  As
        such, it throws a deprecated warning.

        Call signature::

            start_event_loop_default(self,timeout=0)

        This call blocks until a callback function triggers
        stop_event_loop() or *timeout* is reached.  If *timeout* is
        <=0, never timeout.
        """
        str = "Using default event loop until function specific"
        str += " to this GUI is implemented"
        warnings.warn(str,DeprecationWarning)

        if timeout <= 0: timeout = np.inf
        timestep = 0.01
        counter = 0
        self._looping = True
        while self._looping and counter*timestep < timeout:
            self.flush_events()
            time.sleep(timestep)
            counter += 1

    def stop_event_loop_default(self):
        """
        Stop an event loop.  This is used to stop a blocking event
        loop so that interactive functions, such as ginput and
        waitforbuttonpress, can wait for events.

        Call signature::

          stop_event_loop_default(self)
        """
        self._looping = False



class FigureManagerBase:
    """
    Helper class for matlab mode, wraps everything up into a neat bundle

    Public attibutes:

    *canvas*
        A :class:`FigureCanvasBase` instance

    *num*
        The figure nuamber
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

        # *h*ome or *r*eset mnemonic
        elif event.key == 'h' or event.key == 'r' or event.key == "home":
            self.canvas.toolbar.home()
        # c and v to enable left handed quick navigation
        elif event.key == 'left' or event.key == 'c' or event.key == 'backspace':
            self.canvas.toolbar.back()
        elif event.key == 'right' or event.key == 'v':
            self.canvas.toolbar.forward()
        # *p*an mnemonic
        elif event.key == 'p':
            self.canvas.toolbar.pan()
        # z*o*om mnemonic
        elif event.key == 'o':
            self.canvas.toolbar.zoom()
        elif event.key == 's':
            self.canvas.toolbar.save_figure(self.canvas.toolbar)

        if event.inaxes is None:
            return

        # the mouse has to be over an axes to trigger these
        if event.key == 'g':
            event.inaxes.grid()
            self.canvas.draw()
        elif event.key == 'l':
            ax = event.inaxes
            scale = ax.get_yscale()
            if scale=='log':
                ax.set_yscale('linear')
                ax.figure.canvas.draw()
            elif scale=='linear':
                ax.set_yscale('log')
                ax.figure.canvas.draw()

        elif event.key is not None and (event.key.isdigit() and event.key!='0') or event.key=='a':
            # 'a' enables all axes
            if event.key!='a':
                n=int(event.key)-1
            for i, a in enumerate(self.canvas.figure.get_axes()):
                if event.x is not None and event.y is not None and a.in_axes(event):
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
    :meth:`FigureCanvasBase.mpl_connect` for more information


    They must also define

      :meth:`save_figure`
         save the current figure

      :meth:`set_cursor`
         if you want the pointer icon to change

      :meth:`_init_toolbar`
         create your toolbar widget

      :meth:`draw_rubberband` (optional)
         draw the zoom to rect "rubberband" rectangle

      :meth:`press`  (optional)
         whenever a mouse button is pressed, you'll be notified with
         the event

      :meth:`release` (optional)
         whenever a mouse button is released, you'll be notified with
         the event

      :meth:`dynamic_update` (optional)
         dynamically update the window while navigating

      :meth:`set_message` (optional)
         display message

      :meth:`set_history_buttons` (optional)
         you can change the history back / forward buttons to
         indicate disabled / enabled state.

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
        __init__).  The icons ``home.xpm``, ``back.xpm``, ``forward.xpm``,
        ``hand.xpm``, ``zoom_to_rect.xpm`` and ``filesave.xpm`` are standard
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
                    lastx, lasty, a, ind, lim, trans = self._xypress[0]
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
            if x is not None and y is not None and a.in_axes(event) and a.get_navigate():
                a.start_pan(x, y, event.button)
                self._xypress.append((a, i))
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
            if x is not None and y is not None and a.in_axes(event) \
                    and a.get_navigate() and a.can_zoom():
                self._xypress.append(( x, y, a, i, a.viewLim.frozen(), a.transData.frozen()))

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
                    a.get_position(True).frozen(),
                    a.get_position().frozen() ) )
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
        for a, ind in self._xypress:
            a.end_pan()
        if not self._xypress: return
        self._xypress = []
        self._button_pressed=None
        self.push_current()
        self.release(event)
        self.draw()

    def drag_pan(self, event):
        'the drag callback in pan/zoom mode'

        for a, ind in self._xypress:
            #safer to use the recorded button at the press than current button:
            #multiple button can get pressed during motion...
            a.drag_pan(self._button_pressed, event.key, event.x, event.y)
        self.dynamic_update()

    def release_zoom(self, event):
        'the release mouse button callback in zoom to rect mode'
        if not self._xypress: return

        last_a = []

        for cur_xypress in self._xypress:
            x, y = event.x, event.y
            lastx, lasty, a, ind, lim, trans = cur_xypress

            # ignore singular clicks - 5 pixels is a threshold
            if abs(x-lastx)<5 or abs(y-lasty)<5:
                self._xypress = None
                self.release(event)
                self.draw()
                return

            x0, y0, x1, y1 = lim.extents

            # zoom to rect
            inverse = a.transData.inverted()
            lastx, lasty = inverse.transform_point( (lastx, lasty) )
            x, y = inverse.transform_point( (x, y) )
            Xmin,Xmax=a.get_xlim()
            Ymin,Ymax=a.get_ylim()

            # detect twinx,y axes and avoid double zooming
            twinx, twiny = False, False
            if last_a:
                for la in last_a:
                    if a.get_shared_x_axes().joined(a,la): twinx=True
                    if a.get_shared_y_axes().joined(a,la): twiny=True
            last_a.append(a)

            if twinx:
                x0, x1 = Xmin, Xmax
            else:
                if Xmin < Xmax:
                    if x<lastx:  x0, x1 = x, lastx
                    else: x0, x1 = lastx, x
                    if x0 < Xmin: x0=Xmin
                    if x1 > Xmax: x1=Xmax
                else:
                    if x>lastx:  x0, x1 = x, lastx
                    else: x0, x1 = lastx, x
                    if x0 > Xmin: x0=Xmin
                    if x1 < Xmax: x1=Xmax

            if twiny:
                y0, y1 = Ymin, Ymax
            else:
                if Ymin < Ymax:
                    if y<lasty:  y0, y1 = y, lasty
                    else: y0, y1 = lasty, y
                    if y0 < Ymin: y0=Ymin
                    if y1 > Ymax: y1=Ymax
                else:
                    if y>lasty:  y0, y1 = y, lasty
                    else: y0, y1 = lasty, y
                    if y0 > Ymin: y0=Ymin
                    if y1 < Ymax: y1=Ymax

            if self._button_pressed == 1:
                a.set_xlim((x0, x1))
                a.set_ylim((y0, y1))
            elif self._button_pressed == 3:
                if a.get_xscale()=='log':
                    alpha=np.log(Xmax/Xmin)/np.log(x1/x0)
                    rx1=pow(Xmin/x0,alpha)*Xmin
                    rx2=pow(Xmax/x0,alpha)*Xmin
                else:
                    alpha=(Xmax-Xmin)/(x1-x0)
                    rx1=alpha*(Xmin-x0)+Xmin
                    rx2=alpha*(Xmax-x0)+Xmin
                if a.get_yscale()=='log':
                    alpha=np.log(Ymax/Ymin)/np.log(y1/y0)
                    ry1=pow(Ymin/y0,alpha)*Ymin
                    ry2=pow(Ymax/y0,alpha)*Ymin
                else:
                    alpha=(Ymax-Ymin)/(y1-y0)
                    ry1=alpha*(Ymin-y0)+Ymin
                    ry2=alpha*(Ymax-y0)+Ymin
                a.set_xlim((rx1, rx2))
                a.set_ylim((ry1, ry2))

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
        Set the current cursor to one of the :class:`Cursors`
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
