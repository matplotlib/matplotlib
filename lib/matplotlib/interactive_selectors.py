# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import numpy as np

# TODO: convert these to relative when finished
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Rectangle, Ellipse, Polygon
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
from matplotlib import docstring, artist as martist


# These are not available for the object inspector until after the
# class is built so we define an initial set here for the init
# function and they will be overridden after object definition.
docstring.interpd.update(BaseInteractiveTool="""\
    To guarantee that the tool remains responsive and not garbage-collected,
    a reference to the object should be maintained by the user.

    This is necessary because the callback registry
    maintains only weak-refs to the functions, which are member
    functions of the tool.  If there are no references to the tool
    object it may be garbage collected which will disconnect the
    callbacks.

    Use `set_geometry()` to update the tool programmatically.  The geometry
    attributes (verts, center, extents, etc.) are all read-only.

    Attributes
    ----------
    ax: :class:`~matplotlib.axes.Axes`
        The parent axes for the tool.
    canvas: :class:`~matplotlib.backend_bases.FigureCanvasBase` subclass
        The parent figure canvas for the tool.
    active: boolean
        If False, the widget does not respond to events.
    on_select: callable, optional
        A callback for when a selection is made `on_select(tool)`.
    on_motion: callable, optional
        A callback for when the tool is moved `on_motion(tool)`.
    on_accept: callable, optional
        A callback for when the selection is accepted `on_accept(tool)`.
        This is called in response to an 'accept' key event.
    """)


docstring.interpd.update(BasePolygonTool="""\
    interactive: boolean
        Whether to allow interaction with the shape using handles.
    allow_redraw: boolean
        Whether to allow the tool to redraw itself or whether it must be
        drawn programmatically and then dragged.
    focused: boolean
        Whether the tool has focus for keyboard and scroll events.
    polygon: `matplotlib.patches.Polygon`
        The polygon patch.
    center: (x, y)
        The center coordinates of the tool in data units (read-only).
    extents: (x0, y0, width, height) float
        The total geometry of the tool in data units (read-only).
    """)


docstring.interpd.update(BaseInteractiveToolInit="""
    Parameters
    ----------
    ax: :class:`matplotlib.axes.Axes`
        The parent axes for the tool.
    on_select: callable, optional
        A callback for when a selection is made `on_select(tool)`.
    on_motion: callable, optional
        A callback for when the tool is moved `on_motion(tool)`.
    on_accept: callable, optional
        A callback for when the selection is accepted `on_accept(tool)`.
        This is called in response to an 'accept' key event.
    useblit: boolean, optional
        Whether to use blitting while drawing if available.
    button: int or list of int, optional
        Which mouse button(s) should be used.  Typically:
         1 = left mouse button
         2 = center mouse button (scroll wheel)
         3 = right mouse button
    keys: dict, optional
        A mapping of key shortcuts for the tool.
        'move': Move the existing shape.
        'clear': Clear the current shape.
        'square': Makes the shape square.
        'center': Make the initial point the center of the shape.
        'polygon': Draw a polygon shape for the lasso.
        'square' and 'center' can be combined.
        'accept': Trigger an `on_accept` callback.
    """)


docstring.interpd.update(BasePolygonToolInit="""
    interactive: boolean, optional
        Whether to allow interaction with the shape using handles.
    allow_redraw: boolean, optional
        Whether to allow the tool to redraw itself or whether it must be
        drawn programmatically and then dragged.
    shape_props: dict, optional
        The properties of the shape patch.
    handle_props: dict, optional
        The properties of the handle markers.
""")


@docstring.dedent_interpd
class BaseTool(object):

    """Interactive selection tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.

    %(BaseInteractiveTool)s
    """
    def __init__(self, ax, on_select=None, on_motion=None, on_accept=None,
                 useblit=True, button=None, keys=None):
        """Initialize the tool.
        %(BaseInteractiveToolInit)s
        %(BasePolygonToolInit)s
        """
        self.ax = ax
        self.canvas = ax.figure.canvas
        self._active = True

        self.on_motion = _dummy if on_motion is None else on_motion
        self.on_accept = _dummy if on_accept is None else on_accept
        self.on_select = _dummy if on_select is None else on_select

        self._useblit = useblit and self.canvas.supports_blit
        self._keys = dict(move=' ', clear='escape',
                          accept='enter', polygon='shift',
                          square='shift', center='control')
        self._keys.update(keys or {})

        if isinstance(button, int):
            self._buttons = [button]
        else:
            button = button or [1, 2, 3]
            self._buttons = button

        self._artists = []
        self._drawing = False
        self._background = None
        self._prev_evt_xy = None

        # Connect the major canvas events to methods."""
        self._cids = []
        self._connect_event('motion_notify_event', self._handle_event)
        self._connect_event('button_press_event', self._handle_event)
        self._connect_event('button_release_event', self._handle_event)
        self._connect_event('draw_event', self._handle_draw)
        self._connect_event('key_press_event', self._handle_event)
        self._connect_event('key_release_event', self._handle_event)
        self._connect_event('scroll_event', self._handle_event)

    @property
    def active(self):
        """Get the active state of the tool"""
        return self._active

    @active.setter
    def active(self, value):
        """Set the active state of the tool"""
        self._active = value
        if not value:
            for artist in self._artists:
                artist.set_visible(False)
            self.canvas.draw_idle()

    def remove(self):
        """Clean up the tool"""
        for c in self._cids:
            self.canvas.mpl_disconnect(c)
        for artist in self._artists:
            artist.remove()
        self.canvas.draw_idle()

    def _handle_draw(self, event):
        """Update the ax background on a draw event"""
        if self._useblit:
            self._background = self.canvas.copy_from_bbox(self.ax.bbox)

    def _handle_event(self, event):
        """Handle default actions for events and call to event handlers"""
        if self._ignore(event):
            return
        event = self._clean_event(event)
        if event.xdata is None and 'key' not in event.name:
            return

        if event.name == 'button_press_event':
            self._handle_button_press(event)

        elif event.name == 'motion_notify_event':
            self._handle_motion_notify(event)

        elif event.name == 'button_release_event':
            self._handle_button_release(event)

        elif event.name == 'key_press_event':
            self._handle_key_press(event)

        elif event.name == 'key_release_event':
            self._handle_key_release(event)

        elif event.name == 'scroll_event':
            self._handle_scroll(event)

    def _handle_motion_notify(self, event):
        self._on_motion(event)
        self.on_motion(self)

    def _handle_button_release(self, event):
        self._on_release(event)

    def _handle_key_release(self, event):
        self._on_key_release(event)

    def _handle_scroll(self, event):
        self._on_scroll(event)

    def _handle_button_press(self, event):
        self._on_press(event)

    def _handle_key_press(self, event):
        """Handle key_press_event defaults and call to subclass handler"""
        self._on_key_press(event)
        if event.key == self._keys['accept']:
            self.on_accept(self)

    def _clean_event(self, event):
        """Clean up an event.

        Use previous xy data if there is no xdata (the event was outside the
            axes).
        Limit the xdata and ydata to the axes limits.
        """
        event = copy.copy(event)
        if event.xdata is not None:
            x0, x1 = self.ax.get_xbound()
            y0, y1 = self.ax.get_ybound()
            xdata = max(x0, event.xdata)
            event.xdata = min(x1, xdata)
            ydata = max(y0, event.ydata)
            event.ydata = min(y1, ydata)
            self._prev_evt_xy = event.xdata, event.ydata
        elif self._prev_evt_xy is not None:
            event.xdata, event.ydata = self._prev_evt_xy

        event.key = event.key or ''
        event.key = event.key.replace('ctrl', 'control')
        return event

    def _connect_event(self, event, callback):
        """Connect callback with an event"""
        cid = self.canvas.mpl_connect(event, callback)
        self._cids.append(cid)

    def _ignore(self, event):
        """Return *True* if *event* should be ignored"""
        if not self.active or not self.ax.get_visible():
            return True

        # If canvas was locked
        if not self.canvas.widgetlock.available(self):
            return True

        # If we are currently drawing
        if self._drawing:
            return False

        if event.inaxes != self.ax:
            return True

        # If it is an invalid button press
        if self._buttons is not None:
            if getattr(event, 'button', None) not in self._buttons:
                return True

        return False

    def _update(self):
        """Update the artists while drawing"""
        if not self.ax.get_visible():
            return

        if self._useblit and self._drawing:
            if self._background is not None:
                self.canvas.restore_region(self._background)
            for artist in self._artists:
                self.ax.draw_artist(artist)

            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    #############################################################
    # The following are meant to be subclassed as needed.
    #############################################################
    def _on_press(self, event):
        """Handle a button_press_event"""
        pass

    def _on_motion(self, event):
        """Handle a motion_notify_event"""
        pass

    def _on_release(self, event):
        """Handle a button_release_event"""
        pass

    def _on_key_press(self, event):
        """Handle a key_press_event"""
        pass

    def _on_key_release(self, event):
        """Handle a key_release_event"""
        pass

    def _on_scroll(self, event):
        """Handle a scroll_event"""
        pass


def _dummy(tool):
    """A dummy callback for a tool."""
    pass


tooldoc = martist.kwdoc(BaseTool)
for k in ('RectangleTool', 'EllipseTool', 'LineTool', 'BaseTool',
          'PaintTool'):
    docstring.interpd.update({k: tooldoc})

# define BaseTool.__init__ docstring after the class has been added to interpd
docstring.dedent_interpd(BaseTool.__init__)


@docstring.dedent_interpd
class BasePatchTool(BaseTool):

    """Interactive patch selection tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.

    %(BaseInteractiveTool)s
    %(BasePolygonTool)s
    """

    def __init__(self, ax, on_select=None, on_motion=None, on_accept=None,
                 interactive=True, allow_redraw=True, patch_args=None,
                 patch_props=None,
                 handle_props=None, useblit=True, button=None, keys=None):
        super(BasePatchTool, self).__init__(ax, on_select=on_select,
            on_accept=on_accept, on_motion=on_motion, useblit=True,
            keys=None)

        if patch_args is None:
            patch_args = [[[0, 0], [1, 1]], True]
        patch_props = patch_props or {}
        self.patch = self._make_patch(**patch_props)
        self.ax.add_patch(self.patch)

        props = dict(marker='s', markersize=7, mfc='w', ls='none',
                     alpha=0.5, visible=False, label='_nolegend_',
                     picker=10, zorder=2)
        props.update(handle_props or {})
        self._size_handles = Line2D([], [], **props)
        props['marker'] = 'o'
        self._rot_handle = Line2D([], [], **props)

        self.ax.add_line(self._size_handles)
        self.ax.add_line(self._rot_handle)

        self._artists = [self.patch, self._size_handles, self._rot_handle]

        self._interactive = None
        self._modifiers = set()
        self._drawing = False
        self._dragging = False
        self._moving = False
        self._drag_idx = None
        self._has_selected = False
        self._prev_data = None
        self._start_event = None

        self.interactive = interactive
        self.allow_redraw = allow_redraw
        self.focused = True

    @property
    def interactive(self):
        return self._interactive

    @interactive.setter
    def interactive(self, value):
        if not value:
            self._size_handles.set_visible(False)
            self._rot_handle.set_visible(False)
            self.patch.set_visible(False)
            self.canvas.draw_idle()
        self._interactive = value

    def get_geometry(self):
        """Get the tool specific geometry as a dictionary"""
        return dict()

    def set_geometry(self, **kwargs):
        """Set the tool geometry directly"""
        for (key, value) in kwargs.items():
            func = getattr(self.patch, 'set_%s' % key, None)
            if func:
                func(value)
            elif hasattr(self.patch, key):
                setattr(self.patch, key, value)

        self.patch.set_visible(True)
        self.patch.set_animated(self._drawing)

        if self._prev_data is None:
            self._prev_geometry = self.get_geometry()

        size_size_handles, rot_handle = np.asarray(self._get_handle_verts())
        self._size_handles.set_data(size_size_handles[:, 0], size_size_handles[:, 1])
        self._size_handles.set_visible(self.interactive)
        self._size_handles.set_animated(self._drawing)
        if not rot_handle is None:
            self._rot_handle.set_visible(self.interactive)
            self._rot_handle.set_animated(self._drawing)
            self._rot_handle.set_data(*rot_handle)
        self._update()

        if not self._drawing:
            self._has_selected = True

    def _start_drawing(self, event):
        """Start drawing or dragging the shape"""
        self._drawing = True
        self._start_event = event
        if self.interactive:
            # Force a draw_event without our previous state.
            for artist in self._artists:
                artist.set_visible(False)
            self.canvas.draw()
            for artist in self._artists:
                artist.set_animated(self._useblit)
        else:
            self._size_handles.set_visible(False)
            self._rot_handle.set_visible(False)
        # Blit without being visible if not dragging to avoid showing the old
        # shape.
        for artist in self._artists:
            artist.set_visible(self._dragging)
            if artist in [self._size_handles, self._rot_handle]:
                artist.set_visible(self._dragging and self.interactive)
        self._update()

    def _finish_drawing(self, event, selection=False):
        """Finish drawing or dragging the shape"""
        self._drawing = False
        self._dragging = False
        self._start_event = None
        if self.interactive:
            for artist in self._artists:
                artist.set_animated(False)
        else:
            for artist in self._artists:
                artist.set_visible(False)
        self._modifiers = set()
        if selection:
            self._prev_geometry = self.get_geometry()
            self.on_select(self)
            self._has_selected = True
        self.canvas.draw_idle()

    def _handle_button_press(self, event):
        if (not self._drawing and not self.allow_redraw and
                self._has_selected):
            self.focused = self.patch.contains(event)[0]

        if self.interactive and not self._drawing:
            self._dragging, idx = self._size_handles.contains(event)
            if self._dragging:
                self._drag_idx = idx['ind'][0]
            elif self.patch.contains(event)[0]:
                self._moving = True
                self._dragging = True

            self._rotating = self._rot_handle.contains(event)

        if (self._drawing or self._dragging or self._rotating or self.allow_redraw or
                not self._has_selected):
            if self._moving:
                self._start_drawing(event)
            else:
                self._on_press(event)

    def _handle_motion_notify(self, event):
        if self._drawing:
            if self._moving:
                self._move(event)
            else:
                self._on_motion(event)
            self.on_motion(self)

    def _handle_button_release(self, event):
        if self._drawing:
            if self._moving:
                self._finish_drawing(event)
                self._moving = False
            else:
                self._on_release(event)
        self._dragging = False

    def _handle_key_press(self, event):
        """Handle key_press_event defaults and call to subclass handler"""

        if not self._drawing and not self.focused:
            return

        if event.key == self._keys['clear']:
            if self._dragging:
                self.set_geometry(**self._prev_geometry)
                self._finish_drawing(event, False)
            elif self._drawing:
                for artist in self._artists:
                    artist.set_visible(False)
                self._finish_drawing(event, False)
            return

        elif event.key == self._keys['accept']:
            if self._drawing:
                self._finish_drawing(event)

            self.on_accept(self)
            if self.allow_redraw:
                for artist in self._artists:
                    artist.set_visible(False)
                self.canvas.draw_idle()

        for (modifer, key) in self._keys.items():
            if modifer == 'move' and not self.interactive:
                continue
            if key in event.key:
                self._modifiers.add(modifer)
        self._on_key_press(event)

    def _handle_key_release(self, event):
        if self.focused:
            for (modifier, key) in self._keys.items():
                if key in event.key:
                    self._modifiers.discard(modifier)
            self._on_key_release(event)

    def _handle_scroll(self, event):
        if self.focused:
            self._on_scroll(event)

    #############################################################
    # The following are meant to be subclassed as needed.
    #############################################################
    def _make_patch(self, **props):
        """Initialize the patch"""
        return Patch(**props)

    def _get_handle_verts(self):
        """Get the handle vertices for a tool, not including the center.

        Return an (N, 2) array of vertices.
        """
        return None

    def _move(self, event):
        """Move the shape"""
        pass

    def _on_press(self, event):
        """Handle a button_press_event"""
        self._start_drawing(event)

    def _on_release(self, event):
        """Handle a button_release_event"""
        self._finish_drawing(event, True)


HANDLE_ORDER = ['NW', 'NE', 'SE', 'SW', 'W', 'N', 'E', 'S']


@docstring.dedent_interpd
class RectangleTool(BasePatchTool):

    """Interactive rectangle selection tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.

    %(BaseInteractiveTool)s
    %(BasePolygonTool)s
    """

    def get_geometry(self):
        """Get the geometry of the ellipse tool.

        Returns
        -------
        geometry: dict
            xy: The (x0, y0) origin point.
            width: The width of the tool.
            height: The height of the tool
            angle: The angle of the tool (counter-clockwise).
        """
        return dict(xy=self.patch.get_xy(),
                    width=self.patch.get_width(),
                    height=self.patch.get_height(),
                    angle=self.patch.get_angle())

    def _make_patch(self, **overrides):
        props = dict(facecolor='red', edgecolor='black', visible=False,
                     alpha=0.2, fill=True, picker=10, linewidth=2,
                     zorder=1)
        props.update(overrides)
        return Rectangle((0, 0), 0, 0, **props)

    def _get_handle_verts(self):
        geometry = self.get_geometry()
        width, height = geometry['width'], geometry['height']
        if isinstance(self.patch, Ellipse):
            xm, ym = geometry['center']
            x0, y0 = xm - width / 2, ym - height / 2
        else:
            x0, y0 = geometry['xy']
            xm, ym = x0 + width / 2, y0 + height / 2
        w = width / 2
        h = height / 2
        xc = xm - w
        yc = ym - h
        xe = xm - w, xm
        ye = ym, ym - h
        x = np.hstack((xc, xe))
        y = np.hstack((yc, ye))
        pts = np.vstack((x, y)).T
        rot_trans = transforms.Affine2D()
        if isinstance(self.patch, Ellipse):
            rot_trans.rotate_deg_around(xm, ym, geometry['angle'])
        else:
            rot_trans.rotate_deg_around(x0, y0, geometry['angle'])
        return rot_trans.transform(pts), rot_trans.transform((xm, ym + h))

    def _on_motion(self, event):
        # Resize an existing shape.
        if self._dragging:
            geometry = self._prev_geometry
            width, height = geometry['width'], geometry['height']
            if isinstance(self.patch,  Ellipse):
                x0, y0 = geometry['center']
                x0 -= width / 2
                y0 -= height / 2
            else:
                x0, y0 = geometry['xy']
            x1, y1 = x0 + width, y0 + height

            handle = HANDLE_ORDER[self._drag_idx]
            if handle in ['NW', 'SW', 'W']:
                x0 = event.xdata
            elif handle in ['NE', 'SE', 'E']:
                x1 = event.xdata
            if handle in ['NE', 'N', 'NW']:
                y0 = event.ydata
            elif handle in ['SE', 'S', 'SW']:
                y1 = event.ydata

        # Draw new shape.
        else:
            center = [self._start_event.xdata, self._start_event.ydata]
            center_pix = [self._start_event.x, self._start_event.y]
            dx = (event.xdata - center[0]) / 2.
            dy = (event.ydata - center[1]) / 2.

            # Draw a square shape.
            if 'square' in self._modifiers:
                dx_pix = abs(event.x - center_pix[0])
                dy_pix = abs(event.y - center_pix[1])
                if not dx_pix:
                    return
                maxd = max(abs(dx_pix), abs(dy_pix))
                if abs(dx_pix) < maxd:
                    dx *= maxd / (abs(dx_pix) + 1e-6)
                if abs(dy_pix) < maxd:
                    dy *= maxd / (abs(dy_pix) + 1e-6)

            # Draw from center.
            if 'center' in self._modifiers:
                dx *= 2
                dy *= 2

            # Draw from corner.
            else:
                center[0] += dx
                center[1] += dy

            x0, x1, y0, y1 = (center[0] - dx, center[0] + dx,
                              center[1] - dy, center[1] + dy)

        # Update the shape.
        width, height = x1 - x0, y1 - y0
        if isinstance(self.patch, Ellipse):
            self.set_geometry(center=(x0 + width / 2, y0 + height / 2),
                              width=width, height=height)
        else:
            self.set_geometry(xy=(x0, y0), width=width, height=height)

    def _move(self, event):
        geo = self.get_geometry()
        self.set_geometry(xy=(event.xdata - geo['width'] / 2,
                              event.ydata - geo['height'] / 2))


@docstring.dedent_interpd
class EllipseTool(RectangleTool):

    """Interactive ellipse selection tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.

    %(BaseInteractiveTool)s
    %(BasePolygonTool)s
    width: float
        The width of the ellipse in data units (read-only).
    height: float
        The height of the ellipse in data units (read-only).
    """

    def get_geometry(self):
        """Get the geometry of the ellipse tool.

        Returns
        -------
        geometry: dict
            center: The (x0, y0) center point.
            width: The width of the tool.
            height: The height of the tool
            angle: The angle of the tool (counter-clockwise).
        """
        return dict(center=self.patch.center,
                    width=self.patch.width,
                    height=self.patch.height,
                    angle=self.patch.angle)

    def _make_patch(self, **overrides):
        props = dict(facecolor='red', edgecolor='black', visible=False,
                     alpha=0.2, fill=True, picker=10, linewidth=2,
                     zorder=1)
        props.update(overrides)
        return Ellipse((0, 0), 0, 0, **props)

    def _move(self, event):
        self.set_geometry(center=(event.xdata, event.ydata))


@docstring.dedent_interpd
class LineTool(BasePatchTool):

    """Interactive line selection tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.
    %(BaseInteractiveTool)s
    %(BasePolygonTool)s
    """

    def get_geometry(self):
        """Get the geometry of the line tool.

        Returns
        -------
        geometry: dict
            end_points: The [(x0, y0), (x1, y1)] points.
            width: The width of the tool in pixels.
        """
        verts = self.patch.get_xy()
        p0x = (verts[0, 0] + verts[1, 0]) / 2
        p0y = (verts[0, 1] + verts[1, 1]) / 2
        p1x = (verts[3, 0] + verts[2, 0]) / 2
        p1y = (verts[3, 1] + verts[2, 1]) / 2
        pts = np.array([[p0x, p0y], [p1x, p1y]])
        return dict(end_points=pts, width=self._width)

    def set_geometry(self, end_points=None, width=None):
        """Set the geometry of the line tool.

        Parameters
        ----------
        end_points: [(xo, y0), (x1, y1)]
           The end points of the tool
        width: int
            The width in pixels of the line
        """
        geometry = self.get_geometry()
        self.width = width or geometry['width']
        if end_points is None:
            end_points = geometry['end_points']
        pts = np.asarray(end_points)
        # Get the widths in data units.
        xfm = self.ax.transData.inverted()
        x0, y0 = xfm.transform((0, 0))
        x1, y1 = xfm.transform((self.width, self.width))
        wx, wy = abs(x1 - x0), abs(y1 - y0)

        # Find line segments centered on the end points perpendicular to the
        # line and the proper width.
        # http://math.stackexchange.com/a/9375
        if (pts[1, 0] == pts[0, 0]):
            c, s = 0, 1
        elif (pts[1, 1] == pts[0, 1]):
            c, s = 0, 0
        else:
            m = - 1 / ((pts[1, 1] - pts[0, 1]) /
                       (pts[1, 0] - pts[0, 0]))
            c = 1 / np.sqrt(1 + m ** 2)
            s = m / np.sqrt(1 + m ** 2)

        p0 = pts[0, :]
        v00 = p0[0] + wx / 2 * c, p0[1] + wy / 2 * s
        v01 = p0[0] - wx / 2 * c, p0[1] - wy / 2 * s

        p1 = pts[1, :]
        v10 = p1[0] + wx / 2 * c, p1[1] + wy / 2 * s
        v11 = p1[0] - wx / 2 * c, p1[1] - wy / 2 * s

        super(LineTool, self).set_geometry(xy=(v00, v01, v11, v10))

    def _make_patch(self, **overrides):
        self._width = 1
        props = dict(facecolor='red', edgecolor='red', visible=False,
                     alpha=0.5, fill=True, picker=10, linewidth=2,
                     zorder=1)
        props.update(overrides)
        return Polygon([(0, 0), (0, 0), (0, 0), (0, 0)], **props)

    def _get_handle_verts(self):
        return self.get_geometry()['end_points'], None

    def _on_press(self, event):
        if not self._dragging:
            self.set_geometry(end_points=[[event.xdata, event.ydata],
                               [event.xdata, event.ydata]])
            self._dragging = True
            self._drag_idx = 1
        self._start_drawing(event)

    def _on_motion(self, event):
        end_points = self.get_geometry()['end_points']
        end_points[self._drag_idx, :] = event.xdata, event.ydata
        self.set_geometry(end_points=end_points)

    def _move(self, event):
        pts = self.get_geometry()['end_points']
        pts[:, 0] += event.xdata - (pts[1, 0] + pts[0, 0]) / 2
        pts[:, 1] += event.ydata - (pts[1, 1] + pts[0, 1]) / 2
        self.set_geometry(end_points=pts)

    def _on_scroll(self, event):
        if event.button == 'up':
            self.width += 1
        elif event.button == 'down' and self.width > 1:
            self.width -= 1

    def _on_key_press(self, event):
        if event.key == '+':
            self.width += 1
        elif event.key == '-' and self.width > 1:
            self.width -= 1


LABELS_CMAP = mcolors.ListedColormap(['white', 'red', 'dodgerblue', 'gold',
                                      'greenyellow', 'blueviolet'])


@docstring.dedent_interpd
class PaintTool(BaseTool):

    """Interactive paint tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.
    %(BaseInteractiveTool)s
    """

    @docstring.dedent_interpd
    def __init__(self, ax, on_select=None, on_motion=None, on_accept=None,
                 overlay_props=None, cursor_props=None, radius=5,
                 useblit=True, button=None, keys=None):
        """Initialize the tool.
        %(BaseInteractiveToolInit)s
        """
        super(PaintTool, self).__init__(ax, on_select=on_select,
            on_motion=on_motion, on_accept=on_accept,
            useblit=useblit, button=button, keys=keys)
        self.cmap = LABELS_CMAP
        self._useblit = useblit and self.canvas.supports_blit
        self._previous = None
        self._overlay = None
        self._overlay_plot = None
        self._cursor_shape = [0, 0, 0]

        props = dict(edgecolor='r', facecolor='0.7', alpha=1,
                     animated=self._useblit, visible=False, zorder=2)
        props.update(cursor_props or {})
        self._cursor = Rectangle((0, 0), 0, 0, **props)
        self.ax.add_patch(self._cursor)

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        if y0 < y1:
            origin = 'lower'
        else:
            origin = 'upper'
        props = dict(cmap=self.cmap, alpha=0.5, origin=origin,
                     norm=mcolors.NoNorm(), visible=False, zorder=1,
                     extent=(x0, x1, y0, y1), aspect=self.ax.get_aspect())
        props.update(overlay_props or {})

        extents = self.ax.get_window_extent().extents
        self._offsetx = extents[0]
        self._offsety = extents[1]
        self._shape = (extents[3] - extents[1], extents[2] - extents[0])
        self._overlay = np.zeros(self._shape, dtype='uint8')
        self._overlay_plot = self.ax.imshow(self._overlay, **props)

        self._artists = [self._cursor, self._overlay_plot]

        # These must be called last
        self.label = 1
        self.radius = radius
        self._drawing = True
        for artist in self._artists:
            artist.set_visible(True)

    @property
    def overlay(self):
        return self._overlay

    @overlay.setter
    def overlay(self, image):
        self._overlay = image
        if image is None:
            self.ax.images.remove(self._overlay_plot)
            self._update()
            return
        self.ax.set_data(image)
        self._shape = image.shape
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self._overlay_plot.set_extent(x0, x1, y0, y1)
        # Update the radii and window.
        self.radius = self._radius
        self._update()

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        if value >= self.cmap.N:
            raise ValueError('Maximum label value = %s' % len(self.cmap - 1))
        self._label = value
        self._cursor.set_edgecolor(self.cmap(value))

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, r):
        self._radius = r
        xfm = self.ax.transData.inverted()
        x0, y0 = xfm.transform((0, 0))
        x1, y1 = xfm.transform((r, r))
        self._rx, self._ry = abs(x1 - x0), abs(y1 - y0)

        self._cursor.set_width(self._rx * 2)
        self._cursor.set_height(self._ry * 2)

    def _on_press(self, event):
        self._update_cursor(event.xdata, event.ydata)
        self._update_overlay(event.x, event.y)
        self._update()

    def _on_motion(self, event):
        self._update_cursor(event.xdata, event.ydata)
        if event.button and event.button in self._buttons:
            self._update_overlay(event.x, event.y)
        self._update()

    def _on_release(self, event):
        pass

    def _update_overlay(self, x, y):
        col = x - self._offsetx
        row = y - self._offsety

        h, w = self._shape
        r = self._radius

        xmin = int(max(0, col - r))
        xmax = int(min(w, col + r + 1))
        ymin = int(max(0, row - r))
        ymax = int(min(h, row + r + 1))

        self._overlay[slice(ymin, ymax), slice(xmin, xmax)] = self.label
        self._overlay_plot.set_data(self._overlay)

    def _update_cursor(self, x, y):
        x = x - self._rx
        y = y - self._ry
        self._cursor.set_xy((x, y))


"""
RectangleTool and EllipseTool have three scale handles and a rotation
Handle
Get rid of the move handle a use contains for move
RegularPolygonTool has a scale handle and a rotation handle
"""


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = np.random.rand(100, 2)
    data[:, 1] *= 2

    fig, ax = plt.subplots()

    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    """
    ellipse = EllipseTool(ax)
    ellipse.set_geometry(center=(0.4, 0.5), width=0.3, height=0.5, angle=45)
    ax.invert_yaxis()

    def test(tool):
        print(tool.center, tool.width, tool.height)

    ellipse.on_accept = test
    ellipse.interactive = True
    """
    line = LineTool(ax)
    line.set_geometry(end_points=[[0.1, 0.1], [0.5, 0.5]])
    line.interactive = True
    """

    def test(tool):
        print(tool.overlay)

    p = PaintTool(ax)
    p.on_accept = test
    """
    plt.show()
