# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy

import numpy as np

# TODO: convert these to relative when finished
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
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

    Attributes
    ----------
    ax: :class:`~matplotlib.axes.Axes`
        The parent axes for the tool.
    canvas: :class:`~matplotlib.backend_bases.FigureCanvasBase` subclass
        The parent figure canvas for the tool.
    patch: :class:`~matplotlib.patches.Patch`
        The patch object contained by the tool.
    active: boolean
        If False, the widget does not respond to events.
    on_select: callable, optional
        A callback for when a selection is made `on_select(tool)`.
    on_move: callable, optional
        A callback for when the tool is moved `on_move(tool)`.
    on_accept: callable, optional
        A callback for when the selection is accepted `on_accept(tool)`.
        This is called in response to an 'accept' key event.
    interactive: boolean
        Whether to allow interaction with the shape using handles.
    allow_redraw: boolean
        Whether to allow the tool to redraw itself or whether it must be
        drawn programmatically and then dragged.
    verts: nd-array of floats (N, 2)
        The vertices of the tool in data units.
    focused: boolean
        Whether the tool has focus for keyboard and scroll events.""")

docstring.interpd.update(BaseInteractiveToolInit="""
    Parameters
    ----------
    ax: :class:`matplotlib.axes.Axes`
        The parent axes for the tool.
    on_select: callable, optional
        A callback for when a selection is made `on_select(tool)`.
    on_move: callable, optional
        A callback for when the tool is moved `on_move(tool)`.
    on_accept: callable, optional
        A callback for when the selection is accepted `on_accept(tool)`.
        This is called in response to an 'accept' key event.
    interactive: boolean, optional
        Whether to allow interaction with the shape using handles.
    allow_redraw: boolean, optional
        Whether to allow the tool to redraw itself or whether it must be
        drawn programmatically and then dragged.
    shape_props: dict, optional
        The properties of the shape patch.
    handle_props: dict, optional
        The properties of the handle markers.
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


class BaseTool(object):

    """Interactive selection tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.

    %(BaseInteractiveTool)s
    """

    def __init__(self, ax, on_select=None, on_move=None, on_accept=None,
                 interactive=True, allow_redraw=True,
                 shape_props=None, handle_props=None,
                 useblit=True, button=None, keys=None):
        """Initialize the tool.
        %(BaseInteractiveToolInit)s
        """
        self.ax = ax
        self.canvas = ax.figure.canvas
        self._active = True
        self.interactive = interactive
        self.allow_redraw = allow_redraw
        self.focused = True

        self.on_move = _dummy if on_move is None else on_move
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
            self._buttons = button

        props = dict(facecolor='red', edgecolor='black', visible=False,
                     alpha=0.2, fill=True, picker=5, linewidth=2)
        props.update(shape_props or {})
        self.patch = Polygon([[0, 0], [1, 1]], True, **props)
        self.ax.add_patch(self.patch)

        props = dict(marker='o', markersize=7, mfc='w', ls='none',
                     alpha=0.5, visible=False, label='_nolegend_',
                     picker=10)
        props.update(handle_props or {})
        self._handles = Line2D([], [], **props)
        self.ax.add_line(self._handles)

        self._artists = [self.patch, self._handles]
        self._state = set()
        self._drawing = False
        self._dragging = False
        self._drag_idx = None
        self._verts = []
        self._prev_data = None
        self._background = None
        self._prev_evt_xy = None
        self._start_event = None

        # Connect the major canvas events to methods."""
        self._cids = []
        self._connect_event('motion_notify_event', self._handle_event)
        self._connect_event('button_press_event', self._handle_event)
        self._connect_event('button_release_event', self._handle_event)
        self._connect_event('draw_event', self._handle_draw)
        self._connect_event('key_press_event', self._handle_key_press)
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

    @property
    def verts(self):
        """Get the (N, 2) vertices of the tool"""
        return self.patch.get_xy()

    @property
    def center(self):
        """Get the (x, y) center of the tool"""
        return (self._verts.min(axis=0) + self._verts.max(axis=0)) / 2

    @property
    def extents(self):
        """Get the (x0, y0, width, height) extents of the tool"""
        x0, x1 = np.min(self._verts[:, 0]), np.max(self._verts[:, 0])
        y0, y1 = np.min(self._verts[:, 1]), np.max(self._verts[:, 1])
        return x0, y0, x1 - x0, y1 - y0

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

        if event.name == 'button_press_event':

            if not self._drawing and not self.allow_redraw:
                self.focused = self.patch.contains(event)[0]

            if self.interactive and not self._drawing:
                self._dragging, idx = self._handles.contains(event)
                if self._dragging:
                    self._drag_idx = idx['ind'][0]
                    # If the move handle was selected, enter move state.
                    if self._drag_idx == self._handles.get_xdata().size - 1:
                        self._state.add('move')

            if self._drawing or self._dragging or self.allow_redraw:
                if 'move' in self._state:
                    self._start_drawing(event)
                else:
                    self._on_press(event)

        elif event.name == 'motion_notify_event':
            if self._drawing:
                if 'move' in self._state:
                    center = self.center
                    self._verts[:, 0] += event.xdata - center[0]
                    self._verts[:, 1] += event.ydata - center[1]
                    self._set_verts(self._verts)
                else:
                    self._on_motion(event)
                self.on_move(self)

        elif event.name == 'button_release_event':
            if self._drawing:
                if 'move' in self._state:
                    self._finish_drawing(event)
                else:
                    self._on_release(event)
            self._dragging = False

        elif event.name == 'key_release_event' and self.focused:
            for (state, modifier) in self._keys.items():
                # Keep move state locked until drawing finished.
                if state == 'move' and self._drawing:
                    continue
                if modifier in event.key:
                    self._state.discard(state)
            self._on_key_release(event)

        elif event.name == 'scroll_event' and self.focused:
            self._on_scroll(event)

    def _handle_key_press(self, event):
        """Handle key_press_event defaults and call to subclass handler"""
        if not self.focused:
            return

        if event.key == self._keys['clear']:
            if self._dragging:
                self._set_verts(self._prev_verts)
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

        for (state, modifier) in self._keys.items():
            if state == 'move' and not self.interactive:
                continue
            if modifier in event.key:
                self._state.add(state)
        self._on_key_press(event)

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
        else:
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

    def _set_verts(self, value):
        """Commit a change to the tool vertices."""
        value = np.asarray(value)
        assert value.ndim == 2
        assert value.shape[1] == 2
        self._verts = np.array(value)
        if self._prev_data is None:
            self._prev_data = dict(verts=self._verts,
                                   center=self.center,
                                   extents=self.extents)
        self.patch.set_xy(self._verts)
        self.patch.set_visible(True)
        self.patch.set_animated(False)

        handles = self._get_handle_verts()
        handles = np.vstack((handles, self.center))
        self._handles.set_data(handles[:, 0], handles[:, 1])
        self._handles.set_visible(self.interactive)
        self._handles.set_animated(False)
        self._update()

    def _update(self, visible=True):
        """Update the artists while drawing"""
        if not self.ax.get_visible():
            return

        if self._useblit and self._drawing:
            if self._background is not None:
                self.canvas.restore_region(self._background)
            for artist in self._artists:
                artist.set_visible(visible)
                self.ax.draw_artist(artist)

            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

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
            self._handles.set_visible(False)
        # Blit without being visible if not draggin to avoid showing the old
        # shape.
        self._update(self._dragging)

    def _finish_drawing(self, selection=False):
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
        self._state = set()
        if selection:
            self._prev_data = dict(verts=self._verts,
                                   center=self.center,
                                   extents=self.extents)
            self.on_select(self)
        self.canvas.draw_idle()

    #############################################################
    # The following are meant to be subclassed as needed.
    #############################################################
    def _get_handle_verts(self):
        """Get the handle vertices for a tool, not including the center.

        Return an (N, 2) array of vertices.
        """
        return self._verts

    def _on_press(self, event):
        """Handle a button_press_event"""
        self._start_drawing(event)

    def _on_motion(self, event):
        """Handle a motion_notify_event"""
        pass

    def _on_release(self, event):
        """Handle a button_release_event"""
        self._finish_drawing(event)

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
for k in ('RectangleTool', 'EllipseTool', 'LineTool', 'BaseTool'):
    docstring.interpd.update({k: tooldoc})

# define BaseTool.__init__ docstring after the class has been added to interpd
docstring.dedent_interpd(BaseTool.__init__)


HANDLE_ORDER = ['NW', 'NE', 'SE', 'SW', 'W', 'N', 'E', 'S']


@docstring.dedent_interpd
class RectangleTool(BaseTool):

    """Interactive rectangle selection tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.

    %(BaseInteractiveTool)s
    width: float
        The width of the rectangle in data units.
    height: float
        The height of the rectangle in data units.
    """

    @property
    def width(self):
        """Get the width of the tool in data units"""
        return np.ptp(self._verts[:, 0])

    @property
    def height(self):
        """Get the height of the tool in data units"""
        return np.ptp(self._verts[:, 1])

    def set_geometry(self, x0, y0, width, height):
        """Set the geometry of the rectangle tool.

        Parameters
        ----------
        x0: float
            The left coordinate in data units.
        y0: float
            The bottom coordinate in data units.
        width:
            The width in data units.
        height:
            The height in data units.
        """
        radx = width / 2
        rady = height / 2
        center = x0 + width / 2, y0 + height / 2
        self._set_verts([[center - radx, center - rady],
                         [center - radx, center + rady],
                         [center + radx, center + rady],
                         [center + radx, center - rady]])

    def _get_handle_verts(self):
        xm, ym = self.center
        w = self.width / 2
        h = self.height / 2
        xc = xm - w, xm + w, xm + w, xm - w
        yc = ym - h, ym - h, ym + h, ym + h
        xe = xm - w, xm, xm + w, xm
        ye = ym, ym - h, ym, ym + h
        x = np.hstack((xc, xe))
        y = np.hstack((yc, ye))
        return np.vstack((x, y)).T

    def _on_motion(self, event):
        # Resize an existing shape.
        if self._dragging:
            x0, y0, width, height = self._prev_data['extents']
            x1 = x0 + width
            y1 = y0 + height
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
            if 'square' in self._state:
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
            if 'center' in self._state:
                dx *= 2
                dy *= 2

            # Draw from corner.
            else:
                center[0] += dx
                center[1] += dy

            x0, x1, y0, y1 = (center[0] - dx, center[0] + dx,
                              center[1] - dy, center[1] + dy)

        # Update the shape.
        self.set_geometry(x0, y0, x1 - x0, y1 - y0)


@docstring.dedent_interpd
class EllipseTool(RectangleTool):

    """Interactive ellipse selection tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.

    %(BaseInteractiveTool)s
    width: float
        The width of the ellipse in data units.
    height: float
        The height of the ellipse in data units.
    """

    def set_geometry(self, center, width, height):
        """Set the geometry of the ellipse tool.

        Parameters
        ----------
        center: (x, y) float
            The center coordinates in data units.
        width:
            The width in data units.
        height:
            The height in data units.
        """
        rad = np.arange(61) * 6 * np.pi / 180
        x = width / 2 * np.cos(rad) + center[0]
        y = height / 2 * np.sin(rad) + center[1]
        self._set_verts(np.vstack((x, y)).T)


@docstring.dedent_interpd
class LineTool(BaseTool):

    """Interactive line selection tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.
    %(BaseInteractiveTool)s
    width: float
        The width of the line in pixels.
    end_points: (2, 2) float
        The [(x0, y0), (x1, y1)] end points of the line in data units.
    angle: float
        The angle between the left point and the right point in radians in
        pixel space.
    """

    @docstring.dedent_interpd
    def __init__(self, ax, on_select=None, on_move=None, on_accept=None,
             interactive=True, allow_redraw=True,
             shape_props=None, handle_props=None,
             useblit=True, button=None, keys=None):
        """Initialize the tool.
        %(BaseInteractiveToolInit)s
        """
        props = dict(edgecolor='red', visible=False,
                     alpha=0.5, fill=True, picker=5, linewidth=1)
        props.update(shape_props or {})
        super(LineTool, self).__init__(ax, on_select=on_select,
            on_move=on_move, on_accept=on_accept, interactive=interactive,
            allow_redraw=allow_redraw, shape_props=props,
            handle_props=handle_props, useblit=useblit, button=button,
            keys=keys)

    @property
    def width(self):
        """Get the width of the line in pixels."""
        return self._width

    @property
    def end_points(self):
        """Get the end points of the line in data units."""
        p0x = (self._verts[0, 0] + self._verts[1, 0]) / 2
        p0y = (self._verts[0, 1] + self._verts[1, 1]) / 2
        p1x = (self._verts[3, 0] + self._verts[2, 0]) / 2
        p1y = (self._verts[3, 1] + self._verts[2, 1]) / 2
        return np.array([[p0x, p0y], [p1x, p1y]])

    @property
    def angle(self):
        """Find the angle between the left and right points in pixel space."""
        # Convert to pixels.
        pts = self.end_points
        pts = self.ax.transData.inverted().transform(pts)
        if pts[0, 0] < pts[1, 0]:
            return np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])
        else:
            return np.arctan2(pts[0, 1] - pts[1, 1], pts[0, 0] - pts[1, 0])

    def set_geometry(self, end_points, width):
        """Set the geometry of the line tool.

        Parameters
        ----------
        end_points: (2, 2) float
            The coordinates of the end points in data space.
        width:
            The width in pixels.
        """
        pts = np.asarray(end_points)
        self._width = width

        # Get the widths in data units.
        xfm = self.ax.transData.inverted()
        x0, y0 = xfm.transform((0, 0))
        x1, y1 = xfm.transform((width, width))
        wx, wy = x1 - x0, y1 - y0

        # Find line segments centered on the end points perpendicular to the
        # line and the proper width.
        # http://math.stackexchange.com/a/9375
        if (pts[1, 0] == pts[0, 0]):
            c, s = 0, 1
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

        self._set_verts((v00, v01, v11, v10))

    def _get_handle_verts(self):
        return self.end_points

    def _on_press(self, event):
        if not self._dragging:
            self._end_pts = [[event.xdata, event.ydata],
                             [event.xdata, event.ydata]]
            self._dragging = True
            self._drag_idx = 1
        self._start_drawing(event)

    def _on_motion(self, event):
        end_points = self.end_points
        end_points[self._drag_idx, :] = event.xdata, event.ydata
        self.set_geometry(end_points, self._width)

    def _on_scroll(self, event):
        if event.button == 'up':
            self.set_geometry(self.width + 1)
        elif event.button == 'down' and self.width > 1:
            self.set_geometry(self.width - 1)

    def _on_key_press(self, event):
        if event.key == '+':
            self.set_geometry(self.width + 1)
        elif event.key == '-' and self.width > 1:
            self.set_geometry(self.width - 1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = np.random.rand(100, 2)
    data[:, 1] *= 2

    fig, ax = plt.subplots()

    pts = ax.scatter(data[:, 0], data[:, 1], s=80)
    # ellipse = EllipseTool(ax)
    # ellipse.set_geometry((0.6, 1.1), 0.5, 0.5)
    # ax.invert_yaxis()

    # def test(tool):
    #     print(tool.center, tool.width, tool.height)
    # ellipse.on_accept = test
    # ellipse.allow_redraw = False
    line = LineTool(ax)
    line.set_geometry([[0.1, 0.1], [0.5, 0.5]], 10)
    line.allow_redraw = False

    plt.show()
