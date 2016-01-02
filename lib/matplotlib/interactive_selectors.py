# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy

import numpy as np

from .patches import Polygon
from .lines import Line2D


class BaseTool(object):

    """Interactive selection tool that is connected to a single
    :class:`~matplotlib.axes.Axes`.

    To guarantee that the tool remains responsive and not garbage-collected,
    a reference to the object should be maintained by the user.

    This is necessary because the callback registry
    maintains only weak-refs to the functions, which are member
    functions of the tool.  If there are no references to the tool
    object it may be garbage collected which will disconnect the
    callbacks.

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

    Attributes
    ----------
    ax: :class:`matplotlib.axes.Axes`
        The parent axes for the tool.
    canvas: :class:`~matplotlib.backend_bases.FigureCanvasBase` subclass
        The parent figure canvas for the tool.
    active: bool
        If False, the widget does not respond to events.
    interactive: boolean
        Whether to allow interaction with the shape using handles.
    allow_redraw: boolean
        Whether to allow the tool to redraw itself or whether it must be
        drawn programmatically and then dragged.
    verts: nd-array of floats (2, N)
        The vertices of the tool.
    """

    def __init__(self, ax, on_select=None, on_move=None, on_accept=None,
                 interactive=True, allow_redraw=True,
                 shape_props=None, handle_props=None,
                 useblit=True, button=None, keys=None):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.active = True
        self.interactive = interactive
        self.allow_redraw = allow_redraw

        self._callback_on_move = _dummy if on_move is None else on_move
        self._callback_on_accept = _dummy if on_accept is None else on_accept
        self._callback_on_select = _dummy if on_select is None else on_select

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
                     alpha=0.2, fill=True, pickradius=10)
        props.update(shape_props or {})
        self._patch = Polygon([[0, 0], [1, 1]], True, **props)
        self.ax.add_patch(self._patch)

        props = dict(marker='0', markersize=7, mfc='w', ls='none',
                     alpha=0.5, visible=False, label='_nolegend_',
                     pickradius=10)
        props.update(handle_props or {})
        self._handles = Line2D([], [], **props)
        self.ax.add_line(self._handles)

        self._artists = [self._patch, self._handles]
        self._state = set()
        self._drawing = False
        self._dragging = False
        self._verts = []
        self._prev_verts = None
        self._background = None
        self._prevxy = None

        # Connect the major canvas events to methods."""
        self._cids = []
        self._connect_event('pick_event', self._handle_pick)
        self._connect_event('motion_notify_event', self._handle_event)
        self._connect_event('button_press_event', self._handle_event)
        self._connect_event('button_release_event', self._handle_event)
        self._connect_event('draw_event', self._handle_draw)
        self._connect_event('key_press_event', self._handle_key_press)
        self._connect_event('key_release_event', self._handle_event)
        self._connect_event('scroll_event', self._handle_event)

    @property
    def verts(self):
        return self._verts

    @verts.setter
    def verts(self, value):
        value = np.asarray(value)
        assert value.ndim == 2
        assert value.shape[1] == 2
        self._verts = np.array(value)
        self._patch.set_xy(value)
        if self.interactive:
            self._set_handles_xy(value)
            self._handles.set_animated(False)
        self._patch.set_animated(False)
        self.canvas.draw_idle()

    def remove(self):
        """Clean up the tool."""
        for c in self._cids:
            self.canvas.mpl_disconnect(c)
        for artist in self._artists:
            self.ax.remove(artist)
        self.canvas.draw_idle()

    def _handle_pick(self, artist, event):
        if not self.interactive:
            return
        # TODO: implement picking logic.
        pass

    def _handle_draw(self, event):
        """Update the ax background on a draw event"""
        if self._useblit:
            self._background = self.canvas.copy_from_bbox(self.ax.bbox)

    def _handle_event(self, event):
        """Handle default actions for events and call to event handlers"""
        if self._ignore(event):
            return
        event = self._clean_event(event)

        if event.type == 'button_press_event':
            self._on_press(event)

        elif event.type == 'motion_notify_event':
            if self._drawing:
                self._on_move(event)
                self._callback_on_move(self)

        elif event.type == 'button_release_event':
            if self._drawing:
                self._on_release(event)
                if not self.drawing:
                    self._callback_on_select(self)

        elif event.type == 'key_release_event':
            for (state, modifier) in self.state_modifier_keys.items():
                # Keep move state locked until button released.
                if state == 'move' and self._drawing:
                    continue
                if modifier in event.key:
                    self.state.discard(state)
            self._on_key_release(event)

        elif event.type == 'scroll_event':
            self._on_scroll(event)

    def _handle_key_press(self, event):
        """Handle key_press_event defaults and call to subclass handler"""
        if event.key == self._keys['clear']:
            if self._dragging:
                self.verts = self._prev_verts
                self._finish_drawing()
            elif self._drawing:
                for artist in self._artists:
                    artist.set_visible(False)
                self._finish_drawing()
            return

        elif event.key == self._keys['accept']:
            if not self._drawing:
                self._callback_on_accept(self)
                if self.allow_redraw:
                    for artist in self._artists:
                        artist.set_visible(False)
                    self.canvas.draw_idle()

        for (state, modifier) in self._keys.items():
            if modifier in event.key:
                self._state.add(state)
        self._on_key_press(event)

    def _clean_event(self, event):
        """Clean up an event

        Use prev xy data if there is no xdata (left the axes)
        Limit the xdata and ydata to the axes limits
        Set the prev xy data
        """
        event = copy.copy(event)
        if event.xdata is not None:
            x0, x1 = self.ax.get_xbound()
            y0, y1 = self.ax.get_ybound()
            xdata = max(x0, event.xdata)
            event.xdata = min(x1, xdata)
            ydata = max(y0, event.ydata)
            event.ydata = min(y1, ydata)
            self._prevxy = event.xdata, event.ydata
        else:
            event.xdata, event.ydata = self._prevxy

        event.key = event.key or ''
        event.key = event.key.replace('ctrl', 'control')
        return event

    def _connect_event(self, event, callback):
        """Connect callback with an event.

        This should be used in lieu of `figure.canvas.mpl_connect` since this
        function stores callback ids for later clean up.
        """
        cid = self.canvas.mpl_connect(event, callback)
        self._cids.append(cid)

    def _ignore(self, event):
        """Return *True* if *event* should be ignored"""
        if not self._active or not self.ax.get_visible():
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

        if self._useblit:
            if self._background is not None:
                self.canvas.restore_region(self._background)
            for artist in self._artists:
                self.ax.draw_artist(artist)

            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def _start_drawing(self):
        """Start drawing or dragging the shape"""
        self._drawing = True
        if self.interactive:
            for artist in self._artists:
                artist.set_visible(False)
            self.canvas.draw()
            for artist in self._artists:
                artist.set_animated(self._useblit)
                artist.set_visible(True)
        else:
            self._handles.set_visible(False)
        self._update()

    def _finish_drawing(self):
        """Finish drawing or dragging the shape"""
        self._drawing = False
        self._dragging = False
        if self.interactive:
            for artist in self._artists:
                artist.set_animated(False)
        else:
            for artist in self._artists:
                artist.set_visible(False)
        self._state = set()
        self._prev_verts = self._verts
        self.canvas.draw_idle()

    #############################################################
    # The following are meant to be subclassed
    #############################################################
    def _set_handles_xy(self, value):
        """By default use the corners and the center."""
        value = np.vstack((value, np.mean(value, axis=0)))
        self._handles.set_xy(value)

    def _on_press(self, event):
        """Handle a button_press_event"""
        pass

    def _on_move(self, event):
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
