"""
Abstract base classes define the primitives for Tools.
These tools are used by `matplotlib.backend_managers.ToolManager`

:class:`ToolBase`
    Simple stateless tool

:class:`ToolToggleBase`
    Tool that has two states, only one Toggle tool can be
    active at any given time for the same
    `matplotlib.backend_managers.ToolManager`
"""

import six

import functools
import time
import warnings
from weakref import WeakKeyDictionary

import numpy as np

from matplotlib import cbook, rcParams
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import FigureCanvasBase, cursors


_registered_tools = {}


def register_tool(canvas_cls, tool_cls=None):
    """Declares that a class implements a tool for a certain canvas class.

    Can be used as a class decorator.
    """
    if tool_cls is None:
        return functools.partial(register_tool, canvas_cls)
    if (tool_cls.name, canvas_cls) in _registered_tools:
        raise KeyError("Tool {!r} is already registered for canvas class {!r}"
                       .format(tool_cls.name, canvas_cls.__name__))
    _registered_tools[tool_cls.name, canvas_cls] = tool_cls
    return tool_cls


def get_tool(name, canvas_cls):
    """Get the tool class for tool name *name* and backend *backend*.
    """
    for parent in canvas_cls.__mro__:
        try:
            return _registered_tools[name, parent]
        except KeyError:
            pass
    raise KeyError("Tool {!r} is not implemented for canvas class {!r}"
                   .format(name, canvas_cls.__name__))


class ToolBase(object):
    """
    Base tool class

    A base tool, only implements `trigger` method or not method at all.
    The tool is instantiated by `matplotlib.backend_managers.ToolManager`

    Attributes
    ----------
    toolmanager: `matplotlib.backend_managers.ToolManager`
        ToolManager that controls this Tool
    figure: `FigureCanvas`
        Figure instance that is affected by this Tool
    """

    # name = None
    """Tool **id**, must be unique.

    **String**.
    """

    default_keymap = None
    """
    Keymap to associate with this tool

    **String**: List of comma separated keys that will be used to call this
    tool when the keypress event of *self.figure.canvas* is emited
    """

    description = None
    """
    Description of the Tool

    **String**: If the Tool is included in the Toolbar this text is used
    as a Tooltip
    """

    image = None
    """
    Filename of the image

    **String**: Filename of the image to use in the toolbar. If None, the
    `name` is used as a label in the toolbar button
    """

    def __init__(self, toolmanager):
        warnings.warn('Treat the new Tool classes introduced in v1.5 as ' +
                      'experimental for now, the API will likely change in ' +
                      'version 2.1, and some tools might change name')
        self._toolmanager = toolmanager
        self._figure = None

    @property
    def figure(self):
        return self._figure

    @figure.setter
    def figure(self, figure):
        self.set_figure(figure)

    @property
    def canvas(self):
        if not self._figure:
            return None
        return self._figure.canvas

    @property
    def toolmanager(self):
        return self._toolmanager

    def set_figure(self, figure):
        """
        Assign a figure to the tool

        Parameters
        ----------
        figure: `Figure`
        """
        self._figure = figure

    def trigger(self, event):
        """
        Called when this tool gets used

        This method is called by
        `matplotlib.backend_managers.ToolManager.trigger_tool`

        Parameters
        ----------
        event: `Event`
            The Canvas event that caused this tool to be called
        """

    def destroy(self):
        """
        Destroy the tool

        This method is called when the tool is removed by
        `matplotlib.backend_managers.ToolManager.remove_tool`
        """
        pass


class ToolToggleBase(ToolBase):
    """
    Toggleable tool

    Every time it is triggered, it switches between enable and disable

    Parameters
    ----------
    ``*args``
        Variable length argument to be used by the Tool
    ``**kwargs``
        `toggled` if present and True, sets the initial state ot the Tool
        Arbitrary keyword arguments to be consumed by the Tool
    """

    radio_group = None
    """Attribute to group 'radio' like tools (mutually exclusive)

    **String** that identifies the group or **None** if not belonging to a
    group
    """

    cursor = None
    """Cursor to use when the tool is active"""

    default_toggled = False
    """Default of toggled state"""

    def __init__(self, *args, **kwargs):
        self._toggled = kwargs.pop('toggled', self.default_toggled)
        ToolBase.__init__(self, *args, **kwargs)

    def trigger(self, event):
        """Calls `enable` or `disable` based on `toggled` value"""
        if self._toggled:
            self.disable(event)
        else:
            self.enable(event)
        self._toggled = not self._toggled

    def enable(self, event=None):
        """
        Enable the toggle tool

        `trigger` calls this method when `toggled` is False
        """

        pass

    def disable(self, event=None):
        """
        Disable the toggle tool

        `trigger` call this method when `toggled` is True.

        This can happen in different circumstances

        * Click on the toolbar tool button
        * Call to `matplotlib.backend_managers.ToolManager.trigger_tool`
        * Another `ToolToggleBase` derived tool is triggered
          (from the same `ToolManager`)
        """

        pass

    @property
    def toggled(self):
        """State of the toggled tool"""

        return self._toggled

    def set_figure(self, figure):
        toggled = self.toggled
        if toggled:
            if self.figure:
                self.trigger(None)
            else:
                # if no figure the internal state is not changed
                # we change it here so next call to trigger will change it back
                self._toggled = False
        ToolBase.set_figure(self, figure)
        if toggled:
            if figure:
                self.trigger(None)
            else:
                # if there is no figure, triggen wont change the internal state
                # we change it back
                self._toggled = True


class SetCursorBase(ToolBase):
    """
    Change to the current cursor while inaxes

    This tool, keeps track of all `ToolToggleBase` derived tools, and calls
    set_cursor when a tool gets triggered
    """
    name = "cursor"

    def __init__(self, *args, **kwargs):
        ToolBase.__init__(self, *args, **kwargs)
        self._idDrag = None
        self._cursor = None
        self._default_cursor = cursors.POINTER
        self._last_cursor = self._default_cursor
        self.toolmanager.toolmanager_connect('tool_added_event',
                                             self._add_tool_cbk)

        # process current tools
        for tool in self.toolmanager.tools.values():
            self._add_tool(tool)

    def set_figure(self, figure):
        if self._idDrag:
            self.canvas.mpl_disconnect(self._idDrag)
        ToolBase.set_figure(self, figure)
        if figure:
            self._idDrag = self.canvas.mpl_connect(
                'motion_notify_event', self._set_cursor_cbk)

    def _tool_trigger_cbk(self, event):
        if event.tool.toggled:
            self._cursor = event.tool.cursor
        else:
            self._cursor = None

        self._set_cursor_cbk(event.canvasevent)

    def _add_tool(self, tool):
        """set the cursor when the tool is triggered"""
        if getattr(tool, 'cursor', None) is not None:
            self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name,
                                                 self._tool_trigger_cbk)

    def _add_tool_cbk(self, event):
        """Process every newly added tool"""
        if event.tool is self:
            return

        self._add_tool(event.tool)

    def _set_cursor_cbk(self, event):
        if not event:
            return

        if not getattr(event, 'inaxes', False) or not self._cursor:
            if self._last_cursor != self._default_cursor:
                self.set_cursor(self._default_cursor)
                self._last_cursor = self._default_cursor
        elif self._cursor:
            cursor = self._cursor
            if cursor and self._last_cursor != cursor:
                self.set_cursor(cursor)
                self._last_cursor = cursor

    def set_cursor(self, cursor):
        """
        Set the cursor

        This method has to be implemented per backend
        """
        raise NotImplementedError


@register_tool(FigureCanvasBase)
class ToolCursorPosition(ToolBase):
    """
    Send message with the current pointer position

    This tool runs in the background reporting the position of the cursor
    """
    name = "position"

    def __init__(self, *args, **kwargs):
        self._idDrag = None
        ToolBase.__init__(self, *args, **kwargs)

    def set_figure(self, figure):
        if self._idDrag:
            self.canvas.mpl_disconnect(self._idDrag)
        ToolBase.set_figure(self, figure)
        if figure:
            self._idDrag = self.canvas.mpl_connect(
                'motion_notify_event', self.send_message)

    def send_message(self, event):
        """Call `matplotlib.backend_managers.ToolManager.message_event`"""
        if self.toolmanager.messagelock.locked():
            return

        message = ' '

        if event.inaxes and event.inaxes.get_navigate():
            try:
                s = event.inaxes.format_coord(event.xdata, event.ydata)
            except (ValueError, OverflowError):
                pass
            else:
                artists = [a for a in event.inaxes.mouseover_set
                           if a.contains(event) and a.get_visible()]

                if artists:
                    a = cbook._topmost_artist(artists)
                    if a is not event.inaxes.patch:
                        data = a.get_cursor_data(event)
                        if data is not None:
                            s += ' [%s]' % a.format_cursor_data(data)

                message = s
        self.toolmanager.message_event(message)


@register_tool(FigureCanvasBase)
class ToolQuit(ToolBase):
    """Tool to call the figure manager destroy method"""

    name = "quit"
    description = 'Quit the figure'
    default_keymap = rcParams['keymap.quit']

    def trigger(self, event):
        Gcf.destroy_fig(self.figure)


@register_tool(FigureCanvasBase)
class ToolQuitAll(ToolBase):
    """Tool to call the figure manager destroy method"""

    name = "quit_all"
    description = 'Quit all figures'
    default_keymap = rcParams['keymap.quit_all']

    def trigger(self, event):
        Gcf.destroy_all()


@register_tool(FigureCanvasBase)
class ToolEnableAllNavigation(ToolBase):
    """Tool to enable all axes for toolmanager interaction"""

    name = "allnav"
    description = 'Enables all axes toolmanager'
    default_keymap = rcParams['keymap.all_axes']

    def trigger(self, event):
        if event.inaxes is None:
            return

        for a in self.figure.get_axes():
            if (event.x is not None and event.y is not None
                    and a.in_axes(event)):
                a.set_navigate(True)


@register_tool(FigureCanvasBase)
class ToolEnableNavigation(ToolBase):
    """Tool to enable a specific axes for toolmanager interaction"""

    name = "nav"
    description = 'Enables one axes toolmanager'
    default_keymap = (1, 2, 3, 4, 5, 6, 7, 8, 9)

    def trigger(self, event):
        if event.inaxes is None:
            return

        n = int(event.key) - 1
        for i, a in enumerate(self.figure.get_axes()):
            if (event.x is not None and event.y is not None
                    and a.in_axes(event)):
                a.set_navigate(i == n)


class _ToolGridBase(ToolBase):
    """Common functionality between ToolGrid and ToolMinorGrid."""

    _cycle = [(False, False), (True, False), (True, True), (False, True)]

    def trigger(self, event):
        ax = event.inaxes
        if ax is None:
            return
        try:
            x_state, x_which, y_state, y_which = self._get_next_grid_states(ax)
        except ValueError:
            pass
        else:
            ax.grid(x_state, which=x_which, axis="x")
            ax.grid(y_state, which=y_which, axis="y")
            ax.figure.canvas.draw_idle()

    @staticmethod
    def _get_uniform_grid_state(ticks):
        """
        Check whether all grid lines are in the same visibility state.

        Returns True/False if all grid lines are on or off, None if they are
        not all in the same state.
        """
        if all(tick.gridOn for tick in ticks):
            return True
        elif not any(tick.gridOn for tick in ticks):
            return False
        else:
            return None


@register_tool(FigureCanvasBase)
class ToolGrid(_ToolGridBase):
    """Tool to toggle the major grids of the figure"""

    name = "grid"
    description = 'Toogle major grids'
    default_keymap = rcParams['keymap.grid']

    def _get_next_grid_states(self, ax):
        if None in map(self._get_uniform_grid_state,
                       [ax.xaxis.minorTicks, ax.yaxis.minorTicks]):
            # Bail out if minor grids are not in a uniform state.
            raise ValueError
        x_state, y_state = map(self._get_uniform_grid_state,
                               [ax.xaxis.majorTicks, ax.yaxis.majorTicks])
        cycle = self._cycle
        # Bail out (via ValueError) if major grids are not in a uniform state.
        x_state, y_state = (
            cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)])
        return (x_state, "major" if x_state else "both",
                y_state, "major" if y_state else "both")


@register_tool(FigureCanvasBase)
class ToolMinorGrid(_ToolGridBase):
    """Tool to toggle the major and minor grids of the figure"""

    name = "grid_minor"
    description = 'Toogle major and minor grids'
    default_keymap = rcParams['keymap.grid_minor']

    def _get_next_grid_states(self, ax):
        if None in map(self._get_uniform_grid_state,
                       [ax.xaxis.majorTicks, ax.yaxis.majorTicks]):
            # Bail out if major grids are not in a uniform state.
            raise ValueError
        x_state, y_state = map(self._get_uniform_grid_state,
                               [ax.xaxis.minorTicks, ax.yaxis.minorTicks])
        cycle = self._cycle
        # Bail out (via ValueError) if minor grids are not in a uniform state.
        x_state, y_state = (
            cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)])
        return x_state, "both", y_state, "both"


@register_tool(FigureCanvasBase)
class ToolFullScreen(ToolToggleBase):
    """Tool to toggle full screen"""

    name = "fullscreen"
    description = 'Toogle Fullscreen mode'
    default_keymap = rcParams['keymap.fullscreen']

    def enable(self, event):
        self.figure.canvas.manager.full_screen_toggle()

    def disable(self, event):
        self.figure.canvas.manager.full_screen_toggle()


class AxisScaleBase(ToolToggleBase):
    """Base Tool to toggle between linear and logarithmic"""

    def trigger(self, event):
        if event.inaxes is None:
            return
        ToolToggleBase.trigger(self, event)

    def enable(self, event):
        self.set_scale(event.inaxes, 'log')
        self.figure.canvas.draw_idle()

    def disable(self, event):
        self.set_scale(event.inaxes, 'linear')
        self.figure.canvas.draw_idle()


@register_tool(FigureCanvasBase)
class ToolXScale(AxisScaleBase):
    """Tool to toggle between linear and logarithmic scales on the X axis"""

    name = "xscale"
    description = 'Toogle Scale X axis'
    default_keymap = rcParams['keymap.xscale']

    def set_scale(self, ax, scale):
        ax.set_xscale(scale)


@register_tool(FigureCanvasBase)
class ToolYScale(AxisScaleBase):
    """Tool to toggle between linear and logarithmic scales on the Y axis"""

    name = "yscale"
    description = 'Toogle Scale Y axis'
    default_keymap = rcParams['keymap.yscale']

    def set_scale(self, ax, scale):
        ax.set_yscale(scale)


@register_tool(FigureCanvasBase)
class ToolViewsPositions(ToolBase):
    """
    Auxiliary Tool to handle changes in views and positions

    Runs in the background and should get used by all the tools that
    need to access the figure's history of views and positions, e.g.

    * `ToolZoom`
    * `ToolPan`
    * `ToolHome`
    * `ToolBack`
    * `ToolForward`
    """

    name = "viewpos"

    def __init__(self, *args, **kwargs):
        self.views = WeakKeyDictionary()
        self.positions = WeakKeyDictionary()
        self.home_views = WeakKeyDictionary()
        ToolBase.__init__(self, *args, **kwargs)

    def add_figure(self, figure):
        """Add the current figure to the stack of views and positions"""

        if figure not in self.views:
            self.views[figure] = cbook.Stack()
            self.positions[figure] = cbook.Stack()
            self.home_views[figure] = WeakKeyDictionary()
            # Define Home
            self.push_current(figure)
            # Make sure we add a home view for new axes as they're added
            figure.add_axobserver(lambda fig: self.update_home_views(fig))

    def clear(self, figure):
        """Reset the axes stack"""
        if figure in self.views:
            self.views[figure].clear()
            self.positions[figure].clear()
            self.home_views[figure].clear()
            self.update_home_views()

    def update_view(self):
        """
        Update the view limits and position for each axes from the current
        stack position. If any axes are present in the figure that aren't in
        the current stack position, use the home view limits for those axes and
        don't update *any* positions.
        """

        views = self.views[self.figure]()
        if views is None:
            return
        pos = self.positions[self.figure]()
        if pos is None:
            return
        home_views = self.home_views[self.figure]
        all_axes = self.figure.get_axes()
        for a in all_axes:
            if a in views:
                cur_view = views[a]
            else:
                cur_view = home_views[a]
            a._set_view(cur_view)

        if set(all_axes).issubset(pos):
            for a in all_axes:
                # Restore both the original and modified positions
                a.set_position(pos[a][0], 'original')
                a.set_position(pos[a][1], 'active')

        self.figure.canvas.draw_idle()

    def push_current(self, figure=None):
        """
        Push the current view limits and position onto their respective stacks
        """
        if not figure:
            figure = self.figure
        views = WeakKeyDictionary()
        pos = WeakKeyDictionary()
        for a in figure.get_axes():
            views[a] = a._get_view()
            pos[a] = self._axes_pos(a)
        self.views[figure].push(views)
        self.positions[figure].push(pos)

    def _axes_pos(self, ax):
        """
        Return the original and modified positions for the specified axes

        Parameters
        ----------
        ax : (matplotlib.axes.AxesSubplot)
        The axes to get the positions for

        Returns
        -------
        limits : (tuple)
        A tuple of the original and modified positions
        """

        return (ax.get_position(True).frozen(),
                ax.get_position().frozen())

    def update_home_views(self, figure=None):
        """
        Make sure that self.home_views has an entry for all axes present in the
        figure
        """

        if not figure:
            figure = self.figure
        for a in figure.get_axes():
            if a not in self.home_views[figure]:
                self.home_views[figure][a] = a._get_view()

    def refresh_locators(self):
        """Redraw the canvases, update the locators"""
        for a in self.figure.get_axes():
            xaxis = getattr(a, 'xaxis', None)
            yaxis = getattr(a, 'yaxis', None)
            zaxis = getattr(a, 'zaxis', None)
            locators = []
            if xaxis is not None:
                locators.append(xaxis.get_major_locator())
                locators.append(xaxis.get_minor_locator())
            if yaxis is not None:
                locators.append(yaxis.get_major_locator())
                locators.append(yaxis.get_minor_locator())
            if zaxis is not None:
                locators.append(zaxis.get_major_locator())
                locators.append(zaxis.get_minor_locator())

            for loc in locators:
                loc.refresh()
        self.figure.canvas.draw_idle()

    def home(self):
        """Recall the first view and position from the stack"""
        self.views[self.figure].home()
        self.positions[self.figure].home()

    def back(self):
        """Back one step in the stack of views and positions"""
        self.views[self.figure].back()
        self.positions[self.figure].back()

    def forward(self):
        """Forward one step in the stack of views and positions"""
        self.views[self.figure].forward()
        self.positions[self.figure].forward()


class ViewsPositionsBase(ToolBase):
    """Base class for `ToolHome`, `ToolBack` and `ToolForward`"""

    _on_trigger = None

    def trigger(self, event):
        self.toolmanager.get_tool("viewpos").add_figure(self.figure)
        getattr(self.toolmanager.get_tool("viewpos"),
                self._on_trigger)()
        self.toolmanager.get_tool("viewpos").update_view()


@register_tool(FigureCanvasBase)
class ToolHome(ViewsPositionsBase):
    """Restore the original view lim"""

    name = "home"
    description = 'Reset original view'
    image = 'home'
    default_keymap = rcParams['keymap.home']
    _on_trigger = 'home'


@register_tool(FigureCanvasBase)
class ToolBack(ViewsPositionsBase):
    """Move back up the view lim stack"""

    name = "back"
    description = 'Back to previous view'
    image = 'back'
    default_keymap = rcParams['keymap.back']
    _on_trigger = 'back'


@register_tool(FigureCanvasBase)
class ToolForward(ViewsPositionsBase):
    """Move forward in the view lim stack"""

    name = "forward"
    description = 'Forward to next view'
    image = 'forward'
    default_keymap = rcParams['keymap.forward']
    _on_trigger = 'forward'


@register_tool(FigureCanvasBase)
class ConfigureSubplotsBase(ToolBase):
    """Base tool for the configuration of subplots"""

    name = "subplots"
    description = 'Configure subplots'
    image = 'subplots'


@register_tool(FigureCanvasBase)
class SaveFigureBase(ToolBase):
    """Base tool for figure saving"""

    name = "save"
    description = 'Save the figure'
    image = 'filesave'
    default_keymap = rcParams['keymap.save']


class ZoomPanBase(ToolToggleBase):
    """Base class for `ToolZoom` and `ToolPan`"""
    def __init__(self, *args):
        ToolToggleBase.__init__(self, *args)
        self._button_pressed = None
        self._xypress = None
        self._idPress = None
        self._idRelease = None
        self._idScroll = None
        self.base_scale = 2.
        self.scrollthresh = .5  # .5 second scroll threshold
        self.lastscroll = time.time()-self.scrollthresh

    def enable(self, event):
        """Connect press/release events and lock the canvas"""
        self.figure.canvas.widgetlock(self)
        self._idPress = self.figure.canvas.mpl_connect(
            'button_press_event', self._press)
        self._idRelease = self.figure.canvas.mpl_connect(
            'button_release_event', self._release)
        self._idScroll = self.figure.canvas.mpl_connect(
            'scroll_event', self.scroll_zoom)

    def disable(self, event):
        """Release the canvas and disconnect press/release events"""
        self._cancel_action()
        self.figure.canvas.widgetlock.release(self)
        self.figure.canvas.mpl_disconnect(self._idPress)
        self.figure.canvas.mpl_disconnect(self._idRelease)
        self.figure.canvas.mpl_disconnect(self._idScroll)

    def trigger(self, event):
        self.toolmanager.get_tool("viewpos").add_figure(self.figure)
        ToolToggleBase.trigger(self, event)

    def scroll_zoom(self, event):
        # https://gist.github.com/tacaswell/3144287
        if event.inaxes is None:
            return

        if event.button == 'up':
            # deal with zoom in
            scl = self.base_scale
        elif event.button == 'down':
            # deal with zoom out
            scl = 1/self.base_scale
        else:
            # deal with something that should never happen
            scl = 1

        ax = event.inaxes
        ax._set_view_from_bbox([event.x, event.y, scl])

        # If last scroll was done within the timing threshold, delete the
        # previous view
        if (time.time()-self.lastscroll) < self.scrollthresh:
            self.toolmanager.get_tool("viewpos").back()

        self.figure.canvas.draw_idle()  # force re-draw

        self.lastscroll = time.time()
        self.toolmanager.get_tool("viewpos").push_current()


@register_tool(FigureCanvasBase)
class ToolZoom(ZoomPanBase):
    """Zoom to rectangle"""
    # Subclasses should overrider _draw_rubberband and possibly
    # _remove_rubberband.

    name = "zoom"
    description = 'Zoom to rectangle'
    image = 'zoom_to_rect'
    default_keymap = rcParams['keymap.zoom']
    cursor = cursors.SELECT_REGION
    radio_group = 'default'

    def __init__(self, *args):
        ZoomPanBase.__init__(self, *args)
        self._ids_zoom = []

    def _draw_rubberband(self, x1, y1, x2, y2):
        raise NotImplementedError

    def _remove_rubberband(self):
        pass

    def _cancel_action(self):
        for zoom_id in self._ids_zoom:
            self.figure.canvas.mpl_disconnect(zoom_id)
        self._remove_rubberband()
        self.toolmanager.get_tool("viewpos").refresh_locators()
        self._xypress = None
        self._button_pressed = None
        self._ids_zoom = []
        return

    def _press(self, event):
        """the _press mouse button in zoom to rect mode callback"""

        # If we're already in the middle of a zoom, pressing another
        # button works to "cancel"
        if self._ids_zoom != []:
            self._cancel_action()

        if event.button == 1:
            self._button_pressed = 1
        elif event.button == 3:
            self._button_pressed = 3
        else:
            self._cancel_action()
            return

        x, y = event.x, event.y

        self._xypress = []
        for i, a in enumerate(self.figure.get_axes()):
            if (x is not None and y is not None and a.in_axes(event) and
                    a.get_navigate() and a.can_zoom()):
                self._xypress.append((x, y, a, i, a._get_view()))

        id1 = self.figure.canvas.mpl_connect(
            'motion_notify_event', self._mouse_move)
        id2 = self.figure.canvas.mpl_connect(
            'key_press_event', self._switch_on_zoom_mode)
        id3 = self.figure.canvas.mpl_connect(
            'key_release_event', self._switch_off_zoom_mode)

        self._ids_zoom = id1, id2, id3
        self._zoom_mode = event.key

    def _switch_on_zoom_mode(self, event):
        self._zoom_mode = event.key
        self._mouse_move(event)

    def _switch_off_zoom_mode(self, event):
        self._zoom_mode = None
        self._mouse_move(event)

    def _mouse_move(self, event):
        """the drag callback in zoom mode"""

        if self._xypress:
            x, y = event.x, event.y
            lastx, lasty, a, ind, view = self._xypress[0]
            (x1, y1), (x2, y2) = np.clip(
                [[lastx, lasty], [x, y]], a.bbox.min, a.bbox.max)
            if self._zoom_mode == "x":
                y1, y2 = a.bbox.intervaly
            elif self._zoom_mode == "y":
                x1, x2 = a.bbox.intervalx
            self._draw_rubberband(x1, y1, x2, y2)

    def _release(self, event):
        """the release mouse button callback in zoom to rect mode"""

        for zoom_id in self._ids_zoom:
            self.figure.canvas.mpl_disconnect(zoom_id)
        self._ids_zoom = []

        if not self._xypress:
            self._cancel_action()
            return

        last_a = []

        for cur_xypress in self._xypress:
            x, y = event.x, event.y
            lastx, lasty, a, _ind, view = cur_xypress
            # ignore singular clicks - 5 pixels is a threshold
            if abs(x - lastx) < 5 or abs(y - lasty) < 5:
                self._cancel_action()
                return

            # detect twinx,y axes and avoid double zooming
            twinx, twiny = False, False
            if last_a:
                for la in last_a:
                    if a.get_shared_x_axes().joined(a, la):
                        twinx = True
                    if a.get_shared_y_axes().joined(a, la):
                        twiny = True
            last_a.append(a)

            if self._button_pressed == 1:
                direction = 'in'
            elif self._button_pressed == 3:
                direction = 'out'
            else:
                continue

            a._set_view_from_bbox((lastx, lasty, x, y), direction,
                                  self._zoom_mode, twinx, twiny)

        self._zoom_mode = None
        self.toolmanager.get_tool("viewpos").push_current()
        self._cancel_action()


@register_tool(FigureCanvasBase)
class ToolPan(ZoomPanBase):
    """Pan axes with left mouse, zoom with right"""

    name = "pan"
    default_keymap = rcParams['keymap.pan']
    description = 'Pan axes with left mouse, zoom with right'
    image = 'move'
    cursor = cursors.MOVE
    radio_group = 'default'

    def __init__(self, *args):
        ZoomPanBase.__init__(self, *args)
        self._idDrag = None

    def _cancel_action(self):
        self._button_pressed = None
        self._xypress = []
        self.figure.canvas.mpl_disconnect(self._idDrag)
        self.toolmanager.messagelock.release(self)
        self.toolmanager.get_tool("viewpos").refresh_locators()

    def _press(self, event):
        if event.button == 1:
            self._button_pressed = 1
        elif event.button == 3:
            self._button_pressed = 3
        else:
            self._cancel_action()
            return

        x, y = event.x, event.y

        self._xypress = []
        for i, a in enumerate(self.figure.get_axes()):
            if (x is not None and y is not None and a.in_axes(event) and
                    a.get_navigate() and a.can_pan()):
                a.start_pan(x, y, event.button)
                self._xypress.append((a, i))
                self.toolmanager.messagelock(self)
                self._idDrag = self.figure.canvas.mpl_connect(
                    'motion_notify_event', self._mouse_move)

    def _release(self, event):
        if self._button_pressed is None:
            self._cancel_action()
            return

        self.figure.canvas.mpl_disconnect(self._idDrag)
        self.toolmanager.messagelock.release(self)

        for a, _ind in self._xypress:
            a.end_pan()
        if not self._xypress:
            self._cancel_action()
            return

        self.toolmanager.get_tool("viewpos").push_current()
        self._cancel_action()

    def _mouse_move(self, event):
        for a, _ind in self._xypress:
            # safer to use the recorded button at the _press than current
            # button: # multiple button can get pressed during motion...
            a.drag_pan(self._button_pressed, event.key, event.x, event.y)
        self.toolmanager.canvas.draw_idle()


default_tools = [
    'home', 'back', 'forward', 'zoom', 'pan', 'subplots', 'save',
    'grid', 'grid_minor', 'fullscreen', 'quit', 'quit_all', 'allnav', 'nav',
    'xscale', 'yscale', 'position', 'viewpos', 'cursor',
]
"""Default tools"""

default_toolbar_tools = [['navigation', ['home', 'back', 'forward']],
                         ['zoompan', ['pan', 'zoom', 'subplots']],
                         ['io', ['save']]]
"""Default tools in the toolbar"""


def add_tools_to_manager(toolmanager, tools=default_tools):
    """
    Add multiple tools to `ToolManager`

    Parameters
    ----------
    toolmanager: ToolManager
        `backend_managers.ToolManager` object that will get the tools added
    tools : List[str], optional
        The tools to add.
    """

    for tool_name in tools:
        toolmanager.add_tool(tool_name)


def add_tools_to_container(container, tools=default_toolbar_tools):
    """
    Add multiple tools to the container.

    Parameters
    ----------
    container: Container
        `backend_bases.ToolContainerBase` object that will get the tools added
    tools : list, optional
        List in the form
        [[group1, [tool1, tool2 ...]], [group2, [...]]]
        Where the tools given by tool1, and tool2 will display in group1.
        See `add_tool` for details.
    """

    for group, grouptools in tools:
        for position, tool in enumerate(grouptools):
            container.add_tool(tool, group, position)
