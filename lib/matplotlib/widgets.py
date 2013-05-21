"""
GUI Neutral widgets
===================

Widgets that are designed to work for any of the GUI backends.
All of these widgets require you to predefine an :class:`matplotlib.axes.Axes`
instance and pass that as the first arg.  matplotlib doesn't try to
be too smart with respect to layout -- you will have to figure out how
wide and tall you want your Axes to be to accommodate your widget.
"""

from __future__ import print_function
import numpy as np

from mlab import dist
from patches import Circle, Rectangle
from lines import Line2D
from transforms import blended_transform_factory


class LockDraw:
    """
    Some widgets, like the cursor, draw onto the canvas, and this is not
    desirable under all circumstances, like when the toolbar is in
    zoom-to-rect mode and drawing a rectangle.  The module level "lock"
    allows someone to grab the lock and prevent other widgets from
    drawing.  Use ``matplotlib.widgets.lock(someobj)`` to pr
    """
    # FIXME: This docstring ends abruptly without...

    def __init__(self):
        self._owner = None

    def __call__(self, o):
        """reserve the lock for *o*"""
        if not self.available(o):
            raise ValueError('already locked')
        self._owner = o

    def release(self, o):
        """release the lock"""
        if not self.available(o):
            raise ValueError('you do not own this lock')
        self._owner = None

    def available(self, o):
        """drawing is available to *o*"""
        return not self.locked() or self.isowner(o)

    def isowner(self, o):
        """Return True if *o* owns this lock"""
        return self._owner is o

    def locked(self):
        """Return True if the lock is currently held by an owner"""
        return self._owner is not None


class Widget(object):
    """
    Abstract base class for GUI neutral widgets
    """
    drawon = True
    eventson = True


class AxesWidget(Widget):
    """Widget that is connected to a single :class:`~matplotlib.axes.Axes`.

    Attributes:

    *ax* : :class:`~matplotlib.axes.Axes`
        The parent axes for the widget
    *canvas* : :class:`~matplotlib.backend_bases.FigureCanvasBase` subclass
        The parent figure canvas for the widget.
    *active* : bool
        If False, the widget does not respond to events.
    """
    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.cids = []
        self.active = True

    def connect_event(self, event, callback):
        """Connect callback with an event.

        This should be used in lieu of `figure.canvas.mpl_connect` since this
        function stores call back ids for later clean up.
        """
        cid = self.canvas.mpl_connect(event, callback)
        self.cids.append(cid)

    def disconnect_events(self):
        """Disconnect all events created by this widget."""
        for c in self.cids:
            self.canvas.mpl_disconnect(c)

    def ignore(self, event):
        """Return True if event should be ignored.

        This method (or a version of it) should be called at the beginning
        of any event callback.
        """
        return not self.active


class Button(AxesWidget):
    """
    A GUI neutral button

    The following attributes are accessible

      *ax*
        The :class:`matplotlib.axes.Axes` the button renders into.

      *label*
        A :class:`matplotlib.text.Text` instance.

      *color*
        The color of the button when not hovering.

      *hovercolor*
        The color of the button when hovering.

    Call :meth:`on_clicked` to connect to the button
    """

    def __init__(self, ax, label, image=None,
                 color='0.85', hovercolor='0.95'):
        """
        *ax*
            The :class:`matplotlib.axes.Axes` instance the button
            will be placed into.

        *label*
            The button text. Accepts string.

        *image*
            The image to place in the button, if not *None*.
            Can be any legal arg to imshow (numpy array,
            matplotlib Image instance, or PIL image).

        *color*
            The color of the button when not activated

        *hovercolor*
            The color of the button when the mouse is over it
        """
        AxesWidget.__init__(self, ax)

        if image is not None:
            ax.imshow(image)
        self.label = ax.text(0.5, 0.5, label,
                             verticalalignment='center',
                             horizontalalignment='center',
                             transform=ax.transAxes)

        self.cnt = 0
        self.observers = {}

        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)
        ax.set_navigate(False)
        ax.set_axis_bgcolor(color)
        ax.set_xticks([])
        ax.set_yticks([])
        self.color = color
        self.hovercolor = hovercolor

        self._lastcolor = color

    def _click(self, event):
        if self.ignore(event):
            return
        if event.inaxes != self.ax:
            return
        if not self.eventson:
            return
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)

    def _release(self, event):
        if self.ignore(event):
            return
        if event.canvas.mouse_grabber != self.ax:
            return
        event.canvas.release_mouse(self.ax)
        if not self.eventson:
            return
        if event.inaxes != self.ax:
            return
        for cid, func in self.observers.iteritems():
            func(event)

    def _motion(self, event):
        if self.ignore(event):
            return
        if event.inaxes == self.ax:
            c = self.hovercolor
        else:
            c = self.color
        if c != self._lastcolor:
            self.ax.set_axis_bgcolor(c)
            self._lastcolor = c
            if self.drawon:
                self.ax.figure.canvas.draw()

    def on_clicked(self, func):
        """
        When the button is clicked, call this *func* with event

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        """remove the observer with connection id *cid*"""
        try:
            del self.observers[cid]
        except KeyError:
            pass


class Slider(AxesWidget):
    """
    A slider representing a floating point range

    The following attributes are defined
      *ax*        : the slider :class:`matplotlib.axes.Axes` instance

      *val*       : the current slider value

      *vline*     : a :class:`matplotlib.lines.Line2D` instance
                     representing the initial value of the slider

      *poly*      : A :class:`matplotlib.patches.Polygon` instance
                     which is the slider knob

      *valfmt*    : the format string for formatting the slider text

      *label*     : a :class:`matplotlib.text.Text` instance
                     for the slider label

      *closedmin* : whether the slider is closed on the minimum

      *closedmax* : whether the slider is closed on the maximum

      *slidermin* : another slider - if not *None*, this slider must be
                     greater than *slidermin*

      *slidermax* : another slider - if not *None*, this slider must be
                     less than *slidermax*

      *dragging*  : allow for mouse dragging on slider

    Call :meth:`on_changed` to connect to the slider event
    """
    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valfmt='%1.2f',
                 closedmin=True, closedmax=True, slidermin=None,
                 slidermax=None, dragging=True, **kwargs):
        """
        Create a slider from *valmin* to *valmax* in axes *ax*

        *valinit*
            The slider initial position

        *label*
            The slider label

        *valfmt*
            Used to format the slider value

        *closedmin* and *closedmax*
            Indicate whether the slider interval is closed

        *slidermin* and *slidermax*
            Used to constrain the value of this slider to the values
            of other sliders.

        additional kwargs are passed on to ``self.poly`` which is the
        :class:`matplotlib.patches.Rectangle` which draws the slider
        knob.  See the :class:`matplotlib.patches.Rectangle` documentation
        valid property names (e.g., *facecolor*, *edgecolor*, *alpha*, ...)
        """
        AxesWidget.__init__(self, ax)

        self.valmin = valmin
        self.valmax = valmax
        self.val = valinit
        self.valinit = valinit
        self.poly = ax.axvspan(valmin, valinit, 0, 1, **kwargs)

        self.vline = ax.axvline(valinit, 0, 1, color='r', lw=1)

        self.valfmt = valfmt
        ax.set_yticks([])
        ax.set_xlim((valmin, valmax))
        ax.set_xticks([])
        ax.set_navigate(False)

        self.connect_event('button_press_event', self._update)
        self.connect_event('button_release_event', self._update)
        if dragging:
            self.connect_event('motion_notify_event', self._update)
        self.label = ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                             verticalalignment='center',
                             horizontalalignment='right')

        self.valtext = ax.text(1.02, 0.5, valfmt % valinit,
                               transform=ax.transAxes,
                               verticalalignment='center',
                               horizontalalignment='left')

        self.cnt = 0
        self.observers = {}

        self.closedmin = closedmin
        self.closedmax = closedmax
        self.slidermin = slidermin
        self.slidermax = slidermax
        self.drag_active = False

    def _update(self, event):
        """update the slider position"""
        if self.ignore(event):
            return

        if event.button != 1:
            return

        if event.name == 'button_press_event' and event.inaxes == self.ax:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        elif ((event.name == 'button_release_event') or
              (event.name == 'button_press_event' and
               event.inaxes != self.ax)):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return

        val = event.xdata
        if val <= self.valmin:
            if not self.closedmin:
                return
            val = self.valmin
        elif val >= self.valmax:
            if not self.closedmax:
                return
            val = self.valmax

        if self.slidermin is not None and val <= self.slidermin.val:
            if not self.closedmin:
                return
            val = self.slidermin.val

        if self.slidermax is not None and val >= self.slidermax.val:
            if not self.closedmax:
                return
            val = self.slidermax.val

        self.set_val(val)

    def set_val(self, val):
        xy = self.poly.xy
        xy[2] = val, 1
        xy[3] = val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.iteritems():
            func(val)

    def on_changed(self, func):
        """
        When the slider value is changed, call *func* with the new
        slider position

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        """remove the observer with connection id *cid*"""
        try:
            del self.observers[cid]
        except KeyError:
            pass

    def reset(self):
        """reset the slider to the initial value if needed"""
        if (self.val != self.valinit):
            self.set_val(self.valinit)


class CheckButtons(AxesWidget):
    """
    A GUI neutral radio button

    The following attributes are exposed

     *ax*
        The :class:`matplotlib.axes.Axes` instance the buttons are
        located in

     *labels*
        List of :class:`matplotlib.text.Text` instances

     *lines*
        List of (line1, line2) tuples for the x's in the check boxes.
        These lines exist for each box, but have ``set_visible(False)``
        when its box is not checked.

     *rectangles*
        List of :class:`matplotlib.patches.Rectangle` instances

    Connect to the CheckButtons with the :meth:`on_clicked` method
    """
    def __init__(self, ax, labels, actives):
        """
        Add check buttons to :class:`matplotlib.axes.Axes` instance *ax*

        *labels*
            A len(buttons) list of labels as strings

        *actives*
            A len(buttons) list of booleans indicating whether
             the button is active
        """
        AxesWidget.__init__(self, ax)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        if len(labels) > 1:
            dy = 1. / (len(labels) + 1)
            ys = np.linspace(1 - dy, dy, len(labels))
        else:
            dy = 0.25
            ys = [0.5]

        cnt = 0
        axcolor = ax.get_axis_bgcolor()

        self.labels = []
        self.lines = []
        self.rectangles = []

        lineparams = {'color': 'k', 'linewidth': 1.25,
                      'transform': ax.transAxes, 'solid_capstyle': 'butt'}
        for y, label in zip(ys, labels):
            t = ax.text(0.25, y, label, transform=ax.transAxes,
                        horizontalalignment='left',
                        verticalalignment='center')

            w, h = dy / 2., dy / 2.
            x, y = 0.05, y - h / 2.

            p = Rectangle(xy=(x, y), width=w, height=h,
                          facecolor=axcolor,
                          transform=ax.transAxes)

            l1 = Line2D([x, x + w], [y + h, y], **lineparams)
            l2 = Line2D([x, x + w], [y, y + h], **lineparams)

            l1.set_visible(actives[cnt])
            l2.set_visible(actives[cnt])
            self.labels.append(t)
            self.rectangles.append(p)
            self.lines.append((l1, l2))
            ax.add_patch(p)
            ax.add_line(l1)
            ax.add_line(l2)
            cnt += 1

        self.connect_event('button_press_event', self._clicked)

        self.cnt = 0
        self.observers = {}

    def _clicked(self, event):
        if self.ignore(event):
            return
        if event.button != 1:
            return
        if event.inaxes != self.ax:
            return

        for p, t, lines in zip(self.rectangles, self.labels, self.lines):
            if (t.get_window_extent().contains(event.x, event.y) or
                    p.get_window_extent().contains(event.x, event.y)):
                l1, l2 = lines
                l1.set_visible(not l1.get_visible())
                l2.set_visible(not l2.get_visible())
                thist = t
                break
        else:
            return

        if self.drawon:
            self.ax.figure.canvas.draw()

        if not self.eventson:
            return
        for cid, func in self.observers.iteritems():
            func(thist.get_text())

    def on_clicked(self, func):
        """
        When the button is clicked, call *func* with button label

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        """remove the observer with connection id *cid*"""
        try:
            del self.observers[cid]
        except KeyError:
            pass


class RadioButtons(AxesWidget):
    """
    A GUI neutral radio button

    The following attributes are exposed

     *ax*
        The :class:`matplotlib.axes.Axes` instance the buttons are in

     *activecolor*
        The color of the button when clicked

     *labels*
        A list of :class:`matplotlib.text.Text` instances

     *circles*
        A list of :class:`matplotlib.patches.Circle` instances

    Connect to the RadioButtons with the :meth:`on_clicked` method
    """
    def __init__(self, ax, labels, active=0, activecolor='blue'):
        """
        Add radio buttons to :class:`matplotlib.axes.Axes` instance *ax*

        *labels*
            A len(buttons) list of labels as strings

        *active*
            The index into labels for the button that is active

        *activecolor*
            The color of the button when clicked
        """
        AxesWidget.__init__(self, ax)

        self.activecolor = activecolor

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)
        dy = 1. / (len(labels) + 1)
        ys = np.linspace(1 - dy, dy, len(labels))
        cnt = 0
        axcolor = ax.get_axis_bgcolor()

        self.labels = []
        self.circles = []
        for y, label in zip(ys, labels):
            t = ax.text(0.25, y, label, transform=ax.transAxes,
                        horizontalalignment='left',
                        verticalalignment='center')

            if cnt == active:
                facecolor = activecolor
            else:
                facecolor = axcolor

            p = Circle(xy=(0.15, y), radius=0.05, facecolor=facecolor,
                       transform=ax.transAxes)

            self.labels.append(t)
            self.circles.append(p)
            ax.add_patch(p)
            cnt += 1

        self.connect_event('button_press_event', self._clicked)

        self.cnt = 0
        self.observers = {}

    def _clicked(self, event):
        if self.ignore(event):
            return
        if event.button != 1:
            return
        if event.inaxes != self.ax:
            return
        xy = self.ax.transAxes.inverted().transform_point((event.x, event.y))
        pclicked = np.array([xy[0], xy[1]])

        def inside(p):
            pcirc = np.array([p.center[0], p.center[1]])
            return dist(pclicked, pcirc) < p.radius

        for p, t in zip(self.circles, self.labels):
            if t.get_window_extent().contains(event.x, event.y) or inside(p):
                inp = p
                thist = t
                break
        else:
            return

        for p in self.circles:
            if p == inp:
                color = self.activecolor
            else:
                color = self.ax.get_axis_bgcolor()
            p.set_facecolor(color)

        if self.drawon:
            self.ax.figure.canvas.draw()

        if not self.eventson:
            return
        for cid, func in self.observers.iteritems():
            func(thist.get_text())

    def on_clicked(self, func):
        """
        When the button is clicked, call *func* with button label

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        """remove the observer with connection id *cid*"""
        try:
            del self.observers[cid]
        except KeyError:
            pass


class SubplotTool(Widget):
    """
    A tool to adjust to subplot params of a :class:`matplotlib.figure.Figure`
    """
    def __init__(self, targetfig, toolfig):
        """
        *targetfig*
            The figure instance to adjust

        *toolfig*
            The figure instance to embed the subplot tool into. If
            None, a default figure will be created. If you are using
            this from the GUI
        """
        # FIXME: The docstring seems to just abruptly end without...

        self.targetfig = targetfig
        toolfig.subplots_adjust(left=0.2, right=0.9)

        class toolbarfmt:
            def __init__(self, slider):
                self.slider = slider

            def __call__(self, x, y):
                fmt = '%s=%s' % (self.slider.label.get_text(),
                                 self.slider.valfmt)
                return fmt % x

        self.axleft = toolfig.add_subplot(711)
        self.axleft.set_title('Click on slider to adjust subplot param')
        self.axleft.set_navigate(False)

        self.sliderleft = Slider(self.axleft, 'left',
                                 0, 1, targetfig.subplotpars.left,
                                 closedmax=False)
        self.sliderleft.on_changed(self.funcleft)

        self.axbottom = toolfig.add_subplot(712)
        self.axbottom.set_navigate(False)
        self.sliderbottom = Slider(self.axbottom,
                                   'bottom', 0, 1,
                                   targetfig.subplotpars.bottom,
                                   closedmax=False)
        self.sliderbottom.on_changed(self.funcbottom)

        self.axright = toolfig.add_subplot(713)
        self.axright.set_navigate(False)
        self.sliderright = Slider(self.axright, 'right', 0, 1,
                                  targetfig.subplotpars.right,
                                  closedmin=False)
        self.sliderright.on_changed(self.funcright)

        self.axtop = toolfig.add_subplot(714)
        self.axtop.set_navigate(False)
        self.slidertop = Slider(self.axtop, 'top', 0, 1,
                                targetfig.subplotpars.top,
                                closedmin=False)
        self.slidertop.on_changed(self.functop)

        self.axwspace = toolfig.add_subplot(715)
        self.axwspace.set_navigate(False)
        self.sliderwspace = Slider(self.axwspace, 'wspace',
                                   0, 1, targetfig.subplotpars.wspace,
                                   closedmax=False)
        self.sliderwspace.on_changed(self.funcwspace)

        self.axhspace = toolfig.add_subplot(716)
        self.axhspace.set_navigate(False)
        self.sliderhspace = Slider(self.axhspace, 'hspace',
                                   0, 1, targetfig.subplotpars.hspace,
                                   closedmax=False)
        self.sliderhspace.on_changed(self.funchspace)

        # constraints
        self.sliderleft.slidermax = self.sliderright
        self.sliderright.slidermin = self.sliderleft
        self.sliderbottom.slidermax = self.slidertop
        self.slidertop.slidermin = self.sliderbottom

        bax = toolfig.add_axes([0.8, 0.05, 0.15, 0.075])
        self.buttonreset = Button(bax, 'Reset')

        sliders = (self.sliderleft, self.sliderbottom, self.sliderright,
                   self.slidertop, self.sliderwspace, self.sliderhspace,)

        def func(event):
            thisdrawon = self.drawon

            self.drawon = False

            # store the drawon state of each slider
            bs = []
            for slider in sliders:
                bs.append(slider.drawon)
                slider.drawon = False

            # reset the slider to the initial position
            for slider in sliders:
                slider.reset()

            # reset drawon
            for slider, b in zip(sliders, bs):
                slider.drawon = b

            # draw the canvas
            self.drawon = thisdrawon
            if self.drawon:
                toolfig.canvas.draw()
                self.targetfig.canvas.draw()

        # during reset there can be a temporary invalid state
        # depending on the order of the reset so we turn off
        # validation for the resetting
        validate = toolfig.subplotpars.validate
        toolfig.subplotpars.validate = False
        self.buttonreset.on_clicked(func)
        toolfig.subplotpars.validate = validate

    def funcleft(self, val):
        self.targetfig.subplots_adjust(left=val)
        if self.drawon:
            self.targetfig.canvas.draw()

    def funcright(self, val):
        self.targetfig.subplots_adjust(right=val)
        if self.drawon:
            self.targetfig.canvas.draw()

    def funcbottom(self, val):
        self.targetfig.subplots_adjust(bottom=val)
        if self.drawon:
            self.targetfig.canvas.draw()

    def functop(self, val):
        self.targetfig.subplots_adjust(top=val)
        if self.drawon:
            self.targetfig.canvas.draw()

    def funcwspace(self, val):
        self.targetfig.subplots_adjust(wspace=val)
        if self.drawon:
            self.targetfig.canvas.draw()

    def funchspace(self, val):
        self.targetfig.subplots_adjust(hspace=val)
        if self.drawon:
            self.targetfig.canvas.draw()


class Cursor(AxesWidget):
    """
    A horizontal and vertical line span the axes that and move with
    the pointer.  You can turn off the hline or vline spectively with
    the attributes

      *horizOn*
        Controls the visibility of the horizontal line

      *vertOn*
        Controls the visibility of the horizontal line

    and the visibility of the cursor itself with the *visible* attribute
    """
    def __init__(self, ax, horizOn=True, vertOn=True, useblit=False,
                 **lineprops):
        """
        Add a cursor to *ax*.  If ``useblit=True``, use the backend-
        dependent blitting features for faster updates (GTKAgg
        only for now).  *lineprops* is a dictionary of line properties.

        .. plot :: mpl_examples/widgets/cursor.py
        """
        # TODO: Is the GTKAgg limitation still true?
        AxesWidget.__init__(self, ax)

        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.clear)

        self.visible = True
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.useblit = useblit and self.canvas.supports_blit

        if self.useblit:
            lineprops['animated'] = True
        self.lineh = ax.axhline(ax.get_ybound()[0], visible=False, **lineprops)
        self.linev = ax.axvline(ax.get_xbound()[0], visible=False, **lineprops)

        self.background = None
        self.needclear = False

    def clear(self, event):
        """clear the cursor"""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.linev.set_visible(False)
        self.lineh.set_visible(False)

    def onmove(self, event):
        """on mouse motion draw the cursor if visible"""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            self.linev.set_visible(False)
            self.lineh.set_visible(False)

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        if not self.visible:
            return
        self.linev.set_xdata((event.xdata, event.xdata))

        self.lineh.set_ydata((event.ydata, event.ydata))
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)

        self._update()

    def _update(self):

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.canvas.blit(self.ax.bbox)
        else:

            self.canvas.draw_idle()

        return False


class MultiCursor(Widget):
    """
    Provide a vertical (default) and/or horizontal line cursor shared between
    multiple axes

    Example usage::

        from matplotlib.widgets import MultiCursor
        from pylab import figure, show, np

        t = np.arange(0.0, 2.0, 0.01)
        s1 = np.sin(2*np.pi*t)
        s2 = np.sin(4*np.pi*t)
        fig = figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(t, s1)


        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.plot(t, s2)

        multi = MultiCursor(fig.canvas, (ax1, ax2), color='r', lw=1,
                            horizOn=False, vertOn=True)
        show()

    """
    def __init__(self, canvas, axes, useblit=True, horizOn=False, vertOn=True,
                 **lineprops):

        self.canvas = canvas
        self.axes = axes
        self.horizOn = horizOn
        self.vertOn = vertOn

        xmin, xmax = axes[-1].get_xlim()
        ymin, ymax = axes[-1].get_ylim()
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        self.visible = True
        self.useblit = useblit and self.canvas.supports_blit
        self.background = None
        self.needclear = False

        if useblit:
            lineprops['animated'] = True

        if vertOn:
            self.vlines = [ax.axvline(xmid, visible=False, **lineprops)
                           for ax in axes]
        else:
            self.vlines = []

        if horizOn:
            self.hlines = [ax.axhline(ymid, visible=False, **lineprops)
                           for ax in axes]
        else:
            self.hlines = []

        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('draw_event', self.clear)

    def clear(self, event):
        """clear the cursor"""
        if self.useblit:
            self.background = (
                self.canvas.copy_from_bbox(self.canvas.figure.bbox))
        for line in self.vlines + self.hlines:
            line.set_visible(False)

    def onmove(self, event):
        if event.inaxes is None:
            return
        if not self.canvas.widgetlock.available(self):
            return
        self.needclear = True
        if not self.visible:
            return
        if self.vertOn:
            for line in self.vlines:
                line.set_xdata((event.xdata, event.xdata))
                line.set_visible(self.visible)
        if self.horizOn:
            for line in self.hlines:
                line.set_ydata((event.ydata, event.ydata))
                line.set_visible(self.visible)
        self._update()

    def _update(self):
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            if self.vertOn:
                for ax, line in zip(self.axes, self.vlines):
                    ax.draw_artist(line)
            if self.horizOn:
                for ax, line in zip(self.axes, self.hlines):
                    ax.draw_artist(line)
            self.canvas.blit(self.canvas.figure.bbox)
        else:

            self.canvas.draw_idle()


class SpanSelector(AxesWidget):
    """
    Select a min/max range of the x or y axes for a matplotlib Axes

    Example usage::

        ax = subplot(111)
        ax.plot(x,y)

        def onselect(vmin, vmax):
            print vmin, vmax
        span = SpanSelector(ax, onselect, 'horizontal')

    *onmove_callback* is an optional callback that is called on mouse
      move within the span range

    """

    def __init__(self, ax, onselect, direction, minspan=None, useblit=False,
                 rectprops=None, onmove_callback=None):
        """
        Create a span selector in *ax*.  When a selection is made, clear
        the span and call *onselect* with::

            onselect(vmin, vmax)

        and clear the span.

        *direction* must be 'horizontal' or 'vertical'

        If *minspan* is not *None*, ignore events smaller than *minspan*

        The span rectangle is drawn with *rectprops*; default::
          rectprops = dict(facecolor='red', alpha=0.5)

        Set the visible attribute to *False* if you want to turn off
        the functionality of the span selector
        """
        AxesWidget.__init__(self, ax)

        if rectprops is None:
            rectprops = dict(facecolor='red', alpha=0.5)

        assert direction in ['horizontal', 'vertical'], 'Must choose horizontal or vertical for direction'
        self.direction = direction

        self.visible = True

        self.rect = None
        self.background = None
        self.pressv = None

        self.rectprops = rectprops
        self.onselect = onselect
        self.onmove_callback = onmove_callback
        self.minspan = minspan

        # Needed when dragging out of axes
        self.buttonDown = False
        self.prev = (0, 0)

        # Set useblit based on original canvas.
        self.useblit = useblit and self.canvas.supports_blit

        # Reset canvas so that `new_axes` connects events.
        self.canvas = None
        self.new_axes(ax)

    def new_axes(self, ax):
        self.ax = ax
        if self.canvas is not ax.figure.canvas:
            self.disconnect_events()

            self.canvas = ax.figure.canvas
            self.connect_event('motion_notify_event', self.onmove)
            self.connect_event('button_press_event', self.press)
            self.connect_event('button_release_event', self.release)
            self.connect_event('draw_event', self.update_background)

        if self.direction == 'horizontal':
            trans = blended_transform_factory(self.ax.transData,
                                              self.ax.transAxes)
            w, h = 0, 1
        else:
            trans = blended_transform_factory(self.ax.transAxes,
                                              self.ax.transData)
            w, h = 1, 0
        self.rect = Rectangle((0, 0), w, h,
                              transform=trans,
                              visible=False,
                              **self.rectprops)

        if not self.useblit:
            self.ax.add_patch(self.rect)

    def update_background(self, event):
        """force an update of the background"""
        # If you add a call to `ignore` here, you'll want to check edge case:
        # `release` can call a draw event even when `ignore` is True.
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def ignore(self, event):
        """return *True* if *event* should be ignored"""
        widget_off = not self.visible or not self.active
        non_event = event.inaxes != self.ax or event.button != 1
        return widget_off or non_event

    def press(self, event):
        """on button press event"""
        if self.ignore(event):
            return
        self.buttonDown = True

        self.rect.set_visible(self.visible)
        if self.direction == 'horizontal':
            self.pressv = event.xdata
        else:
            self.pressv = event.ydata
        return False

    def release(self, event):
        """on button release event"""
        if self.ignore(event) and not self.buttonDown:
            return
        if self.pressv is None:
            return
        self.buttonDown = False

        self.rect.set_visible(False)
        self.canvas.draw()
        vmin = self.pressv
        if self.direction == 'horizontal':
            vmax = event.xdata or self.prev[0]
        else:
            vmax = event.ydata or self.prev[1]

        if vmin > vmax:
            vmin, vmax = vmax, vmin
        span = vmax - vmin
        if self.minspan is not None and span < self.minspan:
            return
        self.onselect(vmin, vmax)
        self.pressv = None
        return False

    def update(self):
        """
        Draw using newfangled blit or oldfangled draw depending
        on *useblit*
        """
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.rect)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

        return False

    def onmove(self, event):
        """on motion notify event"""
        if self.pressv is None or self.ignore(event):
            return
        x, y = event.xdata, event.ydata
        self.prev = x, y
        if self.direction == 'horizontal':
            v = x
        else:
            v = y

        minv, maxv = v, self.pressv
        if minv > maxv:
            minv, maxv = maxv, minv
        if self.direction == 'horizontal':
            self.rect.set_x(minv)
            self.rect.set_width(maxv - minv)
        else:
            self.rect.set_y(minv)
            self.rect.set_height(maxv - minv)

        if self.onmove_callback is not None:
            vmin = self.pressv
            if self.direction == 'horizontal':
                vmax = event.xdata or self.prev[0]
            else:
                vmax = event.ydata or self.prev[1]

            if vmin > vmax:
                vmin, vmax = vmax, vmin
            self.onmove_callback(vmin, vmax)

        self.update()
        return False


class RectangleSelector(AxesWidget):
    """
    Select a min/max range of the x axes for a matplotlib Axes

    Example usage::

        from matplotlib.widgets import  RectangleSelector
        from pylab import *

        def onselect(eclick, erelease):
          'eclick and erelease are matplotlib events at press and release'
          print ' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata)
          print ' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata)
          print ' used button   : ', eclick.button

        def toggle_selector(event):
            print ' Key pressed.'
            if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                print ' RectangleSelector deactivated.'
                toggle_selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not toggle_selector.RS.active:
                print ' RectangleSelector activated.'
                toggle_selector.RS.set_active(True)

        x = arange(100)/(99.0)
        y = sin(x)
        fig = figure
        ax = subplot(111)
        ax.plot(x,y)

        toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='line')
        connect('key_press_event', toggle_selector)
        show()
    """
    def __init__(self, ax, onselect, drawtype='box',
                 minspanx=None, minspany=None, useblit=False,
                 lineprops=None, rectprops=None, spancoords='data',
                 button=None):

        """
        Create a selector in *ax*.  When a selection is made, clear
        the span and call onselect with::

          onselect(pos_1, pos_2)

        and clear the drawn box/line. The ``pos_1`` and ``pos_2`` are
        arrays of length 2 containing the x- and y-coordinate.

        If *minspanx* is not *None* then events smaller than *minspanx*
        in x direction are ignored (it's the same for y).

        The rectangle is drawn with *rectprops*; default::

          rectprops = dict(facecolor='red', edgecolor = 'black',
                           alpha=0.5, fill=False)

        The line is drawn with *lineprops*; default::

          lineprops = dict(color='black', linestyle='-',
                           linewidth = 2, alpha=0.5)

        Use *drawtype* if you want the mouse to draw a line,
        a box or nothing between click and actual position by setting

        ``drawtype = 'line'``, ``drawtype='box'`` or ``drawtype = 'none'``.

        *spancoords* is one of 'data' or 'pixels'.  If 'data', *minspanx*
        and *minspanx* will be interpreted in the same coordinates as
        the x and y axis. If 'pixels', they are in pixels.

        *button* is a list of integers indicating which mouse buttons should
        be used for rectangle selection.  You can also specify a single
        integer if only a single button is desired.  Default is *None*,
        which does not limit which button can be used.

        Note, typically:
         1 = left mouse button
         2 = center mouse button (scroll wheel)
         3 = right mouse button
        """
        AxesWidget.__init__(self, ax)

        self.visible = True
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('button_press_event', self.press)
        self.connect_event('button_release_event', self.release)
        self.connect_event('draw_event', self.update_background)

        self.active = True                    # for activation / deactivation
        self.to_draw = None
        self.background = None

        if drawtype == 'none':
            drawtype = 'line'                        # draw a line but make it
            self.visible = False                     # invisible

        if drawtype == 'box':
            if rectprops is None:
                rectprops = dict(facecolor='white', edgecolor='black',
                                 alpha=0.5, fill=False)
            self.rectprops = rectprops
            self.to_draw = Rectangle((0, 0),
                                     0, 1, visible=False, **self.rectprops)
            self.ax.add_patch(self.to_draw)
        if drawtype == 'line':
            if lineprops is None:
                lineprops = dict(color='black', linestyle='-',
                                 linewidth=2, alpha=0.5)
            self.lineprops = lineprops
            self.to_draw = Line2D([0, 0], [0, 0], visible=False,
                                  **self.lineprops)
            self.ax.add_line(self.to_draw)

        self.onselect = onselect
        self.useblit = useblit and self.canvas.supports_blit
        self.minspanx = minspanx
        self.minspany = minspany

        if button is None or isinstance(button, list):
            self.validButtons = button
        elif isinstance(button, int):
            self.validButtons = [button]

        assert(spancoords in ('data', 'pixels'))

        self.spancoords = spancoords
        self.drawtype = drawtype
        # will save the data (position at mouseclick)
        self.eventpress = None
        # will save the data (pos. at mouserelease)
        self.eventrelease = None

    def update_background(self, event):
        """force an update of the background"""
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def ignore(self, event):
        """return *True* if *event* should be ignored"""
        if not self.active:
            return True

        # If canvas was locked
        if not self.canvas.widgetlock.available(self):
            return True

        # Only do rectangle selection if event was triggered
        # with a desired button
        if self.validButtons is not None:
            if not event.button in self.validButtons:
                return True

        # If no button was pressed yet ignore the event if it was out
        # of the axes
        if self.eventpress is None:
            return event.inaxes != self.ax

        # If a button was pressed, check if the release-button is the
        # same. If event is out of axis, limit the data coordinates to axes
        # boundaries.
        if event.button == self.eventpress.button and event.inaxes != self.ax:
            (xdata, ydata) = self.ax.transData.inverted().transform_point(
                (event.x, event.y))
            x0, x1 = self.ax.get_xbound()
            y0, y1 = self.ax.get_ybound()
            xdata = max(x0, xdata)
            xdata = min(x1, xdata)
            ydata = max(y0, ydata)
            ydata = min(y1, ydata)
            event.xdata = xdata
            event.ydata = ydata
            return False

        # If a button was pressed, check if the release-button is the
        # same.
        return (event.inaxes != self.ax or
                event.button != self.eventpress.button)

    def press(self, event):
        """on button press event"""
        if self.ignore(event):
            return
        # make the drawed box/line visible get the click-coordinates,
        # button, ...
        self.to_draw.set_visible(self.visible)
        self.eventpress = event
        return False

    def release(self, event):
        """on button release event"""
        if self.eventpress is None or self.ignore(event):
            return
        # make the box/line invisible again
        self.to_draw.set_visible(False)
        self.canvas.draw()
        # release coordinates, button, ...
        self.eventrelease = event

        if self.spancoords == 'data':
            xmin, ymin = self.eventpress.xdata, self.eventpress.ydata
            xmax, ymax = self.eventrelease.xdata, self.eventrelease.ydata
            # calculate dimensions of box or line get values in the right
            # order
        elif self.spancoords == 'pixels':
            xmin, ymin = self.eventpress.x, self.eventpress.y
            xmax, ymax = self.eventrelease.x, self.eventrelease.y
        else:
            raise ValueError('spancoords must be "data" or "pixels"')

        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

        spanx = xmax - xmin
        spany = ymax - ymin
        xproblems = self.minspanx is not None and spanx < self.minspanx
        yproblems = self.minspany is not None and spany < self.minspany

        if (((self.drawtype == 'box') or (self.drawtype == 'line')) and
                (xproblems or yproblems)):
            # check if drawn distance (if it exists) is not too small in
            # neither x nor y-direction
            return

        self.onselect(self.eventpress, self.eventrelease)
                                              # call desired function
        self.eventpress = None                # reset the variables to their
        self.eventrelease = None              # inital values
        return False

    def update(self):
        """draw using newfangled blit or oldfangled draw depending on
        useblit

        """
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.to_draw)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
        return False

    def onmove(self, event):
        """on motion notify event if box/line is wanted"""
        if self.eventpress is None or self.ignore(event):
            return
        x, y = event.xdata, event.ydata             # actual position (with
                                                   #   (button still pressed)
        if self.drawtype == 'box':
            minx, maxx = self.eventpress.xdata, x  # click-x and actual mouse-x
            miny, maxy = self.eventpress.ydata, y  # click-y and actual mouse-y
            if minx > maxx:
                minx, maxx = maxx, minx  # get them in the right order
            if miny > maxy:
                miny, maxy = maxy, miny
            self.to_draw.set_x(minx)             # set lower left of box
            self.to_draw.set_y(miny)
            self.to_draw.set_width(maxx - minx)  # set width and height of box
            self.to_draw.set_height(maxy - miny)
            self.update()
            return False
        if self.drawtype == 'line':
            self.to_draw.set_data([self.eventpress.xdata, x],
                                  [self.eventpress.ydata, y])
            self.update()
            return False

    def set_active(self, active):
        """
        Use this to activate / deactivate the RectangleSelector
        from your program with an boolean parameter *active*.
        """
        self.active = active

    def get_active(self):
        """ Get status of active mode (boolean variable)"""
        return self.active


class LassoSelector(AxesWidget):
    """Selection curve of an arbitrary shape.

    The selected path can be used in conjunction with
    :func:`~matplotlib.path.Path.contains_point` to select
    data points from an image.

    In contrast to :class:`Lasso`, `LassoSelector` is written with an interface
    similar to :class:`RectangleSelector` and :class:`SpanSelector` and will
    continue to interact with the axes until disconnected.

    Parameters:

    *ax* : :class:`~matplotlib.axes.Axes`
        The parent axes for the widget.
    *onselect* : function
        Whenever the lasso is released, the `onselect` function is called and
        passed the vertices of the selected path.

    Example usage::

        ax = subplot(111)
        ax.plot(x,y)

        def onselect(verts):
            print verts
        lasso = LassoSelector(ax, onselect)

    """

    def __init__(self, ax, onselect=None, useblit=True, lineprops=None):
        AxesWidget.__init__(self, ax)

        self.useblit = useblit and self.canvas.supports_blit
        self.onselect = onselect
        self.verts = None

        if lineprops is None:
            lineprops = dict()
        self.line = Line2D([], [], **lineprops)
        self.line.set_visible(False)
        self.ax.add_line(self.line)

        self.connect_event('button_press_event', self.onpress)
        self.connect_event('button_release_event', self.onrelease)
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.update_background)

    def ignore(self, event):
        wrong_button = hasattr(event, 'button') and event.button != 1
        return not self.active or wrong_button

    def onpress(self, event):
        if self.ignore(event) or event.inaxes != self.ax:
            return
        self.verts = [(event.xdata, event.ydata)]
        self.line.set_visible(True)

    def onrelease(self, event):
        if self.ignore(event):
            return
        if self.verts is not None:
            if event.inaxes == self.ax:
                self.verts.append((event.xdata, event.ydata))
            self.onselect(self.verts)
        self.line.set_data([[], []])
        self.line.set_visible(False)
        self.verts = None

    def onmove(self, event):
        if self.ignore(event) or event.inaxes != self.ax:
            return
        if self.verts is None:
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self.verts.append((event.xdata, event.ydata))

        self.line.set_data(zip(*self.verts))

        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def update_background(self, event):
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)


class Lasso(AxesWidget):
    """Selection curve of an arbitrary shape.

    The selected path can be used in conjunction with
    :func:`~matplotlib.path.Path.contains_point` to select data points
    from an image.

    Unlike :class:`LassoSelector`, this must be initialized with a starting
    point `xy`, and the `Lasso` events are destroyed upon release.

    Parameters:

    *ax* : :class:`~matplotlib.axes.Axes`
        The parent axes for the widget.
    *xy* : array
        Coordinates of the start of the lasso.
    *callback* : function
        Whenever the lasso is released, the `callback` function is called and
        passed the vertices of the selected path.

    """

    def __init__(self, ax, xy, callback=None, useblit=True):
        AxesWidget.__init__(self, ax)

        self.useblit = useblit and self.canvas.supports_blit
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        x, y = xy
        self.verts = [(x, y)]
        self.line = Line2D([x], [y], linestyle='-', color='black', lw=2)
        self.ax.add_line(self.line)
        self.callback = callback
        self.connect_event('button_release_event', self.onrelease)
        self.connect_event('motion_notify_event', self.onmove)

    def onrelease(self, event):
        if self.ignore(event):
            return
        if self.verts is not None:
            self.verts.append((event.xdata, event.ydata))
            if len(self.verts) > 2:
                self.callback(self.verts)
            self.ax.lines.remove(self.line)
        self.verts = None
        self.disconnect_events()

    def onmove(self, event):
        if self.ignore(event):
            return
        if self.verts is None:
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self.verts.append((event.xdata, event.ydata))

        self.line.set_data(zip(*self.verts))

        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
