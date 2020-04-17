.. _event-handling-tutorial:

**************************
Event handling and picking
**************************

Matplotlib works with a number of user interface toolkits (wxpython,
tkinter, qt4, gtk, and macosx) and in order to support features like
interactive panning and zooming of figures, it is helpful to the
developers to have an API for interacting with the figure via key
presses and mouse movements that is "GUI neutral" so we don't have to
repeat a lot of code across the different user interfaces.  Although
the event handling API is GUI neutral, it is based on the GTK model,
which was the first user interface matplotlib supported.  The events
that are triggered are also a bit richer vis-a-vis matplotlib than
standard GUI events, including information like which
:class:`matplotlib.axes.Axes` the event occurred in.  The events also
understand the matplotlib coordinate system, and report event
locations in both pixel and data coordinates.

.. _event-connections:

Event connections
=================

To receive events, you need to write a callback function and then
connect your function to the event manager, which is part of the
:class:`~matplotlib.backend_bases.FigureCanvasBase`.  Here is a simple
example that prints the location of the mouse click and which button
was pressed::

    fig, ax = plt.subplots()
    ax.plot(np.random.rand(10))

    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

The ``FigureCanvas`` method
:meth:`~matplotlib.backend_bases.FigureCanvasBase.mpl_connect` returns
a connection id which is simply an integer.  When you want to
disconnect the callback, just call::

    fig.canvas.mpl_disconnect(cid)

.. note::
   The canvas retains only weak references to instance methods used as
   callbacks.  Therefore, you need to retain a reference to instances owning
   such methods.  Otherwise the instance will be garbage-collected and the
   callback will vanish.

   This does not affect free functions used as callbacks.


Here are the events that you can connect to, the class instances that
are sent back to you when the event occurs, and the event descriptions:

====================== ================ ======================================
Event name             Class            Description
====================== ================ ======================================
'button_press_event'   `.MouseEvent`    mouse button is pressed
'button_release_event' `.MouseEvent`    mouse button is released
'close_event'          `.CloseEvent`    figure is closed
'draw_event'           `.DrawEvent`     canvas has been drawn (but screen
                                        widget not updated yet)
'key_press_event'      `.KeyEvent`      key is pressed
'key_release_event'    `.KeyEvent`      key is released
'motion_notify_event'  `.MouseEvent`    mouse moves
'pick_event'           `.PickEvent`     artist in the canvas is selected
'resize_event'         `.ResizeEvent`   figure canvas is resized
'scroll_event'         `.MouseEvent`    mouse scroll wheel is rolled
'figure_enter_event'   `.LocationEvent` mouse enters a new figure
'figure_leave_event'   `.LocationEvent` mouse leaves a figure
'axes_enter_event'     `.LocationEvent` mouse enters a new axes
'axes_leave_event'     `.LocationEvent` mouse leaves an axes
====================== ================ ======================================

.. _event-attributes:

Event attributes
================

All matplotlib events inherit from the base class
:class:`matplotlib.backend_bases.Event`, which store the attributes:

    ``name``
	the event name

    ``canvas``
	the FigureCanvas instance generating the event

    ``guiEvent``
	the GUI event that triggered the matplotlib event


The most common events that are the bread and butter of event handling
are key press/release events and mouse press/release and movement
events.  The :class:`~matplotlib.backend_bases.KeyEvent` and
:class:`~matplotlib.backend_bases.MouseEvent` classes that handle
these events are both derived from the LocationEvent, which has the
following attributes

    ``x``
        x position - pixels from left of canvas

    ``y``
        y position - pixels from bottom of canvas

    ``inaxes``
        the :class:`~matplotlib.axes.Axes` instance if mouse is over axes

    ``xdata``
        x coord of mouse in data coords

    ``ydata``
        y coord of mouse in data coords

Let's look a simple example of a canvas, where a simple line segment
is created every time a mouse is pressed::

    from matplotlib import pyplot as plt

    class LineBuilder:
        def __init__(self, line):
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            print('click', event)
            if event.inaxes!=self.line.axes: return
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click to build line segments')
    line, = ax.plot([0], [0])  # empty line
    linebuilder = LineBuilder(line)

    plt.show()


The :class:`~matplotlib.backend_bases.MouseEvent` that we just used is a
:class:`~matplotlib.backend_bases.LocationEvent`, so we have access to
the data and pixel coordinates in event.x and event.xdata.  In
addition to the ``LocationEvent`` attributes, it has

    ``button``
        button pressed None, 1, 2, 3, 'up', 'down' (up and down are used for scroll events)

    ``key``
        the key pressed: None, any character, 'shift', 'win', or 'control'

Draggable rectangle exercise
----------------------------

Write draggable rectangle class that is initialized with a
:class:`~matplotlib.patches.Rectangle` instance but will move its x,y
location when dragged.  Hint: you will need to store the original
``xy`` location of the rectangle which is stored as rect.xy and
connect to the press, motion and release mouse events.  When the mouse
is pressed, check to see if the click occurs over your rectangle (see
:meth:`matplotlib.patches.Rectangle.contains`) and if it does, store
the rectangle xy and the location of the mouse click in data coords.
In the motion event callback, compute the deltax and deltay of the
mouse movement, and add those deltas to the origin of the rectangle
you stored.  The redraw the figure.  On the button release event, just
reset all the button press data you stored as None.

Here is the solution::

    import numpy as np
    import matplotlib.pyplot as plt

    class DraggableRectangle:
        def __init__(self, rect):
            self.rect = rect
            self.press = None

        def connect(self):
            'connect to all the events we need'
            self.cidpress = self.rect.figure.canvas.mpl_connect(
                'button_press_event', self.on_press)
            self.cidrelease = self.rect.figure.canvas.mpl_connect(
                'button_release_event', self.on_release)
            self.cidmotion = self.rect.figure.canvas.mpl_connect(
                'motion_notify_event', self.on_motion)

        def on_press(self, event):
            'on button press we will see if the mouse is over us and store some data'
            if event.inaxes != self.rect.axes: return

            contains, attrd = self.rect.contains(event)
            if not contains: return
            print('event contains', self.rect.xy)
            x0, y0 = self.rect.xy
            self.press = x0, y0, event.xdata, event.ydata

        def on_motion(self, event):
            'on motion we will move the rect if the mouse is over us'
            if self.press is None: return
            if event.inaxes != self.rect.axes: return
            x0, y0, xpress, ypress = self.press
            dx = event.xdata - xpress
            dy = event.ydata - ypress
            #print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
            #      (x0, xpress, event.xdata, dx, x0+dx))
            self.rect.set_x(x0+dx)
            self.rect.set_y(y0+dy)

            self.rect.figure.canvas.draw()


        def on_release(self, event):
            'on release we reset the press data'
            self.press = None
            self.rect.figure.canvas.draw()

        def disconnect(self):
            'disconnect all the stored connection ids'
            self.rect.figure.canvas.mpl_disconnect(self.cidpress)
            self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
            self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects = ax.bar(range(10), 20*np.random.rand(10))
    drs = []
    for rect in rects:
        dr = DraggableRectangle(rect)
        dr.connect()
        drs.append(dr)

    plt.show()


**Extra credit**: Use blitting to make the animated drawing faster and
smoother.

Extra credit solution::

    # Draggable rectangle with blitting.
    import numpy as np
    import matplotlib.pyplot as plt

    class DraggableRectangle:
        lock = None  # only one can be animated at a time
        def __init__(self, rect):
            self.rect = rect
            self.press = None
            self.background = None

        def connect(self):
            'connect to all the events we need'
            self.cidpress = self.rect.figure.canvas.mpl_connect(
                'button_press_event', self.on_press)
            self.cidrelease = self.rect.figure.canvas.mpl_connect(
                'button_release_event', self.on_release)
            self.cidmotion = self.rect.figure.canvas.mpl_connect(
                'motion_notify_event', self.on_motion)

        def on_press(self, event):
            'on button press we will see if the mouse is over us and store some data'
            if event.inaxes != self.rect.axes: return
            if DraggableRectangle.lock is not None: return
            contains, attrd = self.rect.contains(event)
            if not contains: return
            print('event contains', self.rect.xy)
            x0, y0 = self.rect.xy
            self.press = x0, y0, event.xdata, event.ydata
            DraggableRectangle.lock = self

            # draw everything but the selected rectangle and store the pixel buffer
            canvas = self.rect.figure.canvas
            axes = self.rect.axes
            self.rect.set_animated(True)
            canvas.draw()
            self.background = canvas.copy_from_bbox(self.rect.axes.bbox)

            # now redraw just the rectangle
            axes.draw_artist(self.rect)

            # and blit just the redrawn area
            canvas.blit(axes.bbox)

        def on_motion(self, event):
            'on motion we will move the rect if the mouse is over us'
            if DraggableRectangle.lock is not self:
                return
            if event.inaxes != self.rect.axes: return
            x0, y0, xpress, ypress = self.press
            dx = event.xdata - xpress
            dy = event.ydata - ypress
            self.rect.set_x(x0+dx)
            self.rect.set_y(y0+dy)

            canvas = self.rect.figure.canvas
            axes = self.rect.axes
            # restore the background region
            canvas.restore_region(self.background)

            # redraw just the current rectangle
            axes.draw_artist(self.rect)

            # blit just the redrawn area
            canvas.blit(axes.bbox)

        def on_release(self, event):
            'on release we reset the press data'
            if DraggableRectangle.lock is not self:
                return

            self.press = None
            DraggableRectangle.lock = None

            # turn off the rect animation property and reset the background
            self.rect.set_animated(False)
            self.background = None

            # redraw the full figure
            self.rect.figure.canvas.draw()

        def disconnect(self):
            'disconnect all the stored connection ids'
            self.rect.figure.canvas.mpl_disconnect(self.cidpress)
            self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
            self.rect.figure.canvas.mpl_disconnect(self.cidmotion)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects = ax.bar(range(10), 20*np.random.rand(10))
    drs = []
    for rect in rects:
        dr = DraggableRectangle(rect)
        dr.connect()
        drs.append(dr)

    plt.show()


.. _enter-leave-events:

Mouse enter and leave
======================

If you want to be notified when the mouse enters or leaves a figure or
axes, you can connect to the figure/axes enter/leave events.  Here is
a simple example that changes the colors of the axes and figure
background that the mouse is over::

    """
    Illustrate the figure and axes enter and leave events by changing the
    frame colors on enter and leave
    """
    import matplotlib.pyplot as plt

    def enter_axes(event):
        print('enter_axes', event.inaxes)
        event.inaxes.patch.set_facecolor('yellow')
        event.canvas.draw()

    def leave_axes(event):
        print('leave_axes', event.inaxes)
        event.inaxes.patch.set_facecolor('white')
        event.canvas.draw()

    def enter_figure(event):
        print('enter_figure', event.canvas.figure)
        event.canvas.figure.patch.set_facecolor('red')
        event.canvas.draw()

    def leave_figure(event):
        print('leave_figure', event.canvas.figure)
        event.canvas.figure.patch.set_facecolor('grey')
        event.canvas.draw()

    fig1 = plt.figure()
    fig1.suptitle('mouse hover over figure or axes to trigger events')
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212)

    fig1.canvas.mpl_connect('figure_enter_event', enter_figure)
    fig1.canvas.mpl_connect('figure_leave_event', leave_figure)
    fig1.canvas.mpl_connect('axes_enter_event', enter_axes)
    fig1.canvas.mpl_connect('axes_leave_event', leave_axes)

    fig2 = plt.figure()
    fig2.suptitle('mouse hover over figure or axes to trigger events')
    ax1 = fig2.add_subplot(211)
    ax2 = fig2.add_subplot(212)

    fig2.canvas.mpl_connect('figure_enter_event', enter_figure)
    fig2.canvas.mpl_connect('figure_leave_event', leave_figure)
    fig2.canvas.mpl_connect('axes_enter_event', enter_axes)
    fig2.canvas.mpl_connect('axes_leave_event', leave_axes)

    plt.show()


.. _object-picking:

Object picking
==============

You can enable picking by setting the ``picker`` property of an
:class:`~matplotlib.artist.Artist` (e.g., a matplotlib
:class:`~matplotlib.lines.Line2D`, :class:`~matplotlib.text.Text`,
:class:`~matplotlib.patches.Patch`, :class:`~matplotlib.patches.Polygon`,
:class:`~matplotlib.image.AxesImage`, etc...)

There are a variety of meanings of the ``picker`` property:

    ``None``
	picking is disabled for this artist (default)

    ``boolean``
	if True then picking will be enabled and the artist will fire a
	pick event if the mouse event is over the artist

    ``float``
	if picker is a number it is interpreted as an epsilon tolerance in
	points and the artist will fire off an event if its data is
	within epsilon of the mouse event.  For some artists like lines
	and patch collections, the artist may provide additional data to
	the pick event that is generated, e.g., the indices of the data
	within epsilon of the pick event.

    ``function``
	if picker is callable, it is a user supplied function which
	determines whether the artist is hit by the mouse event.  The
	signature is ``hit, props = picker(artist, mouseevent)`` to
	determine the hit test.  If the mouse event is over the artist,
	return ``hit=True`` and props is a dictionary of properties you
	want added to the :class:`~matplotlib.backend_bases.PickEvent`
	attributes


After you have enabled an artist for picking by setting the ``picker``
property, you need to connect to the figure canvas pick_event to get
pick callbacks on mouse press events.  e.g.::

    def pick_handler(event):
        mouseevent = event.mouseevent
        artist = event.artist
        # now do something with this...


The :class:`~matplotlib.backend_bases.PickEvent` which is passed to
your callback is always fired with two attributes:

    ``mouseevent`` the mouse event that generate the pick event.  The
	mouse event in turn has attributes like ``x`` and ``y`` (the
	coords in display space, e.g., pixels from left, bottom) and xdata,
	ydata (the coords in data space).  Additionally, you can get
	information about which buttons were pressed, which keys were
	pressed, which :class:`~matplotlib.axes.Axes` the mouse is over,
	etc.  See :class:`matplotlib.backend_bases.MouseEvent` for
	details.

    ``artist``
	the :class:`~matplotlib.artist.Artist` that generated the pick
	event.

Additionally, certain artists like :class:`~matplotlib.lines.Line2D`
and :class:`~matplotlib.collections.PatchCollection` may attach
additional meta data like the indices into the data that meet the
picker criteria (e.g., all the points in the line that are within the
specified epsilon tolerance)

Simple picking example
----------------------

In the example below, we set the line picker property to a scalar, so
it represents a tolerance in points (72 points per inch).  The onpick
callback function will be called when the pick event it within the
tolerance distance from the line, and has the indices of the data
vertices that are within the pick distance tolerance.  Our onpick
callback function simply prints the data that are under the pick
location.  Different matplotlib Artists can attach different data to
the PickEvent.  For example, ``Line2D`` attaches the ind property,
which are the indices into the line data under the pick point.  See
:meth:`~matplotlib.lines.Line2D.pick` for details on the ``PickEvent``
properties of the line.  Here is the code::

    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click on points')

    line, = ax.plot(np.random.rand(100), 'o', picker=5)  # 5 points tolerance

    def onpick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        points = tuple(zip(xdata[ind], ydata[ind]))
        print('onpick points:', points)

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()


Picking exercise
----------------

Create a data set of 100 arrays of 1000 Gaussian random numbers and
compute the sample mean and standard deviation of each of them (hint:
NumPy arrays have a mean and std method) and make a xy marker plot of
the 100 means vs. the 100 standard deviations.  Connect the line
created by the plot command to the pick event, and plot the original
time series of the data that generated the clicked on points.  If more
than one point is within the tolerance of the clicked on point, you
can use multiple subplots to plot the multiple time series.

Exercise solution::

    """
    compute the mean and stddev of 100 data sets and plot mean vs. stddev.
    When you click on one of the mu, sigma points, plot the raw data from
    the dataset that generated the mean and stddev
    """
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.random.rand(100, 1000)
    xs = np.mean(X, axis=1)
    ys = np.std(X, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('click on point to plot time series')
    line, = ax.plot(xs, ys, 'o', picker=5)  # 5 points tolerance


    def onpick(event):

        if event.artist!=line: return True

        N = len(event.ind)
        if not N: return True


        figi = plt.figure()
        for subplotnum, dataind in enumerate(event.ind):
            ax = figi.add_subplot(N,1,subplotnum+1)
            ax.plot(X[dataind])
            ax.text(0.05, 0.9, 'mu=%1.3f\nsigma=%1.3f'%(xs[dataind], ys[dataind]),
                    transform=ax.transAxes, va='top')
            ax.set_ylim(-0.5, 1.5)
        figi.show()
        return True

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()
