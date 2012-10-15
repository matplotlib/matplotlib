"""
Displays Agg images in the browser, with interactivity
"""
from __future__ import division, print_function

import cStringIO
import datetime
import errno
import json
import os
import random
import socket

import numpy as np

try:
    import tornado
except ImportError:
    raise RuntimeError("The WebAgg backend requires Tornado.")
import tornado.web
import tornado.ioloop
import tornado.websocket
import tornado.template

import matplotlib
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends import backend_agg
from matplotlib import backend_bases
from matplotlib._pylab_helpers import Gcf
from matplotlib import _png


def draw_if_interactive():
    """
    Is called after every pylab drawing command
    """
    if matplotlib.is_interactive():
        figManager = Gcf.get_active()
        if figManager is not None:
            figManager.canvas.draw_idle()


class Show(backend_bases.ShowBase):
    def mainloop(self):
        WebAggApplication.initialize()
        if rcParams['webagg.open_in_browser']:
            import webbrowser
            print("Opening in webbrowser")
            for manager in Gcf.get_all_fig_managers():
                webbrowser.open("http://127.0.0.1:{0}/{1}/".format(
                    WebAggApplication.port, manager.num))
        WebAggApplication.start()

show = Show()


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasWebAgg(figure)
    manager = FigureManagerWebAgg(canvas, num)
    return manager


class TimerTornado(backend_bases.TimerBase):
    def _timer_start(self):
        self._timer_stop()
        if self._single:
            ioloop = tornado.ioloop.IOLoop.instance()
            self._timer = ioloop.add_timeout(
                datetime.timedelta(milliseconds=self.interval),
                self._on_timer)
        else:
            self._timer = tornado.ioloop.PeriodicCallback(
                self._on_timer,
                self.interval)
        self._timer.start()

    def _timer_stop(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _timer_set_interval(self):
        # Only stop and restart it if the timer has already been started
        if self._timer is not None:
            self._timer_stop()
            self._timer_start()


class FigureCanvasWebAgg(backend_agg.FigureCanvasAgg):
    supports_blit = False

    def __init__(self, *args, **kwargs):
        backend_agg.FigureCanvasAgg.__init__(self, *args, **kwargs)
        self.png_buffer = cStringIO.StringIO()
        self.png_is_old = True
        self.force_full = True
        self.pending_draw = None

    def show(self):
        # show the figure window
        show()

    def draw(self):
        # TODO: Do we just queue the drawing here?  That's what Gtk does
        renderer = self.get_renderer()

        self.png_is_old = True

        backend_agg.RendererAgg.lock.acquire()
        try:
            self.figure.draw(renderer)
        finally:
            backend_agg.RendererAgg.lock.release()
            # Swap the frames
            self.manager.refresh_all()

    def draw_idle(self):
        if self.pending_draw is None:
            ioloop = tornado.ioloop.IOLoop.instance()
            self.pending_draw = ioloop.add_timeout(
                datetime.timedelta(milliseconds=50),
                self._draw_idle_callback)

    def _draw_idle_callback(self):
        try:
            self.draw()
        finally:
            self.pending_draw = None

    def get_diff_image(self):
        if self.png_is_old:
            buffer = np.frombuffer(
                self.renderer.buffer_rgba(), dtype=np.uint32)
            buffer = buffer.reshape(
                (self.renderer.height, self.renderer.width))

            if not self.force_full:
                last_buffer = np.frombuffer(
                    self.last_renderer.buffer_rgba(), dtype=np.uint32)
                last_buffer = last_buffer.reshape(
                    (self.renderer.height, self.renderer.width))

                diff = buffer != last_buffer
                output = np.where(diff, buffer, 0)
            else:
                output = buffer

            self.png_buffer.reset()
            self.png_buffer.truncate()
            # TODO: We should write a new version of write_png that
            # handles the differencing inline
            _png.write_png(
                output.tostring(),
                output.shape[1], output.shape[0],
                self.png_buffer)

            self.renderer, self.last_renderer = \
              self.last_renderer, self.renderer
            self.force_full = False
            self.png_is_old = False
        return self.png_buffer.getvalue()

    def get_renderer(self):
        l, b, w, h = self.figure.bbox.bounds
        key = w, h, self.figure.dpi
        try:
            self._lastKey, self.renderer
        except AttributeError:
            need_new_renderer = True
        else:
            need_new_renderer = (self._lastKey != key)

        if need_new_renderer:
            self.renderer = backend_agg.RendererAgg(w, h, self.figure.dpi)
            self.last_renderer = backend_agg.RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key

        return self.renderer

    def handle_event(self, event):
        type = event['type']
        if type in ('button_press', 'button_release', 'motion_notify'):
            x = event['x']
            y = event['y']
            y = self.get_renderer().height - y

            # Javascript button numbers and matplotlib button numbers are
            # off by 1
            button = event['button'] + 1

            # The right mouse button pops up a context menu, which doesn't
            # work very well, so use the middle mouse button instead
            if button == 2:
                button = 3

            if type == 'button_press':
                self.button_press_event(x, y, button)
            elif type == 'button_release':
                self.button_release_event(x, y, button)
            elif type == 'motion_notify':
                self.motion_notify_event(x, y)
        elif type in ('key_press', 'key_release'):
            key = event['key']

            if type == 'key_press':
                self.key_press_event(key)
            elif type == 'key_release':
                self.key_release_event(key)
        elif type == 'toolbar_button':
            # TODO: Be more suspicious of the input
            getattr(self.toolbar, event['name'])()
        elif type == 'refresh':
            self.force_full = True
            self.draw_idle()

    def send_event(self, event_type, **kwargs):
        self.manager.send_event(event_type, **kwargs)

    def new_timer(self, *args, **kwargs):
        return TimerTornado(*args, **kwargs)

    def start_event_loop(self, timeout):
        backend_bases.FigureCanvasBase.start_event_loop_default(
            self, timeout)
    start_event_loop.__doc__ = backend_bases.FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        backend_bases.FigureCanvasBase.stop_event_loop_default(self)
    stop_event_loop.__doc__ = backend_bases.FigureCanvasBase.stop_event_loop_default.__doc__


class FigureManagerWebAgg(backend_bases.FigureManagerBase):
    def __init__(self, canvas, num):
        backend_bases.FigureManagerBase.__init__(self, canvas, num)

        self.web_sockets = set()

        self.canvas = canvas
        self.num = num

        self.toolbar = self._get_toolbar(canvas)

    def show(self):
        pass

    def add_web_socket(self, web_socket):
        self.web_sockets.add(web_socket)

    def remove_web_socket(self, web_socket):
        self.web_sockets.remove(web_socket)

    def refresh_all(self):
        for s in self.web_sockets:
            s.send_image()

    def send_event(self, event_type, **kwargs):
        for s in self.web_sockets:
            s.send_event(event_type, **kwargs)

    def _get_toolbar(self, canvas):
        toolbar = NavigationToolbar2WebAgg(canvas)
        return toolbar

    def resize(self, w, h):
        self.send_event('resize', size=(w, h))


class NavigationToolbar2WebAgg(backend_bases.NavigationToolbar2):
    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to  previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),
        ('Download', 'Download plot', 'filesave', 'download')
        )

    def _init_toolbar(self):
        self.message = ''
        self.cursor = 0

    def _get_canvas(self, fig):
        return FigureCanvasWebAgg(fig)

    def set_message(self, message):
        if message != self.message:
            self.canvas.send_event("message", message=message)
        self.message = message

    def set_cursor(self, cursor):
        if cursor != self.cursor:
            self.canvas.send_event("cursor", cursor=cursor)
        self.cursor = cursor

    def dynamic_update(self):
        self.canvas.draw_idle()


class WebAggApplication(tornado.web.Application):
    initialized = False
    started = False

    class FavIcon(tornado.web.RequestHandler):
        def get(self):
            self.set_header('Content-Type', 'image/png')
            with open(os.path.join(
                    os.path.dirname(__file__),
                    '../mpl-data/images/matplotlib.png')) as fd:
                self.write(fd.read())

    class IndexPage(tornado.web.RequestHandler):
        def get(self, fignum):
            with open(os.path.join(
                    os.path.dirname(__file__),
                    'web_static', 'index.html')) as fd:
                tpl = fd.read()

            fignum = int(fignum)
            manager = Gcf().get_fig_manager(fignum)

            t = tornado.template.Template(tpl)
            self.write(t.generate(
                toolitems=NavigationToolbar2WebAgg.toolitems,
                canvas=manager.canvas))

    class Download(tornado.web.RequestHandler):
        def get(self, fignum, format):
            self.fignum = int(fignum)
            manager = Gcf().get_fig_manager(self.fignum)

            # TODO: Move this to a central location
            mimetypes = {
                'ps': 'application/postscript',
                'eps': 'application/postscript',
                'pdf': 'application/pdf',
                'svg': 'image/svg+xml',
                'png': 'image/png',
                'jpeg': 'image/jpeg',
                'tif': 'image/tiff',
                'emf': 'application/emf'
                }

            self.set_header('Content-Type', mimetypes.get(format, 'binary'))

            buffer = cStringIO.StringIO()
            manager.canvas.print_figure(buffer, format=format)
            self.write(buffer.getvalue())

    class WebSocket(tornado.websocket.WebSocketHandler):
        def open(self, fignum):
            self.fignum = int(fignum)
            manager = Gcf().get_fig_manager(self.fignum)
            manager.add_web_socket(self)
            l, b, w, h = manager.canvas.figure.bbox.bounds
            manager.resize(w, h)
            self.on_message('{"type":"refresh"}')

        def on_close(self):
            Gcf().get_fig_manager(self.fignum).remove_web_socket(self)

        def on_message(self, message):
            message = json.loads(message)
            canvas = Gcf().get_fig_manager(self.fignum).canvas
            canvas.handle_event(message)

        def send_event(self, event_type, **kwargs):
            payload = {'type': event_type}
            payload.update(kwargs)
            self.write_message(json.dumps(payload))

        def send_image(self):
            canvas = Gcf().get_fig_manager(self.fignum).canvas
            diff = canvas.get_diff_image()
            self.write_message(diff, binary=True)

    def __init__(self):
        super(WebAggApplication, self).__init__([
            # Static files for the CSS and JS
            (r'/static/(.*)', tornado.web.StaticFileHandler,
             {'path':
              os.path.join(os.path.dirname(__file__), 'web_static')}),
            # Static images for toolbar buttons
            (r'/images/(.*)', tornado.web.StaticFileHandler,
             {'path':
              os.path.join(os.path.dirname(__file__), '../mpl-data/images')}),
            # An MPL favicon
            (r'/favicon.ico', self.FavIcon),
            # The page that contains all of the pieces
            (r'/([0-9]+)/', self.IndexPage),
            # Sends images and events to the browser, and receives
            # events from the browser
            (r'/([0-9]+)/ws', self.WebSocket),
            # Handles the downloading (i.e., saving) of static images
            (r'/([0-9]+)/download.([a-z]+)', self.Download)
            ])

    @classmethod
    def initialize(cls):
        if cls.initialized:
            return

        app = cls()

        def random_ports(port, n):
            """Generate a list of n random ports near the given port.

            The first 5 ports will be sequential, and the remaining n-5 will be
            randomly selected in the range [port-2*n, port+2*n].
            """
            for i in range(min(5, n)):
                yield port + i
            for i in range(n - 5):
                yield port + random.randint(-2 * n, 2 * n)

        success = None
        cls.port = rcParams['webagg.port']
        # TODO: Configure port_retrues
        for port in random_ports(cls.port, 50):
            try:
                app.listen(port)
            except socket.error as e:
                if e.errno != errno.EADDRINUSE:
                    raise
            else:
                cls.port = port
                success = True
                break

        if not success:
            raise SystemExit(
                "The webagg server could not be started because an available "
                "port could not be found")

        cls.initialized = True

    @classmethod
    def start(cls):
        if cls.started:
            return

        tornado.ioloop.IOLoop.instance().start()

        cls.started = True
