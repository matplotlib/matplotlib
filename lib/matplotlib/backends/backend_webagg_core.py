"""
Displays Agg images in the browser, with interactivity
"""
# The WebAgg backend is divided into two modules:
#
# - `backend_webagg_core.py` contains code necessary to embed a WebAgg
#   plot inside of a web application, and communicate in an abstract
#   way over a web socket.
#
# - `backend_webagg.py` contains a concrete implementation of a basic
#   application, implemented with tornado.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import io
import json
import os
import time

import numpy as np

from matplotlib.backends import backend_agg
from matplotlib.figure import Figure
from matplotlib import backend_bases
from matplotlib import _png


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
    canvas = FigureCanvasWebAggCore(figure)
    manager = FigureManagerWebAgg(canvas, num)
    return manager


class FigureCanvasWebAggCore(backend_agg.FigureCanvasAgg):
    supports_blit = False

    def __init__(self, *args, **kwargs):
        backend_agg.FigureCanvasAgg.__init__(self, *args, **kwargs)

        # A buffer to hold the PNG data for the last frame.  This is
        # retained so it can be resent to each client without
        # regenerating it.
        self._png_buffer = io.BytesIO()

        # Set to True when the renderer contains data that is newer
        # than the PNG buffer.
        self._png_is_old = True

        # Set to True by the `refresh` message so that the next frame
        # sent to the clients will be a full frame.
        self._force_full = True

    def show(self):
        # show the figure window
        from matplotlib.pyplot import show
        show()

    def draw(self):
        renderer = self.get_renderer()

        self._png_is_old = True

        backend_agg.RendererAgg.lock.acquire()
        try:
            self.figure.draw(renderer)
        finally:
            backend_agg.RendererAgg.lock.release()
            # Swap the frames
            self.manager.refresh_all()

    def draw_idle(self):
        self.send_event("draw")

    def get_diff_image(self):
        if self._png_is_old:
            # The buffer is created as type uint32 so that entire
            # pixels can be compared in one numpy call, rather than
            # needing to compare each plane separately.
            buff = np.frombuffer(
                self.get_renderer().buffer_rgba(), dtype=np.uint32)
            buff.shape = (
                self._renderer.height, self._renderer.width)

            if not self._force_full:
                last_buffer = np.frombuffer(
                    self._last_renderer.buffer_rgba(), dtype=np.uint32)
                last_buffer.shape = (
                    self._renderer.height, self._renderer.width)

                diff = buff != last_buffer
                output = np.where(diff, buff, 0)
            else:
                output = buff

            # Clear out the PNG data buffer rather than recreating it
            # each time.  This reduces the number of memory
            # (de)allocations.
            self._png_buffer.truncate()
            self._png_buffer.seek(0)

            # TODO: We should write a new version of write_png that
            # handles the differencing inline
            _png.write_png(
                output.tostring(),
                output.shape[1], output.shape[0],
                self._png_buffer)

            # Swap the renderer frames
            self._renderer, self._last_renderer = (
                self._last_renderer, self._renderer)
            self._force_full = False
            self._png_is_old = False
        return self._png_buffer.getvalue()

    def get_renderer(self, cleared=None):
        # Mirrors super.get_renderer, but caches the old one
        # so that we can do things such as produce a diff image
        # in get_diff_image
        _, _, w, h = self.figure.bbox.bounds
        key = w, h, self.figure.dpi
        try:
            self._lastKey, self._renderer
        except AttributeError:
            need_new_renderer = True
        else:
            need_new_renderer = (self._lastKey != key)

        if need_new_renderer:
            self._renderer = backend_agg.RendererAgg(
                w, h, self.figure.dpi)
            self._last_renderer = backend_agg.RendererAgg(
                w, h, self.figure.dpi)
            self._lastKey = key

        return self._renderer

    def handle_event(self, event):
        e_type = event['type']
        if e_type == 'ack':
            # Network latency tends to decrease if traffic is flowing
            # in both directions.  Therefore, the browser sends back
            # an "ack" message after each image frame is received.
            # This could also be used as a simple sanity check in the
            # future, but for now the performance increase is enough
            # to justify it, even if the server does nothing with it.
            pass
        elif e_type == 'draw':
            self.draw()
        elif e_type in ('button_press', 'button_release', 'motion_notify'):
            x = event['x']
            y = event['y']
            y = self.get_renderer().height - y

            # Javascript button numbers and matplotlib button numbers are
            # off by 1
            button = event['button'] + 1

            # The right mouse button pops up a context menu, which
            # doesn't work very well, so use the middle mouse button
            # instead.  It doesn't seem that it's possible to disable
            # the context menu in recent versions of Chrome.
            if button == 2:
                button = 3

            if e_type == 'button_press':
                self.button_press_event(x, y, button)
            elif e_type == 'button_release':
                self.button_release_event(x, y, button)
            elif e_type == 'motion_notify':
                self.motion_notify_event(x, y)
        elif e_type in ('key_press', 'key_release'):
            key = event['key']

            if e_type == 'key_press':
                self.key_press_event(key)
            elif e_type == 'key_release':
                self.key_release_event(key)
        elif e_type == 'toolbar_button':
            # TODO: Be more suspicious of the input
            getattr(self.toolbar, event['name'])()
        elif e_type == 'refresh':
            figure_label = self.figure.get_label()
            if not figure_label:
                figure_label = "Figure {0}".format(self.manager.num)
            self.send_event('figure_label', label=figure_label)
            self._force_full = True
            self.draw_idle()
        else:
            handler = getattr(self, 'handle_{}'.format(e_type), None)
            if handler is None:
                import warnings
                warnings.warn('Unhandled message type {}. {}'.format(
                                                        e_type, event))
            else:
                return handler(event)

    def handle_resize(self, event):
        x, y = event.get('width', 800), event.get('height', 800)
        x, y = int(x), int(y)
        fig = self.figure
        # An attempt at approximating the figure size in pixels.
        fig.set_size_inches(x / fig.dpi, y / fig.dpi)

        _, _, w, h = self.figure.bbox.bounds
        # Acknowledge the resize, and force the viewer to update the
        # canvas size to the figure's new size (which is hopefully
        # identical or within a pixel or so).
        self._png_is_old = True
        self.manager.resize(w, h)

    def send_event(self, event_type, **kwargs):
        self.manager._send_event(event_type, **kwargs)

    def start_event_loop(self, timeout):
        backend_bases.FigureCanvasBase.start_event_loop_default(
            self, timeout)
    start_event_loop.__doc__ = \
        backend_bases.FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        backend_bases.FigureCanvasBase.stop_event_loop_default(self)
    stop_event_loop.__doc__ = \
        backend_bases.FigureCanvasBase.stop_event_loop_default.__doc__


class NavigationToolbar2WebAgg(backend_bases.NavigationToolbar2):
    _jquery_icon_classes = {
        'home': 'ui-icon ui-icon-home',
        'back': 'ui-icon ui-icon-circle-arrow-w',
        'forward': 'ui-icon ui-icon-circle-arrow-e',
        'zoom_to_rect': 'ui-icon ui-icon-search',
        'move': 'ui-icon ui-icon-arrow-4',
        'download': 'ui-icon ui-icon-disk',
        None: None,
    }

    # Use the standard toolbar items + download button
    toolitems = [(text, tooltip_text, _jquery_icon_classes[image_file],
                  name_of_method)
                 for text, tooltip_text, image_file, name_of_method
                 in (backend_bases.NavigationToolbar2.toolitems +
                     (('Download', 'Download plot', 'download', 'download'),))
                 if image_file in _jquery_icon_classes]

    def _init_toolbar(self):
        self.message = ''
        self.cursor = 0

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

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.send_event(
            "rubberband", x0=x0, y0=y0, x1=x1, y1=y1)

    def release_zoom(self, event):
        super(NavigationToolbar2WebAgg, self).release_zoom(event)
        self.canvas.send_event(
            "rubberband", x0=-1, y0=-1, x1=-1, y1=-1)


class FigureManagerWebAgg(backend_bases.FigureManagerBase):
    ToolbarCls = NavigationToolbar2WebAgg

    def __init__(self, canvas, num):
        backend_bases.FigureManagerBase.__init__(self, canvas, num)

        self.web_sockets = set()

        self.toolbar = self._get_toolbar(canvas)

    def show(self):
        pass

    def _get_toolbar(self, canvas):
        toolbar = self.ToolbarCls(canvas)
        return toolbar

    def resize(self, w, h):
        self._send_event('resize', size=(w, h))

    def set_window_title(self, title):
        self._send_event('figure_label', label=title)

    # The following methods are specific to FigureManagerWebAgg

    def add_web_socket(self, web_socket):
        assert hasattr(web_socket, 'send_binary')
        assert hasattr(web_socket, 'send_json')

        self.web_sockets.add(web_socket)

        _, _, w, h = self.canvas.figure.bbox.bounds
        self.resize(w, h)
        self._send_event('refresh')

    def remove_web_socket(self, web_socket):
        self.web_sockets.remove(web_socket)

    def handle_json(self, content):
        self.canvas.handle_event(content)

    def refresh_all(self):
        if self.web_sockets:
            diff = self.canvas.get_diff_image()
            for s in self.web_sockets:
                s.send_binary(diff)

    @classmethod
    def get_javascript(cls, stream=None):
        if stream is None:
            output = io.StringIO()
        else:
            output = stream

        with io.open(os.path.join(
                os.path.dirname(__file__),
                "web_backend",
                "mpl.js"), encoding='utf8') as fd:
            output.write(fd.read())

        toolitems = []
        for name, tooltip, image, method in cls.ToolbarCls.toolitems:
            if name is None:
                toolitems.append(['', '', '', ''])
            else:
                toolitems.append([name, tooltip, image, method])
        output.write("mpl.toolbar_items = {0};\n\n".format(
            json.dumps(toolitems)))

        extensions = []
        for filetype, ext in sorted(FigureCanvasWebAggCore.
                                    get_supported_filetypes_grouped().
                                    items()):
            extensions.append(ext[0])
        output.write("mpl.extensions = {0};\n\n".format(
            json.dumps(extensions)))

        output.write("mpl.default_extension = {0};".format(
            json.dumps(FigureCanvasWebAggCore.get_default_filetype())))

        if stream is None:
            return output.getvalue()

    @classmethod
    def get_static_file_path(cls):
        return os.path.join(os.path.dirname(__file__), 'web_backend')

    def _send_event(self, event_type, **kwargs):
        payload = {'type': event_type}
        payload.update(kwargs)
        for s in self.web_sockets:
            s.send_json(payload)
