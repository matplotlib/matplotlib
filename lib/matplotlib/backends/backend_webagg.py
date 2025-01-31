"""Displays Agg images in the browser, with interactivity."""

# The WebAgg backend is divided into two modules:
#
# - `backend_webagg_core.py` contains code necessary to embed a WebAgg
#   plot inside of a web application, and communicate in an abstract
#   way over a web socket.
#
# - `backend_webagg.py` contains a concrete implementation of a basic
#   application, implemented with tornado.

from contextlib import contextmanager
import errno
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import random
import sys
import signal
import threading

from js import document
from pyodide.code import run_js
from pyodide.ffi import create_proxy

import matplotlib as mpl
from matplotlib.backend_bases import _Backend
from matplotlib._pylab_helpers import Gcf
from . import backend_webagg_core as core


class FigureManagerWebAgg(core.FigureManagerWebAgg):
    _toolbar2_class = core.NavigationToolbar2WebAgg

    @classmethod
    def pyplot_show(cls, *, block=None):
        WebAggApplication.initialize()

        managers = Gcf.get_all_fig_managers()
        for manager in managers:
            manager.show()

    def show(self):
        fignum = str(self.num)

        js_code = \
            """
            var websocket_type = mpl.get_websocket_type();
            var fig = new mpl.figure(fig_id, new websocket_type(fig_id), null, document.body);
            fig;
            """
        js_code = f"var fig_id = '{fignum}';" + js_code

        js_fig = run_js(js_code)
        web_socket = WebAggApplication.MockPythonWebSocket(self, js_fig.ws)
        web_socket.open(fignum)


class FigureCanvasWebAgg(core.FigureCanvasWebAggCore):
    manager_class = FigureManagerWebAgg


class WebAggApplication():
    initialized = False

    class MockPythonWebSocket:
        supports_binary = True

        def __init__(self, manager, js_web_socket):
            self.manager = manager
            self.js_web_socket = js_web_socket

        def open(self, fignum):
            self.js_web_socket.open(create_proxy(self.on_message))   # should destroy proxy on close/exit?
            self.fignum = int(fignum)
            self.manager.add_web_socket(self)

        def on_close(self):
            self.manager.remove_web_socket(self)

        def on_message(self, message):
            message = message.as_py_json()

            # The 'supports_binary' message is on a client-by-client
            # basis.  The others affect the (shared) canvas as a
            # whole.
            if message['type'] == 'supports_binary':
                self.supports_binary = message['value']
            else:
                manager = self.manager
                # It is possible for a figure to be closed,
                # but a stale figure UI is still sending messages
                # from the browser.
                if manager is not None:
                    manager.handle_json(message)

        def send_json(self, content):
            self.js_web_socket.receive_json(json.dumps(content))

        def send_binary(self, blob):
            if self.supports_binary:
                self.js_web_socket.receive_binary(blob, binary=True)
            else:
                data_uri = "data:image/png;base64,{}".format(
                    blob.encode('base64').replace('\n', ''))
                self.js_web_socket.receive_binary(data_uri)

    @classmethod
    def initialize(cls, url_prefix='', port=None, address=None):
        if cls.initialized:
            return

        css = (Path(__file__).parent / "web_backend/css/mpl.css").read_text(encoding="utf-8")
        style = document.createElement('style')
        style.textContent = css
        document.head.append(style)

        js_content = core.FigureManagerWebAgg.get_javascript()
        set_toolbar_image_callback = run_js(js_content)
        set_toolbar_image_callback(create_proxy(WebAggApplication.get_toolbar_image))

        cls.initialized = True

    @classmethod
    def get_toolbar_image(cls, image):
        filename = Path(__file__).parent.parent / f"mpl-data/images/{image}.png"
        png_bytes = filename.read_bytes()
        return png_bytes


@_Backend.export
class _BackendWebAgg(_Backend):
    FigureCanvas = FigureCanvasWebAgg
    FigureManager = FigureManagerWebAgg
