"""Interactive backend for pyodide running in main browser thread, based on webagg."""

import base64
from io import BytesIO
import json
import mimetypes
from pathlib import Path

from pyodide.code import run_js
from pyodide.ffi import create_proxy

from matplotlib.backend_bases import _Backend
from matplotlib._pylab_helpers import Gcf
from . import backend_webagg_core as core


class FigureManagerPyodide(core.FigureManagerWebAgg):
    _toolbar2_class = core.NavigationToolbar2WebAgg

    @classmethod
    def pyplot_show(cls, *, block=None):
        PyodideApplication.initialize()
        managers = Gcf.get_all_fig_managers()
        for manager in managers:
            manager.show()

    def show(self):
        fignum = str(self.num)

        js_code = \
            """
            var websocket_type = mpl.get_websocket_type();
            const parent_element = document.pyodideMplTarget ?? document.body;
            const fig = new mpl.figure(fig_id, new websocket_type(fig_id), null, parent_element);
            fig;
            """
        js_code = f"var fig_id = '{fignum}';" + js_code

        self.js_fig = run_js(js_code)
        web_socket = PyodideApplication.MockPythonWebSocket(self, self.js_fig.ws)
        web_socket.open(fignum)


class FigureCanvasPyodide(core.FigureCanvasWebAggCore):
    manager_class = FigureManagerPyodide

    def get_diff_image(self):
        ret = super().get_diff_image()
        self._force_full = True
        return ret

    def handle_save(self, event):
        figure_id = event['figure_id']
        format = event['format']

        try:
            from js import alert, document
        except ImportError:
            raise RuntimeError("Save not supported as cannot import js.alert and js.document")

        mimetype = mimetypes.types_map.get(f".{format}")
        if mimetype is None:
            alert(f"Cannot download plot, unable to determine mimetype for '{format}'")
            return

        element = document.createElement('a')
        data = BytesIO()
        self.figure.savefig(data, format=format)

        element.setAttribute(
            "href",
            "data:{};base64,{}".format(
                mimetype, base64.b64encode(data.getvalue()).decode("ascii")
            ),
        )
        element.setAttribute("download", f"plot{figure_id}.{format}")
        element.style.display = "none"
        document.body.appendChild(element)
        element.click()
        document.body.removeChild(element)

class PyodideApplication():
    initialized = False

    class MockPythonWebSocket:
        supports_binary = True

        def __init__(self, manager, js_web_socket):
            self.manager = manager
            self.js_web_socket = js_web_socket
            self.on_message_proxy = None

        def open(self, fignum):
            self.fignum = int(fignum)
            self.on_message_proxy = create_proxy(self.on_message)
            self.js_web_socket.open(self.on_message_proxy)
            self.manager.add_web_socket(self)

        def on_close(self):
            self.manager.remove_web_socket(self)
            self.on_message_proxy.destroy()
            self.on_message_proxy = None

        def on_message(self, message):
            message = json.loads(message)

            # The 'supports_binary' message is on a client-by-client
            # basis.  The others affect the (shared) canvas as a
            # whole.
            if message['type'] == 'supports_binary':
                self.supports_binary = message['value']
            else:
                # It is possible for a figure to be closed,
                # but a stale figure UI is still sending messages
                # from the browser.
                if self.manager is not None:
                    self.manager.handle_json(message)

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

        try:
            from js import document

            css = (Path(__file__).parent / "web_backend/css/mpl.css").read_text(encoding="utf-8")
            style = document.createElement('style')
            style.textContent = css
            document.head.append(style)
        except ImportError:
            # js.document not available, continue without CSS.
            pass

        js_content = core.FigureManagerWebAgg.get_javascript(pyodide=True)
        set_toolbar_image_callback = run_js(js_content)
        set_toolbar_image_callback(create_proxy(PyodideApplication.get_toolbar_image))

        cls.initialized = True

    @classmethod
    def get_toolbar_image(cls, image):
        filename = Path(__file__).parent.parent / f"mpl-data/images/{image}.png"
        png_bytes = filename.read_bytes()
        return png_bytes


@_Backend.export
class _BackendPyodide(_Backend):
    FigureCanvas = FigureCanvasPyodide
    FigureManager = FigureManagerPyodide
