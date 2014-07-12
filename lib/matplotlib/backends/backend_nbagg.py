"""Interactive figures in the IPython notebook"""
from base64 import b64encode
import json
import io
import os
from uuid import uuid4 as uuid

from IPython.display import display, Javascript, HTML
from IPython.kernel.comm import Comm

from matplotlib.figure import Figure
from matplotlib.backends.backend_webagg_core import (FigureManagerWebAgg,
                                                     FigureCanvasWebAggCore,
                                                     NavigationToolbar2WebAgg)
from matplotlib.backend_bases import ShowBase, NavigationToolbar2


class Show(ShowBase):
    def __call__(self, block=None):
        import matplotlib._pylab_helpers as pylab_helpers
        from matplotlib import is_interactive

        managers = pylab_helpers.Gcf.get_all_fig_managers()
        if not managers:
            return

        interactive = is_interactive()

        for manager in managers:
            manager.show()
            if not interactive and manager in pylab_helpers.Gcf._activeQue:
                pylab_helpers.Gcf._activeQue.remove(manager)


show = Show()


def draw_if_interactive():
    from matplotlib import is_interactive
    import matplotlib._pylab_helpers as pylab_helpers

    if is_interactive():
        manager = pylab_helpers.Gcf.get_active()
        if manager is not None:
            manager.show()


def connection_info():
    """
    Return a string showing the figure and connection status for
    the backend.

    """
    # TODO: Make this useful!
    import matplotlib._pylab_helpers as pylab_helpers
    result = []
    for manager in pylab_helpers.Gcf.get_all_fig_managers():
        fig = manager.canvas.figure
        result.append('{} - {}'.format((fig.get_label() or
                                        "Figure {0}".format(manager.num)),
                                       manager.web_sockets))
    result.append('Figures pending show: ' +
                  str(len(pylab_helpers.Gcf._activeQue)))
    return '\n'.join(result)


class NavigationIPy(NavigationToolbar2WebAgg):
    # Note: Version 3.2 icons, not the later 4.0 ones.
    # http://fontawesome.io/3.2.1/icons/
    _font_awesome_classes = {
        'home': 'icon-home',
        'back': 'icon-arrow-left',
        'forward': 'icon-arrow-right',
        'zoom_to_rect': 'icon-check-empty',
        'move': 'icon-move',
        None: None
    }

    # Use the standard toolbar items + download button
    toolitems = [(text, tooltip_text,
                  _font_awesome_classes[image_file], name_of_method)
                 for text, tooltip_text, image_file, name_of_method
                 in NavigationToolbar2.toolitems
                 if image_file in _font_awesome_classes]


class FigureManagerNbAgg(FigureManagerWebAgg):
    ToolbarCls = NavigationIPy

    def __init__(self, canvas, num):
        self._shown = False
        FigureManagerWebAgg.__init__(self, canvas, num)

    def display_js(self):
        # XXX How to do this just once? It has to deal with multiple
        # browser instances using the same kernel.
        display(Javascript(FigureManagerNbAgg.get_javascript()))

    def show(self):
        if not self._shown:
            self.display_js()
            self._create_comm()
        else:
            self.canvas.draw_idle()
        self._shown = True

    def reshow(self):
        self._shown = False
        self.show()

    @property
    def connected(self):
        return bool(self.web_sockets)

    @classmethod
    def get_javascript(cls, stream=None):
        if stream is None:
            output = io.StringIO()
        else:
            output = stream
        super(FigureManagerNbAgg, cls).get_javascript(stream=output)
        with io.open(os.path.join(
                os.path.dirname(__file__),
                "web_backend",
                "nbagg_mpl.js"), encoding='utf8') as fd:
            output.write(fd.read())
        if stream is None:
            return output.getvalue()

    def _create_comm(self):
        comm = CommSocket(self)
        self.add_web_socket(comm)
        return comm

    def destroy(self):
        self._send_event('close')
        for comm in self.web_sockets.copy():
            comm.on_close()


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
    manager = FigureManagerNbAgg(canvas, num)
    return manager


class CommSocket(object):
    """
    Manages the Comm connection between IPython and the browser (client).

    Comms are 2 way, with the CommSocket being able to publish a message
    via the send_json method, and handle a message with on_message. On the
    JS side figure.send_message and figure.ws.onmessage do the sending and
    receiving respectively.

    """
    def __init__(self, manager):
        self.supports_binary = None
        self.manager = manager
        self.uuid = str(uuid())
        display(HTML("<div id=%r></div>" % self.uuid))
        try:
            self.comm = Comm('matplotlib', data={'id': self.uuid})
        except AttributeError:
            raise RuntimeError('Unable to create an IPython notebook Comm '
                               'instance. Are you in the IPython notebook?')
        self.comm.on_msg(self.on_message)

    def on_close(self):
        # When the socket is closed, deregister the websocket with
        # the FigureManager.
        if self.comm in self.manager.web_sockets:
            self.manager.remove_web_socket(self)
        self.comm.close()

    def send_json(self, content):
        self.comm.send({'data': json.dumps(content)})

    def send_binary(self, blob):
        # The comm is ascii, so we always send the image in base64
        # encoded data URL form.
        data_uri = "data:image/png;base64,{0}".format(b64encode(blob))
        self.comm.send({'data': data_uri})

    def on_message(self, message):
        # The 'supports_binary' message is relevant to the
        # websocket itself.  The other messages get passed along
        # to matplotlib as-is.

        # Every message has a "type" and a "figure_id".
        message = json.loads(message['content']['data'])
        if message['type'] == 'closing':
            self.on_close()
        elif message['type'] == 'supports_binary':
            self.supports_binary = message['value']
        else:
            self.manager.handle_json(message)
