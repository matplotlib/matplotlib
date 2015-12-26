"""Interactive figures in the IPython notebook"""
# Note: There is a notebook in
# lib/matplotlib/backends/web_backend/nbagg_uat.ipynb to help verify
# that changes made maintain expected behaviour.

import datetime
from base64 import b64encode
import json
import io
import os
from matplotlib.externals import six
from uuid import uuid4 as uuid

import tornado.ioloop

from IPython.display import display, Javascript, HTML
try:
    # Jupyter/IPython 4.x or later
    from ipywidgets import DOMWidget
    from traitlets import Unicode, Instance, Bool
except ImportError:
    # Jupyter/IPython 3.x or earlier
    from IPython.html.widgets import DOMWidget
    from IPython.utils.traitlets import Unicode, Instance, Bool

from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib import is_interactive
from matplotlib.backends.backend_webagg_core import (FigureManagerWebAgg,
                                                     FigureCanvasWebAggCore,
                                                     NavigationToolbar2WebAgg,
                                                     TimerTornado)
from matplotlib.backend_bases import (ShowBase, NavigationToolbar2,
                                      FigureCanvasBase)


class Show(ShowBase):
    def __call__(self, block=None):
        from matplotlib._pylab_helpers import Gcf

        managers = Gcf.get_all_fig_managers()
        if not managers:
            return

        interactive = is_interactive()

        for manager in managers:
            manager.show()

            # plt.figure adds an event which puts the figure in focus
            # in the activeQue. Disable this behaviour, as it results in
            # figures being put as the active figure after they have been
            # shown, even in non-interactive mode.
            if hasattr(manager, '_cidgcf'):
                manager.canvas.mpl_disconnect(manager._cidgcf)

            if not interactive and manager in Gcf._activeQue:
                Gcf._activeQue.remove(manager)

show = Show()


def draw_if_interactive():
    import matplotlib._pylab_helpers as pylab_helpers

    if is_interactive():
        manager = pylab_helpers.Gcf.get_active()
        if manager is not None:
            manager.show()


def connection_info():
    """
    Return a string showing the figure and connection status for
    the backend. This is intended as a diagnostic tool, and not for general
    use.

    """
    from matplotlib._pylab_helpers import Gcf
    result = []
    for manager in Gcf.get_all_fig_managers():
        fig = manager.canvas.figure
        result.append('{0} - {0}'.format((fig.get_label() or
                                          "Figure {0}".format(manager.num)),
                                         manager.web_sockets))
    if not is_interactive():
        result.append('Figures pending show: {0}'.format(len(Gcf._activeQue)))
    return '\n'.join(result)


# Note: Version 3.2 and 4.x icons
# http://fontawesome.io/3.2.1/icons/
# http://fontawesome.io/
# the `fa fa-xxx` part targets font-awesome 4, (IPython 3.x)
# the icon-xxx targets font awesome 3.21 (IPython 2.x)
_FONT_AWESOME_CLASSES = {
    'home': 'fa fa-home icon-home',
    'back': 'fa fa-arrow-left icon-arrow-left',
    'forward': 'fa fa-arrow-right icon-arrow-right',
    'zoom_to_rect': 'fa fa-square-o icon-check-empty',
    'move': 'fa fa-arrows icon-move',
    'download': 'fa fa-floppy-o icon-save',
    None: None
}


class NavigationIPy(NavigationToolbar2WebAgg):

    # Use the standard toolbar items + download button
    toolitems = [(text, tooltip_text,
                  _FONT_AWESOME_CLASSES[image_file], name_of_method)
                 for text, tooltip_text, image_file, name_of_method
                 in (NavigationToolbar2.toolitems +
                     (('Download', 'Download plot', 'download', 'download'),))
                 if image_file in _FONT_AWESOME_CLASSES]


class MPLCanvasWidget(DOMWidget):
    _view_module = Unicode("nbextensions/matplotlib/canvas.widget", sync=True)
    _view_name = Unicode('MPLCanvasView', sync=True)
    manager = Instance('FigureManagerNbAgg')
    supports_binary = Bool(False)
    closed = Bool(False)

    def on_msg(self, message):
        # The 'supports_binary' message is relevant to the
        # websocket itself.  The other messages get passed along
        # to matplotlib as-is.

        # Every message has a "type" and a "figure_id".
        message = json.loads(message['content']['data'])
        if message['type'] == 'closing':
            self.closed = True
            self.manager.clearup_closed()
        elif message['type'] == 'supports_binary':
            self.supports_binary = message['value']
        else:
            self.manager.handle_json(message)

    def send_json(self, content):
        self.send({'data': json.dumps(content)})

    def send_binary(self, blob):
        # The comm is ascii, so we always send the image in base64
        # encoded data URL form.
        data = b64encode(blob)
        if six.PY3:
            data = data.decode('ascii')
        data_uri = "data:image/png;base64,{0}".format(data)
        self.send({'data': data_uri})


class FigureManagerNbAgg(FigureManagerWebAgg):
    ToolbarCls = NavigationIPy

    def __init__(self, canvas, num):
        self._shown = False
        FigureManagerWebAgg.__init__(self, canvas, num)
        self.widgets = []

    def show(self):
        if not self._shown:
            widget = MPLCanvasWidget(manager=self)
            self.widgets.append(widget)
            self.add_web_socket(widget)
            display(widget)
        else:
            self.canvas.draw_idle()
        self._shown = True

    def reshow(self):
        """
        A special method to re-show the figure in the notebook.

        """
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

    def destroy(self):
        self._send_event('close')
        # need to copy comms as callbacks will modify this list
        for comm in list(self.web_sockets):
            comm.closed = True
        self.clearup_closed()

    def clearup_closed(self):
        """Clear up any closed Comms."""
        self.web_sockets = set([socket for socket in self.web_sockets
                                if not socket.closed])

        if len(self.web_sockets) == 0:
            self.canvas.close_event()

    def remove_comm(self, comm_id):
        self.web_sockets = set([socket for socket in self.web_sockets
                                if not socket.comm.comm_id == comm_id])


class FigureCanvasNbAgg(FigureCanvasWebAggCore):
    def new_timer(self, *args, **kwargs):
        return TimerTornado(*args, **kwargs)

    def start_event_loop(self, timeout):
        FigureCanvasBase.start_event_loop_default(self, timeout)

    def stop_event_loop(self):
        FigureCanvasBase.stop_event_loop_default(self)


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
    from .._pylab_helpers import Gcf

    def closer(event):
        Gcf.destroy(num)

    canvas = FigureCanvasNbAgg(figure)
    if rcParams['nbagg.transparent']:
        figure.patch.set_alpha(0)
    manager = FigureManagerNbAgg(canvas, num)

    if is_interactive():
        manager.show()
        figure.canvas.draw_idle()

    canvas.mpl_connect('close_event', closer)

    return manager
