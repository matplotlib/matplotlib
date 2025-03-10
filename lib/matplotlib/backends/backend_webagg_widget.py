"""Display a WebAgg HTML Widget in a Jupyter Notebook."""

import io
from base64 import b64encode

import matplotlib as mpl
from matplotlib.backend_bases import _Backend
from matplotlib._pylab_helpers import Gcf
from matplotlib.backends import backend_webagg_core as core
from matplotlib.backends.backend_webagg import WebAggApplication

try:
    from IPython.display import display
except ImportError as err:
    raise RuntimeError("The WebAggWidget backend requires IPython.") from err

try:
    from ipywidgets import HTML, Layout
except ImportError as err:
    raise RuntimeError("The WebAggWidget backend requires ipywidgets.") from err


class WebAggFigureWidget(HTML):
    _margin_x, _margin_y = (20, 80)

    def __init__(self, f=None):
        super().__init__()

        self.f = f
        self._webagg_address = (
            "http://{0}:{1}/{2}".format(
                mpl.rcParams["webagg.address"],
                WebAggApplication.port,
                self.f.number
                )
            )
        self._setup_widget()

    def _setup_widget(self):
        # TODO find a better way to get the required size to show the full figure
        w = int(self.f.dpi * self.f.get_figwidth() + self._margin_x)
        # 40 px is appended as space for the toolbar.
        h = int((self.f.dpi * self.f.get_figheight() + self._margin_y))
        layout = Layout(width=f"{w}px", height=f"{h}px")

        self.value = (
            f"<iframe src={self._webagg_address} "
            "style='width: 100%; height: 100%; border:none;' scrolling='no' "
            "frameborder='0' allowtransparency='true'></iframe>"
            )
        self.layout = layout

        # Add a callback to dynamically adjust the widget layout on resize of the figure
        def cb(event):
            self.layout.width = f"{(event.width + self._margin_x):.0f}px"
            self.layout.height = f"{(event.height + self._margin_y):.0f}px"
        self.f.canvas.mpl_connect("resize_event", cb)

    def _repr_mimebundle_(self, **kwargs):
        # attach a png of the figure for static display
        buf = io.BytesIO()
        self.f.savefig(buf, format='png', dpi='figure')
        data_url = b64encode(buf.getvalue()).decode('utf-8')

        data = {
            'text/plain': str(self.f),
            'image/png': data_url,
            'application/vnd.jupyter.widget-view+json': {
                'version_major': 2,
                'version_minor': 0,
                'model_id': self._model_id
            }
        }
        return data


class FigureManagerWebAggWidget(core.FigureManagerWebAgg):
    _toolbar2_class = core.NavigationToolbar2WebAgg

    @classmethod
    def pyplot_show(cls, *, block=None):
        managers = Gcf.get_all_fig_managers()
        for m in managers:
            # Only display figures that have not yet been shown
            if m.canvas._webagg_widget is None:
                display(m.canvas._get_widget())


class FigureCanvasWebAggWidget(core.FigureCanvasWebAggCore):
    manager_class = FigureManagerWebAggWidget

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._webagg_widget = None

    def _get_widget(self):
        # Return cached widget if it already exists
        if self._webagg_widget is None:
            WebAggApplication.initialize()
            WebAggApplication.start()

            self._webagg_widget = WebAggFigureWidget(f=self.figure)

        return self._webagg_widget

    def show(self):
        return self._get_widget()

@_Backend.export
class _BackendWebAggWidget(_Backend):
    FigureCanvas = FigureCanvasWebAggWidget
    FigureManager = FigureManagerWebAggWidget
