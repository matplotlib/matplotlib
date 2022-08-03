import io
import os
import shutil
import sys
from unittest.mock import MagicMock

from PIL import Image
import pytest

import matplotlib.backends.backend_webagg_core
from matplotlib.backends.backend_webagg_core import (
    FigureCanvasWebAggCore, NavigationToolbar2WebAgg,
)
from matplotlib.testing import subprocess_run_for_testing

import matplotlib.pyplot as plt
from matplotlib.backends.backend_webagg import WebAggApplication
from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import _image_directories
from matplotlib.testing.exceptions import ImageComparisonFailure


pytest.importorskip('tornado')


try:
    import pytest_playwright  # noqa
except ImportError:
    @pytest.fixture
    def page():
        pytest.skip(reason='Missing pytest-playwright')


@pytest.mark.parametrize("backend", ["webagg", "nbagg"])
def test_webagg_fallback(backend):
    if backend == "nbagg":
        pytest.importorskip("IPython")
    env = dict(os.environ)
    if sys.platform != "win32":
        env["DISPLAY"] = ""

    env["MPLBACKEND"] = backend

    test_code = (
        "import os;"
        + f"assert os.environ['MPLBACKEND'] == '{backend}';"
        + "import matplotlib.pyplot as plt; "
        + "print(plt.get_backend());"
        f"assert '{backend}' == plt.get_backend().lower();"
    )
    subprocess_run_for_testing([sys.executable, "-c", test_code], env=env, check=True)


def test_webagg_core_no_toolbar():
    fm = matplotlib.backends.backend_webagg_core.FigureManagerWebAgg
    assert fm._toolbar2_class is None


def test_toolbar_button_dispatch_allowlist():
    """Only declared toolbar items should be dispatched."""
    fig = MagicMock()
    canvas = FigureCanvasWebAggCore(fig)
    canvas.toolbar = MagicMock(spec=NavigationToolbar2WebAgg)
    canvas.toolbar.toolitems = NavigationToolbar2WebAgg.toolitems

    # Valid toolbar action should be dispatched.
    canvas.handle_toolbar_button({'name': 'home'})
    canvas.toolbar.home.assert_called_once()

    # Invalid names should be silently ignored.
    canvas.toolbar.reset_mock()
    canvas.handle_toolbar_button({'name': '__init__'})
    canvas.handle_toolbar_button({'name': 'not_a_real_button'})
    # No methods should have been called.
    assert canvas.toolbar.method_calls == []


@pytest.mark.parametrize("host, origin, allowed", [
    ("localhost:8988", "http://localhost:8988", True),
    ("localhost:8988", "http://evil.com", False),
    ("localhost:8988", "http://127.0.0.1:8988", False),
    ("localhost:8988", "http://[::1]:8988", False),
    ("127.0.0.1:8988", "http://127.0.0.1:8988", True),
    ("127.0.0.1:8988", "http://localhost:8988", False),
    ("127.0.0.1:8988", "http://[::1]:8988", False),
    ("[::1]:8988", "http://[::1]:8988", True),
    ("[::1]:8988", "http://[::2]:8988", False),
    ("[::1]:8988", "http://localhost:8988", False),
    ("[::1]:8988", "http://evil.com", False),
])
def test_websocket_rejects_cross_origin(host, origin, allowed):
    """Verify Tornado's default check_origin rejects cross-origin requests."""
    from matplotlib.backends.backend_webagg import WebAggApplication

    ws = WebAggApplication.WebSocket.__new__(WebAggApplication.WebSocket)
    ws.request = MagicMock()
    ws.request.headers = {"Host": host}
    assert ws.check_origin(origin) is allowed


@pytest.mark.backend('webagg')
def test_webagg_general(page):
    from playwright.sync_api import expect

    # Listen for all console logs.
    page.on('console', lambda msg: print(f'CONSOLE: {msg.text}'))

    fig, ax = plt.subplots(facecolor='w')

    # Don't start the Tornado event loop, but use the existing event loop
    # started by the `page` fixture.
    WebAggApplication.initialize()
    WebAggApplication.started = True

    page.goto(f'http://{WebAggApplication.address}:{WebAggApplication.port}/')
    expect(page).to_have_title('MPL | WebAgg current figures')

    # Check title.
    expect(page.locator('div.ui-dialog-title')).to_have_text('Figure 1')

    # Check toolbar buttons.
    assert page.locator('button.mpl-widget').count() == len([
        name for name, *_ in fig.canvas.manager.ToolbarCls.toolitems
        if name is not None])

    # Check canvas actually contains something.
    baseline_dir, result_dir = _image_directories(test_webagg_general)
    browser = page.context.browser.browser_type.name
    actual = result_dir / f'{browser}.png'
    expected = result_dir / f'{browser}-expected.png'

    canvas = page.locator('canvas.mpl-canvas')
    actual_bytes = canvas.screenshot()
    im = Image.open(io.BytesIO(actual_bytes))
    # Hide the resize grip, which varies across OS/browser.
    if browser == 'firefox':
        im.paste((255, 255, 255),
                 box=(im.width - 20, im.height - 20, im.width, im.height))
    im.save(actual)
    shutil.copyfile(baseline_dir / f'{browser}.png', expected)

    err = compare_images(expected, actual, tol=0)
    if err:
        raise ImageComparisonFailure(err)
