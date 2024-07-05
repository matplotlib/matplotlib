import os
import re
import shutil
import subprocess
import sys
import warnings

import pytest

import matplotlib.backends.backend_webagg_core
from matplotlib.testing import subprocess_run_for_testing

import matplotlib.pyplot as plt
from matplotlib.backends.backend_webagg import WebAggApplication
from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import _image_directories
from matplotlib.testing.exceptions import ImageComparisonFailure


pytest.importorskip("tornado")


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


@pytest.mark.backend('webagg')
def test_webagg_general(random_port, page):
    from playwright.sync_api import expect

    # Listen for all console logs.
    page.on('console', lambda msg: print(f'CONSOLE: {msg.text}'))

    fig, ax = plt.subplots(facecolor='w')

    # Don't start the Tornado event loop, but use the existing event loop
    # started by the `page` fixture.
    WebAggApplication.initialize(port=random_port)
    WebAggApplication.started = True

    page.goto(f'http://{WebAggApplication.address}:{WebAggApplication.port}/')
    expect(page).to_have_title('MPL | WebAgg current figures')

    # Check title.
    expect(page.locator('div.ui-dialog-title')).to_have_text('Figure 1')

    # Check canvas actually contains something.
    baseline_dir, result_dir = _image_directories(test_webagg_general)
    browser = page.context.browser.browser_type.name
    actual = result_dir / f'{browser}.png'
    expected = result_dir / f'{browser}-expected.png'

    canvas = page.locator('canvas.mpl-canvas')
    canvas.screenshot(path=actual)
    shutil.copyfile(baseline_dir / f'{browser}.png', expected)

    err = compare_images(expected, actual, tol=0)
    if err:
        raise ImageComparisonFailure(err)


@pytest.mark.backend('webagg')
def test_webagg_resize(random_port, page):
    # Listen for all console logs.
    page.on('console', lambda msg: print(f'CONSOLE: {msg.text}'))

    fig, ax = plt.subplots(facecolor='w')
    orig_bbox = fig.bbox.frozen()

    # Don't start the Tornado event loop, but use the existing event loop
    # started by the `page` fixture.
    WebAggApplication.initialize(port=random_port)
    WebAggApplication.started = True

    page.goto(f'http://{WebAggApplication.address}:{WebAggApplication.port}/')

    canvas = page.locator('canvas.mpl-canvas')

    print(f'{orig_bbox=}')
    # Resize the canvas to be twice as big.
    bbox = canvas.bounding_box()
    print(f'{bbox=}')
    x, y = bbox['x'] + bbox['width'] - 1, bbox['y'] + bbox['height'] - 1
    print(f'{x=} {y=}')
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + bbox['width'], y + bbox['height'])
    print(f'{x + bbox["width"]=} {y + bbox["height"]=}')
    page.mouse.up()

    assert fig.bbox.height == orig_bbox.height * 2
    assert fig.bbox.width == orig_bbox.width * 2


@pytest.mark.backend('webagg')
@pytest.mark.parametrize('toolbar', ['toolbar2', 'toolmanager'])
def test_webagg_toolbar(random_port, page, toolbar):
    from playwright.sync_api import expect

    # Listen for all console logs.
    page.on('console', lambda msg: print(f'CONSOLE: {msg.text}'))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Treat the new Tool classes',
                                category=UserWarning)
        plt.rcParams['toolbar'] = toolbar

    fig, ax = plt.subplots(facecolor='w')

    # Don't start the Tornado event loop, but use the existing event loop
    # started by the `page` fixture.
    WebAggApplication.initialize(port=random_port)
    WebAggApplication.started = True

    page.goto(f'http://{WebAggApplication.address}:{WebAggApplication.port}/')

    expect(page.locator('button.mpl-widget')).to_have_count(
        len([
            name for name, *_ in fig.canvas.manager.ToolbarCls.toolitems
            if name is not None]))

    home = page.locator('button.mpl-widget').nth(0)
    expect(home).to_be_visible()

    back = page.locator('button.mpl-widget').nth(1)
    expect(back).to_be_visible()
    forward = page.locator('button.mpl-widget').nth(2)
    expect(forward).to_be_visible()
    if toolbar == 'toolbar2':
        # ToolManager doesn't implement history button disabling.
        # https://github.com/matplotlib/matplotlib/issues/17979
        expect(back).to_be_disabled()
        expect(forward).to_be_disabled()

    pan = page.locator('button.mpl-widget').nth(3)
    expect(pan).to_be_visible()
    zoom = page.locator('button.mpl-widget').nth(4)
    expect(zoom).to_be_visible()

    save = page.locator('button.mpl-widget').nth(5)
    expect(save).to_be_visible()
    format_dropdown = page.locator('select.mpl-widget')
    expect(format_dropdown).to_be_visible()

    if toolbar == 'toolmanager':
        # Location in status bar is not supported by ToolManager.
        return

    ax.set_position([0, 0, 1, 1])
    bbox = page.locator('canvas.mpl-canvas').bounding_box()
    x, y = bbox['x'] + bbox['width'] / 2, bbox['y'] + bbox['height'] / 2
    page.mouse.move(x, y, steps=2)
    message = page.locator('span.mpl-message')
    expect(message).to_have_text('x=0.500 y=0.500')


@pytest.mark.backend('webagg')
def test_webagg_toolbar_save(random_port, page):
    from playwright.sync_api import expect

    # Listen for all console logs.
    page.on('console', lambda msg: print(f'CONSOLE: {msg.text}'))

    fig, ax = plt.subplots(facecolor='w')

    # Don't start the Tornado event loop, but use the existing event loop
    # started by the `page` fixture.
    WebAggApplication.initialize(port=random_port)
    WebAggApplication.started = True

    page.goto(f'http://{WebAggApplication.address}:{WebAggApplication.port}/')

    save = page.locator('button.mpl-widget').nth(5)
    expect(save).to_be_visible()

    with page.context.expect_page() as new_page_info:
        save.click()
    new_page = new_page_info.value

    new_page.wait_for_load_state()
    assert new_page.url.endswith('download.png')


@pytest.mark.backend('webagg')
def test_webagg_toolbar_pan(random_port, page):
    from playwright.sync_api import expect

    # Listen for all console logs.
    page.on('console', lambda msg: print(f'CONSOLE: {msg.text}'))

    fig, ax = plt.subplots(facecolor='w')
    ax.plot([3, 2, 1])
    orig_lim = ax.viewLim.frozen()
    # Make figure coords ~= axes coords, with ticks visible for inspection.
    ax.set_position([0, 0, 1, 1])
    ax.tick_params(axis='y', direction='in', pad=-22)
    ax.tick_params(axis='x', direction='in', pad=-15)

    # Don't start the Tornado event loop, but use the existing event loop
    # started by the `page` fixture.
    WebAggApplication.initialize()
    WebAggApplication.started = True

    page.goto(f'http://{WebAggApplication.address}:{WebAggApplication.port}/')

    canvas = page.locator('canvas.mpl-canvas')
    expect(canvas).to_be_visible()
    home = page.locator('button.mpl-widget').nth(0)
    expect(home).to_be_visible()
    pan = page.locator('button.mpl-widget').nth(3)
    expect(pan).to_be_visible()
    zoom = page.locator('button.mpl-widget').nth(4)
    expect(zoom).to_be_visible()

    active_re = re.compile(r'active')
    expect(pan).not_to_have_class(active_re)
    expect(zoom).not_to_have_class(active_re)
    assert ax.get_navigate_mode() is None
    pan.click()
    expect(pan).to_have_class(active_re)
    expect(zoom).not_to_have_class(active_re)
    assert ax.get_navigate_mode() == 'PAN'

    # Pan 50% of the figure diagonally toward bottom-right.
    bbox = canvas.bounding_box()
    x, y = bbox['x'] + bbox['width'] / 4, bbox['y'] + bbox['height'] / 4
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + bbox['width'] / 2, y + bbox['height'] / 2,
                    steps=20)
    page.mouse.up()

    assert ax.get_xlim() == (orig_lim.x0 - orig_lim.width / 2,
                             orig_lim.x1 - orig_lim.width / 2)
    assert ax.get_ylim() == (orig_lim.y0 + orig_lim.height / 2,
                             orig_lim.y1 + orig_lim.height / 2)

    # Reset.
    home.click()
    assert ax.viewLim.bounds == orig_lim.bounds

    # Pan 50% of the figure diagonally toward bottom-right, while holding 'x'
    # key, to constrain the pan horizontally.
    bbox = canvas.bounding_box()
    x, y = bbox['x'] + bbox['width'] / 4, bbox['y'] + bbox['height'] / 4
    page.mouse.move(x, y)
    page.mouse.down()
    page.keyboard.down('x')
    page.mouse.move(x + bbox['width'] / 2, y + bbox['height'] / 2,
                    steps=20)
    page.mouse.up()
    page.keyboard.up('x')

    assert ax.get_xlim() == (orig_lim.x0 - orig_lim.width / 2,
                             orig_lim.x1 - orig_lim.width / 2)
    assert ax.get_ylim() == (orig_lim.y0, orig_lim.y1)

    # Reset.
    home.click()
    assert ax.viewLim.bounds == orig_lim.bounds

    # Pan 50% of the figure diagonally toward bottom-right, while holding 'y'
    # key, to constrain the pan vertically.
    bbox = canvas.bounding_box()
    x, y = bbox['x'] + bbox['width'] / 4, bbox['y'] + bbox['height'] / 4
    page.mouse.move(x, y)
    page.mouse.down()
    page.keyboard.down('y')
    page.mouse.move(x + bbox['width'] / 2, y + bbox['height'] / 2,
                    steps=20)
    page.mouse.up()
    page.keyboard.up('y')

    assert ax.get_xlim() == (orig_lim.x0, orig_lim.x1)
    assert ax.get_ylim() == (orig_lim.y0 + orig_lim.height / 2,
                             orig_lim.y1 + orig_lim.height / 2)

    # Reset.
    home.click()
    assert ax.viewLim.bounds == orig_lim.bounds

    # Zoom 50% of the figure diagonally toward bottom-right.
    bbox = canvas.bounding_box()
    x, y = bbox['x'], bbox['y']
    page.mouse.move(x, y)
    page.mouse.down(button='right')
    page.mouse.move(x + bbox['width'] / 2, y + bbox['height'] / 2,
                    steps=20)
    page.mouse.up(button='right')

    # Expands in x-direction.
    assert ax.viewLim.x0 == orig_lim.x0
    assert ax.viewLim.x1 < orig_lim.x1 - orig_lim.width / 2
    # Contracts in y-direction.
    assert ax.viewLim.y1 == orig_lim.y1
    assert ax.viewLim.y0 < orig_lim.y0 - orig_lim.height / 2


@pytest.mark.backend('webagg')
def test_webagg_toolbar_zoom(random_port, page):
    from playwright.sync_api import expect

    # Listen for all console logs.
    page.on('console', lambda msg: print(f'CONSOLE: {msg.text}'))

    fig, ax = plt.subplots(facecolor='w')
    ax.plot([3, 2, 1])
    orig_lim = ax.viewLim.frozen()
    # Make figure coords ~= axes coords, with ticks visible for inspection.
    ax.set_position([0, 0, 1, 1])
    ax.tick_params(axis='y', direction='in', pad=-22)
    ax.tick_params(axis='x', direction='in', pad=-15)

    # Don't start the Tornado event loop, but use the existing event loop
    # started by the `page` fixture.
    WebAggApplication.initialize()
    WebAggApplication.started = True

    page.goto(f'http://{WebAggApplication.address}:{WebAggApplication.port}/')

    canvas = page.locator('canvas.mpl-canvas')
    expect(canvas).to_be_visible()
    home = page.locator('button.mpl-widget').nth(0)
    expect(home).to_be_visible()
    pan = page.locator('button.mpl-widget').nth(3)
    expect(pan).to_be_visible()
    zoom = page.locator('button.mpl-widget').nth(4)
    expect(zoom).to_be_visible()

    active_re = re.compile(r'active')
    expect(pan).not_to_have_class(active_re)
    expect(zoom).not_to_have_class(active_re)
    assert ax.get_navigate_mode() is None
    zoom.click()
    expect(pan).not_to_have_class(active_re)
    expect(zoom).to_have_class(active_re)
    assert ax.get_navigate_mode() == 'ZOOM'

    # Zoom 25% in on each side.
    bbox = canvas.bounding_box()
    x, y = bbox['x'] + bbox['width'] / 4, bbox['y'] + bbox['height'] / 4
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + bbox['width'] / 2, y + bbox['height'] / 2,
                    steps=20)
    page.mouse.up()

    assert ax.get_xlim() == (orig_lim.x0 + orig_lim.width / 4,
                             orig_lim.x1 - orig_lim.width / 4)
    assert ax.get_ylim() == (orig_lim.y0 + orig_lim.height / 4,
                             orig_lim.y1 - orig_lim.height / 4)

    # Reset.
    home.click()

    # Zoom 25% in on each side, while holding 'x' key, to constrain the zoom
    # horizontally..
    bbox = canvas.bounding_box()
    x, y = bbox['x'] + bbox['width'] / 4, bbox['y'] + bbox['height'] / 4
    page.mouse.move(x, y)
    page.mouse.down()
    page.keyboard.down('x')
    page.mouse.move(x + bbox['width'] / 2, y + bbox['height'] / 2,
                    steps=20)
    page.mouse.up()
    page.keyboard.up('x')

    assert ax.get_xlim() == (orig_lim.x0 + orig_lim.width / 4,
                             orig_lim.x1 - orig_lim.width / 4)
    assert ax.get_ylim() == (orig_lim.y0, orig_lim.y1)

    # Reset.
    home.click()

    # Zoom 25% in on each side, while holding 'y' key, to constrain the zoom
    # vertically.
    bbox = canvas.bounding_box()
    x, y = bbox['x'] + bbox['width'] / 4, bbox['y'] + bbox['height'] / 4
    page.mouse.move(x, y)
    page.mouse.down()
    page.keyboard.down('y')
    page.mouse.move(x + bbox['width'] / 2, y + bbox['height'] / 2,
                    steps=20)
    page.mouse.up()
    page.keyboard.up('y')

    assert ax.get_xlim() == (orig_lim.x0, orig_lim.x1)
    assert ax.get_ylim() == (orig_lim.y0 + orig_lim.height / 4,
                             orig_lim.y1 - orig_lim.height / 4)

    # Reset.
    home.click()

    # Zoom 25% out on each side.
    bbox = canvas.bounding_box()
    x, y = bbox['x'] + bbox['width'] / 4, bbox['y'] + bbox['height'] / 4
    page.mouse.move(x, y)
    page.mouse.down(button='right')
    page.mouse.move(x + bbox['width'] / 2, y + bbox['height'] / 2,
                    steps=20)
    page.mouse.up(button='right')

    # Limits were doubled, but based on the central point.
    cx = orig_lim.x0 + orig_lim.width / 2
    x0 = cx - orig_lim.width
    x1 = cx + orig_lim.width
    assert ax.get_xlim() == (x0, x1)
    cy = orig_lim.y0 + orig_lim.height / 2
    y0 = cy - orig_lim.height
    y1 = cy + orig_lim.height
    assert ax.get_ylim() == (y0, y1)
