import asyncio
from contextlib import closing

import numpy as np
import pytest

from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure

pytest.importorskip("jupyter_rfb")


def test_double_click_sequence():
    from matplotlib.backends.backend_anywidget import FigureCanvas, _FigureWidget

    canvas = FigureCanvas(Figure())
    events = []
    for name in ("button_press_event", "button_release_event", "motion_notify_event"):
        canvas.mpl_connect(name, events.append)

    with closing(_FigureWidget(canvas)) as widget:
        for kind, timestamp in [
            ("pointer_down", 1.0), ("pointer_up", 1.1),
            ("pointer_down", 1.2), ("pointer_up", 1.3), ("double_click", 1.3),
            ("pointer_move", 1.4),
        ]:
            widget.handle_event(dict(type=kind, timestamp=timestamp, x=20, y=20,
                                     button=1, buttons=(), modifiers=()))

    left = MouseButton.LEFT
    assert [(e.name, e.button, e.dblclick) for e in events] == [
        ("button_press_event", left, False),
        ("button_release_event", left, False),
        ("button_press_event", left, True),
        ("button_release_event", left, False),
        ("motion_notify_event", None, False),
    ]


def test_scroll_updates_widget_frame():
    from matplotlib.backends.backend_anywidget import FigureCanvas, _FigureWidget

    fig = Figure(figsize=(1, 1), dpi=40)
    canvas = FigureCanvas(fig)
    slices = [[[0]], [[1]]]
    image = fig.subplots().imshow(slices[0], cmap="gray", vmin=0, vmax=1)

    def on_scroll(event):
        image.set_data(slices[1 if event.button == "up" else 0])
        canvas.draw()

    canvas.mpl_connect("scroll_event", on_scroll)

    async def scroll():
        with closing(_FigureWidget(canvas)) as widget:
            # A visible frontend can receive frames on the next event-loop turn.
            widget.set_state({"_has_visible_views": True})
            widget.request_draw()
            await asyncio.sleep(0)
            np.testing.assert_array_equal(widget.snapshot().data[20, 20, :3],
                                          [0, 0, 0])

            widget.handle_event(dict(type="wheel", x=20, y=20, dy=-120, modifiers=()))
            await asyncio.sleep(0)
            np.testing.assert_array_equal(widget.snapshot().data[20, 20, :3],
                                          [255, 255, 255])

    asyncio.run(scroll())
