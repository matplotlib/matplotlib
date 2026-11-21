from contextlib import closing

import pytest

from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure


def test_double_click_sequence():
    pytest.importorskip("jupyter_rfb")
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
