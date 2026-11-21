"""
Display an Agg figure in a notebook using jupyter_rfb.

Install the widget extra, then run this in a notebook::

    import matplotlib
    matplotlib.use("module://matplotlib.backends.backend_anywidget")
    import matplotlib.pyplot as plt

    plt.plot([0, 1, 2], [0, 1, 0])
    plt.show()
"""

import numpy as np
from jupyter_rfb import RemoteFrameBuffer

from matplotlib.backend_bases import (
    FigureManagerBase, LocationEvent, MouseButton, MouseEvent,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg


_BUTTONS = {
    1: MouseButton.LEFT,
    2: MouseButton.RIGHT,
    3: MouseButton.MIDDLE,
    4: MouseButton.BACK,
    5: MouseButton.FORWARD,
}


def _mpl_modifiers(modifiers: tuple[str, ...]) -> list[str]:
    return ["ctrl" if mod == "Control" else mod.lower() for mod in modifiers]


class _FigureWidget(RemoteFrameBuffer):
    def __init__(self, canvas):
        self.canvas = canvas
        self._last_press = None
        width, height = canvas.get_width_height()
        self._display_size = width, height
        super().__init__(
            css_width=f"{width}px",
            css_height=f"{height}px",
            resizable=False,
            quality=100,
        )

    def get_frame(self):
        self.canvas.draw()
        # RFB retains the frame, while Agg reuses its pixel buffer.
        return np.asarray(self.canvas.buffer_rgba()).copy()

    def _near_last_press(self, *, x: float, y: float) -> bool:
        if self._last_press is None:
            return False
        last_x, last_y, _ = self._last_press
        return (x - last_x) ** 2 + (y - last_y) ** 2 <= 5 ** 2

    def _is_double_click(
        self, *, x: float, y: float, button: int, timestamp: float | None,
    ) -> bool:
        # RFB sends double_click after the second release. Matplotlib needs
        # dblclick on the second press, so use the frontend's clock (seconds).
        if button != 1 or timestamp is None:
            self._last_press = None
            return False
        if (self._near_last_press(x=x, y=y)
                and 0 <= timestamp - self._last_press[2] <= 0.5):
            self._last_press = None
            return True
        self._last_press = x, y, timestamp
        return False

    def handle_event(self, event):
        match event:
            case {"type": "resize", "width": width, "height": height}:
                self._display_size = width, height
            case {
                "type": "pointer_enter",
                "x": x,
                "y": y,
                "modifiers": modifiers,
            }:
                self._handle_location_event(
                    name="figure_enter_event",
                    x=x,
                    y=y,
                    modifiers=modifiers,
                )
            case {
                "type": "pointer_down",
                "x": x,
                "y": y,
                "button": button,
                "modifiers": modifiers,
            }:
                self._handle_mouse_event(
                    name="button_press_event",
                    x=x,
                    y=y,
                    button=button,
                    modifiers=modifiers,
                    dblclick=self._is_double_click(
                        x=x, y=y, button=button, timestamp=event.get("timestamp")),
                )
            case {
                "type": "pointer_up",
                "x": x,
                "y": y,
                "button": button,
                "modifiers": modifiers,
            }:
                if not self._near_last_press(x=x, y=y):
                    self._last_press = None
                self._handle_mouse_event(
                    name="button_release_event",
                    x=x,
                    y=y,
                    button=button,
                    modifiers=modifiers,
                )
            case {
                "type": "pointer_move",
                "x": x,
                "y": y,
                "buttons": buttons,
                "modifiers": modifiers,
            }:
                if buttons and not self._near_last_press(x=x, y=y):
                    self._last_press = None
                self._handle_mouse_event(
                    name="motion_notify_event",
                    x=x,
                    y=y,
                    buttons=buttons,
                    modifiers=modifiers,
                )
            case {
                "type": "wheel",
                "x": x,
                "y": y,
                "dy": dy,
                "modifiers": modifiers,
            } if dy != 0:
                self._handle_mouse_event(
                    name="scroll_event",
                    x=x,
                    y=y,
                    step=1 if dy < 0 else -1,
                    modifiers=modifiers,
                )

    def _canvas_xy(self, x: float, y: float) -> tuple[float, float] | None:
        width, height = self._display_size
        if not width or not height:
            return None
        # Map top-left CSS coordinates into the bottom-left Agg frame.
        frame_width, frame_height = self.canvas.get_width_height(physical=True)
        return frame_width * x / width, frame_height * (1 - y / height)

    def _handle_location_event(
        self, *, name: str, x: float, y: float, modifiers: tuple[str, ...],
    ) -> None:
        xy = self._canvas_xy(x, y)
        if xy is not None:
            LocationEvent(
                name, self.canvas, *xy, modifiers=_mpl_modifiers(modifiers),
            )._process()

    def _handle_mouse_event(
        self,
        *,
        name: str,
        x: float,
        y: float,
        modifiers: tuple[str, ...],
        button: int | None = None,
        buttons: tuple[int, ...] = (),
        dblclick: bool = False,
        step: float = 0,
    ) -> None:
        xy = self._canvas_xy(x, y)
        if xy is None or (button is not None and button not in _BUTTONS):
            return
        MouseEvent(
            name, self.canvas, *xy,
            button=_BUTTONS[button] if button is not None else None,
            buttons=[_BUTTONS[b] for b in buttons if b in _BUTTONS],
            dblclick=dblclick,
            step=step,
            modifiers=_mpl_modifiers(modifiers),
        )._process()


class FigureManager(FigureManagerBase):
    def __init__(self, canvas, num):
        super().__init__(canvas, num)
        self._widget = None

    def show(self):
        if self._widget is None:
            self._widget = _FigureWidget(self.canvas)
            self._widget.display()
        self._widget.request_draw()

    def destroy(self):
        if self._widget is not None:
            self._widget.close()
        super().destroy()


class FigureCanvas(FigureCanvasAgg):
    manager_class = FigureManager
    supports_blit = False
