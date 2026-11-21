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

from matplotlib.backend_bases import FigureManagerBase, MouseButton, MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg


_BUTTONS = {
    1: MouseButton.LEFT,
    2: MouseButton.RIGHT,
    3: MouseButton.MIDDLE,
    4: MouseButton.BACK,
    5: MouseButton.FORWARD,
}


class _FigureWidget(RemoteFrameBuffer):
    def __init__(self, canvas):
        self.canvas = canvas
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

    def handle_event(self, event):
        match event:
            case {"type": "resize", "width": width, "height": height}:
                self._display_size = width, height
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
                )
            case {
                "type": "pointer_up",
                "x": x,
                "y": y,
                "button": button,
                "modifiers": modifiers,
            }:
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
                self._handle_mouse_event(
                    name="motion_notify_event",
                    x=x,
                    y=y,
                    buttons=buttons,
                    modifiers=modifiers,
                )

    def _handle_mouse_event(
        self,
        *,
        name: str,
        x: float,
        y: float,
        modifiers: tuple[str, ...],
        button: int | None = None,
        buttons: tuple[int, ...] = (),
    ) -> None:
        width, height = self._display_size
        if not width or not height or (button is not None and button not in _BUTTONS):
            return
        # Map top-left CSS coordinates into the bottom-left Agg frame.
        frame_width, frame_height = self.canvas.get_width_height(physical=True)
        x = frame_width * x / width
        y = frame_height * (1 - y / height)
        mpl_modifiers = [
            "ctrl" if mod == "Control" else mod.lower() for mod in modifiers
        ]
        MouseEvent(
            name, self.canvas, x, y,
            button=_BUTTONS[button] if button is not None else None,
            buttons=[_BUTTONS[b] for b in buttons if b in _BUTTONS],
            modifiers=mpl_modifiers,
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
