"""
=========================================================
Select four control points to define a cubic Bézier curve
=========================================================

Shows how one can use `.widgets.PolylineSelector` to create an interactive
cubic Bézier curve visualizer. Select four control points to define the curve.
"""

import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.widgets import PolylineSelector

EMPTY_PATH = Path([(0, 0)], [Path.MOVETO])


class CubicBezierCurveVisualizer:
    """
    Interactive cubic Bézier curve visualizer.

    This tool allows you to define a cubic Bézier curve by selecting
    four control points using `PolylineSelector`. The resulting curve
    is displayed using `PathPatch` and updates as the selection changes,
    provided that a valid set of control points is selected. Press the
    'enter' key to complete the curve after selecting four control points.
    Pressing 'esc' clears the current curve so a new one can be created.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    props : dict, optional
        Properties with which the polyline selector lines will be drawn.
    handle_props : dict, optional
        Properties for the control point handles.
    """
    def __init__(self, ax, props=None, handle_props=None):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.curve = patches.PathPatch(EMPTY_PATH, facecolor='none', lw=2)
        self.ax.add_patch(self.curve)

        if props is None:
            props = dict(ls='--', c='gray')
        if handle_props is None:
            handle_props = dict(marker='o', ms=8, mfc='gray', mec='black')

        self.selector = PolylineSelector(ax, self.onselect,
                                         props=props, handle_props=handle_props)

        self.cid = self.canvas.mpl_connect('key_press_event', self.on_key_press)

    def onselect(self, verts):
        if len(verts) != 4:
            print("Select exactly 4 points for a cubic Bézier curve.")
            self.clear()
            return
        path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        self.curve.set_path(path)
        self.canvas.draw_idle()

    def clear(self):
        self.curve.set_path(EMPTY_PATH)
        self.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == 'escape':
            self.clear()

    def disconnect(self):
        self.selector.disconnect_events()
        self.canvas.mpl_disconnect(self.cid)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.5)

    bezier_visualizer = CubicBezierCurveVisualizer(ax)

    print("Select exactly 4 control points to define a cubic Bézier curve.")
    print("Press the 'enter' key to complete the selection.")
    print("Hold 'ctrl' to reposition a single point while polyline is incomplete.")
    print("Hold 'shift' to move all control points.")
    print("Left click and drag a point to reposition it.")
    print("Press the 'esc' key to start a new curve.")

    plt.show()

    bezier_visualizer.disconnect()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.widgets.PolylineSelector`
#    - `matplotlib.path.Path`
#    - `matplotlib.patches.PathPatch`
