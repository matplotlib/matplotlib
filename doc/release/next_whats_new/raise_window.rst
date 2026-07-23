Raise a figure window on demand with ``raise_window``
-----------------------------------------------------

The figure manager now exposes a public ``raise_window`` method, letting you
bring a figure window to the front at any point in your program rather than
only as a side effect of `~.pyplot.show` and :rc:`figure.raise_window`::

    fig.canvas.manager.raise_window()

By default, the window is raised in the stacking order *without* stealing
keyboard focus from the currently active application. This is convenient in
interactive workflows (for example from an IPython session), where you want to
glance at a figure without leaving your terminal. Pass ``with_focus=True`` to
additionally activate the window and give it keyboard focus::

    fig.canvas.manager.raise_window(with_focus=True)

Raising and focusing are now treated as independent operations with a
consistent default across the GUI backends, so code that raises a window
behaves the same way regardless of which backend is in use.
