"""
Prototype project for new Matplotlib GUI management.

The pyplot module current serves two critical, but unrelated functions:

1. provide a state-full implicit API that rhymes / was inspired by MATLAB
2. provide the management of interaction between Matplotlib and the GUI event
   loop

This project is prototype for separating the second function from the first.
This will enable users to both only use the explicit API (nee OO interface) and
to have smooth integration with the GUI event loop as with pyplot.

"""
from collections import Counter
from itertools import count
import functools
import logging
import warnings

from matplotlib.backend_bases import FigureCanvasBase as _FigureCanvasBase

from ._figure import Figure  # noqa: F401

from ._manage_interactive import ion, ioff, is_interactive  # noqa: F401
from ._manage_backend import select_gui_toolkit  # noqa: F401
from ._manage_backend import current_backend_module as _cbm
from ._promotion import promote_figure as promote_figure
from ._creation import figure, subplots, subplot_mosaic  # noqa: F401

_log = logging.getLogger(__name__)


def show(figs, *, block=None, timeout=0):
    """
    Show the figures and maybe block.

    Parameters
    ----------
    figs : List[Figure]
        The figures to show.  If they do not currently have a GUI aware
        canvas + manager attached they will be promoted.

    block : bool, optional
        Whether to wait for all figures to be closed before returning.

        If `True` block and run the GUI main loop until all figure windows
        are closed.

        If `False` ensure that all figure windows are displayed and return
        immediately.  In this case, you are responsible for ensuring
        that the event loop is running to have responsive figures.

        Defaults to True in non-interactive mode and to False in interactive
        mode (see `.is_interactive`).

    """
    # TODO handle single figure

    # call this to ensure a backend is indeed selected
    backend = _cbm()
    managers = []
    for fig in figs:
        if fig.canvas.manager is not None:
            managers.append(fig.canvas.manager)
        else:
            managers.append(promote_figure(fig))

    if block is None:
        block = not is_interactive()

    if block and len(managers):
        if timeout == 0:
            backend.show_managers(managers=managers, block=block)
        elif len(managers):
            manager, *_ = managers
            manager.canvas.start_event_loop(timeout=timeout)


class FigureRegistry:
    """
    A registry to wrap the creation of figures and track them.

    This instance will keep a hard reference to created Figures to ensure
    that they do not get garbage collected.

    Parameters
    ----------
    block : bool, optional
        Whether to wait for all figures to be closed before returning from
        show_all.

        If `True` block and run the GUI main loop until all figure windows
        are closed.

        If `False` ensure that all figure windows are displayed and return
        immediately.  In this case, you are responsible for ensuring
        that the event loop is running to have responsive figures.

        Defaults to True in non-interactive mode and to False in interactive
        mode (see `.is_interactive`).

    timeout : float, optional
        Default time to wait for all of the Figures to be closed if blocking.

        If 0 block forever.

    """

    def __init__(self, *, block=None, timeout=0, prefix="Figure "):
        # settings stashed to set defaults on show
        self._timeout = timeout
        self._block = block
        # Settings / state to control the default figure label
        self._count = count()
        self._prefix = prefix
        # the canonical location for storing the Figures this registry owns.
        # any additional views must never include a figure not in the list but
        # may omit figures
        self.figures = []

    def _register_fig(self, fig):
        # if the user closes the figure by any other mechanism, drop our
        # reference to it.  This is important for getting a "pyplot" like user
        # experience
        fig.canvas.mpl_connect(
            "close_event",
            lambda e: self.figures.remove(fig) if fig in self.figures else None,
        )
        # hold a hard reference to the figure.
        self.figures.append(fig)
        # Make sure we give the figure a quasi-unique label.  We will never set
        # the same label twice, but will not over-ride any user label (but
        # empty string) on a Figure so if they provide duplicate labels, change
        # the labels under us, or provide a label that will be shadowed in the
        # future it will be what it is.
        fignum = next(self._count)
        if fig.get_label() == "":
            fig.set_label(f"{self._prefix}{fignum:d}")
        # TODO: is there a better way to track this than monkey patching?
        fig._mpl_gui_fignum = fignum
        return fig

    @property
    def by_label(self):
        """
        Return a dictionary of the current mapping labels -> figures.

        If there are duplicate labels, newer figures will take precedence.
        """
        mapping = {fig.get_label(): fig for fig in self.figures}
        if len(mapping) != len(self.figures):
            counts = Counter(fig.get_label() for fig in self.figures)
            multiples = {k: v for k, v in counts.items() if v > 1}
            warnings.warn(
                (
                    f"There are repeated labels ({multiples!r}), but only the newest"
                    "figure with that label can be returned. "
                ),
                stacklevel=2,
            )
        return mapping

    @property
    def by_number(self):
        """
        Return a dictionary of the current mapping number -> figures.

        """
        self._ensure_all_figures_promoted()
        return {fig.canvas.manager.num: fig for fig in self.figures}

    @functools.wraps(figure)
    def figure(self, *args, **kwargs):
        fig = figure(*args, **kwargs)
        return self._register_fig(fig)

    @functools.wraps(subplots)
    def subplots(self, *args, **kwargs):
        fig, axs = subplots(*args, **kwargs)
        return self._register_fig(fig), axs

    @functools.wraps(subplot_mosaic)
    def subplot_mosaic(self, *args, **kwargs):
        fig, axd = subplot_mosaic(*args, **kwargs)
        return self._register_fig(fig), axd

    def _ensure_all_figures_promoted(self):
        for f in self.figures:
            if f.canvas.manager is None:
                promote_figure(f, num=f._mpl_gui_fignum)

    def show_all(self, *, block=None, timeout=None):
        """
        Show all of the Figures that the FigureRegistry knows about.

        Parameters
        ----------
        block : bool, optional
            Whether to wait for all figures to be closed before returning from
            show_all.

            If `True` block and run the GUI main loop until all figure windows
            are closed.

            If `False` ensure that all figure windows are displayed and return
            immediately.  In this case, you are responsible for ensuring
            that the event loop is running to have responsive figures.

            Defaults to the value set on the Registry at init

        timeout : float, optional
            time to wait for all of the Figures to be closed if blocking.

            If 0 block forever.

            Defaults to the timeout set on the Registry at init
        """
        if block is None:
            block = self._block

        if timeout is None:
            timeout = self._timeout
        self._ensure_all_figures_promoted()
        show(self.figures, block=self._block, timeout=self._timeout)

    # alias to easy pyplot compatibility
    show = show_all

    def close_all(self):
        """
        Close all Figures know to this Registry.

        This will do four things:

        1. call the ``.destroy()`` method on the manager
        2. clears the Figure on the canvas instance
        3. replace the canvas on each Figure with a new `~matplotlib.backend_bases.
           FigureCanvasBase` instance
        4. drops its hard reference to the Figure

        If the user still holds a reference to the Figure it can be revived by
        passing it to `show`.

        """
        for fig in list(self.figures):
            self.close(fig)

    def close(self, val):
        """
        Close (meaning destroy the UI) and forget a managed Figure.

        This will do two things:

        - start the destruction process of an UI (the event loop may need to
          run to complete this process and if the user is holding hard
          references to any of the UI elements they may remain alive).
        - Remove the `Figure` from this Registry.

        We will no longer have any hard references to the Figure, but if
        the user does the `Figure` (and its components) will not be garbage
        collected.  Due to the circular references in Matplotlib these
        objects may not be collected until the full cyclic garbage collection
        runs.

        If the user still has a reference to the `Figure` they can re-show the
        figure via `show`, but the `FigureRegistry` will not be aware of it.

        Parameters
        ----------
        val : 'all' or int or str or Figure

            - The special case of 'all' closes all open Figures
            - If any other string is passed, it is interpreted as a key in
              `by_label` and that Figure is closed
            - If an integer it is interpreted as a key in `by_number` and that
              Figure is closed
            - If it is a `Figure` instance, then that figure is closed

        """
        if val == "all":
            return self.close_all()
        # or do we want to close _all_ of the figures with a given label / number?
        if isinstance(val, str):
            fig = self.by_label[val]
        elif isinstance(val, int):
            fig = self.by_number[val]
        else:
            fig = val
            if fig not in self.figures:
                raise ValueError(
                    "Trying to close a figure not associated with this Registry."
                )
        if fig.canvas.manager is not None:
            fig.canvas.manager.destroy()
            # disconnect figure from canvas
            fig.canvas.figure = None
            # disconnect canvas from figure
            _FigureCanvasBase(figure=fig)
        assert fig.canvas.manager is None
        if fig in self.figures:
            self.figures.remove(fig)


class FigureContext(FigureRegistry):
    """
    Extends FigureRegistry to be used as a context manager.

    All figures known to the Registry will be shown on exiting the context.

    Parameters
    ----------
    block : bool, optional
        Whether to wait for all figures to be closed before returning from
        show_all.

        If `True` block and run the GUI main loop until all figure windows
        are closed.

        If `False` ensure that all figure windows are displayed and return
        immediately.  In this case, you are responsible for ensuring
        that the event loop is running to have responsive figures.

        Defaults to True in non-interactive mode and to False in interactive
        mode (see `.is_interactive`).

    timeout : float, optional
        Default time to wait for all of the Figures to be closed if blocking.

        If 0 block forever.

    forgive_failure : bool, optional
        If True, block to show the figure before letting the exception
        propagate

    """

    def __init__(self, *, forgive_failure=False, **kwargs):
        super().__init__(**kwargs)
        self._forgive_failure = forgive_failure

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None and not self._forgive_failure:
            return
        show(self.figures, block=self._block, timeout=self._timeout)


# from mpl_gui import * # is a language mis-feature
__all__ = []
