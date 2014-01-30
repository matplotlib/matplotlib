from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import xrange

from .backends.base_backend_bases import *

from matplotlib._pylab_helpers import Gcf


class ShowBase(object):
    """
    Simple base class to generate a show() callable in backends.

    Subclass must override mainloop() method.
    """
    def __call__(self, block=None):
        """
        Show all figures.  If *block* is not None, then
        it is a boolean that overrides all other factors
        determining whether show blocks by calling mainloop().
        The other factors are:
        it does not block if run inside "ipython --pylab";
        it does not block in interactive mode.
        """
        managers = Gcf.get_all_fig_managers()
        if not managers:
            return

        for manager in managers:
            manager.show()

        if block is not None:
            if block:
                self.mainloop()
                return
            else:
                return

        # Hack: determine at runtime whether we are
        # inside ipython in pylab mode.
        from matplotlib import pyplot
        try:
            ipython_pylab = not pyplot.show._needmain
            # IPython versions >= 0.10 tack the _needmain
            # attribute onto pyplot.show, and always set
            # it to False, when in --pylab mode.
            ipython_pylab = ipython_pylab and get_backend() != 'WebAgg'
            # TODO: The above is a hack to get the WebAgg backend
            # working with `ipython --pylab` until proper integration
            # is implemented.
        except AttributeError:
            ipython_pylab = False

        # Leave the following as a separate step in case we
        # want to control this behavior with an rcParam.
        if ipython_pylab:
            return

        if not is_interactive() or get_backend() == 'WebAgg':
            self.mainloop()

    def mainloop(self):
        pass


def key_press_handler(event, canvas, toolbar=None):
    """
    Implement the default mpl key bindings for the canvas and toolbar
    described at :ref:`key-event-handling`

    *event*
      a :class:`KeyEvent` instance
    *canvas*
      a :class:`FigureCanvasBase` instance
    *toolbar*
      a :class:`NavigationToolbar2` instance

    """
    # these bindings happen whether you are over an axes or not

    if event.key is None:
        return

    # Load key-mappings from your matplotlibrc file.
    fullscreen_keys = rcParams['keymap.fullscreen']
    home_keys = rcParams['keymap.home']
    back_keys = rcParams['keymap.back']
    forward_keys = rcParams['keymap.forward']
    pan_keys = rcParams['keymap.pan']
    zoom_keys = rcParams['keymap.zoom']
    save_keys = rcParams['keymap.save']
    quit_keys = rcParams['keymap.quit']
    grid_keys = rcParams['keymap.grid']
    toggle_yscale_keys = rcParams['keymap.yscale']
    toggle_xscale_keys = rcParams['keymap.xscale']
    all = rcParams['keymap.all_axes']

    # toggle fullscreen mode (default key 'f')
    if event.key in fullscreen_keys:
        canvas.manager.full_screen_toggle()

    # quit the figure (defaut key 'ctrl+w')
    if event.key in quit_keys:
        Gcf.destroy_fig(canvas.figure)

    if toolbar is not None:
        # home or reset mnemonic  (default key 'h', 'home' and 'r')
        if event.key in home_keys:
            toolbar.home()
        # forward / backward keys to enable left handed quick navigation
        # (default key for backward: 'left', 'backspace' and 'c')
        elif event.key in back_keys:
            toolbar.back()
        # (default key for forward: 'right' and 'v')
        elif event.key in forward_keys:
            toolbar.forward()
        # pan mnemonic (default key 'p')
        elif event.key in pan_keys:
            toolbar.pan()
        # zoom mnemonic (default key 'o')
        elif event.key in zoom_keys:
            toolbar.zoom()
        # saving current figure (default key 's')
        elif event.key in save_keys:
            toolbar.save_figure()

    if event.inaxes is None:
        return

    # these bindings require the mouse to be over an axes to trigger

    # switching on/off a grid in current axes (default key 'g')
    if event.key in grid_keys:
        event.inaxes.grid()
        canvas.draw()
    # toggle scaling of y-axes between 'log and 'linear' (default key 'l')
    elif event.key in toggle_yscale_keys:
        ax = event.inaxes
        scale = ax.get_yscale()
        if scale == 'log':
            ax.set_yscale('linear')
            ax.figure.canvas.draw()
        elif scale == 'linear':
            ax.set_yscale('log')
            ax.figure.canvas.draw()
    # toggle scaling of x-axes between 'log and 'linear' (default key 'k')
    elif event.key in toggle_xscale_keys:
        ax = event.inaxes
        scalex = ax.get_xscale()
        if scalex == 'log':
            ax.set_xscale('linear')
            ax.figure.canvas.draw()
        elif scalex == 'linear':
            ax.set_xscale('log')
            ax.figure.canvas.draw()

    elif (event.key.isdigit() and event.key != '0') or event.key in all:
        # keys in list 'all' enables all axes (default key 'a'),
        # otherwise if key is a number only enable this particular axes
        # if it was the axes, where the event was raised
        if not (event.key in all):
            n = int(event.key) - 1
        for i, a in enumerate(canvas.figure.get_axes()):
            # consider axes, in which the event was raised
            # FIXME: Why only this axes?
            if event.x is not None and event.y is not None \
                    and a.in_axes(event):
                if event.key in all:
                    a.set_navigate(True)
                else:
                    a.set_navigate(i == n)


# 'register' the keypress handler
FigureManagerBase._key_press_handler = staticmethod(key_press_handler)
