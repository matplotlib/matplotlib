from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
from matplotlib.externals.six import unichr
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import cleanup, switch_backend
from matplotlib.testing.decorators import knownfailureif
from matplotlib._pylab_helpers import Gcf
import copy

try:
    # mock in python 3.3+
    from unittest import mock
except ImportError:
    import mock

try:
    from matplotlib.backends.gtk3_compat import Gtk, Gdk, GObject, GLib
    HAS_GTK3 = True
except ImportError:
    HAS_GTK3 = False


def simulate_key_press(canvas, key, modifiers=[]):
    event = mock.Mock()

    keyval = [k for k, v in six.iteritems(canvas.keyvald) if v == key]
    if keyval:
        keyval = keyval[0]
    else:
        keyval = ord(key)
    event.keyval = keyval

    event.state = 0
    for key_mask, prefix in canvas.modifier_keys:
        if prefix in modifiers:
            event.state |= key_mask

    canvas.key_press_event(None, event)


@cleanup
#@knownfailureif(not HAS_GTK3)
@switch_backend('GTK3Agg')
def test_fig_close():
    #save the state of Gcf.figs
    init_figs = copy.copy(Gcf.figs)

    # make a figure using pyplot interface
    fig = plt.figure()

    # simulate user pressing the close shortcut
    #simulate_key_press(fig.canvas, 'w', ['ctrl'])

    plt.show()

    # assert that we have removed the reference to the FigureManager
    # that got added by plt.figure()
    assert(init_figs == Gcf.figs)
