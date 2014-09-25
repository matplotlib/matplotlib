from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six import unichr
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import cleanup
from matplotlib.testing.decorators import knownfailureif
from matplotlib._pylab_helpers import Gcf
import copy

try:
    # mock in python 3.3+
    from unittest import mock
except ImportError:
    import mock

try:
    from matplotlib.backends.qt_compat import QtCore
    from matplotlib.backends.backend_qt4 import (MODIFIER_KEYS,
                                                 SUPER, ALT, CTRL, SHIFT)

    _, ControlModifier, ControlKey = MODIFIER_KEYS[CTRL]
    _, AltModifier, AltKey = MODIFIER_KEYS[ALT]
    _, SuperModifier, SuperKey = MODIFIER_KEYS[SUPER]
    _, ShiftModifier, ShiftKey = MODIFIER_KEYS[SHIFT]
    HAS_QT = True
except ImportError:
    HAS_QT = False


@cleanup
@knownfailureif(not HAS_QT)
def test_fig_close():
    # force switch to the Qt4 backend
    plt.switch_backend('Qt4Agg')

    #save the state of Gcf.figs
    init_figs = copy.copy(Gcf.figs)

    # make a figure using pyplot interface
    fig = plt.figure()

    # simulate user clicking the close button by reaching in
    # and calling close on the underlying Qt object
    fig.canvas.manager.window.close()

    # assert that we have removed the reference to the FigureManager
    # that got added by plt.figure()
    assert(init_figs == Gcf.figs)


def assert_correct_key(qt_key, qt_mods, answer):
    """
    Make a figure
    Send a key_press_event event (using non-public, qt4 backend specific api)
    Catch the event
    Assert sent and caught keys are the same
    """
    plt.switch_backend('Qt4Agg')
    qt_canvas = plt.figure().canvas

    event = mock.Mock()
    event.isAutoRepeat.return_value = False
    event.key.return_value = qt_key
    event.modifiers.return_value = qt_mods

    def receive(event):
        assert event.key == answer

    qt_canvas.mpl_connect('key_press_event', receive)
    qt_canvas.keyPressEvent(event)


@cleanup
@knownfailureif(not HAS_QT)
def test_shift():
    assert_correct_key(QtCore.Qt.Key_A,
                       ShiftModifier,
                       'A')


@cleanup
@knownfailureif(not HAS_QT)
def test_lower():
    assert_correct_key(QtCore.Qt.Key_A,
                       QtCore.Qt.NoModifier,
                       'a')


@cleanup
@knownfailureif(not HAS_QT)
def test_control():
    assert_correct_key(QtCore.Qt.Key_A,
                       ControlModifier,
                       'ctrl+a')


@cleanup
@knownfailureif(not HAS_QT)
def test_unicode_upper():
    assert_correct_key(QtCore.Qt.Key_Aacute,
                       ShiftModifier,
                       unichr(193))


@cleanup
@knownfailureif(not HAS_QT)
def test_unicode_lower():
    assert_correct_key(QtCore.Qt.Key_Aacute,
                       QtCore.Qt.NoModifier,
                       unichr(225))


@cleanup
@knownfailureif(not HAS_QT)
def test_alt_control():
    assert_correct_key(ControlKey,
                       AltModifier,
                       'alt+control')


@cleanup
@knownfailureif(not HAS_QT)
def test_control_alt():
    assert_correct_key(AltKey,
                       ControlModifier,
                       'ctrl+alt')


@cleanup
@knownfailureif(not HAS_QT)
def test_modifier_order():
    assert_correct_key(QtCore.Qt.Key_Aacute,
                       (ControlModifier | AltModifier | SuperModifier),
                       'ctrl+alt+super+' + unichr(225))


@cleanup
@knownfailureif(not HAS_QT)
def test_backspace():
    assert_correct_key(QtCore.Qt.Key_Backspace,
                       QtCore.Qt.NoModifier,
                       'backspace')


@cleanup
@knownfailureif(not HAS_QT)
def test_backspace_mod():
    assert_correct_key(QtCore.Qt.Key_Backspace,
                       ControlModifier,
                       'ctrl+backspace')


@cleanup
@knownfailureif(not HAS_QT)
def test_non_unicode_key():
    assert_correct_key(QtCore.Qt.Key_Play,
                       QtCore.Qt.NoModifier,
                       None)
