from __future__ import absolute_import, division, print_function, unicode_literals

import six

from matplotlib import pyplot as plt
from matplotlib.testing.decorators import cleanup
from matplotlib.testing.decorators import knownfailureif
from matplotlib._pylab_helpers import Gcf
import copy

import mock

try:
    from matplotlib.backends.qt4_compat import QtCore
    from matplotlib.backends.backend_qt import MODIFIER_KEYS
    HAS_QT = True

    ShiftModifier = QtCore.Qt.ShiftModifier
    ControlModifier, ControlKey = MODIFIER_KEYS['ctrl']
    AltModifier, AltKey = MODIFIER_KEYS['alt']
    SuperModifier, SuperKey = MODIFIER_KEYS['super']
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


def assert_correct_key(qt_key, qt_mods, qt_text, answer):
    plt.switch_backend('Qt4Agg')
    qt_canvas = plt.figure().canvas

    event = mock.Mock()
    event.isAutoRepeat.return_value = False
    event.key.return_value = qt_key
    event.modifiers.return_value = qt_mods
    event.text.return_value = qt_text

    def receive(event):
        assert event.key == answer

    qt_canvas.mpl_connect('key_press_event', receive)
    qt_canvas.keyPressEvent(event)


@cleanup
@knownfailureif(not HAS_QT)
def test_shift():
    assert_correct_key(QtCore.Qt.Key_A,
                       ShiftModifier,
                       u'A',
                       u'A')


@cleanup
@knownfailureif(not HAS_QT)
def test_noshift():
    assert_correct_key(QtCore.Qt.Key_A,
                       QtCore.Qt.NoModifier,
                       u'a',
                       u'a')


@cleanup
@knownfailureif(not HAS_QT)
def test_control():
    assert_correct_key(QtCore.Qt.Key_A,
                       ControlModifier,
                       u'',
                       u'ctrl+a')


@cleanup
@knownfailureif(not HAS_QT)
def test_unicode():
    assert_correct_key(QtCore.Qt.Key_Aacute,
                       ShiftModifier,
                       unichr(193),
                       unichr(193))


@cleanup
@knownfailureif(not HAS_QT)
def test_unicode_noshift():
    assert_correct_key(QtCore.Qt.Key_Aacute,
                       QtCore.Qt.NoModifier,
                       unichr(193),
                       unichr(225))


@cleanup
@knownfailureif(not HAS_QT)
def test_alt_control():
    assert_correct_key(QtCore.Qt.Key_Control,
                       AltModifier,
                       u'',
                       u'alt+control')


@cleanup
@knownfailureif(not HAS_QT)
def test_control_alt():
    assert_correct_key(QtCore.Qt.Key_Alt,
                       ControlModifier,
                       u'',
                       u'ctrl+alt')


@cleanup
@knownfailureif(not HAS_QT)
def test_modifier_order():
    assert_correct_key(QtCore.Qt.Key_Aacute,
                       (ControlModifier & AltModifier & SuperModifier),
                       u'',
                       u'ctrl+alt+super+' + unichr(225))
