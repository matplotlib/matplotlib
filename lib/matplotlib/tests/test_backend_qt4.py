import copy
from unittest import mock

import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf

import pytest


@pytest.fixture(autouse=True)
def mpl_test_settings(qt4_module, mpl_test_settings):
    """
    Ensure qt4_module fixture is *first* fixture.

    We override the `mpl_test_settings` fixture and depend on the `qt4_module`
    fixture first. It is very important that it is first, because it skips
    tests when Qt4 is not available, and if not, then the main
    `mpl_test_settings` fixture will try to switch backends before the skip can
    be triggered.
    """
    pass


@pytest.fixture
def qt4_module():
    try:
        import PyQt4
    # RuntimeError if PyQt5 already imported.
    except (ImportError, RuntimeError):
        try:
            import PySide
        except ImportError:
            pytest.skip("Failed to import a Qt4 binding.")

    qt_compat = pytest.importorskip('matplotlib.backends.qt_compat')
    QtCore = qt_compat.QtCore

    try:
        py_qt_ver = int(QtCore.PYQT_VERSION_STR.split('.')[0])
    except AttributeError:
        py_qt_ver = QtCore.__version_info__[0]

    if py_qt_ver != 4:
        pytest.skip(reason='Qt4 is not available')

    from matplotlib.backends.backend_qt4 import (
        MODIFIER_KEYS, SUPER, ALT, CTRL, SHIFT)  # noqa

    mods = {}
    keys = {}
    for name, index in zip(['Alt', 'Control', 'Shift', 'Super'],
                           [ALT, CTRL, SHIFT, SUPER]):
        _, mod, key = MODIFIER_KEYS[index]
        mods[name + 'Modifier'] = mod
        keys[name + 'Key'] = key

    return QtCore, mods, keys


@pytest.fixture
def qt_key(request):
    QtCore, _, keys = request.getfixturevalue('qt4_module')
    if request.param.startswith('Key'):
        return getattr(QtCore.Qt, request.param)
    else:
        return keys[request.param]


@pytest.fixture
def qt_mods(request):
    QtCore, mods, _ = request.getfixturevalue('qt4_module')
    result = QtCore.Qt.NoModifier
    for mod in request.param:
        result |= mods[mod]
    return result


@pytest.mark.backend('Qt4Agg')
def test_fig_close():
    # save the state of Gcf.figs
    init_figs = copy.copy(Gcf.figs)

    # make a figure using pyplot interface
    fig = plt.figure()

    # simulate user clicking the close button by reaching in
    # and calling close on the underlying Qt object
    fig.canvas.manager.window.close()

    # assert that we have removed the reference to the FigureManager
    # that got added by plt.figure()
    assert init_figs == Gcf.figs


@pytest.mark.parametrize(
    'qt_key, qt_mods, answer',
    [
        ('Key_A', ['ShiftModifier'], 'A'),
        ('Key_A', [], 'a'),
        ('Key_A', ['ControlModifier'], 'ctrl+a'),
        ('Key_Aacute', ['ShiftModifier'],
         '\N{LATIN CAPITAL LETTER A WITH ACUTE}'),
        ('Key_Aacute', [],
         '\N{LATIN SMALL LETTER A WITH ACUTE}'),
        ('ControlKey', ['AltModifier'], 'alt+control'),
        ('AltKey', ['ControlModifier'], 'ctrl+alt'),
        ('Key_Aacute', ['ControlModifier', 'AltModifier', 'SuperModifier'],
         'ctrl+alt+super+\N{LATIN SMALL LETTER A WITH ACUTE}'),
        ('Key_Backspace', [], 'backspace'),
        ('Key_Backspace', ['ControlModifier'], 'ctrl+backspace'),
        ('Key_Play', [], None),
    ],
    indirect=['qt_key', 'qt_mods'],
    ids=[
        'shift',
        'lower',
        'control',
        'unicode_upper',
        'unicode_lower',
        'alt_control',
        'control_alt',
        'modifier_order',
        'backspace',
        'backspace_mod',
        'non_unicode_key',
    ]
)
@pytest.mark.backend('Qt4Agg')
def test_correct_key(qt_key, qt_mods, answer):
    """
    Make a figure
    Send a key_press_event event (using non-public, qt4 backend specific api)
    Catch the event
    Assert sent and caught keys are the same
    """
    qt_canvas = plt.figure().canvas

    event = mock.Mock()
    event.isAutoRepeat.return_value = False
    event.key.return_value = qt_key
    event.modifiers.return_value = qt_mods

    def receive(event):
        assert event.key == answer

    qt_canvas.mpl_connect('key_press_event', receive)
    qt_canvas.keyPressEvent(event)
