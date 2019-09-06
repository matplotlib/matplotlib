import copy
import sys
from unittest import mock

import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf

import pytest


@pytest.fixture(autouse=True)
def mpl_test_settings(qt_module, mpl_test_settings):
    """
    Ensure qt_module fixture is *first* fixture.

    We override the `mpl_test_settings` fixture and depend on the `qt_module`
    fixture first. It is very important that it is first, because it skips
    tests when Qt is not available, and if not, then the main
    `mpl_test_settings` fixture will try to switch backends before the skip can
    be triggered.
    """
    pass


@pytest.fixture
def qt_module(request):
    backend, = request.node.get_closest_marker('backend').args
    if backend == 'Qt4Agg':
        if any(k in sys.modules for k in ('PyQt5', 'PySide2')):
            pytest.skip('Qt5 binding already imported')
        try:
            import PyQt4
        # RuntimeError if PyQt5 already imported.
        except (ImportError, RuntimeError):
            try:
                import PySide
            except ImportError:
                pytest.skip("Failed to import a Qt4 binding.")
    elif backend == 'Qt5Agg':
        if any(k in sys.modules for k in ('PyQt4', 'PySide')):
            pytest.skip('Qt4 binding already imported')
        try:
            import PyQt5
        # RuntimeError if PyQt4 already imported.
        except (ImportError, RuntimeError):
            try:
                import PySide2
            except ImportError:
                pytest.skip("Failed to import a Qt5 binding.")
    else:
        raise ValueError('Backend marker has unknown value: ' + backend)

    qt_compat = pytest.importorskip('matplotlib.backends.qt_compat')
    QtCore = qt_compat.QtCore

    if backend == 'Qt4Agg':
        try:
            py_qt_ver = int(QtCore.PYQT_VERSION_STR.split('.')[0])
        except AttributeError:
            py_qt_ver = QtCore.__version_info__[0]

        if py_qt_ver != 4:
            pytest.skip('Qt4 is not available')

        from matplotlib.backends.backend_qt4 import (
            MODIFIER_KEYS, SUPER, ALT, CTRL, SHIFT)
    elif backend == 'Qt5Agg':
        from matplotlib.backends.backend_qt5 import (
            MODIFIER_KEYS, SUPER, ALT, CTRL, SHIFT)

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
    QtCore, _, keys = request.getfixturevalue('qt_module')
    if request.param.startswith('Key'):
        return getattr(QtCore.Qt, request.param)
    else:
        return keys[request.param]


@pytest.fixture
def qt_mods(request):
    QtCore, mods, _ = request.getfixturevalue('qt_module')
    result = QtCore.Qt.NoModifier
    for mod in request.param:
        result |= mods[mod]
    return result


@pytest.mark.parametrize('backend', [
    # Note: the value is irrelevant; the important part is the marker.
    pytest.param('Qt4Agg', marks=pytest.mark.backend('Qt4Agg')),
    pytest.param('Qt5Agg', marks=pytest.mark.backend('Qt5Agg')),
])
def test_fig_close(backend):
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


@pytest.mark.backend('Qt5Agg')
def test_fig_signals(qt_module):
    # Create a figure
    fig = plt.figure()

    # Access QtCore
    QtCore = qt_module[0]

    # Access signals
    import signal
    event_loop_signal = None

    # Callback to fire during event loop: save SIGINT handler, then exit
    def fire_signal_and_quit():
        # Save event loop signal
        nonlocal event_loop_signal
        event_loop_signal = signal.getsignal(signal.SIGINT)

        # Request event loop exit
        QtCore.QCoreApplication.exit()

    # Timer to exit event loop
    QtCore.QTimer.singleShot(0, fire_signal_and_quit)

    # Save original SIGINT handler
    original_signal = signal.getsignal(signal.SIGINT)

    # Use our own SIGINT handler to be 100% sure this is working
    def CustomHandler(signum, frame):
        pass

    signal.signal(signal.SIGINT, CustomHandler)

    # mainloop() sets SIGINT, starts Qt event loop (which triggers timer and
    # exits) and then mainloop() resets SIGINT
    matplotlib.backends.backend_qt5._BackendQT5.mainloop()

    # Assert: signal handler during loop execution is signal.SIG_DFL
    assert event_loop_signal == signal.SIG_DFL

    # Assert: current signal handler is the same as the one we set before
    assert CustomHandler == signal.getsignal(signal.SIGINT)

    # Reset SIGINT handler to what it was before the test
    signal.signal(signal.SIGINT, original_signal)


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
@pytest.mark.parametrize('backend', [
    # Note: the value is irrelevant; the important part is the marker.
    pytest.param('Qt4Agg', marks=pytest.mark.backend('Qt4Agg')),
    pytest.param('Qt5Agg', marks=pytest.mark.backend('Qt5Agg')),
])
def test_correct_key(backend, qt_key, qt_mods, answer):
    """
    Make a figure
    Send a key_press_event event (using non-public, qtX backend specific api)
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


@pytest.mark.backend('Qt5Agg')
def test_dpi_ratio_change():
    """
    Make sure that if _dpi_ratio changes, the figure dpi changes but the
    widget remains the same physical size.
    """

    prop = 'matplotlib.backends.backend_qt5.FigureCanvasQT._dpi_ratio'

    with mock.patch(prop, new_callable=mock.PropertyMock) as p:

        p.return_value = 3

        fig = plt.figure(figsize=(5, 2), dpi=120)
        qt_canvas = fig.canvas
        qt_canvas.show()

        from matplotlib.backends.backend_qt5 import qApp

        # Make sure the mocking worked
        assert qt_canvas._dpi_ratio == 3

        size = qt_canvas.size()

        qt_canvas.manager.show()
        qt_canvas.draw()
        qApp.processEvents()

        # The DPI and the renderer width/height change
        assert fig.dpi == 360
        assert qt_canvas.renderer.width == 1800
        assert qt_canvas.renderer.height == 720

        # The actual widget size and figure physical size don't change
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()

        p.return_value = 2

        assert qt_canvas._dpi_ratio == 2

        qt_canvas.draw()
        qApp.processEvents()
        # this second processEvents is required to fully run the draw.
        # On `update` we notice the DPI has changed and trigger a
        # resize event to refresh, the second processEvents is
        # required to process that and fully update the window sizes.
        qApp.processEvents()

        # The DPI and the renderer width/height change
        assert fig.dpi == 240
        assert qt_canvas.renderer.width == 1200
        assert qt_canvas.renderer.height == 480

        # The actual widget size and figure physical size don't change
        assert size.width() == 600
        assert size.height() == 240
        assert qt_canvas.get_width_height() == (600, 240)
        assert (fig.get_size_inches() == (5, 2)).all()


@pytest.mark.backend('Qt5Agg')
def test_subplottool():
    fig, ax = plt.subplots()
    with mock.patch(
            "matplotlib.backends.backend_qt5.SubplotToolQt.exec_",
            lambda self: None):
        fig.canvas.manager.toolbar.configure_subplots()


@pytest.mark.backend('Qt5Agg')
def test_figureoptions():
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    ax.imshow([[1]])
    ax.scatter(range(3), range(3), c=range(3))
    with mock.patch(
            "matplotlib.backends.qt_editor._formlayout.FormDialog.exec_",
            lambda self: None):
        fig.canvas.manager.toolbar.edit_parameters()


@pytest.mark.backend("Qt5Agg")
def test_canvas_reinit():
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from functools import partial

    called = False

    def crashing_callback(fig, stale):
        nonlocal called
        fig.canvas.draw_idle()
        called = True

    fig, ax = plt.subplots()
    fig.stale_callback = crashing_callback
    # this should not raise
    canvas = FigureCanvasQTAgg(fig)
    assert called
