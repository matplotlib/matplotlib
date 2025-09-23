import copy
import importlib
import os
import signal
import sys

from datetime import date, datetime
from unittest import mock

import pytest

import matplotlib
from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib import _c_internal_utils

try:
    from matplotlib.backends.qt_compat import QtCore  # type: ignore[attr-defined]
    from matplotlib.backends.qt_compat import QtGui  # type: ignore[attr-defined]  # noqa: E501, F401
    from matplotlib.backends.qt_compat import QtWidgets  # type: ignore[attr-defined]
    from matplotlib.backends.qt_editor import _formlayout
except ImportError:
    pytestmark = pytest.mark.skip('No usable Qt bindings')


_test_timeout = 60  # A reasonably safe value for slower architectures.


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
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
    "qt_key, qt_mods, answer",
    [
        ("Key_A", ["ShiftModifier"], "A"),
        ("Key_A", [], "a"),
        ("Key_A", ["ControlModifier"], ("ctrl+a")),
        (
            "Key_Aacute",
            ["ShiftModifier"],
            "\N{LATIN CAPITAL LETTER A WITH ACUTE}",
        ),
        ("Key_Aacute", [], "\N{LATIN SMALL LETTER A WITH ACUTE}"),
        ("Key_Control", ["AltModifier"], ("alt+control")),
        ("Key_Alt", ["ControlModifier"], "ctrl+alt"),
        (
            "Key_Aacute",
            ["ControlModifier", "AltModifier", "MetaModifier"],
            ("ctrl+alt+meta+\N{LATIN SMALL LETTER A WITH ACUTE}"),
        ),
        # We do not currently map the media keys, this may change in the
        # future.  This means the callback will never fire
        ("Key_Play", [], None),
        ("Key_Backspace", [], "backspace"),
        (
            "Key_Backspace",
            ["ControlModifier"],
            "ctrl+backspace",
        ),
    ],
    ids=[
        'shift',
        'lower',
        'control',
        'unicode_upper',
        'unicode_lower',
        'alt_control',
        'control_alt',
        'modifier_order',
        'non_unicode_key',
        'backspace',
        'backspace_mod',
    ]
)
@pytest.mark.parametrize('backend', [
    # Note: the value is irrelevant; the important part is the marker.
    pytest.param(
        'Qt5Agg',
        marks=pytest.mark.backend('Qt5Agg', skip_on_importerror=True)),
    pytest.param(
        'QtAgg',
        marks=pytest.mark.backend('QtAgg', skip_on_importerror=True)),
])
def test_correct_key(backend, qt_key, qt_mods, answer, monkeypatch):
    """
    Make a figure.
    Send a key_press_event event (using non-public, qtX backend specific api).
    Catch the event.
    Assert sent and caught keys are the same.
    """
    from matplotlib.backends.qt_compat import _to_int, QtCore

    if sys.platform == "darwin" and answer is not None:
        answer = answer.replace("ctrl", "cmd")
        answer = answer.replace("control", "cmd")
        answer = answer.replace("meta", "ctrl")
    result = None
    qt_mod = QtCore.Qt.KeyboardModifier.NoModifier
    for mod in qt_mods:
        qt_mod |= getattr(QtCore.Qt.KeyboardModifier, mod)

    class _Event:
        def isAutoRepeat(self): return False
        def key(self): return _to_int(getattr(QtCore.Qt.Key, qt_key))

    monkeypatch.setattr(QtWidgets.QApplication, "keyboardModifiers",
                        lambda self: qt_mod)

    def on_key_press(event):
        nonlocal result
        result = event.key

    qt_canvas = plt.figure().canvas
    qt_canvas.mpl_connect('key_press_event', on_key_press)
    qt_canvas.keyPressEvent(_Event())
    assert result == answer


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_device_pixel_ratio_change():
    """
    Make sure that if the pixel ratio changes, the figure dpi changes but the
    widget remains the same logical size.
    """

    prop = 'matplotlib.backends.backend_qt.FigureCanvasQT.devicePixelRatioF'
    with mock.patch(prop) as p:
        p.return_value = 3

        fig = plt.figure(figsize=(5, 2), dpi=120)
        qt_canvas = fig.canvas
        qt_canvas.show()

        def set_device_pixel_ratio(ratio):
            p.return_value = ratio

            window = qt_canvas.window().windowHandle()
            current_version = tuple(int(x) for x in QtCore.qVersion().split('.', 2)[:2])
            if current_version >= (6, 6):
                QtCore.QCoreApplication.sendEvent(
                    window,
                    QtCore.QEvent(QtCore.QEvent.Type.DevicePixelRatioChange))
            else:
                # The value here doesn't matter, as we can't mock the C++ QScreen
                # object, but can override the functional wrapper around it.
                # Emitting this event is simply to trigger the DPI change handler
                # in Matplotlib in the same manner that it would occur normally.
                window.screen().logicalDotsPerInchChanged.emit(96)

            qt_canvas.draw()
            qt_canvas.flush_events()

            # Make sure the mocking worked
            assert qt_canvas.device_pixel_ratio == ratio

        qt_canvas.manager.show()
        qt_canvas.draw()
        qt_canvas.flush_events()
        size = qt_canvas.size()

        options = [
            (None, 360, 1800, 720),  # Use ratio at startup time.
            (3, 360, 1800, 720),  # Change to same ratio.
            (2, 240, 1200, 480),  # Change to different ratio.
            (1.5, 180, 900, 360),  # Fractional ratio.
        ]
        for ratio, dpi, width, height in options:
            if ratio is not None:
                set_device_pixel_ratio(ratio)

            # The DPI and the renderer width/height change
            assert fig.dpi == dpi
            assert qt_canvas.renderer.width == width
            assert qt_canvas.renderer.height == height

            # The actual widget size and figure logical size don't change.
            assert size.width() == 600
            assert size.height() == 240
            assert qt_canvas.get_width_height() == (600, 240)
            assert (fig.get_size_inches() == (5, 2)).all()

        # check that closing the figure restores the original dpi
        plt.close(fig)
        assert fig.dpi == 120


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_subplottool():
    fig, ax = plt.subplots()
    with mock.patch("matplotlib.backends.qt_compat._exec", lambda obj: None):
        tool = fig.canvas.manager.toolbar.configure_subplots()
        assert tool is not None
        assert tool == fig.canvas.manager.toolbar.configure_subplots()


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_figureoptions():
    fig, ax = plt.subplots()
    ax.plot([1, 2])
    ax.imshow([[1]])
    ax.scatter(range(3), range(3), c=range(3))
    with mock.patch("matplotlib.backends.qt_compat._exec", lambda obj: None):
        fig.canvas.manager.toolbar.edit_parameters()


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_save_figure_return(tmp_path):
    fig, ax = plt.subplots()
    ax.imshow([[1]])
    expected = tmp_path / "foobar.png"
    prop = "matplotlib.backends.qt_compat.QtWidgets.QFileDialog.getSaveFileName"
    with mock.patch(prop, return_value=(str(expected), None)):
        fname = fig.canvas.manager.toolbar.save_figure()
        assert fname == str(expected)
        assert expected.exists()
    with mock.patch(prop, return_value=(None, None)):
        fname = fig.canvas.manager.toolbar.save_figure()
        assert fname is None


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_figureoptions_with_datetime_axes():
    fig, ax = plt.subplots()
    xydata = [
        datetime(year=2021, month=1, day=1),
        datetime(year=2021, month=2, day=1)
    ]
    ax.plot(xydata, xydata)
    with mock.patch("matplotlib.backends.qt_compat._exec", lambda obj: None):
        fig.canvas.manager.toolbar.edit_parameters()


@pytest.mark.backend("QtAgg", skip_on_importerror=True)
@pytest.mark.parametrize(
    "has_legend_initial,target_visible",
    [
        (False, True),  # No legend initially -> should create and be visible
        (False, False),  # No legend initially -> still no legend or invisible
        (True, True),  # Legend exists -> set to visible
        (True, False),  # Legend exists -> set to invisible
    ],
)
def test_legend_present_absent_toggle(has_legend_initial, target_visible):
    import numpy as np

    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots()
    try:
        # Add a line that can go into legend
        ax.plot([1, 2, 3], label="Line 1")

        # Add an image mappable (covers mappables branch; custom cmap not in registry)
        data = np.arange(4).reshape(2, 2)
        custom_cmap = ListedColormap(["#000000", "#ffffff"], name="__unit_custom__")
        img = ax.imshow(data, cmap=custom_cmap)
        img.set_label("heat")

        # Initial legend setup
        if has_legend_initial:
            leg = ax.legend(ncols=2)
            leg.set_draggable(True)
            ax.legend_.set_visible(not target_visible)
        else:
            assert ax.legend_ is None

        with (
            mock.patch.object(fig.canvas, "draw") as mock_draw,
            mock.patch.object(fig.canvas, "toolbar", create=True),
        ):
            callback = None

            def fake_fedit(datalist, title, parent=None, icon=None, apply=None):
                nonlocal callback
                callback = apply
                return datalist

            with mock.patch.object(
                matplotlib.backends.qt_editor.figureoptions._formlayout,
                "fedit",
                side_effect=fake_fedit,
            ):
                matplotlib.backends.qt_editor.figureoptions.figure_edit(ax)

            # —— General properties —— #
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            xscale_now = ax.get_xscale()
            yscale_now = ax.get_yscale()

            # Switch scale to trigger symlog branch safely (avoid negative issues)
            xscale_new = "symlog" if xscale_now != "symlog" else "linear"

            # Force xlim changes to trigger push_current
            new_xmin = xlim[0] - 1.0
            new_xmax = xlim[1] + 1.0

            general_block = [
                "Title",  # title
                new_xmin,
                new_xmax,  # X: Min/Max
                xlabel,
                xscale_new,  # X: Label/Scale
                ylim[0],
                ylim[1],  # Y: Min/Max (unchanged)
                ylabel,
                yscale_now,  # Y: Label/Scale (unchanged)
                target_visible,  # Legend visible (second last popped)
                True,  # (Re-)Generate automatic legend (last popped)
            ]

            # —— Curves block —— #
            # marker='none' covers False branch for marker handling
            curves_block = [
                [
                    "Line 1",
                    "-",
                    "default",
                    1.5,
                    "#1f77b4ff",
                    "none",
                    6.0,
                    "#1f77b4ff",
                    "#1f77b4ff",
                ]
            ]

            # —— Mappables block (Image 6-tuple) —— #
            low, high = img.get_clim()
            mappables_block = [
                [
                    "heat",
                    custom_cmap,
                    low,
                    high,
                    "nearest",
                    "auto",
                ]
            ]

            form_data = [general_block, curves_block, mappables_block]

            assert callback is not None
            callback(form_data)

            # —— Assertions for legend —— #
            if target_visible:
                assert ax.legend_ is not None
                assert ax.legend_.get_visible() is True
                if has_legend_initial:
                    # ncols should not exceed number of visible items
                    handle_count = len(ax.legend_.texts)
                    expected_ncols = min(2, max(1, handle_count))
                    assert ax.legend_._ncols == expected_ncols
                    # Draggability is preserved
                    assert ax.legend_._draggable is not None
            else:
                # Legend should be None or invisible
                assert (ax.legend_ is None) or (ax.legend_.get_visible() is False)

            # —— Assertions for mappables writeback —— #
            assert img.get_label() == "heat"
            assert img.get_cmap().name == custom_cmap.name
            assert img.get_clim() == tuple(sorted((low, high)))
            if hasattr(img, "get_interpolation"):
                assert img.get_interpolation() == "nearest"
            if hasattr(img, "get_interpolation_stage"):
                assert img.get_interpolation_stage() == "auto"

            # —— Assertions for scale/limits, redraw, and navigation stack —— #
            assert ax.get_xscale() == xscale_new
            assert ax.get_xlim() != xlim  # changed
            assert mock_draw.called
            assert fig.canvas.toolbar.push_current.called

    finally:
        plt.close(fig)


@pytest.mark.backend("QtAgg", skip_on_importerror=True)
def test_figure_edit_with_date_axis():
    import datetime as dt

    fig, ax = plt.subplots()
    try:
        xs = [dt.datetime(2024, 1, i + 1) for i in range(3)]
        ax.plot(xs, [1, 2, 3], label="d")

        cb = {}

        def fake_fedit(datalist, title, parent=None, icon=None, apply=None):
            cb["apply"] = apply
            return datalist

        with (
            mock.patch.object(fig.canvas, "draw"),
            mock.patch.object(fig.canvas, "toolbar", create=True),
            mock.patch.object(
                matplotlib.backends.qt_editor.figureoptions._formlayout,
                "fedit",
                side_effect=fake_fedit,
            ),
        ):
            matplotlib.backends.qt_editor.figureoptions.figure_edit(ax)
            # Keep original values; scale unchanged; legend False, regenerate False
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            form = [
                [
                    "",
                    xlim[0],
                    xlim[1],
                    ax.get_xlabel(),
                    ax.get_xscale(),
                    ylim[0],
                    ylim[1],
                    ax.get_ylabel(),
                    ax.get_yscale(),
                    False,
                    False,
                ],
                [
                    [
                        "d",
                        "-",
                        "default",
                        1.5,
                        "#1f77b4ff",
                        "none",
                        6.0,
                        "#1f77b4ff",
                        "#1f77b4ff",
                    ]
                ],
            ]
            cb["apply"](form)
    finally:
        plt.close(fig)


@pytest.mark.backend("QtAgg", skip_on_importerror=True)
def test_skip_nolegend_line():
    fig, ax = plt.subplots()
    try:
        # Lines with label "_nolegend_" should be skipped from legend
        ax.plot([1, 2, 3], label="_nolegend_")
        apply_ref = {}

        def fake_fedit(datalist, title, parent=None, icon=None, apply=None):
            apply_ref["apply"] = apply
            return datalist

        with (
            mock.patch.object(fig.canvas, "draw"),
            mock.patch.object(fig.canvas, "toolbar", create=True),
            mock.patch.object(
                matplotlib.backends.qt_editor.figureoptions._formlayout,
                "fedit",
                side_effect=fake_fedit,
            ),
        ):
            matplotlib.backends.qt_editor.figureoptions.figure_edit(ax)
            # No curves block provided; only general
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            form = [
                [
                    "",
                    xlim[0],
                    xlim[1],
                    "",
                    ax.get_xscale(),
                    ylim[0],
                    ylim[1],
                    "",
                    ax.get_yscale(),
                    False,
                    False,
                ]
            ]
            apply_ref["apply"](form)
    finally:
        plt.close(fig)


@pytest.mark.backend("QtAgg", skip_on_importerror=True)
def test_skip_mappable_without_array():
    fig, ax = plt.subplots()
    try:
        # A mappable without array (PathCollection):
        # get_array() is None if c= not provided
        coll = ax.scatter([0, 1], [0, 1], label="noarray")
        assert coll.get_array() is None

        with (
            mock.patch.object(fig.canvas, "draw") as _draw,
            mock.patch.object(fig.canvas, "toolbar", create=True),
        ):
            cb = None

            def fake_fedit(datalist, title, parent=None, icon=None, apply=None):
                nonlocal cb
                cb = apply
                return datalist

            with mock.patch.object(
                matplotlib.backends.qt_editor.figureoptions._formlayout,
                "fedit",
                side_effect=fake_fedit,
            ):
                matplotlib.backends.qt_editor.figureoptions.figure_edit(ax)

            # Minimal apply: only provide General block, keep legend unchanged
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            general = [
                "T",
                xlim[0],
                xlim[1],
                ax.get_xlabel(),
                ax.get_xscale(),
                ylim[0],
                ylim[1],
                ax.get_ylabel(),
                ax.get_yscale(),
                bool(ax.legend_),  # Legend visible
                False,  # Do not regenerate legend
            ]
            assert cb is not None
            cb([general])  # No curves/mappables blocks triggers skip path

            assert _draw.called
    finally:
        plt.close(fig)


@pytest.mark.backend("QtAgg", skip_on_importerror=True)
def test_generate_legend_false_keeps_old_legend():
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0, 1], label="L1")
        old = ax.legend(ncols=2)
        old.set_draggable(True)

        with (
            mock.patch.object(fig.canvas, "draw") as _draw,
            mock.patch.object(fig.canvas, "toolbar", create=True),
        ):
            cb = None

            def fake_fedit(datalist, title, parent=None, icon=None, apply=None):
                nonlocal cb
                cb = apply
                return datalist

            with mock.patch.object(
                matplotlib.backends.qt_editor.figureoptions._formlayout,
                "fedit",
                side_effect=fake_fedit,
            ):
                matplotlib.backends.qt_editor.figureoptions.figure_edit(ax)

            # —— General block —— #
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            general = [
                "T",
                xlim[0],
                xlim[1],
                ax.get_xlabel(),
                ax.get_xscale(),
                ylim[0],
                ylim[1],
                ax.get_ylabel(),
                ax.get_yscale(),
                True,  # Legend visible
                False,  # Key: do not regenerate legend
            ]

            # —— Curves block: fill from existing line, no change —— #
            line = ax.lines[0]
            from matplotlib import colors as mcolors

            color_hex = mcolors.to_hex(
                mcolors.to_rgba(line.get_color(), line.get_alpha()), keep_alpha=True
            )
            face_hex = mcolors.to_hex(
                mcolors.to_rgba(line.get_markerfacecolor(), line.get_alpha()),
                keep_alpha=True,
            )
            edge_hex = mcolors.to_hex(
                mcolors.to_rgba(line.get_markeredgecolor(), line.get_alpha()),
                keep_alpha=True,
            )
            curves_block = [
                [
                    line.get_label(),
                    line.get_linestyle(),
                    line.get_drawstyle(),
                    line.get_linewidth(),
                    color_hex,
                    line.get_marker() or "none",
                    line.get_markersize(),
                    face_hex,
                    edge_hex,
                ]
            ]

            assert cb is not None
            cb([general, curves_block])  # Pass General + Curves (no mappables)

            # —— Assertions: legend object not replaced —— #
            assert ax.legend_ is old  # Critical: legend not rebuilt
            assert ax.legend_.get_visible() is True
            assert ax.legend_._draggable is not None  # draggable preserved
            assert _draw.called
    finally:
        plt.close(fig)


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_double_resize():
    # Check that resizing a figure twice keeps the same window size
    fig, ax = plt.subplots()
    fig.canvas.draw()
    window = fig.canvas.manager.window

    w, h = 3, 2
    fig.set_size_inches(w, h)
    assert fig.canvas.width() == w * matplotlib.rcParams['figure.dpi']
    assert fig.canvas.height() == h * matplotlib.rcParams['figure.dpi']

    old_width = window.width()
    old_height = window.height()

    fig.set_size_inches(w, h)
    assert window.width() == old_width
    assert window.height() == old_height


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_canvas_reinit():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    called = False

    def crashing_callback(fig, stale):
        nonlocal called
        fig.canvas.draw_idle()
        called = True

    fig, ax = plt.subplots()
    fig.stale_callback = crashing_callback
    # this should not raise
    canvas = FigureCanvasQTAgg(fig)
    fig.stale = True
    assert called


@pytest.mark.backend('Qt5Agg', skip_on_importerror=True)
def test_form_widget_get_with_datetime_and_date_fields():
    from matplotlib.backends.backend_qt import _create_qApp
    _create_qApp()

    form = [
        ("Datetime field", datetime(year=2021, month=3, day=11)),
        ("Date field", date(year=2021, month=3, day=11))
    ]
    widget = _formlayout.FormWidget(form)
    widget.setup()
    values = widget.get()
    assert values == [
        datetime(year=2021, month=3, day=11),
        date(year=2021, month=3, day=11)
    ]


def _get_testable_qt_backends():
    envs = []
    for deps, env in [
            ([qt_api], {"MPLBACKEND": "qtagg", "QT_API": qt_api})
            for qt_api in ["PyQt6", "PySide6", "PyQt5", "PySide2"]
    ]:
        reason = None
        missing = [dep for dep in deps if not importlib.util.find_spec(dep)]
        if (sys.platform == "linux" and
                not _c_internal_utils.display_is_valid()):
            reason = "$DISPLAY and $WAYLAND_DISPLAY are unset"
        elif missing:
            reason = "{} cannot be imported".format(", ".join(missing))
        elif env["MPLBACKEND"] == 'macosx' and os.environ.get('TF_BUILD'):
            reason = "macosx backend fails on Azure"
        marks = []
        if reason:
            marks.append(pytest.mark.skip(
                reason=f"Skipping {env} because {reason}"))
        envs.append(pytest.param(env, marks=marks, id=str(env)))
    return envs


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_fig_sigint_override():
    from matplotlib.backends.backend_qt5 import _BackendQT5
    # Create a figure
    plt.figure()

    # Variable to access the handler from the inside of the event loop
    event_loop_handler = None

    # Callback to fire during event loop: save SIGINT handler, then exit
    def fire_signal_and_quit():
        # Save event loop signal
        nonlocal event_loop_handler
        event_loop_handler = signal.getsignal(signal.SIGINT)

        # Request event loop exit
        QtCore.QCoreApplication.exit()

    # Timer to exit event loop
    QtCore.QTimer.singleShot(0, fire_signal_and_quit)

    # Save original SIGINT handler
    original_handler = signal.getsignal(signal.SIGINT)

    # Use our own SIGINT handler to be 100% sure this is working
    def custom_handler(signum, frame):
        pass

    signal.signal(signal.SIGINT, custom_handler)

    try:
        # mainloop() sets SIGINT, starts Qt event loop (which triggers timer
        # and exits) and then mainloop() resets SIGINT
        matplotlib.backends.backend_qt._BackendQT.mainloop()

        # Assert: signal handler during loop execution is changed
        # (can't test equality with func)
        assert event_loop_handler != custom_handler

        # Assert: current signal handler is the same as the one we set before
        assert signal.getsignal(signal.SIGINT) == custom_handler

        # Repeat again to test that SIG_DFL and SIG_IGN will not be overridden
        for custom_handler in (signal.SIG_DFL, signal.SIG_IGN):
            QtCore.QTimer.singleShot(0, fire_signal_and_quit)
            signal.signal(signal.SIGINT, custom_handler)

            _BackendQT5.mainloop()

            assert event_loop_handler == custom_handler
            assert signal.getsignal(signal.SIGINT) == custom_handler

    finally:
        # Reset SIGINT handler to what it was before the test
        signal.signal(signal.SIGINT, original_handler)


@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_ipython():
    from matplotlib.testing import ipython_in_subprocess
    ipython_in_subprocess("qt", {(8, 24): "qtagg", (8, 15): "QtAgg", (7, 0): "Qt5Agg"})
