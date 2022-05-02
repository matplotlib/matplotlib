import importlib
import importlib.util
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

import matplotlib as mpl
from matplotlib import _c_internal_utils
from matplotlib.testing import subprocess_run_helper as _run_helper


# Minimal smoke-testing of the backends for which the dependencies are
# PyPI-installable on CI.  They are not available for all tested Python
# versions so we don't fail on missing backends.

def _get_testable_interactive_backends():
    envs = []
    for deps, env in [
            *[([qt_api],
               {"MPLBACKEND": "qtagg", "QT_API": qt_api})
              for qt_api in ["PyQt6", "PySide6", "PyQt5", "PySide2"]],
            *[([qt_api, "cairocffi"],
               {"MPLBACKEND": "qtcairo", "QT_API": qt_api})
              for qt_api in ["PyQt6", "PySide6", "PyQt5", "PySide2"]],
            *[(["cairo", "gi"], {"MPLBACKEND": f"gtk{version}{renderer}"})
              for version in [3, 4] for renderer in ["agg", "cairo"]],
            (["tkinter"], {"MPLBACKEND": "tkagg"}),
            (["wx"], {"MPLBACKEND": "wx"}),
            (["wx"], {"MPLBACKEND": "wxagg"}),
            (["matplotlib.backends._macosx"], {"MPLBACKEND": "macosx"}),
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
        elif env["MPLBACKEND"].startswith('gtk'):
            import gi
            version = env["MPLBACKEND"][3]
            repo = gi.Repository.get_default()
            if f'{version}.0' not in repo.enumerate_versions('Gtk'):
                reason = "no usable GTK bindings"
        marks = []
        if reason:
            marks.append(pytest.mark.skip(
                reason=f"Skipping {env} because {reason}"))
        elif env["MPLBACKEND"].startswith('wx') and sys.platform == 'darwin':
            # ignore on OSX because that's currently broken (github #16849)
            marks.append(pytest.mark.xfail(reason='github #16849'))
        envs.append(pytest.param(env, marks=marks, id=str(env)))
    return envs


_test_timeout = 60  # A reasonably safe value for slower architectures.


# The source of this function gets extracted and run in another process, so it
# must be fully self-contained.
# Using a timer not only allows testing of timers (on other backends), but is
# also necessary on gtk3 and wx, where a direct call to key_press_event("q")
# from draw_event causes breakage due to the canvas widget being deleted too
# early.  Also, gtk3 redefines key_press_event with a different signature, so
# we directly invoke it from the superclass instead.
def _test_interactive_impl():
    import importlib.util
    import io
    import json
    import sys
    from unittest import TestCase

    import matplotlib as mpl
    from matplotlib import pyplot as plt, rcParams
    from matplotlib.backend_bases import FigureCanvasBase

    rcParams.update({
        "webagg.open_in_browser": False,
        "webagg.port_retries": 1,
    })

    rcParams.update(json.loads(sys.argv[1]))
    backend = plt.rcParams["backend"].lower()
    assert_equal = TestCase().assertEqual
    assert_raises = TestCase().assertRaises

    if backend.endswith("agg") and not backend.startswith(("gtk", "web")):
        # Force interactive framework setup.
        plt.figure()

        # Check that we cannot switch to a backend using another interactive
        # framework, but can switch to a backend using cairo instead of agg,
        # or a non-interactive backend.  In the first case, we use tkagg as
        # the "other" interactive backend as it is (essentially) guaranteed
        # to be present.  Moreover, don't test switching away from gtk3 (as
        # Gtk.main_level() is not set up at this point yet) and webagg (which
        # uses no interactive framework).

        if backend != "tkagg":
            with assert_raises(ImportError):
                mpl.use("tkagg", force=True)

        def check_alt_backend(alt_backend):
            mpl.use(alt_backend, force=True)
            fig = plt.figure()
            assert_equal(
                type(fig.canvas).__module__,
                "matplotlib.backends.backend_{}".format(alt_backend))

        if importlib.util.find_spec("cairocffi"):
            check_alt_backend(backend[:-3] + "cairo")
        check_alt_backend("svg")

    mpl.use(backend, force=True)

    fig, ax = plt.subplots()
    assert_equal(
        type(fig.canvas).__module__,
        "matplotlib.backends.backend_{}".format(backend))

    ax.plot([0, 1], [2, 3])
    if fig.canvas.toolbar:  # i.e toolbar2.
        fig.canvas.toolbar.draw_rubberband(None, 1., 1, 2., 2)

    timer = fig.canvas.new_timer(1.)  # Test floats casting to int as needed.
    timer.add_callback(FigureCanvasBase.key_press_event, fig.canvas, "q")
    # Trigger quitting upon draw.
    fig.canvas.mpl_connect("draw_event", lambda event: timer.start())
    fig.canvas.mpl_connect("close_event", print)

    result = io.BytesIO()
    fig.savefig(result, format='png')

    plt.show()

    # Ensure that the window is really closed.
    plt.pause(0.5)

    # Test that saving works after interactive window is closed, but the figure
    # is not deleted.
    result_after = io.BytesIO()
    fig.savefig(result_after, format='png')

    if not backend.startswith('qt5') and sys.platform == 'darwin':
        # FIXME: This should be enabled everywhere once Qt5 is fixed on macOS
        # to not resize incorrectly.
        assert_equal(result.getvalue(), result_after.getvalue())


@pytest.mark.parametrize("env", _get_testable_interactive_backends())
@pytest.mark.parametrize("toolbar", ["toolbar2", "toolmanager"])
@pytest.mark.flaky(reruns=3)
def test_interactive_backend(env, toolbar):
    if env["MPLBACKEND"] == "macosx":
        if toolbar == "toolmanager":
            pytest.skip("toolmanager is not implemented for macosx.")
    proc = _run_helper(_test_interactive_impl,
                       json.dumps({"toolbar": toolbar}),
                       timeout=_test_timeout,
                       **env)

    assert proc.stdout.count("CloseEvent") == 1


def _test_thread_impl():
    from concurrent.futures import ThreadPoolExecutor

    from matplotlib import pyplot as plt, rcParams

    rcParams.update({
        "webagg.open_in_browser": False,
        "webagg.port_retries": 1,
    })

    # Test artist creation and drawing does not crash from thread
    # No other guarantees!
    fig, ax = plt.subplots()
    # plt.pause needed vs plt.show(block=False) at least on toolbar2-tkagg
    plt.pause(0.5)

    future = ThreadPoolExecutor().submit(ax.plot, [1, 3, 6])
    future.result()  # Joins the thread; rethrows any exception.

    fig.canvas.mpl_connect("close_event", print)
    future = ThreadPoolExecutor().submit(fig.canvas.draw)
    plt.pause(0.5)  # flush_events fails here on at least Tkagg (bpo-41176)
    future.result()  # Joins the thread; rethrows any exception.
    plt.close()  # backend is responsible for flushing any events here
    if plt.rcParams["backend"].startswith("WX"):
        # TODO: debug why WX needs this only on py3.8
        fig.canvas.flush_events()


_thread_safe_backends = _get_testable_interactive_backends()
# Known unsafe backends. Remove the xfails if they start to pass!
for param in _thread_safe_backends:
    backend = param.values[0]["MPLBACKEND"]
    if "cairo" in backend:
        # Cairo backends save a cairo_t on the graphics context, and sharing
        # these is not threadsafe.
        param.marks.append(
            pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == "wx":
        param.marks.append(
            pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == "macosx":
        from packaging.version import parse
        mac_ver = platform.mac_ver()[0]
        # Note, macOS Big Sur is both 11 and 10.16, depending on SDK that
        # Python was compiled against.
        if mac_ver and parse(mac_ver) < parse('10.16'):
            param.marks.append(
                pytest.mark.xfail(raises=subprocess.TimeoutExpired,
                                  strict=True))
    elif param.values[0].get("QT_API") == "PySide2":
        param.marks.append(
            pytest.mark.xfail(raises=subprocess.CalledProcessError))
    elif backend == "tkagg" and platform.python_implementation() != 'CPython':
        param.marks.append(
            pytest.mark.xfail(
                reason='PyPy does not support Tkinter threading: '
                       'https://foss.heptapod.net/pypy/pypy/-/issues/1929',
                strict=True))


@pytest.mark.parametrize("env", _thread_safe_backends)
@pytest.mark.flaky(reruns=3)
def test_interactive_thread_safety(env):
    proc = _run_helper(_test_thread_impl,
                       timeout=_test_timeout, **env)
    assert proc.stdout.count("CloseEvent") == 1


def _impl_test_lazy_auto_backend_selection():
    import matplotlib
    import matplotlib.pyplot as plt
    # just importing pyplot should not be enough to trigger resolution
    bk = dict.__getitem__(matplotlib.rcParams, 'backend')
    assert not isinstance(bk, str)
    assert plt._backend_mod is None
    # but actually plotting should
    plt.plot(5)
    assert plt._backend_mod is not None
    bk = dict.__getitem__(matplotlib.rcParams, 'backend')
    assert isinstance(bk, str)


def test_lazy_auto_backend_selection():
    _run_helper(_impl_test_lazy_auto_backend_selection,
                timeout=_test_timeout)


def _implqt5agg():
    import matplotlib.backends.backend_qt5agg  # noqa
    import sys

    assert 'PyQt6' not in sys.modules
    assert 'pyside6' not in sys.modules
    assert 'PyQt5' in sys.modules or 'pyside2' in sys.modules

    import matplotlib.backends.backend_qt5
    matplotlib.backends.backend_qt5.qApp


def _implcairo():
    import matplotlib.backends.backend_qt5cairo # noqa
    import sys

    assert 'PyQt6' not in sys.modules
    assert 'pyside6' not in sys.modules
    assert 'PyQt5' in sys.modules or 'pyside2' in sys.modules

    import matplotlib.backends.backend_qt5
    matplotlib.backends.backend_qt5.qApp


def _implcore():
    import matplotlib.backends.backend_qt5
    import sys

    assert 'PyQt6' not in sys.modules
    assert 'pyside6' not in sys.modules
    assert 'PyQt5' in sys.modules or 'pyside2' in sys.modules
    matplotlib.backends.backend_qt5.qApp


def test_qt5backends_uses_qt5():
    qt5_bindings = [
        dep for dep in ['PyQt5', 'pyside2']
        if importlib.util.find_spec(dep) is not None
    ]
    qt6_bindings = [
        dep for dep in ['PyQt6', 'pyside6']
        if importlib.util.find_spec(dep) is not None
    ]
    if len(qt5_bindings) == 0 or len(qt6_bindings) == 0:
        pytest.skip('need both QT6 and QT5 bindings')
    _run_helper(_implqt5agg, timeout=_test_timeout)
    if importlib.util.find_spec('pycairo') is not None:
        _run_helper(_implcairo, timeout=_test_timeout)
    _run_helper(_implcore, timeout=_test_timeout)


def _impl_test_cross_Qt_imports():
    import sys
    import importlib
    import pytest

    _, host_binding, mpl_binding = sys.argv
    # import the mpl binding.  This will force us to use that binding
    importlib.import_module(f'{mpl_binding}.QtCore')
    mpl_binding_qwidgets = importlib.import_module(f'{mpl_binding}.QtWidgets')
    import matplotlib.backends.backend_qt
    host_qwidgets = importlib.import_module(f'{host_binding}.QtWidgets')

    host_app = host_qwidgets.QApplication(["mpl testing"])
    with pytest.warns(UserWarning, match="Mixing Qt major"):
        matplotlib.backends.backend_qt._create_qApp()


def test_cross_Qt_imports():
    qt5_bindings = [
        dep for dep in ['PyQt5', 'PySide2']
        if importlib.util.find_spec(dep) is not None
    ]
    qt6_bindings = [
        dep for dep in ['PyQt6', 'PySide6']
        if importlib.util.find_spec(dep) is not None
    ]
    if len(qt5_bindings) == 0 or len(qt6_bindings) == 0:
        pytest.skip('need both QT6 and QT5 bindings')

    for qt5 in qt5_bindings:
        for qt6 in qt6_bindings:
            for pair in ([qt5, qt6], [qt6, qt5]):
                try:
                    _run_helper(_impl_test_cross_Qt_imports,
                                *pair,
                                timeout=_test_timeout)
                except subprocess.CalledProcessError as ex:
                    # if segfault, carry on.  We do try to warn the user they
                    # are doing something that we do not expect to work
                    if ex.returncode == -11:
                        continue
                    raise


@pytest.mark.skipif('TF_BUILD' in os.environ,
                    reason="this test fails an azure for unknown reasons")
@pytest.mark.skipif(os.name == "nt", reason="Cannot send SIGINT on Windows.")
def test_webagg():
    pytest.importorskip("tornado")
    proc = subprocess.Popen(
        [sys.executable, "-c",
         inspect.getsource(_test_interactive_impl)
         + "\n_test_interactive_impl()", "{}"],
        env={**os.environ, "MPLBACKEND": "webagg", "SOURCE_DATE_EPOCH": "0"})
    url = "http://{}:{}".format(
        mpl.rcParams["webagg.address"], mpl.rcParams["webagg.port"])
    timeout = time.perf_counter() + _test_timeout
    while True:
        try:
            retcode = proc.poll()
            # check that the subprocess for the server is not dead
            assert retcode is None
            conn = urllib.request.urlopen(url)
            break
        except urllib.error.URLError:
            if time.perf_counter() > timeout:
                pytest.fail("Failed to connect to the webagg server.")
            else:
                continue
    conn.close()
    proc.send_signal(signal.SIGINT)
    assert proc.wait(timeout=_test_timeout) == 0


def _lazy_headless():
    import os
    import sys

    # make it look headless
    os.environ.pop('DISPLAY', None)
    os.environ.pop('WAYLAND_DISPLAY', None)

    # we should fast-track to Agg
    import matplotlib.pyplot as plt
    plt.get_backend() == 'agg'
    assert 'PyQt5' not in sys.modules

    # make sure we really have pyqt installed
    import PyQt5  # noqa
    assert 'PyQt5' in sys.modules

    # try to switch and make sure we fail with ImportError
    try:
        plt.switch_backend('qt5agg')
    except ImportError:
        ...
    else:
        sys.exit(1)


@pytest.mark.skipif(sys.platform != "linux", reason="this a linux-only test")
@pytest.mark.backend('QtAgg', skip_on_importerror=True)
def test_lazy_linux_headless():
    proc = _run_helper(_lazy_headless, timeout=_test_timeout, MPLBACKEND="")


# The source of this function gets extracted and run in another process, so it
# must be fully self-contained.
def _test_figure_leak():
    import gc
    import sys

    import psutil
    from matplotlib import pyplot as plt
    # Second argument is pause length, but if zero we should skip pausing
    t = float(sys.argv[1])
    p = psutil.Process()

    # Warmup cycle, this reasonably allocates a lot
    for _ in range(2):
        fig = plt.figure()
        if t:
            plt.pause(t)
        plt.close(fig)
    mem = p.memory_info().rss
    gc.collect()

    for _ in range(5):
        fig = plt.figure()
        if t:
            plt.pause(t)
        plt.close(fig)
        gc.collect()
    growth = p.memory_info().rss - mem

    print(growth)


# TODO: "0.1" memory threshold could be reduced 10x by fixing tkagg
@pytest.mark.parametrize("env", _get_testable_interactive_backends())
@pytest.mark.parametrize("time_mem", [(0.0, 2_000_000), (0.1, 30_000_000)])
def test_figure_leak_20490(env, time_mem):
    pytest.importorskip("psutil", reason="psutil needed to run this test")

    # We haven't yet directly identified the leaks so test with a memory growth
    # threshold.
    pause_time, acceptable_memory_leakage = time_mem
    if env["MPLBACKEND"] == "macosx":
        acceptable_memory_leakage += 11_000_000

    result = _run_helper(
        _test_figure_leak, str(pause_time), timeout=_test_timeout, **env
    )

    growth = int(result.stdout)
    assert growth <= acceptable_memory_leakage
