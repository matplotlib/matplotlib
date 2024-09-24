"""Timeout for tests to stop hanging testruns.

This plugin will dump the stack and terminate the test.  This can be
useful when running tests on a continuous integration server.

If the platform supports SIGALRM this is used to raise an exception in
the test, otherwise os._exit(1) is used.
"""
import inspect
import os
import signal
import sys
import threading
import time
import traceback
from collections import namedtuple

import pytest


__all__ = ("is_debugging", "Settings")
SESSION_TIMEOUT_KEY = pytest.StashKey[float]()
SESSION_EXPIRE_KEY = pytest.StashKey[float]()


HAVE_SIGALRM = hasattr(signal, "SIGALRM")
if HAVE_SIGALRM:
    DEFAULT_METHOD = "signal"
else:
    DEFAULT_METHOD = "thread"
TIMEOUT_DESC = """
Timeout in seconds before dumping the stacks.  Default is 0 which
means no timeout.
""".strip()
METHOD_DESC = """
Timeout mechanism to use.  'signal' uses SIGALRM, 'thread' uses a timer
thread.  If unspecified 'signal' is used on platforms which support
SIGALRM, otherwise 'thread' is used.
""".strip()
FUNC_ONLY_DESC = """
When set to True, defers the timeout evaluation to only the test
function body, ignoring the time it takes when evaluating any fixtures
used in the test.
""".strip()
DISABLE_DEBUGGER_DETECTION_DESC = """
When specified, disables debugger detection. breakpoint(), pdb.set_trace(), etc.
will be interrupted by the timeout.
""".strip()
SESSION_TIMEOUT_DESC = """
Timeout in seconds for entire session.  Default is None which
means no timeout. Timeout is checked between tests, and will not interrupt a test
in progress.
""".strip()

# bdb covers pdb, ipdb, and possibly others
# pydevd covers PyCharm, VSCode, and possibly others
KNOWN_DEBUGGING_MODULES = {"pydevd", "bdb", "pydevd_frame_evaluator"}
Settings = namedtuple(
    "Settings", ["timeout", "method", "func_only", "disable_debugger_detection"]
)


@pytest.hookimpl
def pytest_addoption(parser):
    """Add options to control the timeout plugin."""
    group = parser.getgroup(
        "timeout",
        "Interrupt test run and dump stacks of all threads after a test times out",
    )
    group.addoption("--timeout", type=float, help=TIMEOUT_DESC)
    group.addoption(
        "--timeout_method",
        action="store",
        choices=["signal", "thread"],
        help="Deprecated, use --timeout-method",
    )
    group.addoption(
        "--timeout-method",
        dest="timeout_method",
        action="store",
        choices=["signal", "thread"],
        help=METHOD_DESC,
    )
    group.addoption(
        "--timeout-disable-debugger-detection",
        dest="timeout_disable_debugger_detection",
        action="store_true",
        help=DISABLE_DEBUGGER_DETECTION_DESC,
    )
    group.addoption(
        "--session-timeout",
        action="store",
        dest="session_timeout",
        default=None,
        type=float,
        metavar="SECONDS",
        help=SESSION_TIMEOUT_DESC,
    )
    parser.addini("timeout", TIMEOUT_DESC)
    parser.addini("timeout_method", METHOD_DESC)
    parser.addini("timeout_func_only", FUNC_ONLY_DESC, type="bool", default=False)
    parser.addini(
        "timeout_disable_debugger_detection",
        DISABLE_DEBUGGER_DETECTION_DESC,
        type="bool",
        default=False,
    )
    parser.addini("session_timeout", SESSION_TIMEOUT_DESC)


class TimeoutHooks:
    """Timeout specific hooks."""

    @pytest.hookspec(firstresult=True)
    def pytest_timeout_set_timer(item, settings):
        """Called at timeout setup.

        'item' is a pytest node to setup timeout for.

        Can be overridden by plugins for alternative timeout implementation strategies.

        """

    @pytest.hookspec(firstresult=True)
    def pytest_timeout_cancel_timer(item):
        """Called at timeout teardown.

        'item' is a pytest node which was used for timeout setup.

        Can be overridden by plugins for alternative timeout implementation strategies.

        """


def pytest_addhooks(pluginmanager):
    """Register timeout-specific hooks."""
    pluginmanager.add_hookspecs(TimeoutHooks)


@pytest.hookimpl
def pytest_configure(config):
    """Register the marker so it shows up in --markers output."""
    config.addinivalue_line(
        "markers",
        "timeout(timeout, method=None, func_only=False, "
        "disable_debugger_detection=False): Set a timeout, timeout "
        "method and func_only evaluation on just one test item.  The first "
        "argument, *timeout*, is the timeout in seconds while the keyword, "
        "*method*, takes the same values as the --timeout-method option. The "
        "*func_only* keyword, when set to True, defers the timeout evaluation "
        "to only the test function body, ignoring the time it takes when "
        "evaluating any fixtures used in the test. The "
        "*disable_debugger_detection* keyword, when set to True, disables "
        "debugger detection, allowing breakpoint(), pdb.set_trace(), etc. "
        "to be interrupted",
    )

    settings = get_env_settings(config)
    config._env_timeout = settings.timeout
    config._env_timeout_method = settings.method
    config._env_timeout_func_only = settings.func_only
    config._env_timeout_disable_debugger_detection = settings.disable_debugger_detection

    timeout = config.getoption("session_timeout")
    if timeout is None:
        ini = config.getini("session_timeout")
        if ini:
            timeout = _validate_timeout(config.getini("session_timeout"), "config file")
    if timeout is not None:
        expire_time = time.time() + timeout
    else:
        expire_time = 0
        timeout = 0
    config.stash[SESSION_TIMEOUT_KEY] = timeout
    config.stash[SESSION_EXPIRE_KEY] = expire_time


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item):
    """Hook in timeouts to the runtest protocol.

    If the timeout is set on the entire test, including setup and
    teardown, then this hook installs the timeout.  Otherwise
    pytest_runtest_call is used.
    """
    hooks = item.config.pluginmanager.hook
    settings = _get_item_settings(item)
    is_timeout = settings.timeout is not None and settings.timeout > 0
    if is_timeout and settings.func_only is False:
        hooks.pytest_timeout_set_timer(item=item, settings=settings)
    yield
    if is_timeout and settings.func_only is False:
        hooks.pytest_timeout_cancel_timer(item=item)

    #  check session timeout
    expire_time = item.session.config.stash[SESSION_EXPIRE_KEY]
    if expire_time and (expire_time < time.time()):
        timeout = item.session.config.stash[SESSION_TIMEOUT_KEY]
        item.session.shouldfail = f"session-timeout: {timeout} sec exceeded"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Hook in timeouts to the test function call only.

    If the timeout is set on only the test function this hook installs
    the timeout, otherwise pytest_runtest_protocol is used.
    """
    hooks = item.config.pluginmanager.hook
    settings = _get_item_settings(item)
    is_timeout = settings.timeout is not None and settings.timeout > 0
    if is_timeout and settings.func_only is True:
        hooks.pytest_timeout_set_timer(item=item, settings=settings)
    yield
    if is_timeout and settings.func_only is True:
        hooks.pytest_timeout_cancel_timer(item=item)


@pytest.hookimpl(tryfirst=True)
def pytest_report_header(config):
    """Add timeout config to pytest header."""
    timeout_header = []

    if config._env_timeout:
        timeout_header.append(
            "timeout: %ss\ntimeout method: %s\ntimeout func_only: %s"
            % (
                config._env_timeout,
                config._env_timeout_method,
                config._env_timeout_func_only,
            )
        )

    session_timeout = config.getoption("session_timeout")
    if session_timeout:
        timeout_header.append("session timeout: %ss" % session_timeout)
    if timeout_header:
        return timeout_header


@pytest.hookimpl(tryfirst=True)
def pytest_exception_interact(node):
    """Stop the timeout when pytest enters pdb in post-mortem mode."""
    hooks = node.config.pluginmanager.hook
    hooks.pytest_timeout_cancel_timer(item=node)


@pytest.hookimpl
def pytest_enter_pdb():
    """Stop the timeouts when we entered pdb.

    This stops timeouts from triggering when pytest's builting pdb
    support notices we entered pdb.
    """
    # Since pdb.set_trace happens outside of any pytest control, we don't have
    # any pytest ``item`` here, so we cannot use timeout_teardown. Thus, we
    # need another way to signify that the timeout should not be performed.
    global SUPPRESS_TIMEOUT
    SUPPRESS_TIMEOUT = True


def is_debugging(trace_func=None):
    """Detect if a debugging session is in progress.

    This looks at both pytest's builtin pdb support as well as
    externally installed debuggers using some heuristics.

     This is done by checking if either of the following conditions is
     true:

     1. Examines the trace function to see if the module it originates
        from is in KNOWN_DEBUGGING_MODULES.
     2. Check is SUPPRESS_TIMEOUT is set to True.

    :param trace_func: the current trace function, if not given will use
        sys.gettrace(). Used to unit-test this function.
    """
    global SUPPRESS_TIMEOUT, KNOWN_DEBUGGING_MODULES
    if SUPPRESS_TIMEOUT:
        return True
    if trace_func is None:
        trace_func = sys.gettrace()
    trace_module = None
    if trace_func:
        trace_module = inspect.getmodule(trace_func) or inspect.getmodule(
            trace_func.__class__
        )
    if trace_module:
        parts = trace_module.__name__.split(".")
        for name in KNOWN_DEBUGGING_MODULES:
            if any(part.startswith(name) for part in parts):
                return True
    return False


SUPPRESS_TIMEOUT = False


@pytest.hookimpl(trylast=True)
def pytest_timeout_set_timer(item, settings):
    """Setup up a timeout trigger and handler."""
    timeout_method = settings.method
    if (
        timeout_method == "signal"
        and threading.current_thread() is not threading.main_thread()
    ):
        timeout_method = "thread"

    if timeout_method == "signal":

        def handler(signum, frame):
            __tracebackhide__ = True
            timeout_sigalrm(item, settings)

        def cancel():
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

        item.cancel_timeout = cancel
        signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, settings.timeout)
    elif timeout_method == "thread":
        timer = threading.Timer(settings.timeout, timeout_timer, (item, settings))
        timer.name = "%s %s" % (__name__, item.nodeid)

        def cancel():
            timer.cancel()
            timer.join()

        item.cancel_timeout = cancel
        timer.start()
    return True


@pytest.hookimpl(trylast=True)
def pytest_timeout_cancel_timer(item):
    """Cancel the timeout trigger if it was set."""
    # When skipping is raised from a pytest_runtest_setup function
    # (as is the case when using the pytest.mark.skipif marker) we
    # may be called without our setup counterpart having been
    # called.
    cancel = getattr(item, "cancel_timeout", None)
    if cancel:
        cancel()
    return True


def get_env_settings(config):
    """Return the configured timeout settings.

    This looks up the settings in the environment and config file.
    """
    timeout = config.getvalue("timeout")
    if timeout is None:
        timeout = _validate_timeout(
            os.environ.get("PYTEST_TIMEOUT"), "PYTEST_TIMEOUT environment variable"
        )
    if timeout is None:
        ini = config.getini("timeout")
        if ini:
            timeout = _validate_timeout(ini, "config file")

    method = config.getvalue("timeout_method")
    if method is None:
        ini = config.getini("timeout_method")
        if ini:
            method = _validate_method(ini, "config file")
    if method is None:
        method = DEFAULT_METHOD

    func_only = config.getini("timeout_func_only")

    disable_debugger_detection = config.getvalue("timeout_disable_debugger_detection")
    if disable_debugger_detection is None:
        ini = config.getini("timeout_disable_debugger_detection")
        if ini:
            disable_debugger_detection = _validate_disable_debugger_detection(
                ini, "config file"
            )

    return Settings(timeout, method, func_only, disable_debugger_detection)


def _get_item_settings(item, marker=None):
    """Return (timeout, method) for an item."""
    timeout = method = func_only = disable_debugger_detection = None
    if not marker:
        marker = item.get_closest_marker("timeout")
    if marker is not None:
        settings = _parse_marker(item.get_closest_marker(name="timeout"))
        timeout = _validate_timeout(settings.timeout, "marker")
        method = _validate_method(settings.method, "marker")
        func_only = _validate_func_only(settings.func_only, "marker")
        disable_debugger_detection = _validate_disable_debugger_detection(
            settings.disable_debugger_detection, "marker"
        )
    if timeout is None:
        timeout = item.config._env_timeout
    if method is None:
        method = item.config._env_timeout_method
    if func_only is None:
        func_only = item.config._env_timeout_func_only
    if disable_debugger_detection is None:
        disable_debugger_detection = item.config._env_timeout_disable_debugger_detection
    return Settings(timeout, method, func_only, disable_debugger_detection)


def _parse_marker(marker):
    """Return (timeout, method) tuple from marker.

    Either could be None.  The values are not interpreted, so
    could still be bogus and even the wrong type.
    """
    if not marker.args and not marker.kwargs:
        raise TypeError("Timeout marker must have at least one argument")
    timeout = method = func_only = NOTSET = object()
    for kw, val in marker.kwargs.items():
        if kw == "timeout":
            timeout = val
        elif kw == "method":
            method = val
        elif kw == "func_only":
            func_only = val
        else:
            raise TypeError("Invalid keyword argument for timeout marker: %s" % kw)
    if len(marker.args) >= 1 and timeout is not NOTSET:
        raise TypeError("Multiple values for timeout argument of timeout marker")
    elif len(marker.args) >= 1:
        timeout = marker.args[0]
    if len(marker.args) >= 2 and method is not NOTSET:
        raise TypeError("Multiple values for method argument of timeout marker")
    elif len(marker.args) >= 2:
        method = marker.args[1]
    if len(marker.args) > 2:
        raise TypeError("Too many arguments for timeout marker")
    if timeout is NOTSET:
        timeout = None
    if method is NOTSET:
        method = None
    if func_only is NOTSET:
        func_only = None
    return Settings(timeout, method, func_only, None)


def _validate_timeout(timeout, where):
    if timeout is None:
        return None
    try:
        return float(timeout)
    except ValueError:
        raise ValueError("Invalid timeout %s from %s" % (timeout, where))


def _validate_method(method, where):
    if method is None:
        return None
    if method not in ["signal", "thread"]:
        raise ValueError("Invalid method %s from %s" % (method, where))
    return method


def _validate_func_only(func_only, where):
    if func_only is None:
        return None
    if not isinstance(func_only, bool):
        raise ValueError("Invalid func_only value %s from %s" % (func_only, where))
    return func_only


def _validate_disable_debugger_detection(disable_debugger_detection, where):
    if disable_debugger_detection is None:
        return None
    if not isinstance(disable_debugger_detection, bool):
        raise ValueError(
            "Invalid disable_debugger_detection value %s from %s"
            % (disable_debugger_detection, where)
        )
    return disable_debugger_detection


def timeout_sigalrm(item, settings):
    """Dump stack of threads and raise an exception.

    This will output the stacks of any threads other then the
    current to stderr and then raise an AssertionError, thus
    terminating the test.
    """
    if not settings.disable_debugger_detection and is_debugging():
        return
    __tracebackhide__ = True
    nthreads = len(threading.enumerate())
    terminal = item.config.get_terminal_writer()
    if nthreads > 1:
        terminal.sep("+", title="Timeout")
    dump_stacks(terminal)
    if nthreads > 1:
        terminal.sep("+", title="Timeout")
    pytest.fail("Timeout >%ss" % settings.timeout)


def timeout_timer(item, settings):
    """Dump stack of threads and call os._exit().

    This disables the capturemanager and dumps stdout and stderr.
    Then the stacks are dumped and os._exit(1) is called.
    """
    if not settings.disable_debugger_detection and is_debugging():
        return
    terminal = item.config.get_terminal_writer()
    try:
        capman = item.config.pluginmanager.getplugin("capturemanager")
        if capman:
            capman.suspend_global_capture(item)
            stdout, stderr = capman.read_global_capture()
        else:
            stdout, stderr = None, None
        terminal.sep("+", title="Timeout")
        caplog = item.config.pluginmanager.getplugin("_capturelog")
        if caplog and hasattr(item, "capturelog_handler"):
            log = item.capturelog_handler.stream.getvalue()
            if log:
                terminal.sep("~", title="Captured log")
                terminal.write(log)
        if stdout:
            terminal.sep("~", title="Captured stdout")
            terminal.write(stdout)
        if stderr:
            terminal.sep("~", title="Captured stderr")
            terminal.write(stderr)
        dump_stacks(terminal)
        terminal.sep("+", title="Timeout")
    except Exception:
        traceback.print_exc()
    finally:
        terminal.flush()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def dump_stacks(terminal):
    """Dump the stacks of all threads except the current thread."""
    current_ident = threading.current_thread().ident
    for thread_ident, frame in sys._current_frames().items():
        if thread_ident == current_ident:
            continue
        for t in threading.enumerate():
            if t.ident == thread_ident:
                thread_name = t.name
                break
        else:
            thread_name = "<unknown>"
        terminal.sep("~", title="Stack of %s (%s)" % (thread_name, thread_ident))
        terminal.write("".join(traceback.format_stack(frame)))
