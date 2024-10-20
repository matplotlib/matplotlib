from __future__ import annotations

import atexit
import os
import os.path
import sys

import pytest
import pyvirtualdisplay

xvfb_instance = None


def shutdown_xvfb() -> None:
    if xvfb_instance is not None:
        xvfb_instance.stop()


# This needs to be done as early as possible (before importing QtWebEngine for
# example), so that Xvfb gets shut down as late as possible.
atexit.register(shutdown_xvfb)


def is_xdist_master(config: pytest.Config) -> bool:
    return config.getoption("dist", "no") != "no" and not os.environ.get(
        "PYTEST_XDIST_WORKER"
    )


def has_executable(name: str) -> bool:
    # http://stackoverflow.com/a/28909933/2085149
    return any(
        os.access(os.path.join(path, name), os.X_OK)
        for path in os.environ["PATH"].split(os.pathsep)
    )


class XvfbExitedError(Exception):
    pass


class Xvfb:
    def __init__(self, config: pytest.Config) -> None:
        self.width = int(config.getini("xvfb_width"))
        self.height = int(config.getini("xvfb_height"))
        self.colordepth = int(config.getini("xvfb_colordepth"))
        self.args = config.getini("xvfb_args") or []
        self.xauth = config.getini("xvfb_xauth")
        self.backend = config.getoption("--xvfb-backend")
        self.display = None
        self._virtual_display = None

    def start(self) -> None:
        self._virtual_display = pyvirtualdisplay.Display(  # type: ignore[attr-defined]
            backend=self.backend,
            size=(self.width, self.height),
            color_depth=self.colordepth,
            use_xauth=self.xauth,
            extra_args=self.args,
        )
        assert self._virtual_display is not None  # mypy
        self._virtual_display.start()
        self.display = self._virtual_display.display
        assert self._virtual_display.is_alive()

    def stop(self) -> None:
        if self.display is not None:  # starting worked
            self._virtual_display.stop()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("xvfb")
    group.addoption("--no-xvfb", action="store_true", help="Disable Xvfb for tests.")
    group.addoption(
        "--xvfb-backend",
        action="store",
        choices=["xvfb", "xvnc", "xephyr"],
        help="Use Xephyr or Xvnc instead of Xvfb for tests. Will be ignored if --no-xvfb is given.",
    )

    parser.addini("xvfb_width", "Width of the Xvfb display", default="800")
    parser.addini("xvfb_height", "Height of the Xvfb display", default="600")
    parser.addini("xvfb_colordepth", "Color depth of the Xvfb display", default="16")
    parser.addini("xvfb_args", "Additional arguments for Xvfb", type="args")
    parser.addini(
        "xvfb_xauth",
        "Generate an Xauthority token for Xvfb. Needs xauth.",
        default=False,
        type="bool",
    )


def pytest_configure(config: pytest.Config) -> None:
    global xvfb_instance

    no_xvfb = config.getoption("--no-xvfb") or is_xdist_master(config)
    backend = config.getoption("--xvfb-backend")

    if no_xvfb:
        pass
    elif backend is None and not has_executable("Xvfb"):
        # soft fail
        if sys.platform.startswith("linux") and "DISPLAY" in os.environ:
            print(
                "pytest-xvfb could not find Xvfb. "
                "You can install it to prevent windows from being shown."
            )
    elif (
        backend == "xvfb"
        and not has_executable("Xvfb")
        or backend == "xvnc"
        and not has_executable("Xvnc")
        or backend == "xephyr"
        and not has_executable("Xephyr")
    ):
        raise pytest.UsageError(f"xvfb backend {backend} requested but not installed.")
    else:
        xvfb_instance = Xvfb(config)
        xvfb_instance.start()

    config.addinivalue_line("markers", "no_xvfb: Skip test when using Xvfb")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("no_xvfb") and xvfb_instance is not None:
            skipif_marker = pytest.mark.skipif(True, reason="Skipped with Xvfb")
            item.add_marker(skipif_marker)


@pytest.fixture(scope="session")
def xvfb() -> Xvfb | None:
    return xvfb_instance
