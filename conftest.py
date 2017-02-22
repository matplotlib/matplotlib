from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


def pytest_addoption(parser):
    group = parser.getgroup("matplotlib", "matplotlib custom options")
    group.addoption("--conversion-cache-max-size", action="store",
                    help="conversion cache maximum size in bytes")
    group.addoption("--conversion-cache-report-misses",
                    action="store_true",
                    help="report conversion cache misses")
