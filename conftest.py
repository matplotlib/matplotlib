from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import inspect
import pytest

import matplotlib
matplotlib.use('agg')

from matplotlib.testing.decorators import ImageComparisonTest


def pytest_configure(config):
    matplotlib._called_from_pytest = True


def pytest_unconfigure(config):
    matplotlib._called_from_pytest = False


def pytest_pycollect_makeitem(collector, name, obj):
    if inspect.isclass(obj):
        if issubclass(obj, ImageComparisonTest):
            # Workaround `image_compare` decorator as it returns class
            # instead of function and this confuses pytest because it crawls
            # original names and sees 'test_*', but not 'Test*' in that case
            return pytest.Class(name, parent=collector)
