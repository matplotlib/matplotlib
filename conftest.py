from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import inspect
import os
import pytest
import unittest

import matplotlib
matplotlib.use('agg')

from matplotlib import default_test_modules


IGNORED_TESTS = {
    'matplotlib': [],
}


def blacklist_check(path):
    """Check if test is blacklisted and should be ignored"""
    head, tests_dir = os.path.split(path.dirname)
    if tests_dir != 'tests':
        return True
    head, top_module = os.path.split(head)
    return path.purebasename in IGNORED_TESTS.get(top_module, [])


def whitelist_check(path):
    """Check if test is not whitelisted and should be ignored"""
    left = path.dirname
    last_left = None
    module_path = path.purebasename
    while len(left) and left != last_left:
        last_left = left
        left, tail = os.path.split(left)
        module_path = '.'.join([tail, module_path])
        if module_path in default_test_modules:
            return False
    return True


COLLECT_FILTERS = {
    'none': lambda _: False,
    'blacklist': blacklist_check,
    'whitelist': whitelist_check,
}


def is_nose_class(cls):
    """Check if supplied class looks like Nose testcase"""
    return any(name in ['setUp', 'tearDown']
               for name, _ in inspect.getmembers(cls))


def pytest_addoption(parser):
    group = parser.getgroup("matplotlib", "matplotlib custom options")

    group.addoption('--collect-filter', action='store',
                    choices=COLLECT_FILTERS, default='blacklist',
                    help='filter tests during collection phase')

    group.addoption('--no-pep8', action='store_true',
                    help='skip PEP8 compliance tests')


def pytest_configure(config):
    matplotlib._called_from_pytest = True
    matplotlib._init_tests()

    if config.getoption('--no-pep8'):
        default_test_modules.remove('matplotlib.tests.test_coding_standards')
        IGNORED_TESTS['matplotlib'] += 'test_coding_standards'


def pytest_unconfigure(config):
    matplotlib._called_from_pytest = False


def pytest_ignore_collect(path, config):
    if path.ext == '.py':
        collect_filter = config.getoption('--collect-filter')
        return COLLECT_FILTERS[collect_filter](path)


def pytest_pycollect_makeitem(collector, name, obj):
    if inspect.isclass(obj):
        if is_nose_class(obj) and not issubclass(obj, unittest.TestCase):
            # Workaround unittest-like setup/teardown names in pure classes
            setup = getattr(obj, 'setUp', None)
            if setup is not None:
                obj.setup_method = lambda self, _: obj.setUp(self)
            tearDown = getattr(obj, 'tearDown', None)
            if tearDown is not None:
                obj.teardown_method = lambda self, _: obj.tearDown(self)
            setUpClass = getattr(obj, 'setUpClass', None)
            if setUpClass is not None:
                obj.setup_class = obj.setUpClass
            tearDownClass = getattr(obj, 'tearDownClass', None)
            if tearDownClass is not None:
                obj.teardown_class = obj.tearDownClass

            return pytest.Class(name, parent=collector)
