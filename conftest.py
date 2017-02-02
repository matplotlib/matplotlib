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
        IGNORED_TESTS['matplotlib'] += 'test_coding_standards'


def pytest_unconfigure(config):
    matplotlib._called_from_pytest = False


def pytest_ignore_collect(path, config):
    if path.ext == '.py':
        collect_filter = config.getoption('--collect-filter')
        return COLLECT_FILTERS[collect_filter](path)
