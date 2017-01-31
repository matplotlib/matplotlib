from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import inspect
import os
import pytest
import unittest

import matplotlib
matplotlib.use('agg')


def pytest_configure(config):
    matplotlib._called_from_pytest = True
    matplotlib._init_tests()


def pytest_unconfigure(config):
    matplotlib._called_from_pytest = False
