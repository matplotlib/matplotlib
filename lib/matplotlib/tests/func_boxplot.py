from lib.matplotlib.cbook import boxplot_stats
from matplotlib.tests.conftest import boxplotlist
import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real

import contextlib
from collections import namedtuple
import datetime
from decimal import Decimal
from functools import partial
import inspect
import io
from itertools import product
import platform
from types import SimpleNamespace
import numpy as np
from numpy import ma
import pytest

def test_boxplot_settings():
    boxprops = {'color': 'red'}
    sym = ''
    flierprops = {'linestyle': 'none', 'marker': 'o', 'color': 'blue'}
    showfliers = True

    # Apply the boxplot settings
    if 'color' in boxprops:
        boxprops['edgecolor'] = boxprops.pop('color')
    if sym == '':
        flierprops = dict(linestyle='none', marker='', color='none')
        showfliers = False

    # Check that the boxprops were modified correctly
    assert boxprops == {'edgecolor': 'red'}

    # Check that the flierprops and showfliers were modified correctly
    assert flierprops == {'linestyle': 'none', 'marker': '', 'color': 'none'}
    assert showfliers == False

