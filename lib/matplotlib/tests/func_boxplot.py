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
import dateutil.tz
import numpy as np
from numpy import ma
from cycler import cycler
import pytest

boxprops = {'color': 'red'}
boxplot = BoxPlot()
boxplot.boxplot(x=[1,2,3,4], boxprops=boxprops)
assert 'edgecolor' in boxprops
assert boxprops['edgecolor'] == 'red'
assert 'color' not in boxprops


sym = ''
flierprops = {'linestyle': 'dashed', 'marker': '+', 'color': 'blue'}
boxplot = BoxPlot()
boxplot.boxplot(x=[1,2,3,4], sym=sym, flierprops=flierprops)
assert 'linestyle' in flierprops
assert flierprops['linestyle'] == 'none'
assert 'marker' in flierprops
assert flierprops['marker'] == ''
assert 'color' in flierprops
assert flierprops['color'] == 'none'
assert showfliers is False
