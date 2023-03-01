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

import matplotlib
import matplotlib as mpl
from matplotlib import rc_context
from matplotlib._api import MatplotlibDeprecationWarning
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.font_manager as mfont_manager
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.projections.geo import HammerAxes
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import mpl_toolkits.axisartist as AA
from numpy.testing import (
    assert_allclose, assert_array_equal, assert_array_almost_equal)
from matplotlib.testing.decorators import (
    image_comparison, check_figures_equal, remove_ticks_and_titles)



def test_scatter_nofacecolor_issue():
    x = np.arange(10)
    fig, ax1 = plt.subplots()
    pc1 = plt.scatter(x, x, c=[(1.0,0.0,0.0)], facecolors='none')
    fig2, ax2 = plt.subplots()
    pc2 = plt.scatter(x, x, facecolors='none', edgecolors=(1.0,0.0,0.0))

    assert_array_equal(pc1.get_edgecolor(), pc2.get_edgecolor())


@check_figures_equal(extensions=["png"])
def test_scatter_nofacecolor_issue_example_plt(fig_test, fig_ref):
    x = np.arange(0, 10)
    norm = plt.Normalize(0, 10)
    cmap = mpl.colormaps['viridis'].resampled(10)
    cols = cmap(norm(x))

    fig1, ax1 = fig_ref.subplots()
    pc1 = ax1.scatter(x, x, c=x, facecolors='none')
    
    fig2, ax2 = fig_test.subplots()
    pc2 = ax2.scatter(x, x, facecolors='none', edgecolors=cols)



@check_figures_equal(extensions=["png"])
def test_scatter_nofacecolor_issue_example(fig_test, fig_ref):
    x = np.arange(0, 10)
    norm = plt.Normalize(0, 10)
    cmap = mpl.colormaps['viridis'].resampled(10)
    cols = cmap(norm(x))

    ax = fig_test.subplots()
    pc1 = ax.scatter(x, x, c=x, facecolors='none')
    ax = fig_ref.subplots()
    pc2 = ax.scatter(x, x, facecolors='none', edgecolors=cols)

    
