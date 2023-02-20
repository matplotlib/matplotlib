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


"""
Tests the creation of a log log horizontal histogram
with the same coordinates and opposite orientations.
Log is enabled for both axes to take a new branch within
the function.

Precondition: There exists an x-vector with with three integers
              in two calls of the hist() function where one has
              a horizontal orientation and one vertical.
Postcondition: The x- and y-axis limits are equivalent. 
"""
def test_hist_log_orientation():
    fig, axs = plt.subplots(2)
    axs[0].hist([0, 0, 1], orientation='horizontal', align='right', log=True)
    axs[0].set_xscale("log")
    axs[1].hist([0, 0, 1], orientation='vertical', align='right', log=True)
    axs[1].set_xscale("log")
    fig.canvas.draw()
    assert axs[0].get_xlim() == axs[1].get_ylim()


"""
Tests the creation of a log log horizontal histogram
with the same coordinates and opposite orientations.
Density is enabled to reach a new branch within the function.

Precondition: There exists an x-vector with with three integers
              in two calls of the hist() function where one has
              a horizontal orientation and one vertical.
Postcondition: The x- and y-axis limits are equivalent. 
"""
def test_hist_log_orientation_density():
    fig, axs = plt.subplots(2)
    #axs[0].hist([0, 0, 1], orientation='horizontal', align='right', log=True)
    axs[0].hist([0, 0, 1], orientation='horizontal', density=True, log=True, cumulative=True, stacked=True)
    axs[0].set_xscale("log")
    axs[1].hist([0, 0, 1], orientation='vertical', density=True, log=True, cumulative=True, stacked=True)
    axs[1].set_xscale("log")
    fig.canvas.draw()
    assert axs[0].get_xlim() == axs[1].get_ylim()


"""
Tests the creation of a log log horizontal histogram
with the same coordinates and opposite orientations.
Cumulative is enabled to reach a new branch within the function.

Precondition: There exists an x-vector with with three integers
              in two calls of the hist() function where one has
              a horizontal orientation and one vertical.
Postcondition: The x- and y-axis limits are equivalent. 
"""
def test_hist_orientation_density_left():
    fig, axs = plt.subplots(2)
    #axs[0].hist([0, 0, 1], orientation='horizontal', align='right', log=True)
    axs[0].hist([0, 0, 1], orientation='horizontal', density=True, cumulative=True, stacked=True, align='left')
    axs[1].hist([0, 0, 1], orientation='vertical', density=True, cumulative=True, stacked=True, align='left')
    fig.canvas.draw()
    assert axs[0].get_xlim() == axs[1].get_ylim()



"""
_axes.py Hist(), reach branch #15
Tests the creation of a log log horizontal histogram
with too many colors that raises an exception to reach 
a new branch.

Precondition: There exists a valid x-vector with with three integers
              and a colors vector that includes too many colors, 6, for
              only one dataset. 
Postcondition: The hist function raises a ValueError with the correct
               string within that branch.
"""
def test_hist_orientation_density_right():
    fig, axs = plt.subplots(2)
    #axs[0].hist([0, 0, 1], orientation='horizontal', align='right', log=True)
    with pytest.raises(ValueError, match=(f"The 'color' keyword argument must have one "
                                 f"color per dataset, but {1} datasets and "
                                 f"{6} colors were provided")):
        axs[0].hist([0, 0, 1], orientation='horizontal', density=True, cumulative=True, stacked=True, color=['green', 'red', 'blue', 'black', 'white', 'pink'])
    