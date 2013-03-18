"""
Tests specific to the lines module.
"""

from nose.tools import assert_true
from timeit import repeat
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup

@cleanup
def test_invisible_Line_rendering():
    """
    Github issue #1256 identified a bug in Line.draw method
    
    Despite visibility attribute set to False, the draw method was not
    returning early enough and some pre-rendering code was executed
    though not necessary.

    Consequence was an excessive draw time for invisible Line instances
    holding a large number of points (Npts> 10**6)
    """
    # Creates big x and y data:
    N = 10**7
    x = np.linspace(0,1,N)
    y = np.random.normal(size=N)

    # Create a plot figure:
    fig = plt.figure()
    ax = plt.subplot(111)

    # Create a "big" Line instance:
    l = mpl.lines.Line2D(x,y)
    l.set_visible(False)
    # but don't add it to the Axis instance `ax`

    # [here Interactive panning and zooming is pretty responsive]
    # Time the canvas drawing:
    t_no_line = min(repeat(fig.canvas.draw, number=1, repeat=3))
    # (gives about 25 ms)

    # Add the big invisible Line:
    ax.add_line(l)

    # [Now interactive panning and zooming is very slow]
    # Time the canvas drawing:
    t_unvisible_line = min(repeat(fig.canvas.draw, number=1, repeat=3))
    # gives about 290 ms for N = 10**7 pts

    slowdown_factor = (t_unvisible_line/t_no_line)
    slowdown_threshold = 2 # trying to avoid false positive failures
    assert_true(slowdown_factor < slowdown_threshold)
