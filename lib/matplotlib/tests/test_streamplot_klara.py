import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.transforms as mtransforms
import matplotlib.lines as mlines 

def test_streamplot_zorder_not_none(): 
    zord = mlines.Line2D.zorder 
    with pytest.raises(ValueError) as excinfo:
        plt.streamplot(np.arange(3), np.arange(3),
                       np.full((3, 3), np.nan), np.full((3, 3), np.nan),
                       color=np.random.rand(3, 3), zorder=zord)
    
def test_streamplot_color_grid_shapes_not_matching():
    with pytest.raises(ValueError) as excinfo:
        plt.streamplot(np.arange(3), np.arange(3),
                       np.full((3, 3), np.nan), np.full((3, 3), np.nan),
                       color=np.random.rand(5, 5))
    assert str(excinfo.value) == "If 'color' is given, it must match the shape of the (x, y) grid"

def test_streamplot_linewidth_grid_shape_not_matching():
    with pytest.raises(ValueError) as excinfo:
        plt.streamplot(np.arange(3), np.arange(3),
                       np.full((3, 3), np.nan), np.full((3, 3), np.nan),
                       color=np.random.rand(3, 3), linewidth= np.full((5, 5), np.nan))
    assert str(excinfo.value) == "If 'linewidth' is given, it must match the shape of the (x, y) grid"

def test_streamplot_linewidth_check_u_grid_shape_not_matching():
    with pytest.raises(ValueError) as excinfo:
        plt.streamplot(np.arange(3), np.arange(3),
                       np.full((5, 5), np.nan), np.full((3, 3), np.nan),
                       color=np.random.rand(3, 3))
    assert str(excinfo.value) == "'u' and 'v' must match the shape of the (x, y) grid"

def test_streamplot_():
    with pytest.raises(ValueError) as excinfo:
        plt.streamplot(np.arange(3), np.arange(3),
                       np.full((3, 3), np.nan), np.full((3, 3), np.nan),
                       color=np.random.rand(3, 3), start_points=np.full((3, 3), 10))
    assert str(excinfo.value) == "Starting point (10, 10) outside of data boundaries"
