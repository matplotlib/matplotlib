import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.decorators import check_figures_equal


def test_parallel_coordinates_basic():
    """Test basic parallel coordinates plot with numpy array."""
    np.random.seed(42)
    data = np.random.randn(20, 4)
    fig, ax = plt.subplots()
    collections = ax.parallel_coordinates(data)
    assert len(collections) == 1
    assert len(ax.collections) == 1
    assert len(ax.xaxis.get_ticklabels()) == 4


def test_parallel_coordinates_class_column():
    """Test with class_column parameter."""
    np.random.seed(42)
    X = np.random.randn(40, 3)
    classes = np.array(['cat'] * 20 + ['dog'] * 20)
    data = np.column_stack([X, classes])
    fig, ax = plt.subplots()
    collections = ax.parallel_coordinates(data, class_column=3, cols=[0, 1, 2])
    assert len(collections) == 2
    assert ax.get_legend() is not None


def test_parallel_coordinates_color_cmap():
    """Test color and cmap parameters."""
    np.random.seed(42)
    data = np.random.randn(20, 4)
    fig, ax = plt.subplots()
    collections = ax.parallel_coordinates(data, color='red')
    assert len(collections) == 1

    fig2, ax2 = plt.subplots()
    collections2 = ax2.parallel_coordinates(data, cmap='viridis')
    assert len(collections2) == 1


def test_parallel_coordinates_style_params():
    """Test alpha, linewidth, linestyle parameters."""
    np.random.seed(42)
    fig, ax = plt.subplots()
    data = np.random.randn(20, 4)
    collections = ax.parallel_coordinates(data, alpha=0.3, linewidth=2,
                                           linestyle='--')
    assert len(collections) == 1


def test_parallel_coordinates_too_few_dims():
    """Test that 1D data raises ValueError."""
    fig, ax = plt.subplots()
    try:
        ax.parallel_coordinates(np.random.randn(10, 1))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_parallel_coordinates_invalid_data():
    """Test that 1-D data raises ValueError."""
    fig, ax = plt.subplots()
    try:
        ax.parallel_coordinates(np.random.randn(10))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
