import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d


def test_line3dcollection_autolim_ragged():
    """Test Line3DCollection with autolim=True and lines of different lengths."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Create lines with different numbers of points (ragged arrays)
    edges = [
        [(0, 0, 0), (1, 1, 1), (2, 2, 2)],  # 3 points
        [(0, 1, 0), (1, 2, 1)],             # 2 points
        [(1, 0, 1), (2, 1, 2), (3, 2, 3), (4, 3, 4)]  # 4 points
    ]

    # This should not raise an exception with the fix
    collections = ax.add_collection3d(art3d.Line3DCollection(edges), autolim=True)

    # Check that limits were computed correctly with margins
    # The limits should include all points with default margins
    assert np.allclose(ax.get_xlim3d(), (-0.08333333333333333, 4.083333333333333))
    assert np.allclose(ax.get_ylim3d(), (-0.0625, 3.0625))
    assert np.allclose(ax.get_zlim3d(), (-0.08333333333333333, 4.083333333333333))

    plt.close(fig)
