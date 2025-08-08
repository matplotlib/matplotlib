import pytest
from matplotlib import _api
import matplotlib.pyplot as plt
import warnings


def test_set_aspect_anchor_deprecation():
    """Test that using the 'anchor' kwarg in set_aspect raises a warning."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # This should raise a warning
    with pytest.warns(_api.MatplotlibDeprecationWarning,
                     match="anchor"):
        ax.set_aspect('equal', anchor='C')

# This should NOT raise a warning
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ax.set_aspect('equal')
        # Filter out known, unrelated warnings if necessary
        # For now, let's assume no warnings should be present
    assert len(record) == 0
