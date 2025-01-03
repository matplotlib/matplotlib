from matplotlib.testing.conftest import (  # noqa
    mpl_test_settings, pytest_configure, pytest_unconfigure, pd, xr)
import pytest

@pytest.fixture
def mock_axes():
    class MockAxes:
        def get_title_top(self):
            return 1.0
    return MockAxes()
