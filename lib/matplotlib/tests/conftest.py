from matplotlib.testing.conftest import (  # noqa
    mpl_test_settings, pytest_configure, pytest_unconfigure, pd, text_placeholders, xr)
import sys
import os
import pytest

def pytest_collection_modifyitems(config, items):
    """
    On Windows CI, force tests that spawn fresh Python processes to run serially.

    Rationale:
    Tests that spawn a new Python process (via sys.executable) compete for
    Desktop Heap / Handles on Windows. Running them in parallel causes
    resource exhaustion and timeouts.
    """
    # 1. Only apply on Windows
    if sys.platform != "win32":
        return

    # 2. Only apply on CI (let local devs run fast)
    if not os.environ.get("CI"):
        return

    # 3. The Specific "Python Spawning" Files
    # Identified via static analysis of `sys.executable` usage.
    serial_files = {
        'test_backends_interactive.py',
        'test_backend_webagg.py',
        'test_basic.py',
        'test_determinism.py',
        'test_font_manager.py',
        'test_matplotlib.py',
        'test_preprocess_data.py',
        'test_pyplot.py',
        'test_rcparams.py',
        'test_sphinxext.py',
        'test_texmanager.py',
    }

    # 4. Apply the Single Group (Strict Serialization)
    for item in items:
        # Check if the test belongs to one of the identified files
        if any(f in str(item.fspath) for f in serial_files):
            item.add_marker(pytest.mark.xdist_group(name="serial_python_spawn"))