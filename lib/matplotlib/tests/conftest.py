from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest

import matplotlib


@pytest.fixture(autouse=True)
def mpl_test_settings(request):
    from matplotlib.testing.decorators import _do_cleanup

    original_units_registry = matplotlib.units.registry.copy()
    original_settings = matplotlib.rcParams.copy()

    style = 'classic'
    style_marker = request.keywords.get('style')
    if style_marker is not None:
        assert len(style_marker.args) == 1, \
            "Marker 'style' must specify 1 style."
        style = style_marker.args[0]

    matplotlib.style.use(style)
    try:
        yield
    finally:
        _do_cleanup(original_units_registry,
                    original_settings)
