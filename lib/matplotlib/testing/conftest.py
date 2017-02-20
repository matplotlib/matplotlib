from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest

import matplotlib


def pytest_configure(config):
    matplotlib.use('agg')
    matplotlib._called_from_pytest = True
    matplotlib._init_tests()


def pytest_unconfigure(config):
    matplotlib._called_from_pytest = False


@pytest.fixture(autouse=True)
def mpl_test_settings(request):
    from matplotlib.testing.decorators import _do_cleanup

    original_units_registry = matplotlib.units.registry.copy()
    original_settings = matplotlib.rcParams.copy()

    backend = None
    backend_marker = request.keywords.get('backend')
    if backend_marker is not None:
        assert len(backend_marker.args) == 1, \
            "Marker 'backend' must specify 1 backend."
        backend = backend_marker.args[0]
        prev_backend = matplotlib.get_backend()

    style = 'classic'
    style_marker = request.keywords.get('style')
    if style_marker is not None:
        assert len(style_marker.args) == 1, \
            "Marker 'style' must specify 1 style."
        style = style_marker.args[0]

    matplotlib.testing.setup()
    if backend is not None:
        # This import must come after setup() so it doesn't load the default
        # backend prematurely.
        import matplotlib.pyplot as plt
        plt.switch_backend(backend)
    matplotlib.style.use(style)
    try:
        yield
    finally:
        if backend is not None:
            import matplotlib.pyplot as plt
            plt.switch_backend(prev_backend)
        _do_cleanup(original_units_registry,
                    original_settings)
