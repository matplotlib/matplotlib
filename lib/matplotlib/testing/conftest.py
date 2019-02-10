import pytest

import matplotlib
from matplotlib import cbook


def pytest_configure(config):
    matplotlib.use('agg', force=True)
    matplotlib._called_from_pytest = True
    matplotlib._init_tests()


def pytest_unconfigure(config):
    matplotlib._called_from_pytest = False


@pytest.fixture(autouse=True)
def mpl_test_settings(request):
    from matplotlib.testing.decorators import _cleanup_cm

    with _cleanup_cm():

        backend = None
        backend_marker = request.node.get_closest_marker('backend')
        if backend_marker is not None:
            assert len(backend_marker.args) == 1, \
                "Marker 'backend' must specify 1 backend."
            backend, = backend_marker.args
            prev_backend = matplotlib.get_backend()

        style = '_classic_test'  # Default of cleanup and image_comparison too.
        style_marker = request.node.get_closest_marker('style')
        if style_marker is not None:
            assert len(style_marker.args) == 1, \
                "Marker 'style' must specify 1 style."
            style, = style_marker.args

        matplotlib.testing.setup()
        if backend is not None:
            # This import must come after setup() so it doesn't load the
            # default backend prematurely.
            import matplotlib.pyplot as plt
            try:
                plt.switch_backend(backend)
            except ImportError as exc:
                # Should only occur for the cairo backend tests, if neither
                # pycairo nor cairocffi are installed.
                if 'cairo' in backend.lower():
                    pytest.skip("Failed to switch to backend {} ({})."
                                .format(backend, exc))
                else:
                    raise
        with cbook._suppress_matplotlib_deprecation_warning():
            matplotlib.style.use(style)
        try:
            yield
        finally:
            if backend is not None:
                plt.switch_backend(prev_backend)


@pytest.fixture
def mpl_image_comparison_parameters(request, extension):
    # This fixture is applied automatically by the image_comparison decorator.
    #
    # The sole purpose of this fixture is to provide an indirect method of
    # obtaining parameters *without* modifying the decorated function
    # signature. In this way, the function signature can stay the same and
    # pytest won't get confused.
    # We annotate the decorated function with any parameters captured by this
    # fixture so that they can be used by the wrapper in image_comparison.
    baseline_images, = request.node.get_closest_marker('baseline_images').args
    if baseline_images is None:
        # Allow baseline image list to be produced on the fly based on current
        # parametrization.
        baseline_images = request.getfixturevalue('baseline_images')

    func = request.function
    with cbook._setattr_cm(func.__wrapped__,
                           parameters=(baseline_images, extension)):
        yield


@pytest.fixture
def pd():
    """Fixture to import and configure pandas."""
    pd = pytest.importorskip('pandas')
    try:
        from pandas.plotting import (
            deregister_matplotlib_converters as deregister)
        deregister()
    except ImportError:
        pass
    return pd
