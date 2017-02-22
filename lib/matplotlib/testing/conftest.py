from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from six.moves import reduce
import pytest

import matplotlib
from matplotlib.testing import _conversion_cache as ccache


def pytest_configure(config):
    matplotlib.use('agg')
    matplotlib._called_from_pytest = True
    matplotlib._init_tests()

    max_size = config.getoption('--conversion-cache-max-size')
    if max_size is not None:
        ccache._conversion_cache = \
            ccache._ConversionCache(max_size=int(max_size))
    else:
        ccache._conversion_cache = ccache._ConversionCache()
    if config.pluginmanager.hasplugin('xdist'):
        config.pluginmanager.register(DeferPlugin())


def pytest_unconfigure(config):
    ccache._conversion_cache.expire()
    matplotlib._called_from_pytest = False


def pytest_sessionfinish(session):
    if hasattr(session.config, 'slaveoutput'):
        session.config.slaveoutput['cache-report'] = \
            ccache._conversion_cache.report()


def pytest_terminal_summary(terminalreporter):
    tr = terminalreporter
    if hasattr(tr.config, 'cache_reports'):
        reports = tr.config.cache_reports
        data = {'hits': reduce(lambda x, y: x.union(y),
                               (rep['hits'] for rep in reports)),
                'gets': reduce(lambda x, y: x.union(y),
                               (rep['gets'] for rep in reports))}
    else:
        data = ccache._conversion_cache.report()
    tr.write_sep('=', 'Image conversion cache report')
    tr.write_line('Hit rate: %d/%d' % (len(data['hits']), len(data['gets'])))
    if tr.config.getoption('--conversion-cache-report-misses'):
        tr.write_line('Missed files:')
        for filename in sorted(data['gets'].difference(data['hits'])):
            tr.write_line('  %s' % filename)


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


class DeferPlugin(object):
    def pytest_testnodedown(self, node, error):
        if not hasattr(node.config, 'cache_reports'):
            node.config.cache_reports = []
        node.config.cache_reports.append(node.slaveoutput['cache-report'])
