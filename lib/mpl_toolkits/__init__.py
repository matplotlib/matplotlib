import sys
from matplotlib import Verbose
from matplotlib import rcParams
from matplotlib import __version__
from matplotlib import use

verbose = Verbose()

try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    pass # must not have setuptools

toolkits_test_modules = [
    'mpl_toolkits.tests.test_axesgrid1',
    ]

def test(verbosity=1):
    """run the matplotlib toolkits test suite"""
    old_backend = rcParams['backend']
    try:
        use('agg')
        import nose
        import nose.plugins.builtin
        from matplotlib.testing.noseclasses import KnownFailure
        from nose.plugins.manager import PluginManager

        # store the old values before overriding
        plugins = []
        plugins.append( KnownFailure() )
        plugins.extend( [plugin() for plugin in nose.plugins.builtin.plugins] )

        manager = PluginManager(plugins = plugins)
        config = nose.config.Config(verbosity = verbosity, plugins = manager)

        success = nose.run( defaultTest = toolkits_test_modules,
                            config=config,
                            )
    finally:
        if old_backend.lower() != 'agg':
            use(old_backend)

    return success

test.__test__ = False # nose: this function is not a test

verbose.report('matplotlib version %s'%__version__)
verbose.report('verbose.level %s'%verbose.level)
verbose.report('interactive is %s'%rcParams['interactive'])
verbose.report('platform is %s'%sys.platform)
verbose.report('loaded modules: %s'%sys.modules.iterkeys(), 'debug')
