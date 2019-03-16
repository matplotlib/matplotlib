from collections import OrderedDict
from contextlib import contextmanager
from distutils.version import LooseVersion
import gc
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import matplotlib as mpl
from matplotlib import pyplot as plt, style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION
from matplotlib.testing.decorators import check_figures_equal


PARAM = 'image.cmap'
VALUE = 'pink'
DUMMY_SETTINGS = {PARAM: VALUE}


@contextmanager
def temp_style(style_name, settings=None):
    """Context manager to create a style sheet in a temporary directory."""
    if not settings:
        settings = DUMMY_SETTINGS
    temp_file = '%s.%s' % (style_name, STYLE_EXTENSION)
    try:
        with TemporaryDirectory() as tmpdir:
            # Write style settings to file in the tmpdir.
            Path(tmpdir, temp_file).write_text(
                "\n".join("{}: {}".format(k, v) for k, v in settings.items()))
            # Add tmpdir to style path and reload so we can access this style.
            USER_LIBRARY_PATHS.append(tmpdir)
            style.reload_library()
            yield
    finally:
        style.reload_library()


def test_invalid_rc_warning_includes_filename(capsys):
    SETTINGS = {'foo': 'bar'}
    basename = 'basename'
    with temp_style(basename, SETTINGS):
        # style.reload_library() in temp_style() triggers the warning
        pass
    assert basename in capsys.readouterr().err


def test_available():
    with temp_style('_test_', DUMMY_SETTINGS):
        assert '_test_' in style.available


def test_use():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            assert mpl.rcParams[PARAM] == VALUE


@pytest.mark.network
def test_use_url():
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('https://gist.github.com/adrn/6590261/raw'):
            assert mpl.rcParams['axes.facecolor'] == "#adeade"


def test_context():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            assert mpl.rcParams[PARAM] == VALUE
    # Check that this value is reset after the exiting the context.
    assert mpl.rcParams[PARAM] == 'gray'


def test_context_with_dict():
    original_value = 'gray'
    other_value = 'blue'
    mpl.rcParams[PARAM] = original_value
    with style.context({PARAM: other_value}):
        assert mpl.rcParams[PARAM] == other_value
    assert mpl.rcParams[PARAM] == original_value


def test_context_with_dict_after_namedstyle():
    # Test dict after style name where dict modifies the same parameter.
    original_value = 'gray'
    other_value = 'blue'
    mpl.rcParams[PARAM] = original_value
    with temp_style('test', DUMMY_SETTINGS):
        with style.context(['test', {PARAM: other_value}]):
            assert mpl.rcParams[PARAM] == other_value
    assert mpl.rcParams[PARAM] == original_value


def test_context_with_dict_before_namedstyle():
    # Test dict before style name where dict modifies the same parameter.
    original_value = 'gray'
    other_value = 'blue'
    mpl.rcParams[PARAM] = original_value
    with temp_style('test', DUMMY_SETTINGS):
        with style.context([{PARAM: other_value}, 'test']):
            assert mpl.rcParams[PARAM] == VALUE
    assert mpl.rcParams[PARAM] == original_value


def test_context_with_union_of_dict_and_namedstyle():
    # Test dict after style name where dict modifies the a different parameter.
    original_value = 'gray'
    other_param = 'text.usetex'
    other_value = True
    d = {other_param: other_value}
    mpl.rcParams[PARAM] = original_value
    mpl.rcParams[other_param] = (not other_value)
    with temp_style('test', DUMMY_SETTINGS):
        with style.context(['test', d]):
            assert mpl.rcParams[PARAM] == VALUE
            assert mpl.rcParams[other_param] == other_value
    assert mpl.rcParams[PARAM] == original_value
    assert mpl.rcParams[other_param] == (not other_value)


def test_context_with_badparam():
    original_value = 'gray'
    other_value = 'blue'
    d = OrderedDict([(PARAM, original_value), ('badparam', None)])
    with style.context({PARAM: other_value}):
        assert mpl.rcParams[PARAM] == other_value
        x = style.context([d])
        with pytest.raises(KeyError):
            with x:
                pass
        assert mpl.rcParams[PARAM] == other_value


@pytest.mark.parametrize('equiv_styles',
                         [('mpl20', 'default'),
                          ('mpl15', 'classic')],
                         ids=['mpl20', 'mpl15'])
def test_alias(equiv_styles):
    rc_dicts = []
    for sty in equiv_styles:
        with style.context(sty):
            rc_dicts.append(dict(mpl.rcParams))

    rc_base = rc_dicts[0]
    for nm, rc in zip(equiv_styles[1:], rc_dicts[1:]):
        assert rc_base == rc


def test_xkcd_no_cm():
    assert mpl.rcParams["path.sketch"] is None
    plt.xkcd()
    assert mpl.rcParams["path.sketch"] == (1, 100, 2)
    gc.collect()
    assert mpl.rcParams["path.sketch"] == (1, 100, 2)


def test_xkcd_cm():
    assert mpl.rcParams["path.sketch"] is None
    with plt.xkcd():
        assert mpl.rcParams["path.sketch"] == (1, 100, 2)
    assert mpl.rcParams["path.sketch"] is None


@check_figures_equal(extensions=["png"])
def test_seaborn_style(fig_test, fig_ref):
    seaborn = pytest.importorskip('seaborn')
    if LooseVersion(seaborn.__version__) < LooseVersion('0.9'):
        pytest.skip('seaborn style comparisons need at least seaborn 0.9')
    return seaborn

    def make_plot(fig):
        ax1 = fig.add_subplot(121)
        x = np.linspace(0, 14, 100)
        for i in range(1, 7):
            ax1.plot(x, np.sin(x + i * .5) * (7 - i) + 10)
        heights = [1, 3, 8, 4, 2]
        ax1.bar(range(5), heights)
        ax1.bar(range(5, 10), heights)
        ax1.bar(range(10, 15), heights)
        ax1.set_title('lines and bars')

        ax2 = fig.add_subplot(122)
        x = np.tile(np.linspace(0, 1, 20), 2).reshape(-1, 2)
        x[:, 1] /= 2
        x[0, 1] = 0.8
        ax2.boxplot(x)
        x = np.linspace(0.8, 2.2, 8)
        ax2.plot(x, np.exp(-x), 'o')
        ax2.set_title('markers and boxes')

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('seaborn')
    make_plot(fig_test)

    mpl.rcParams.update(mpl.rcParamsDefault)
    seaborn.set()
    make_plot(fig_ref)


@pytest.mark.parametrize(
    'style_name', ['dark', 'darkgrid', 'ticks', 'white', 'whitegrid'])
def test_seaborn_styles(seaborn, style_name):
    """
    Test that after applying a style the style-related rcParams are identical
    to the ones seaborn will define seaborn.rcmod.axes_style().
    """
    import seaborn.rcmod
    style_dict = seaborn.rcmod.axes_style(style_name)

    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.style.use(f'seaborn-{style_name}')
    for key, val in style_dict.items():
        if key == 'image.cmap':
            continue  # we don't have the seaborn color maps
        assert mpl.rcParams[key] == val, \
            f"Style '{style_name}' deviates in key '{key}'"


@pytest.mark.parametrize(
    'context_name', ['paper', 'notebook', 'talk', 'poster'])
def test_seaborn_context(seaborn, context_name):
    """
    Test that after applying a style the context-related rcParams are identical
    to the ones seaborn will define seaborn.rcmod.plotting_context().
    """
    import seaborn.rcmod
    mpl.style.use(f'seaborn-{context_name}')
    for key, val in seaborn.rcmod.plotting_context(context_name).items():
        assert mpl.rcParams[key] == pytest.approx(val), \
            f"Context '{context_name}' deviates in key '{key}'"


def test_seaborn_palettes(seaborn):
    """
    Test that after applying a style rcParams['axes.prop_cycle'] is set to the
    values defined in seaborn.palettes.SEABORN_PALETTES.
    Test that rcParams['patch.facecolor'] is set to the first color in the
    cycle.
    """
    from seaborn.palettes import SEABORN_PALETTES
    for name, colors in SEABORN_PALETTES.items():
        # seaborn-dark was already used for the dark style.
        # Therefore, Matplotlib uses seaborn-dark-palette as mpl style name.
        deviating_mpl_style_name = {
            'dark': 'dark-palette',
            'dark6': 'dark6-palette',
        }
        name = deviating_mpl_style_name.get(name, name)
        mpl.style.use(f'seaborn-{name}')
        assert mpl.rcParams['axes.prop_cycle'].by_key()['color'] == colors
        assert mpl.rcParams['patch.facecolor'] == colors[0]
