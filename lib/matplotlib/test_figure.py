import pytest
from PIL import Image


import matplotlib as mpl
from matplotlib import cbook, rcParams
from matplotlib._api.deprecation import MatplotlibDeprecationWarning
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubplotParams
from matplotlib.ticker import AutoMinorLocator, FixedFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig.set_figsize(2, 4)
assert fig.get_figsize()[0] == 2
assert fig.get_figsize()[1] == 4
 fig = Figure(layout='tight')
    with pytest.warns(UserWarning, match="Figure parameters 'layout'=='tight' "
                      "and 'tight_layout'==False cannot"):
        fig.set_layout(layout='tight', tight_layout=False)
    assert_is_tight(fig)

    with pytest.warns(UserWarning, match="Figure parameters "
                      "'layout'=='constrained' and "
                      "'constrained_layout'==False cannot"):
        fig.set_layout(layout='constrained', constrained_layout=False)
    assert_is_constrained(fig)

    with pytest.warns(UserWarning, match="Figure parameters "
                      "'layout'=='tight' and "
                      "'constrained_layout'!=False cannot"):
        fig.set_layout(layout='tight', constrained_layout=True)
    assert_is_tight(fig)

    with pytest.warns(UserWarning, match="Figure parameters "
                      "'layout'=='constrained' and "
                      "'tight_layout'!=False cannot"):
        fig.set_layout(layout='constrained', tight_layout=True)
    assert_is_constrained(fig)

    with pytest.warns(UserWarning, match="Figure parameters "
                      "'layout'=='tight' and "
                      "'constrained_layout'!=False cannot"):
        fig.set_layout(layout='tight', constrained_layout={'pad': 1})
    assert_is_tight(fig)
    with pytest.warns(UserWarning, match="Figure parameters "
                      "'layout'=='constrained' and "
                      "'tight_layout'!=False cannot"):
        fig.set_layout(layout='constrained', tight_layout={'pad': 1})
    assert_is_constrained(fig)

    with pytest.warns(Warning) as warninfo:
        fig.set_layout(layout='tight',
                       tight_layout=False,
                       constrained_layout=True)
    warns = {(warn.category, warn.message.args[0]) for warn in warninfo}
    expected = {
        (UserWarning, "Figure parameters 'layout'=='tight' "
         "and 'tight_layout'==False cannot be used together. "
         "Please use 'layout' only."),
        (UserWarning, "Figure parameters 'layout'=='tight' "
         "and 'constrained_layout'!=False cannot be used together. "
         "Please use 'layout' only.")}
    assert_is_tight(fig)
    assert warns == expected
    with pytest.warns(Warning) as warninfo:
        fig.set_layout(layout='constrained',
                       tight_layout=True,
                       constrained_layout=False)
    warns = {(warn.category, warn.message.args[0]) for warn in warninfo}
    expected = {
        (UserWarning, "Figure parameters 'layout'=='constrained' "
         "and 'tight_layout'!=False cannot be used together. "
         "Please use 'layout' only."),
        (UserWarning, "Figure parameters 'layout'=='constrained' "
         "and 'constrained_layout'==False cannot be used together. "
         "Please use 'layout' only.")}
    assert_is_constrained(fig)
    assert warns == expected

    with pytest.raises(ValueError,
                       match="Cannot set 'tight_layout' and "
                       "'constrained_layout' simultaneously."):
        fig = Figure(tight_layout={'w': 1}, constrained_layout={'w_pad': 1})
    with pytest.raises(ValueError,
                       match="Cannot set 'tight_layout' and "
                       "'constrained_layout' simultaneously."):
        fig = Figure(tight_layout=True, constrained_layout={'w_pad': 1})
    with pytest.raises(ValueError,
                       match="Cannot set 'tight_layout' and "
                       "'constrained_layout' simultaneously."):
        fig = Figure(tight_layout=True, constrained_layout=True)


def test_set_subplotpars():
    subplotparams_keys = ["left", "bottom", "right", "top", "wspace", "hspace"]
    fig = plt.figure()
    subplotparams = fig.get_subplotpars()
    test_dict = {}
    default_dict = {}
    for key in subplotparams_keys:
        attr = getattr(subplotparams, key)
        assert attr == mpl.rcParams[f"figure.subplot.{key}"]
        default_dict[key] = attr
        test_dict[key] = attr * 2

    subplotparams.update(left=test_dict['left'])
    assert fig.get_subplotpars().left == test_dict['left']

    fig.subplots_adjust(**default_dict)
    assert fig.get_subplotpars().left == default_dict['left']

    fig.set_subplotpars(test_dict)
    for key, value in test_dict.items():
        assert getattr(fig.get_subplotpars(), key) == value

    test_subplotparams = SubplotParams()
    fig.set_subplotpars(test_subplotparams)
    for key, value in default_dict.items():
        assert getattr(fig.get_subplotpars(), key) == value

    fig.set_subplotpars(test_dict)
    for key, value in test_dict.items():
        assert getattr(fig.get_subplotpars(), key) == value

    test_dict['foo'] = 'bar'
    with pytest.warns(UserWarning,
                      match="'foo' is not a valid key for set_subplotpars;"
                      " this key was ignored"):
        fig.set_subplotpars(test_dict)

    with pytest.raises(TypeError,
                       match="subplotpars must be a dictionary of "
                       "keyword-argument pairs or "
                       "an instance of SubplotParams()"):
        fig.set_subplotpars(['foo'])

    fig.set_subplotpars({})
    with pytest.raises(AttributeError):  # test_dict['foo'] = 'bar'
        # but fig.get_subplotpars().foo should be invalid
        for key, value in test_dict.items():
            assert getattr(fig.get_subplotpars(), key) == value

def test_fig_get_set():
    varnames = filter(lambda var: var not in ['self', 'kwargs', 'args'],
                      Figure.__init__.__code__.co_varnames)
    fig = plt.figure()
    for var in varnames:
        # if getattr fails then the getter and setter does not exist
        getfunc = getattr(fig, f"get_{var}")
        setfunc = getattr(fig, f"set_{var}")