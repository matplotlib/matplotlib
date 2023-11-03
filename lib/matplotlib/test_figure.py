import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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

assert test_fig_get_set() is None