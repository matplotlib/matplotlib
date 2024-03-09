from collections.abc import Sequence
from typing import Any

import pytest

import matplotlib as mpl
from matplotlib.backends import BackendFilter, backend_registry


def has_duplicates(seq: Sequence[Any]) -> bool:
    return len(seq) > len(set(seq))


@pytest.mark.parametrize(
    'framework,expected',
    [
        ('qt', 'qtagg'),
        ('gtk3', 'gtk3agg'),
        ('gtk4', 'gtk4agg'),
        ('wx', 'wxagg'),
        ('tk', 'tkagg'),
        ('macosx', 'macosx'),
        ('headless', 'agg'),
        ('does not exist', None),
    ]
)
def test_backend_for_gui_framework(framework, expected):
    assert backend_registry.backend_for_gui_framework(framework) == expected


def test_list_builtin():
    backends = backend_registry.list_builtin()
    assert not has_duplicates(backends)
    # Compare using sets as order is not important
    assert {*backends} == {
        'GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg',
        'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg',
        'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template',
    }


@pytest.mark.parametrize(
    'filter,expected',
    [
        (BackendFilter.INTERACTIVE,
         ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg',
          'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg',
           'WXCairo']),
        (BackendFilter.NON_INTERACTIVE,
         ['agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']),
    ]
)
def test_list_builtin_with_filter(filter, expected):
    backends = backend_registry.list_builtin(filter)
    assert not has_duplicates(backends)
    # Compare using sets as order is not important
    assert {*backends} == {*expected}


def test_deprecated_rcsetup_attributes():
    match = "was deprecated in Matplotlib 3.9"
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match=match):
        mpl.rcsetup.interactive_bk
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match=match):
        mpl.rcsetup.non_interactive_bk
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match=match):
        mpl.rcsetup.all_backends
