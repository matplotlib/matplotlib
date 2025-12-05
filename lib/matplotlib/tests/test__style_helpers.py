import pytest

import matplotlib.colors as mcolors
from matplotlib.lines import _get_dash_pattern
from matplotlib._style_helpers import iterate_styles


@pytest.mark.parametrize('key, value', [('facecolor', ["b", "g", "r"]),
                                        ('edgecolor', ["b", "g", "r"]),
                                        ('hatch', ["/", "\\", "."]),
                                        ('linestyle', ["-", "--", ":"]),
                                        ('linewidth', [1, 1.5, 2])])
def test_iterate_styles_list(key, value):
    kw = {'foo': 12, key: value}
    gen_kw = iterate_styles(kw)

    for v in value * 2:  # Result should repeat
        kw_dict = next(gen_kw)
        assert len(kw_dict) == 2
        assert kw_dict['foo'] == 12
        if key.endswith('color'):
            assert mcolors.same_color(v, kw_dict[key])
        elif key == 'linestyle':
            assert _get_dash_pattern(v) == kw_dict[key]
        else:
            assert v == kw_dict[key]


@pytest.mark.parametrize('key, value', [('facecolor', "b"),
                                        ('edgecolor', "b"),
                                        ('hatch', "/"),
                                        ('linestyle', "-"),
                                        ('linewidth', 1)])
def test_iterate_styles_single(key, value):
    kw = {'foo': 12, key: value}
    gen_kw = iterate_styles(kw)

    for _ in range(2):  # Result should repeat
        kw_dict = next(gen_kw)
        assert len(kw_dict) == 2
        assert kw_dict['foo'] == 12
        if key.endswith('color'):
            assert mcolors.same_color(value, kw_dict[key])
        else:
            assert value == kw_dict[key]


@pytest.mark.parametrize('key', ['facecolor', 'hatch', 'linestyle'])
def test_iterate_styles_empty(key):
    kw = {key: []}
    gen_kw = iterate_styles(kw)
    with pytest.raises(TypeError, match=f'{key} must not be an empty sequence'):
        next(gen_kw)


def test_iterate_styles_sequence_type_styles():
    kw = {'facecolor':  ('r', 0.5),
          'edgecolor': [0.5, 0.5, 0.5],
          'linestyle': (0, (1, 1))}

    gen_kw = iterate_styles(kw)
    for _ in range(2):  # Result should repeat
        kw_dict = next(gen_kw)
        mcolors.same_color(kw['facecolor'], kw_dict['facecolor'])
        mcolors.same_color(kw['edgecolor'], kw_dict['edgecolor'])
        kw['linestyle'] == kw_dict['linestyle']


def test_iterate_styles_none():
    kw = {'facecolor': 'none',
          'edgecolor': 'none'}
    gen_kw = iterate_styles(kw)
    for _ in range(2):  # Result should repeat
        kw_dict = next(gen_kw)
        assert kw_dict['facecolor'] == 'none'
        assert kw_dict['edgecolor'] == 'none'
