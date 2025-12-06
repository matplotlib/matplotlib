import pytest

import matplotlib.colors as mcolors
from matplotlib.lines import _get_dash_pattern
from matplotlib._style_helpers import style_generator


@pytest.mark.parametrize('key, value', [('facecolor', ["b", "g", "r"]),
                                        ('edgecolor', ["b", "g", "r"]),
                                        ('hatch', ["/", "\\", "."]),
                                        ('linestyle', ["-", "--", ":"]),
                                        ('linewidth', [1, 1.5, 2])])
def test_style_generator_list(key, value):
    kw = {'foo': 12, key: value}
    new_kw, gen = style_generator(kw)

    assert new_kw == {'foo': 12}

    for v in value * 2:  # Result should repeat
        style_dict = next(gen)
        assert len(style_dict) == 1
        if key.endswith('color'):
            assert mcolors.same_color(v, style_dict[key])
        elif key == 'linestyle':
            assert _get_dash_pattern(v) == style_dict[key]
        else:
            assert v == style_dict[key]


@pytest.mark.parametrize('key, value', [('facecolor', "b"),
                                        ('edgecolor', "b"),
                                        ('hatch', "/"),
                                        ('linestyle', "-"),
                                        ('linewidth', 1)])
def test_style_generator_single(key, value):
    kw = {'foo': 12, key: value}
    new_kw, gen = style_generator(kw)

    assert new_kw == {'foo': 12}
    for _ in range(2):  # Result should repeat
        style_dict = next(gen)
        if key.endswith('color'):
            assert mcolors.same_color(value, style_dict[key])
        elif key == 'linestyle':
            assert _get_dash_pattern(value) == style_dict[key]
        else:
            assert value == style_dict[key]


@pytest.mark.parametrize('key', ['facecolor', 'hatch', 'linestyle'])
def test_style_generator_empty(key):
    kw = {key: []}
    with pytest.raises(TypeError, match=f'{key} must not be an empty sequence'):
        style_generator(kw)


def test_style_generator_sequence_type_styles():
    kw = {'facecolor':  ('r', 0.5),
          'edgecolor': [0.5, 0.5, 0.5],
          'linestyle': (0, (1, 1))}

    _, gen = style_generator(kw)
    for _ in range(2):  # Result should repeat
        style_dict = next(gen)
        mcolors.same_color(kw['facecolor'], style_dict['facecolor'])
        mcolors.same_color(kw['edgecolor'], style_dict['edgecolor'])
        kw['linestyle'] == style_dict['linestyle']


def test_style_generator_none():
    kw = {'facecolor': 'none',
          'edgecolor': 'none'}
    _, gen = style_generator(kw)
    for _ in range(2):  # Result should repeat
        style_dict = next(gen)
        assert style_dict['facecolor'] == 'none'
        assert style_dict['edgecolor'] == 'none'
