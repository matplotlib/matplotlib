from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import zip
from matplotlib.cycler import cycler
from nose.tools import assert_equal, assert_raises
from itertools import product
from operator import add, iadd, mul, imul


def _cycler_helper(c, length, keys, values):
    assert_equal(len(c), length)
    assert_equal(len(c), len(list(c.finite_iter())))
    assert_equal(len(c), len(c.to_list()))
    assert_equal(c.keys, set(keys))

    for k, vals in zip(keys, values):
        for v, v_target in zip(c, vals):
            assert_equal(v[k], v_target)


def test_creation():
    c = cycler('c', 'rgb')
    yield _cycler_helper, c, 3, ['c'], [['r', 'g', 'b']]
    c = cycler('c', list('rgb'))
    yield _cycler_helper, c, 3, ['c'], [['r', 'g', 'b']]


def test_compose():
    c1 = cycler('c', 'rgb')
    c2 = cycler('lw', range(3))
    c3 = cycler('lw', range(15))
    # addition
    yield _cycler_helper, c1+c2, 3, ['c', 'lw'], [list('rgb'), range(3)]
    yield _cycler_helper, c2+c1, 3, ['c', 'lw'], [list('rgb'), range(3)]
    # miss-matched add lengths
    yield _cycler_helper, c1+c3, 3, ['c', 'lw'], [list('rgb'), range(3)]
    yield _cycler_helper, c3+c1, 3, ['c', 'lw'], [list('rgb'), range(3)]

    # multiplication
    target = zip(*product(list('rgb'), range(3)))
    yield (_cycler_helper, c1 * c2, 9, ['c', 'lw'], target)

    target = zip(*product(range(3), list('rgb')))
    yield (_cycler_helper, c2 * c1, 9, ['lw', 'c'], target)

    target = zip(*product(range(15), list('rgb')))
    yield (_cycler_helper, c3 * c1, 45, ['lw', 'c'], target)


def test_inplace():
    c1 = cycler('c', 'rgb')
    c2 = cycler('lw', range(3))
    c2 += c1
    yield _cycler_helper, c2, 3, ['c', 'lw'], [list('rgb'), range(3)]

    c3 = cycler('c', 'rgb')
    c4 = cycler('lw', range(3))
    c3 *= c4
    target = zip(*product(list('rgb'), range(3)))
    yield (_cycler_helper, c3, 9, ['c', 'lw'], target)


def test_constructor():
    c1 = cycler('c', 'rgb')
    c2 = cycler('ec', c1)
    yield _cycler_helper, c1+c2, 3, ['c', 'ec'], [['r', 'g', 'b']]*2
    c3 = cycler('c', c1)
    yield _cycler_helper, c3+c2, 3, ['c', 'ec'], [['r', 'g', 'b']]*2


def test_failures():
    c1 = cycler('c', 'rgb')
    c2 = cycler('c', c1)
    assert_raises(ValueError, add, c1, c2)
    assert_raises(ValueError, iadd, c1, c2)
    assert_raises(ValueError, mul, c1, c2)
    assert_raises(ValueError, imul, c1, c2)

    c3 = cycler('ec', c1)

    assert_raises(ValueError, cycler, 'c', c2 + c3)
