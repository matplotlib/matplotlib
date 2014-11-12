from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from matplotlib.testing.decorators import cleanup
import six


from nose.tools import assert_equal, assert_raises

from matplotlib.axes._axes import _process_plot_format


@cleanup
def _fmt_parsing_helper(fmt, ref):
    assert_equal(ref, _process_plot_format(fmt))


@cleanup
def _fmt_parsing_invalid_helper(fmt):
    assert_raises(ValueError, _process_plot_format, fmt)


def test_fmt_parsing():
    test_cases = (('-', ('-', 'None', None)),
                  ('--', ('--', 'None', None)),
                  ('-.', ('-.', 'None', None)),
                  (':', (':', 'None', None)),
                  ('-g', ('-', 'None', 'g')),
                  ('--r', ('--', 'None', 'r')),
                  ('-.k', ('-.', 'None', 'k')),
                  (':b', (':', 'None', 'b')),
                  ('-o', ('-', 'o', None)),
                  ('--,', ('--', ',', None)),
                  ('-..', ('-.', '.', None)),
                  ('v:', (':', 'v', None)),
                  ('-g^', ('-', '^', 'g')),
                  ('--r<', ('--', '<', 'r')),
                  (':>g', (':', '>', 'g')),
                  ('.-.k', ('-.', '.', 'k')),
                  ('-c', ('-', 'None', 'c')),
                  ('-1m', ('-', '1', 'm')),
                  ('-y2', ('-', '2', 'y')),
                  ('w-3', ('-', '3', 'w')),
                  ('--.', ('--', '.', None)),
                  ('-4k', ('-', '4', 'k')),
                  ('sk', ('None', 's', 'k')),
                  ('r', (None, None, 'r')),
                  ('1.0', (None, None, '1.0')),
                  ('0.8', (None, None, '0.8')),
                  ('p', ('None', 'p', None)),
                  ('*--', ('--', '*', None)),
                  ('kh:', (':', 'h', 'k')),
                  ('H-r', ('-', 'H', 'r')),
                  ('--+g', ('--', '+', 'g')),
                  (':xb', (':', 'x', 'b')),
                  ('cD-.', ('-.', 'D', 'c')),
                  ('m--d', ('--', 'd', 'm')),
                  (':y|', (':', '|', 'y')),
                  ('w_-', ('-', '_', 'w'))
                  )

    for fmt, target in test_cases:
        yield _fmt_parsing_helper, fmt, target


def test_fmt_parsing_fail():
    test_strings = ('bb-',
                    'xog',
                    '---.',
                    ':-',
                    '12c',
                    '--ph',
                    ',u',
                    ',1.0',
                    '.0.8')
    for fmt in test_strings:
        yield _fmt_parsing_invalid_helper, fmt
