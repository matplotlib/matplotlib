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
                  ('-g', ('-', 'None', 'g')),
                  ('-og', ('-', 'o', 'g')),
                  ('-or', ('-', 'o', 'r')),
                  ('-ok', ('-', 'o', 'k'))
                  )

    for fmt, target in test_cases:
        yield _fmt_parsing_helper, fmt, target


def test_fmt_parsing_fail():
    test_strings = ('bb-',
                    'xog')
    for fmt in test_strings:
        yield _fmt_parsing_invalid_helper, fmt
