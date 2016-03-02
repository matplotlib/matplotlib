from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import io
import os
import sys
import warnings

from cycler import cycler, Cycler

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.tests import assert_str_equal
from matplotlib.testing.decorators import cleanup, knownfailureif
import matplotlib.colors as mcolors
from nose.tools import assert_true, assert_raises, assert_equal
from nose.plugins.skip import SkipTest
import nose
from itertools import chain
import numpy as np
from matplotlib.rcsetup import (validate_bool_maybe_none,
                                validate_stringlist,
                                validate_colorlist,
                                validate_bool,
                                validate_nseq_int,
                                validate_nseq_float,
                                validate_cycler,
                                validate_hatch,
                                validate_hist_bins)


mpl.rc('text', usetex=False)
mpl.rc('lines', linewidth=22)

fname = os.path.join(os.path.dirname(__file__), 'test_rcparams.rc')


def test_rcparams():
    usetex = mpl.rcParams['text.usetex']
    linewidth = mpl.rcParams['lines.linewidth']

    # test context given dictionary
    with mpl.rc_context(rc={'text.usetex': not usetex}):
        assert mpl.rcParams['text.usetex'] == (not usetex)
    assert mpl.rcParams['text.usetex'] == usetex

    # test context given filename (mpl.rc sets linewdith to 33)
    with mpl.rc_context(fname=fname):
        assert mpl.rcParams['lines.linewidth'] == 33
    assert mpl.rcParams['lines.linewidth'] == linewidth

    # test context given filename and dictionary
    with mpl.rc_context(fname=fname, rc={'lines.linewidth': 44}):
        assert mpl.rcParams['lines.linewidth'] == 44
    assert mpl.rcParams['lines.linewidth'] == linewidth

    # test rc_file
    try:
        mpl.rc_file(fname)
        assert mpl.rcParams['lines.linewidth'] == 33
    finally:
        mpl.rcParams['lines.linewidth'] = linewidth


def test_RcParams_class():
    rc = mpl.RcParams({'font.cursive': ['Apple Chancery',
                                        'Textile',
                                        'Zapf Chancery',
                                        'cursive'],
                       'font.family': 'sans-serif',
                       'font.weight': 'normal',
                       'font.size': 12})

    if six.PY3:
        expected_repr = """
RcParams({'font.cursive': ['Apple Chancery',
                           'Textile',
                           'Zapf Chancery',
                           'cursive'],
          'font.family': ['sans-serif'],
          'font.size': 12.0,
          'font.weight': 'normal'})""".lstrip()
    else:
        expected_repr = """
RcParams({u'font.cursive': [u'Apple Chancery',
                            u'Textile',
                            u'Zapf Chancery',
                            u'cursive'],
          u'font.family': [u'sans-serif'],
          u'font.size': 12.0,
          u'font.weight': u'normal'})""".lstrip()

    assert_str_equal(expected_repr, repr(rc))

    if six.PY3:
        expected_str = """
font.cursive: ['Apple Chancery', 'Textile', 'Zapf Chancery', 'cursive']
font.family: ['sans-serif']
font.size: 12.0
font.weight: normal""".lstrip()
    else:
        expected_str = """
font.cursive: [u'Apple Chancery', u'Textile', u'Zapf Chancery', u'cursive']
font.family: [u'sans-serif']
font.size: 12.0
font.weight: normal""".lstrip()

    assert_str_equal(expected_str, str(rc))

    # test the find_all functionality
    assert ['font.cursive', 'font.size'] == sorted(rc.find_all('i[vz]').keys())
    assert ['font.family'] == list(six.iterkeys(rc.find_all('family')))


def test_rcparams_update():
    if sys.version_info[:2] < (2, 7):
        raise nose.SkipTest("assert_raises as context manager "
                            "not supported with Python < 2.7")
    rc = mpl.RcParams({'figure.figsize': (3.5, 42)})
    bad_dict = {'figure.figsize': (3.5, 42, 1)}
    # make sure validation happens on input
    with assert_raises(ValueError):

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                message='.*(validate)',
                                category=UserWarning)
            rc.update(bad_dict)


def test_rcparams_init():
    if sys.version_info[:2] < (2, 7):
        raise nose.SkipTest("assert_raises as context manager "
                            "not supported with Python < 2.7")
    with assert_raises(ValueError):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                message='.*(validate)',
                                category=UserWarning)
            mpl.RcParams({'figure.figsize': (3.5, 42, 1)})


@cleanup
def test_Bug_2543():
    # Test that it possible to add all values to itself / deepcopy
    # This was not possible because validate_bool_maybe_none did not
    # accept None as an argument.
    # https://github.com/matplotlib/matplotlib/issues/2543
    # We filter warnings at this stage since a number of them are raised
    # for deprecated rcparams as they should. We dont want these in the
    # printed in the test suite.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='.*(deprecated|obsolete)',
                                category=UserWarning)
        with mpl.rc_context():
            _copy = mpl.rcParams.copy()
            for key in six.iterkeys(_copy):
                mpl.rcParams[key] = _copy[key]
            mpl.rcParams['text.dvipnghack'] = None
        with mpl.rc_context():
            from copy import deepcopy
            _deep_copy = deepcopy(mpl.rcParams)
        # real test is that this does not raise
        assert_true(validate_bool_maybe_none(None) is None)
        assert_true(validate_bool_maybe_none("none") is None)
        _fonttype = mpl.rcParams['svg.fonttype']
        assert_true(_fonttype == mpl.rcParams['svg.embed_char_paths'])
        with mpl.rc_context():
            mpl.rcParams['svg.embed_char_paths'] = False
            assert_true(mpl.rcParams['svg.fonttype'] == "none")


@cleanup
def test_Bug_2543_newer_python():
    # only split from above because of the usage of assert_raises
    # as a context manager, which only works in 2.7 and above
    if sys.version_info[:2] < (2, 7):
        raise nose.SkipTest("assert_raises as context manager not supported with Python < 2.7")
    from matplotlib.rcsetup import validate_bool_maybe_none, validate_bool
    with assert_raises(ValueError):
        validate_bool_maybe_none("blah")
    with assert_raises(ValueError):
        validate_bool(None)
    with assert_raises(ValueError):
        with mpl.rc_context():
            mpl.rcParams['svg.fonttype'] = True


@cleanup
def _legend_rcparam_helper(param_dict, target, get_func):
    with mpl.rc_context(param_dict):
        _, ax = plt.subplots()
        ax.plot(range(3), label='test')
        leg = ax.legend()
        assert_equal(getattr(leg.legendPatch, get_func)(), target)


def test_legend_facecolor():
    get_func = 'get_facecolor'
    rcparam = 'legend.facecolor'
    test_values = [({rcparam: 'r'},
                    mcolors.colorConverter.to_rgba('r')),
                   ({rcparam: 'inherit',
                     'axes.facecolor': 'r'
                     },
                    mcolors.colorConverter.to_rgba('r')),
                   ({rcparam: 'g',
                     'axes.facecolor': 'r'},
                   mcolors.colorConverter.to_rgba('g'))
                   ]
    for rc_dict, target in test_values:
        yield _legend_rcparam_helper, rc_dict, target, get_func


def test_legend_edgecolor():
    get_func = 'get_edgecolor'
    rcparam = 'legend.edgecolor'
    test_values = [({rcparam: 'r'},
                    mcolors.colorConverter.to_rgba('r')),
                   ({rcparam: 'inherit',
                     'axes.edgecolor': 'r'
                     },
                    mcolors.colorConverter.to_rgba('r')),
                   ({rcparam: 'g',
                     'axes.facecolor': 'r'},
                   mcolors.colorConverter.to_rgba('g'))
                   ]
    for rc_dict, target in test_values:
        yield _legend_rcparam_helper, rc_dict, target, get_func


def test_Issue_1713():
    utf32_be = os.path.join(os.path.dirname(__file__),
                           'test_utf32_be_rcparams.rc')
    old_lang = os.environ.get('LANG', None)
    os.environ['LANG'] = 'en_US.UTF-32-BE'
    rc = mpl.rc_params_from_file(utf32_be, True)
    if old_lang:
        os.environ['LANG'] = old_lang
    else:
        del os.environ['LANG']
    assert rc.get('timezone') == 'UTC'


def _validation_test_helper(validator, arg, target):
    res = validator(arg)
    if isinstance(target, np.ndarray):
        assert_true(np.all(res == target))
    elif not isinstance(target, Cycler):
        assert_equal(res, target)
    else:
        # Cyclers can't simply be asserted equal. They don't implement __eq__
        assert_equal(list(res), list(target))


def _validation_fail_helper(validator, arg, exception_type):
    if sys.version_info[:2] < (2, 7):
        raise nose.SkipTest("assert_raises as context manager not "
                            "supported with Python < 2.7")
    with assert_raises(exception_type):
        validator(arg)


def test_validators():
    validation_tests = (
        {'validator': validate_bool,
         'success': chain(((_, True) for _ in
                           ('t', 'y', 'yes', 'on', 'true', '1', 1, True)),
                           ((_, False) for _ in
                            ('f', 'n', 'no', 'off', 'false', '0', 0, False))),
        'fail': ((_, ValueError)
                 for _ in ('aardvark', 2, -1, [], ))},
        {'validator': validate_stringlist,
         'success': (('', []),
                     ('a,b', ['a', 'b']),
                     ('aardvark', ['aardvark']),
                     ('aardvark, ', ['aardvark']),
                     ('aardvark, ,', ['aardvark']),
                     (['a', 'b'], ['a', 'b']),
                     (('a', 'b'), ['a', 'b']),
                     (iter(['a', 'b']), ['a', 'b']),
                     (np.array(['a', 'b']), ['a', 'b']),
                     ((1, 2), ['1', '2']),
                     (np.array([1, 2]), ['1', '2']),
                    ),
         'fail': ((dict(), ValueError),
                  (1, ValueError),
                 )
        },
        {'validator': validate_nseq_int(2),
         'success': ((_, [1, 2])
                     for _ in ('1, 2', [1.5, 2.5], [1, 2],
                               (1, 2), np.array((1, 2)))),
         'fail': ((_, ValueError)
                  for _ in ('aardvark', ('a', 1),
                            (1, 2, 3)
                            ))
        },
        {'validator': validate_nseq_float(2),
         'success': ((_, [1.5, 2.5])
                     for _ in ('1.5, 2.5', [1.5, 2.5], [1.5, 2.5],
                               (1.5, 2.5), np.array((1.5, 2.5)))),
         'fail': ((_, ValueError)
                  for _ in ('aardvark', ('a', 1),
                            (1, 2, 3)
                            ))
        },
        {'validator': validate_cycler,
         'success': (('cycler("color", "rgb")',
                      cycler("color", 'rgb')),
                     (cycler('linestyle', ['-', '--']),
                      cycler('linestyle', ['-', '--'])),
                     ("""(cycler("color", ["r", "g", "b"]) +
                          cycler("mew", [2, 3, 5]))""",
                      (cycler("color", 'rgb') +
                          cycler("markeredgewidth", [2, 3, 5]))),
                     ("cycler(c='rgb', lw=[1, 2, 3])",
                      cycler('color', 'rgb') + cycler('linewidth', [1, 2, 3])),
                     ("cycler('c', 'rgb') * cycler('linestyle', ['-', '--'])",
                      (cycler('color', 'rgb') *
                          cycler('linestyle', ['-', '--']))),
                    ),
         # This is *so* incredibly important: validate_cycler() eval's
         # an arbitrary string! I think I have it locked down enough,
         # and that is what this is testing.
         # TODO: Note that these tests are actually insufficient, as it may
         # be that they raised errors, but still did an action prior to
         # raising the exception. We should devise some additional tests
         # for that...
         'fail': ((4, ValueError),  # Gotta be a string or Cycler object
                  ('cycler("bleh, [])', ValueError),  # syntax error
                  ('Cycler("linewidth", [1, 2, 3])',
                      ValueError),  # only 'cycler()' function is allowed
                  ('1 + 2', ValueError),  # doesn't produce a Cycler object
                  ('os.system("echo Gotcha")', ValueError),  # os not available
                  ('import os', ValueError),  # should not be able to import
                  ('def badjuju(a): return a; badjuju(cycler("color", "rgb"))',
                      ValueError),  # Should not be able to define anything
                                    # even if it does return a cycler
                  ('cycler("waka", [1, 2, 3])', ValueError),  # not a property
                  ('cycler(c=[1, 2, 3])', ValueError),  # invalid values
                  ("cycler(lw=['a', 'b', 'c'])", ValueError),  # invalid values
                 )
        },
        {'validator': validate_hatch,
         'success': (('--|', '--|'), ('\\oO', '\\oO'),
                     ('/+*/.x', '/+*/.x'), ('', '')),
         'fail': (('--_', ValueError),
                  (8, ValueError),
                  ('X', ValueError)),
        },
        {'validator': validate_colorlist,
         'success': (('r,g,b', ['r', 'g', 'b']),
                     (['r', 'g', 'b'], ['r', 'g', 'b']),
                     ('r, ,', ['r']),
                     (['', 'g', 'blue'], ['g', 'blue']),
                     ([np.array([1, 0, 0]), np.array([0, 1, 0])],
                         np.array([[1, 0, 0], [0, 1, 0]])),
                     (np.array([[1, 0, 0], [0, 1, 0]]),
                         np.array([[1, 0, 0], [0, 1, 0]])),
                    ),
         'fail': (('fish', ValueError),
                 ),
        },
        {'validator': validate_hist_bins,
         'success': (('auto', 'auto'),
                     ('10', 10),
                     ('1, 2, 3', [1, 2, 3]),
                     ([1, 2, 3], [1, 2, 3]),
                     (np.arange(15), np.arange(15))
                     ),
         'fail': (('aardvark', ValueError),
                  )
         }
    )

    for validator_dict in validation_tests:
        validator = validator_dict['validator']
        for arg, target in validator_dict['success']:
            yield _validation_test_helper, validator, arg, target
        for arg, error_type in validator_dict['fail']:
            yield _validation_fail_helper, validator, arg, error_type


def test_keymaps():
    key_list = [k for k in mpl.rcParams if 'keymap' in k]
    for k in key_list:
        assert(isinstance(mpl.rcParams[k], list))


def test_rcparams_reset_after_fail():

    # There was previously a bug that meant that if rc_context failed and
    # raised an exception due to issues in the supplied rc parameters, the
    # global rc parameters were left in a modified state.

    if sys.version_info[:2] >= (2, 7):
        from collections import OrderedDict
    else:
        raise SkipTest("Test can only be run in Python >= 2.7 as it requires OrderedDict")

    with mpl.rc_context(rc={'text.usetex': False}):

        assert mpl.rcParams['text.usetex'] is False

        with assert_raises(KeyError):
            with mpl.rc_context(rc=OrderedDict([('text.usetex', True),('test.blah', True)])):
                pass

        assert mpl.rcParams['text.usetex'] is False


if __name__ == '__main__':
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
