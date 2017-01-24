from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import itertools
import pickle
from weakref import ref
import warnings

import six

from datetime import datetime

import numpy as np
from numpy.testing.utils import (assert_array_equal, assert_approx_equal,
                                 assert_array_almost_equal)
from nose.tools import (assert_equal, assert_not_equal, raises, assert_true,
                        assert_raises)

import matplotlib.cbook as cbook
import matplotlib.colors as mcolors
from matplotlib.cbook import delete_masked_points as dmp


def test_is_string_like():
    y = np.arange(10)
    assert_equal(cbook.is_string_like(y), False)
    y.shape = 10, 1
    assert_equal(cbook.is_string_like(y), False)
    y.shape = 1, 10
    assert_equal(cbook.is_string_like(y), False)

    assert cbook.is_string_like("hello world")
    assert_equal(cbook.is_string_like(10), False)

    y = ['a', 'b', 'c']
    assert_equal(cbook.is_string_like(y), False)

    y = np.array(y)
    assert_equal(cbook.is_string_like(y), False)

    y = np.array(y, dtype=object)
    assert cbook.is_string_like(y)


def test_is_sequence_of_strings():
    y = ['a', 'b', 'c']
    assert cbook.is_sequence_of_strings(y)

    y = np.array(y, dtype=object)
    assert cbook.is_sequence_of_strings(y)


def test_is_hashable():
    s = 'string'
    assert cbook.is_hashable(s)

    lst = ['list', 'of', 'stings']
    assert not cbook.is_hashable(lst)


def test_restrict_dict():
    d = {'foo': 'bar', 1: 2}
    d1 = cbook.restrict_dict(d, ['foo', 1])
    assert_equal(d1, d)
    d2 = cbook.restrict_dict(d, ['bar', 2])
    assert_equal(d2, {})
    d3 = cbook.restrict_dict(d, {'foo': 1})
    assert_equal(d3, {'foo': 'bar'})
    d4 = cbook.restrict_dict(d, {})
    assert_equal(d4, {})
    d5 = cbook.restrict_dict(d, set(['foo', 2]))
    assert_equal(d5, {'foo': 'bar'})
    # check that d was not modified
    assert_equal(d, {'foo': 'bar', 1: 2})


class Test_delete_masked_points(object):
    def setUp(self):
        self.mask1 = [False, False, True, True, False, False]
        self.arr0 = np.arange(1.0, 7.0)
        self.arr1 = [1, 2, 3, np.nan, np.nan, 6]
        self.arr2 = np.array(self.arr1)
        self.arr3 = np.ma.array(self.arr2, mask=self.mask1)
        self.arr_s = ['a', 'b', 'c', 'd', 'e', 'f']
        self.arr_s2 = np.array(self.arr_s)
        self.arr_dt = [datetime(2008, 1, 1), datetime(2008, 1, 2),
                       datetime(2008, 1, 3), datetime(2008, 1, 4),
                       datetime(2008, 1, 5), datetime(2008, 1, 6)]
        self.arr_dt2 = np.array(self.arr_dt)
        self.arr_colors = ['r', 'g', 'b', 'c', 'm', 'y']
        self.arr_rgba = mcolors.to_rgba_array(self.arr_colors)

    @raises(ValueError)
    def test_bad_first_arg(self):
        dmp('a string', self.arr0)

    def test_string_seq(self):
        actual = dmp(self.arr_s, self.arr1)
        ind = [0, 1, 2, 5]
        expected = (self.arr_s2.take(ind), self.arr2.take(ind))
        assert_array_equal(actual[0], expected[0])
        assert_array_equal(actual[1], expected[1])

    def test_datetime(self):
        actual = dmp(self.arr_dt, self.arr3)
        ind = [0, 1,  5]
        expected = (self.arr_dt2.take(ind),
                    self.arr3.take(ind).compressed())
        assert_array_equal(actual[0], expected[0])
        assert_array_equal(actual[1], expected[1])

    def test_rgba(self):
        actual = dmp(self.arr3, self.arr_rgba)
        ind = [0, 1, 5]
        expected = (self.arr3.take(ind).compressed(),
                    self.arr_rgba.take(ind, axis=0))
        assert_array_equal(actual[0], expected[0])
        assert_array_equal(actual[1], expected[1])


def test_allequal():
    assert(cbook.allequal([1, 1, 1]))
    assert(not cbook.allequal([1, 1, 0]))
    assert(cbook.allequal([]))
    assert(cbook.allequal(('a', 'a')))
    assert(not cbook.allequal(('a', 'b')))


class Test_boxplot_stats(object):
    def setup(self):
        np.random.seed(937)
        self.nrows = 37
        self.ncols = 4
        self.data = np.random.lognormal(size=(self.nrows, self.ncols),
                                        mean=1.5, sigma=1.75)
        self.known_keys = sorted([
            'mean', 'med', 'q1', 'q3', 'iqr',
            'cilo', 'cihi', 'whislo', 'whishi',
            'fliers', 'label'
        ])
        self.std_results = cbook.boxplot_stats(self.data)

        self.known_nonbootstrapped_res = {
            'cihi': 6.8161283264444847,
            'cilo': -0.1489815330368689,
            'iqr': 13.492709959447094,
            'mean': 13.00447442387868,
            'med': 3.3335733967038079,
            'fliers': np.array([
                92.55467075,  87.03819018,  42.23204914,  39.29390996
            ]),
            'q1': 1.3597529879465153,
            'q3': 14.85246294739361,
            'whishi': 27.899688243699629,
            'whislo': 0.042143774965502923
        }

        self.known_bootstrapped_ci = {
            'cihi': 8.939577523357828,
            'cilo': 1.8692703958676578,
        }

        self.known_whis3_res = {
            'whishi': 42.232049135969874,
            'whislo': 0.042143774965502923,
            'fliers': np.array([92.55467075, 87.03819018]),
        }

        self.known_res_percentiles = {
            'whislo':   0.1933685896907924,
            'whishi':  42.232049135969874
        }

        self.known_res_range = {
            'whislo': 0.042143774965502923,
            'whishi': 92.554670752188699

        }

    def test_form_main_list(self):
        assert_true(isinstance(self.std_results, list))

    def test_form_each_dict(self):
        for res in self.std_results:
            assert_true(isinstance(res, dict))

    def test_form_dict_keys(self):
        for res in self.std_results:
            keys = sorted(list(res.keys()))
            for key in keys:
                assert_true(key in self.known_keys)

    def test_results_baseline(self):
        res = self.std_results[0]
        for key in list(self.known_nonbootstrapped_res.keys()):
            if key != 'fliers':
                assert_statement = assert_approx_equal
            else:
                assert_statement = assert_array_almost_equal

            assert_statement(
                res[key],
                self.known_nonbootstrapped_res[key]
            )

    def test_results_bootstrapped(self):
        results = cbook.boxplot_stats(self.data, bootstrap=10000)
        res = results[0]
        for key in list(self.known_bootstrapped_ci.keys()):
            assert_approx_equal(
                res[key],
                self.known_bootstrapped_ci[key]
            )

    def test_results_whiskers_float(self):
        results = cbook.boxplot_stats(self.data, whis=3)
        res = results[0]
        for key in list(self.known_whis3_res.keys()):
            if key != 'fliers':
                assert_statement = assert_approx_equal
            else:
                assert_statement = assert_array_almost_equal

            assert_statement(
                res[key],
                self.known_whis3_res[key]
            )

    def test_results_whiskers_range(self):
        results = cbook.boxplot_stats(self.data, whis='range')
        res = results[0]
        for key in list(self.known_res_range.keys()):
            if key != 'fliers':
                assert_statement = assert_approx_equal
            else:
                assert_statement = assert_array_almost_equal

            assert_statement(
                res[key],
                self.known_res_range[key]
            )

    def test_results_whiskers_percentiles(self):
        results = cbook.boxplot_stats(self.data, whis=[5, 95])
        res = results[0]
        for key in list(self.known_res_percentiles.keys()):
            if key != 'fliers':
                assert_statement = assert_approx_equal
            else:
                assert_statement = assert_array_almost_equal

            assert_statement(
                res[key],
                self.known_res_percentiles[key]
            )

    def test_results_withlabels(self):
        labels = ['Test1', 2, 'ardvark', 4]
        results = cbook.boxplot_stats(self.data, labels=labels)
        res = results[0]
        for lab, res in zip(labels, results):
            assert_equal(res['label'], lab)

        results = cbook.boxplot_stats(self.data)
        for res in results:
            assert('label' not in res)

    @raises(ValueError)
    def test_label_error(self):
        labels = [1, 2]
        results = cbook.boxplot_stats(self.data, labels=labels)

    @raises(ValueError)
    def test_bad_dims(self):
        data = np.random.normal(size=(34, 34, 34))
        results = cbook.boxplot_stats(data)

    def test_boxplot_stats_autorange_false(self):
        x = np.zeros(shape=140)
        x = np.hstack([-25, x, 25])
        bstats_false = cbook.boxplot_stats(x, autorange=False)
        bstats_true = cbook.boxplot_stats(x, autorange=True)

        assert_equal(bstats_false[0]['whislo'], 0)
        assert_equal(bstats_false[0]['whishi'], 0)
        assert_array_almost_equal(bstats_false[0]['fliers'], [-25, 25])

        assert_equal(bstats_true[0]['whislo'], -25)
        assert_equal(bstats_true[0]['whishi'], 25)
        assert_array_almost_equal(bstats_true[0]['fliers'], [])


class Test_callback_registry(object):
    def setup(self):
        self.signal = 'test'
        self.callbacks = cbook.CallbackRegistry()

    def connect(self, s, func):
        return self.callbacks.connect(s, func)

    def is_empty(self):
        assert_equal(self.callbacks._func_cid_map, {})
        assert_equal(self.callbacks.callbacks, {})

    def is_not_empty(self):
        assert_not_equal(self.callbacks._func_cid_map, {})
        assert_not_equal(self.callbacks.callbacks, {})

    def test_callback_complete(self):
        # ensure we start with an empty registry
        self.is_empty()

        # create a class for testing
        mini_me = Test_callback_registry()

        # test that we can add a callback
        cid1 = self.connect(self.signal, mini_me.dummy)
        assert_equal(type(cid1), int)
        self.is_not_empty()

        # test that we don't add a second callback
        cid2 = self.connect(self.signal, mini_me.dummy)
        assert_equal(cid1, cid2)
        self.is_not_empty()
        assert_equal(len(self.callbacks._func_cid_map), 1)
        assert_equal(len(self.callbacks.callbacks), 1)

        del mini_me

        # check we now have no callbacks registered
        self.is_empty()

    def dummy(self):
        pass

    def test_pickling(self):
        assert hasattr(pickle.loads(pickle.dumps(cbook.CallbackRegistry())),
                       "callbacks")


def _kwarg_norm_helper(inp, expected, kwargs_to_norm, warn_count=0):

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert expected == cbook.normalize_kwargs(inp, **kwargs_to_norm)
        assert len(w) == warn_count


def _kwarg_norm_fail_helper(inp, kwargs_to_norm):
    assert_raises(TypeError, cbook.normalize_kwargs, inp, **kwargs_to_norm)


def test_normalize_kwargs():
    fail_mapping = (
        ({'a': 1}, {'forbidden': ('a')}),
        ({'a': 1}, {'required': ('b')}),
        ({'a': 1, 'b': 2}, {'required': ('a'), 'allowed': ()})
    )

    for inp, kwargs in fail_mapping:
        yield _kwarg_norm_fail_helper, inp, kwargs

    warn_passing_mapping = (
        ({'a': 1, 'b': 2}, {'a': 1}, {'alias_mapping': {'a': ['b']}}, 1),
        ({'a': 1, 'b': 2}, {'a': 1}, {'alias_mapping': {'a': ['b']},
                                      'allowed': ('a',)}, 1),
        ({'a': 1, 'b': 2}, {'a': 2}, {'alias_mapping': {'a': ['a', 'b']}}, 1),

        ({'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'c': 3},
         {'alias_mapping': {'a': ['b']}, 'required': ('a', )}, 1),

    )

    for inp, exp, kwargs, wc in warn_passing_mapping:
        yield _kwarg_norm_helper, inp, exp, kwargs, wc

    pass_mapping = (
        ({'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {}),
        ({'b': 2}, {'a': 2}, {'alias_mapping': {'a': ['a', 'b']}}),
        ({'b': 2}, {'a': 2}, {'alias_mapping': {'a': ['b']},
                              'forbidden': ('b', )}),

        ({'a': 1, 'c': 3}, {'a': 1, 'c': 3}, {'required': ('a', ),
                                              'allowed': ('c', )}),

        ({'a': 1, 'c': 3}, {'a': 1, 'c': 3}, {'required': ('a', 'c'),
                                              'allowed': ('c', )}),
        ({'a': 1, 'c': 3}, {'a': 1, 'c': 3}, {'required': ('a', 'c'),
                                              'allowed': ('a', 'c')}),
        ({'a': 1, 'c': 3}, {'a': 1, 'c': 3}, {'required': ('a', 'c'),
                                              'allowed': ()}),

        ({'a': 1, 'c': 3}, {'a': 1, 'c': 3}, {'required': ('a', 'c')}),
        ({'a': 1, 'c': 3}, {'a': 1, 'c': 3}, {'allowed': ('a', 'c')}),

    )

    for inp, exp, kwargs in pass_mapping:
        yield _kwarg_norm_helper, inp, exp, kwargs


def test_to_prestep():
    x = np.arange(4)
    y1 = np.arange(4)
    y2 = np.arange(4)[::-1]

    xs, y1s, y2s = cbook.pts_to_prestep(x, y1, y2)

    x_target = np.asarray([0, 0, 1, 1, 2, 2, 3], dtype='float')
    y1_target = np.asarray([0, 1, 1, 2, 2, 3, 3], dtype='float')
    y2_target = np.asarray([3, 2, 2, 1, 1, 0, 0], dtype='float')

    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)
    assert_array_equal(y2_target, y2s)

    xs, y1s = cbook.pts_to_prestep(x, y1)
    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)


def test_to_poststep():
    x = np.arange(4)
    y1 = np.arange(4)
    y2 = np.arange(4)[::-1]

    xs, y1s, y2s = cbook.pts_to_poststep(x, y1, y2)

    x_target = np.asarray([0, 1, 1, 2, 2, 3, 3], dtype='float')
    y1_target = np.asarray([0, 0, 1, 1, 2, 2, 3], dtype='float')
    y2_target = np.asarray([3, 3, 2, 2, 1, 1, 0], dtype='float')

    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)
    assert_array_equal(y2_target, y2s)

    xs, y1s = cbook.pts_to_poststep(x, y1)
    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)


def test_to_midstep():
    x = np.arange(4)
    y1 = np.arange(4)
    y2 = np.arange(4)[::-1]

    xs, y1s, y2s = cbook.pts_to_midstep(x, y1, y2)

    x_target = np.asarray([0, .5, .5, 1.5, 1.5, 2.5, 2.5, 3], dtype='float')
    y1_target = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype='float')
    y2_target = np.asarray([3, 3, 2, 2, 1, 1, 0, 0], dtype='float')

    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)
    assert_array_equal(y2_target, y2s)

    xs, y1s = cbook.pts_to_midstep(x, y1)
    assert_array_equal(x_target, xs)
    assert_array_equal(y1_target, y1s)


def test_step_fails():
    assert_raises(ValueError, cbook._step_validation,
                  np.arange(12).reshape(3, 4), 'a')
    assert_raises(ValueError, cbook._step_validation,
                  np.arange(12), 'a')
    assert_raises(ValueError, cbook._step_validation,
                  np.arange(12))
    assert_raises(ValueError, cbook._step_validation,
                  np.arange(12), np.arange(3))


def test_grouper():
    class dummy():
        pass
    a, b, c, d, e = objs = [dummy() for j in range(5)]
    g = cbook.Grouper()
    g.join(*objs)
    assert set(list(g)[0]) == set(objs)
    assert set(g.get_siblings(a)) == set(objs)

    for other in objs[1:]:
        assert g.joined(a, other)

    g.remove(a)
    for other in objs[1:]:
        assert not g.joined(a, other)

    for A, B in itertools.product(objs[1:], objs[1:]):
        assert g.joined(A, B)


def test_grouper_private():
    class dummy():
        pass
    objs = [dummy() for j in range(5)]
    g = cbook.Grouper()
    g.join(*objs)
    # reach in and touch the internals !
    mapping = g._mapping

    for o in objs:
        assert ref(o) in mapping

    base_set = mapping[ref(objs[0])]
    for o in objs[1:]:
        assert mapping[ref(o)] is base_set


def test_flatiter():
    x = np.arange(5)
    it = x.flat
    assert 0 == next(it)
    assert 1 == next(it)
    ret = cbook.safe_first_element(it)
    assert ret == 0

    assert 0 == next(it)
    assert 1 == next(it)
