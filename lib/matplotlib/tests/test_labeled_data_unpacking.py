from __future__ import (absolute_import, division, print_function)

from nose.tools import (assert_raises, assert_equal)
from nose.plugins.skip import SkipTest

try:
    # 3.2+ versions
    from nose.tools import assert_regex, assert_not_regex
except ImportError:
    # 2.7 versions
    from nose.tools import assert_regexp_matches, assert_not_regexp_matches
    assert_regex = assert_regexp_matches
    assert_not_regex = assert_not_regexp_matches

from ..testing import assert_produces_warning

from .. import unpack_labeled_data


# Notes on testing the plotting functions itself
# *   the individual decorated plotting functions are tested in 'test_axes.py'
# *   that pyplot functions accept a data kwarg is only tested in
#     test_axes.test_pie_linewidth_0


# these two get used in multiple tests, so define them here
@unpack_labeled_data(replace_names=["x", "y"], label_namer="y")
def plot_func(ax, x, y, ls="x", label=None, w="xyz"):
    return ("x: %s, y: %s, ls: %s, w: %s, label: %s" % (
        list(x), list(y), ls, w, label))


@unpack_labeled_data(replace_names=["x", "y"], label_namer="y",
                     positional_parameter_names=["x", "y", "ls", "label", "w"])
def plot_func_varags(ax, *args, **kwargs):
    all_args = [None, None, "x", None, "xyz"]
    for i, v in enumerate(args):
        all_args[i] = v
    for i, k in enumerate(["x", "y", "ls", "label", "w"]):
        if k in kwargs:
            all_args[i] = kwargs[k]
    x, y, ls, label, w = all_args
    return ("x: %s, y: %s, ls: %s, w: %s, label: %s" % (
        list(x), list(y), ls, w, label))


all_funcs = [plot_func, plot_func_varags]


def test_compiletime_checks():
    """test decorator invocations -> no replacements"""

    def func(ax, x, y): pass

    def func_args(ax, x, y, *args): pass

    def func_kwargs(ax, x, y, **kwargs): pass

    def func_no_ax_args(*args, **kwargs): pass

    # this is ok
    unpack_labeled_data(replace_names=["x", "y"])(func)
    unpack_labeled_data(replace_names=["x", "y"])(func_kwargs)
    # this has "enough" information to do all the replaces
    unpack_labeled_data(replace_names=["x", "y"])(func_args)

    # no positional_parameter_names but needed due to replaces
    def f():
        # z is unknown
        unpack_labeled_data(replace_names=["x", "y", "z"])(func_args)

    assert_raises(AssertionError, f)

    def f():
        unpack_labeled_data(replace_names=["x", "y"])(func_no_ax_args)

    assert_raises(AssertionError, f)

    # no replacements at all -> all ok...
    unpack_labeled_data(replace_names=[], label_namer=None)(func)
    unpack_labeled_data(replace_names=[], label_namer=None)(func_args)
    unpack_labeled_data(replace_names=[], label_namer=None)(func_kwargs)
    unpack_labeled_data(replace_names=[], label_namer=None)(func_no_ax_args)

    # label namer is unknown
    def f():
        unpack_labeled_data(label_namer="z")(func)

    assert_raises(AssertionError, f)

    def f():
        unpack_labeled_data(label_namer="z")(func_args)

    assert_raises(AssertionError, f)
    # but "ok-ish", if func has kwargs -> will show up at runtime :-(
    unpack_labeled_data(label_namer="z")(func_kwargs)
    unpack_labeled_data(label_namer="z")(func_no_ax_args)


def test_label_problems_at_runtime():
    """Tests for behaviour which would actually be nice to get rid of."""

    @unpack_labeled_data(label_namer="z")
    def func(*args, **kwargs):
        pass

    def f():
        func(None, x="a", y="b")

    # This is a programming mistake: the parameter which should add the
    # label is not present in the function call. Unfortunately this was masked
    # due to the **kwargs useage
    # This would be nice to handle as a compiletime check (see above...)
    with assert_produces_warning(RuntimeWarning):
        f()

    def real_func(x, y):
        pass

    @unpack_labeled_data(label_namer="x")
    def func(*args, **kwargs):
        real_func(**kwargs)

    def f():
        func(None, x="a", y="b")

    # This sets a label although the function can't handle it.
    assert_raises(TypeError, f)


def test_function_call_without_data():
    """test without data -> no replacements"""
    for func in all_funcs:
        assert_equal(func(None, "x", "y"),
                     "x: ['x'], y: ['y'], ls: x, w: xyz, label: None")
        assert_equal(func(None, x="x", y="y"),
                     "x: ['x'], y: ['y'], ls: x, w: xyz, label: None")
        assert_equal(func(None, "x", "y", label=""),
                     "x: ['x'], y: ['y'], ls: x, w: xyz, label: ")
        assert_equal(func(None, "x", "y", label="text"),
                     "x: ['x'], y: ['y'], ls: x, w: xyz, label: text")
        assert_equal(func(None, x="x", y="y", label=""),
                     "x: ['x'], y: ['y'], ls: x, w: xyz, label: ")
        assert_equal(func(None, x="x", y="y", label="text"),
                     "x: ['x'], y: ['y'], ls: x, w: xyz, label: text")


def test_function_call_with_dict_data():
    """Test with dict data -> label comes from the value of 'x' parameter """
    data = {"a": [1, 2], "b": [8, 9], "w": "NOT"}
    for func in all_funcs:
        assert_equal(func(None, "a", "b", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
        assert_equal(func(None, x="a", y="b", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
        assert_equal(func(None, "a", "b", label="", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
        assert_equal(func(None, "a", "b", label="text", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")
        assert_equal(func(None, x="a", y="b", label="", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
        assert_equal(func(None, x="a", y="b", label="text", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")


def test_function_call_with_dict_data_not_in_data():
    "test for the case that one var is not in data -> half replaces, half kept"
    data = {"a": [1, 2], "w": "NOT"}
    for func in all_funcs:
        assert_equal(func(None, "a", "b", data=data),
                     "x: [1, 2], y: ['b'], ls: x, w: xyz, label: b")
        assert_equal(func(None, x="a", y="b", data=data),
                     "x: [1, 2], y: ['b'], ls: x, w: xyz, label: b")
        assert_equal(func(None, "a", "b", label="", data=data),
                     "x: [1, 2], y: ['b'], ls: x, w: xyz, label: ")
        assert_equal(func(None, "a", "b", label="text", data=data),
                     "x: [1, 2], y: ['b'], ls: x, w: xyz, label: text")
        assert_equal(func(None, x="a", y="b", label="", data=data),
                     "x: [1, 2], y: ['b'], ls: x, w: xyz, label: ")
        assert_equal(func(None, x="a", y="b", label="text", data=data),
                     "x: [1, 2], y: ['b'], ls: x, w: xyz, label: text")


def test_function_call_with_pandas_data():
    """test with pandas dataframe -> label comes from data["col"].name """
    try:
        import pandas as pd
    except ImportError:
        raise SkipTest("Pandas not installed")

    data = pd.DataFrame({"a": [1, 2], "b": [8, 9], "w": ["NOT", "NOT"]})

    for func in all_funcs:
        assert_equal(func(None, "a", "b", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
        assert_equal(func(None, x="a", y="b", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
        assert_equal(func(None, "a", "b", label="", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
        assert_equal(func(None, "a", "b", label="text", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")
        assert_equal(func(None, x="a", y="b", label="", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
        assert_equal(func(None, x="a", y="b", label="text", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")


def test_function_call_replace_all():
    """Test without a "replace_names" argument, all vars should be replaced"""
    data = {"a": [1, 2], "b": [8, 9], "x": "xyz"}

    @unpack_labeled_data(label_namer="y")
    def func_replace_all(ax, x, y, ls="x", label=None, w="NOT"):
        return "x: %s, y: %s, ls: %s, w: %s, label: %s" % (
            list(x), list(y), ls, w, label)

    assert_equal(func_replace_all(None, "a", "b", w="x", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
    assert_equal(func_replace_all(None, x="a", y="b", w="x", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
    assert_equal(func_replace_all(None, "a", "b", w="x", label="", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert_equal(
        func_replace_all(None, "a", "b", w="x", label="text", data=data),
        "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")
    assert_equal(
        func_replace_all(None, x="a", y="b", w="x", label="", data=data),
        "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert_equal(
        func_replace_all(None, x="a", y="b", w="x", label="text", data=data),
        "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")

    @unpack_labeled_data(label_namer="y")
    def func_varags_replace_all(ax, *args, **kwargs):
        all_args = [None, None, "x", None, "xyz"]
        for i, v in enumerate(args):
            all_args[i] = v
        for i, k in enumerate(["x", "y", "ls", "label", "w"]):
            if k in kwargs:
                all_args[i] = kwargs[k]
        x, y, ls, label, w = all_args
        return "x: %s, y: %s, ls: %s, w: %s, label: %s" % (
            list(x), list(y), ls, w, label)

    # in the first case, we can't get a "y" argument,
    # as we don't know the names of the *args
    assert_equal(func_varags_replace_all(None, x="a", y="b", w="x", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
    assert_equal(
        func_varags_replace_all(None, "a", "b", w="x", label="", data=data),
        "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert_equal(
        func_varags_replace_all(None, "a", "b", w="x", label="text",
                                data=data),
        "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")
    assert_equal(
        func_varags_replace_all(None, x="a", y="b", w="x", label="",
                                data=data),
        "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert_equal(
        func_varags_replace_all(None, x="a", y="b", w="x", label="text",
                                data=data),
        "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")

    with assert_produces_warning():
        assert_equal(func_varags_replace_all(None, "a", "b", w="x", data=data),
                     "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: None")


def test_no_label_replacements():
    """Test with "label_namer=None" -> no label replacement at all"""

    @unpack_labeled_data(replace_names=["x", "y"], label_namer=None)
    def func_no_label(ax, x, y, ls="x", label=None, w="xyz"):
        return "x: %s, y: %s, ls: %s, w: %s, label: %s" % (
            list(x), list(y), ls, w, label)

    data = {"a": [1, 2], "b": [8, 9], "w": "NOT"}
    assert_equal(func_no_label(None, "a", "b", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: None")
    assert_equal(func_no_label(None, x="a", y="b", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: None")
    assert_equal(func_no_label(None, "a", "b", label="", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert_equal(func_no_label(None, "a", "b", label="text", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")


def test_more_args_than_pos_parameter():
    @unpack_labeled_data(replace_names=["x", "y"], label_namer="y")
    def func(ax, x, y, z=1):
        pass

    data = {"a": [1, 2], "b": [8, 9], "w": "NOT"}

    def f():
        func(None, "a", "b", "z", "z", data=data)

    assert_raises(RuntimeError, f)


def test_function_call_with_replace_all_args():
    """Test with a "replace_all_args" argument, all *args should be replaced"""
    data = {"a": [1, 2], "b": [8, 9], "x": "xyz"}

    def funcy(ax, *args, **kwargs):
        all_args = [None, None, "x", None, "NOT"]
        for i, v in enumerate(args):
            all_args[i] = v
        for i, k in enumerate(["x", "y", "ls", "label", "w"]):
            if k in kwargs:
                all_args[i] = kwargs[k]
        x, y, ls, label, w = all_args
        return "x: %s, y: %s, ls: %s, w: %s, label: %s" % (
            list(x), list(y), ls, w, label)

    func = unpack_labeled_data(replace_all_args=True, replace_names=["w"],
                               label_namer="y")(funcy)

    assert_equal(func(None, "a", "b", w="x", label="", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert_equal(func(None, "a", "b", w="x", label="text", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")

    func2 = unpack_labeled_data(replace_all_args=True, replace_names=["w"],
                                label_namer="y",
                                positional_parameter_names=["x", "y", "ls",
                                                            "label", "w"])(
        funcy)

    assert_equal(func2(None, "a", "b", w="x", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b")
    assert_equal(func2(None, "a", "b", w="x", label="", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: ")
    assert_equal(func2(None, "a", "b", w="x", label="text", data=data),
                 "x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text")


def test_docstring_addition():
    @unpack_labeled_data()
    def funcy(ax, *args, **kwargs):
        """Funcy does nothing"""
        pass

    assert_regex(funcy.__doc__,
                          r".*All positional and all keyword arguments\.")
    assert_not_regex(funcy.__doc__, r".*All positional arguments\.")
    assert_not_regex(funcy.__doc__,
                              r".*All arguments with the following names: .*")

    @unpack_labeled_data(replace_all_args=True, replace_names=[])
    def funcy(ax, x, y, z, bar=None):
        """Funcy does nothing"""
        pass

    assert_regex(funcy.__doc__, r".*All positional arguments\.")
    assert_not_regex(funcy.__doc__,
                              r".*All positional and all keyword arguments\.")
    assert_not_regex(funcy.__doc__,
                              r".*All arguments with the following names: .*")

    @unpack_labeled_data(replace_all_args=True, replace_names=["bar"])
    def funcy(ax, x, y, z, bar=None):
        """Funcy does nothing"""
        pass

    assert_regex(funcy.__doc__, r".*All positional arguments\.")
    assert_regex(funcy.__doc__,
                          r".*All arguments with the following names: 'bar'\.")
    assert_not_regex(funcy.__doc__,
                              r".*All positional and all keyword arguments\.")

    @unpack_labeled_data(replace_names=["x", "bar"])
    def funcy(ax, x, y, z, bar=None):
        """Funcy does nothing"""
        pass

    # lists can print in any order, so test for both x,bar and bar,x
    assert_regex(funcy.__doc__,
                    r".*All arguments with the following names: '.*', '.*'\.")
    assert_regex(funcy.__doc__, r".*'x'.*")
    assert_regex(funcy.__doc__, r".*'bar'.*")
    assert_not_regex(funcy.__doc__,
                              r".*All positional and all keyword arguments\.")
    assert_not_regex(funcy.__doc__, r".*All positional arguments\.")


def test_positional_parameter_names_as_function():
    # Also test the _plot_arg_replacer for plot...
    from matplotlib.axes._axes import _plot_args_replacer

    @unpack_labeled_data(replace_names=["x", "y"],
                         positional_parameter_names=_plot_args_replacer)
    def funcy(ax, *args, **kwargs):
        return "{args} | {kwargs}".format(args=args, kwargs=kwargs)

    # the normal case...
    data = {"x": "X", "y1": "Y"}
    assert_equal(funcy(None, "x", "y1", data=data),
                 "('X', 'Y') | {}")
    assert_equal(funcy(None, "x", "y1", "c", data=data),
                 "('X', 'Y', 'c') | {}")

    # no arbitrary long args with data
    def f():
        assert_equal(funcy(None, "x", "y", "c", "x", "y", "x", "y", data=data),
                     "('X', 'Y', 'c', 'X', 'Y', 'X', 'Y') | {}")
    assert_raises(ValueError, f)

    # In the two arg case, if a valid color spec is in data, we warn but use
    # it as data...
    data = {"x": "X", "y": "Y", "ro": "!!"}
    with assert_produces_warning(RuntimeWarning):
        assert_equal(funcy(None, "y", "ro", data=data),
                     "('Y', '!!') | {}")
