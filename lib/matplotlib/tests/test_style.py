from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import shutil
import tempfile
import warnings
from collections import OrderedDict
from contextlib import contextmanager

from nose import SkipTest
from nose.tools import assert_raises, assert_equal
from nose.plugins.attrib import attr

import matplotlib as mpl
from matplotlib import style
from matplotlib.style.core import (USER_LIBRARY_PATHS,
                                   STYLE_EXTENSION,
                                   BASE_LIBRARY_PATH,
                                   flatten_inheritance_dict, get_style_dict)

import six

PARAM = 'image.cmap'
VALUE = 'pink'
DUMMY_SETTINGS = {PARAM: VALUE}


@contextmanager
def temp_style(style_name, settings=None):
    """Context manager to create a style sheet in a temporary directory."""
    if not settings:
        settings = DUMMY_SETTINGS
    temp_file = '%s.%s' % (style_name, STYLE_EXTENSION)

    # Write style settings to file in the temp directory.
    tempdir = tempfile.mkdtemp()
    with open(os.path.join(tempdir, temp_file), 'w') as f:
        for k, v in six.iteritems(settings):
            f.write('%s: %s' % (k, v))

    # Add temp directory to style path and reload so we can access this style.
    USER_LIBRARY_PATHS.append(tempdir)
    style.reload_library()

    try:
        yield
    finally:
        shutil.rmtree(tempdir)
        style.reload_library()


def test_deprecated_rc_warning_includes_filename():
    SETTINGS = {'axes.color_cycle': 'ffffff'}
    basename = 'color_cycle'
    with warnings.catch_warnings(record=True) as warns:
        with temp_style(basename, SETTINGS):
            # style.reload_library() in temp_style() triggers the warning
            pass

    for w in warns:
        assert basename in str(w.message)


def test_available():
    with temp_style('_test_', DUMMY_SETTINGS):
        assert '_test_' in style.available


def test_use():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            assert mpl.rcParams[PARAM] == VALUE


@attr('network')
def test_use_url():
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('https://gist.github.com/adrn/6590261/raw'):
            assert mpl.rcParams['axes.facecolor'] == "#adeade"


def test_context():
    mpl.rcParams[PARAM] = 'gray'
    with temp_style('test', DUMMY_SETTINGS):
        with style.context('test'):
            assert mpl.rcParams[PARAM] == VALUE
    # Check that this value is reset after the exiting the context.
    assert mpl.rcParams[PARAM] == 'gray'


def test_context_with_dict():
    original_value = 'gray'
    other_value = 'blue'
    mpl.rcParams[PARAM] = original_value
    with style.context({PARAM: other_value}):
        assert mpl.rcParams[PARAM] == other_value
    assert mpl.rcParams[PARAM] == original_value


def test_context_with_dict_after_namedstyle():
    # Test dict after style name where dict modifies the same parameter.
    original_value = 'gray'
    other_value = 'blue'
    mpl.rcParams[PARAM] = original_value
    with temp_style('test', DUMMY_SETTINGS):
        with style.context(['test', {PARAM: other_value}]):
            assert mpl.rcParams[PARAM] == other_value
    assert mpl.rcParams[PARAM] == original_value


def test_context_with_dict_before_namedstyle():
    # Test dict before style name where dict modifies the same parameter.
    original_value = 'gray'
    other_value = 'blue'
    mpl.rcParams[PARAM] = original_value
    with temp_style('test', DUMMY_SETTINGS):
        with style.context([{PARAM: other_value}, 'test']):
            assert mpl.rcParams[PARAM] == VALUE
    assert mpl.rcParams[PARAM] == original_value


def test_context_with_union_of_dict_and_namedstyle():
    # Test dict after style name where dict modifies the a different parameter.
    original_value = 'gray'
    other_param = 'text.usetex'
    other_value = True
    d = {other_param: other_value}
    mpl.rcParams[PARAM] = original_value
    mpl.rcParams[other_param] = (not other_value)
    with temp_style('test', DUMMY_SETTINGS):
        with style.context(['test', d]):
            assert mpl.rcParams[PARAM] == VALUE
            assert mpl.rcParams[other_param] == other_value
    assert mpl.rcParams[PARAM] == original_value
    assert mpl.rcParams[other_param] == (not other_value)


def test_context_with_badparam():
    original_value = 'gray'
    other_value = 'blue'
    d = OrderedDict([(PARAM, original_value), ('badparam', None)])
    with style.context({PARAM: other_value}):
        assert mpl.rcParams[PARAM] == other_value
        x = style.context([d])
        assert_raises(KeyError, x.__enter__)
        assert mpl.rcParams[PARAM] == other_value


def test_get_style_dict():
    style_dict = get_style_dict('bmh')
    assert(isinstance(style_dict, dict))


def test_get_style_dict_from_lib():
    style_dict = get_style_dict('bmh')
    assert_equal(style_dict['lines.linewidth'], 2.0)


def test_get_style_dict_from_file():
    style_dict = get_style_dict(os.path.join(BASE_LIBRARY_PATH,
                                             'bmh.mplstyle'))
    assert_equal(style_dict['lines.linewidth'], 2.0)


def test_parent_stylesheet():
    parent_value = 'blue'
    parent = {PARAM: parent_value}
    child = {'style': parent}
    with style.context(child):
        assert_equal(mpl.rcParams[PARAM], parent_value)


def test_parent_stylesheet_children_override():
    parent_value = 'blue'
    child_value = 'gray'
    parent = {PARAM: parent_value}
    child = {'style': parent, PARAM: child_value}
    with style.context(child):
        assert_equal(mpl.rcParams[PARAM], child_value)


def test_grandparent_stylesheet():
    grandparent_value = 'blue'
    grandparent = {PARAM: grandparent_value}
    parent = {'style': grandparent}
    child = {'style': parent}
    with style.context(child):
        assert_equal(mpl.rcParams[PARAM], grandparent_value)


def test_parent_stylesheet_from_string():
    parent_param = 'lines.linewidth'
    parent_value = 2.0
    parent = {parent_param: parent_value}
    child = {'style': ['parent']}
    with temp_style('parent', settings=parent):
        with style.context(child):
            assert_equal(mpl.rcParams[parent_param], parent_value)


def test_parent_stylesheet_brothers():
    parent_param = PARAM
    parent_value1 = 'blue'
    parent_value2 = 'gray'
    parent1 = {parent_param: parent_value1}
    parent2 = {parent_param: parent_value2}
    child = {'style': [parent1, parent2]}
    with style.context(child):
        assert_equal(mpl.rcParams[parent_param], parent_value2)


# Dictionnary flattening function tests
def test_empty_dict():
    child = {}
    flattened = flatten_inheritance_dict(child, 'parents')
    assert_equal(flattened, child)


def test_no_parent():
    child = {'my-key': 'my-value'}
    flattened = flatten_inheritance_dict(child, 'parents')
    assert_equal(flattened, child)
    # Verify that flatten_inheritance_dict always returns a copy.
    assert(flattened is not child)


def test_non_list_raises():
    child = {'parents': 'parent-value'}
    assert_raises(ValueError, flatten_inheritance_dict, child,
                  'parents')


def test_child_with_no_unique_values():
    parent = {'a': 1}
    child = {'parents': [parent]}
    flattened = flatten_inheritance_dict(child, 'parents')
    assert_equal(flattened, parent)


def test_child_overrides_parent_value():
    parent = {'a': 'old-value'}
    child = {'parents': [parent], 'a': 'new-value'}
    flattened = flatten_inheritance_dict(child, 'parents')
    assert_equal(flattened, {'a': 'new-value'})


def test_parents_with_distinct_values():
    child = {'parents': [{'a': 1}, {'b': 2}]}
    flattened = flatten_inheritance_dict(child, 'parents')
    assert_equal(flattened, {'a': 1, 'b': 2})


def test_later_parent_overrides_former():
    child = {'parents': [{'a': 1}, {'a': 2}]}
    flattened = flatten_inheritance_dict(child, 'parents')
    assert_equal(flattened, {'a': 2})


def test_grandparent():
    grandparent = {'a': 1}
    parent = {'parents': [grandparent]}
    child = {'parents': [parent]}
    flattened = flatten_inheritance_dict(child, 'parents')
    assert_equal(flattened, grandparent)


def test_custom_expand_parent():
    parent_map = {'a-pointer': {'a': 1}, 'b-pointer': {'b': 2}}

    def expand_parent(key):
        return parent_map[key]

    child = {'parents': ['a-pointer', 'b-pointer']}
    flattened = flatten_inheritance_dict(child, 'parents',
                                         expand_parent=expand_parent)
    assert_equal(flattened, {'a': 1, 'b': 2})


def test_circular_parents():
    parent_map = {'a-pointer': {'parents': ['b-pointer']},
                  'b-pointer': {'parents': ['a-pointer']}}

    def expand_parent(key):
        return parent_map[key]

    child = {'parents': ['a-pointer']}
    assert_raises(RuntimeError, flatten_inheritance_dict, child,
                 'parents', expand_parent=expand_parent)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
