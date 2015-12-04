from __future__ import absolute_import

from nose.tools import *
from unittest import TestCase
try:
    from traitlets import TraitError, HasTraits
except ImportError:
    from IPython.utils.traitlets import TraitError, HasTraits

from matplotlib.traitlets import (Color, exdict, OnGetMixin,
                                  PrivateMethodMixin, Int,
                                  Configurable, observe,
                                  validate, retrieve)


def test_exdict():
    e = exdict()
    assert_equal(e.ex, {})
    e['attr'] = 1
    assert_equal(e.ex, {})
    e['attr'] = 2
    assert_equal(e.ex, {'attr': 1})


def test_getter():

    class gInt(OnGetMixin, Int):
        pass

    class A(PrivateMethodMixin, Configurable):

        attr = gInt(0)

        @retrieve('attr')
        def _attr_getter(self, value, trait):
            return value + 1

    assert_equal(A().attr, 1)


class PrivateMethodTestCase(TestCase):
    """Tests private attribute access, assignment, and callback forcing"""

    def test_private_assignment(self):

        class A(PrivateMethodMixin, Configurable):

            attr = Int(0)
            # callbacks shouldn't be envoked

            @validate('attr')
            def _attr_validate(self, commit):
                # should never be reached
                self.assertTrue(False)

            @observe('attr')
            def _attr_changed(self, change):
                # should never be reached
                self.assertTrue(False)

        a = A()
        a.private('attr', 1)
        self.assertEqual(a.attr, 1)

    def test_private_access(self):

        class gInt(OnGetMixin, Int):
            pass

        class A(PrivateMethodMixin, Configurable):

            attr = gInt(0)

            @retrieve('attr')
            def _attr_getter(self, value, trait):
                return value + 1

        self.assertEqual(A().private('attr'), 0)

    def test_callback_forcing(self):

        class A(PrivateMethodMixin, Configurable):

            attr = Int(1)

            @validate('attr')
            def _attr_validate(self, commit):
                return proposal['value']+1

            @observe('attr')
            def _attr_changed(self, change):
                # `private` avoids infinite recursion
                new = change['old']+change['new']
                self.private(change['name'], new)

        a = A()
        a.private('attr', 2)
        self.assertEqual(a.attr, 2)
        a.force_callbacks('attr')
        self.assertEqual(a.attr, 4)


class ColorTestCase(TestCase):
    """Tests for the Color traits"""

    def setUp(self):
        self.transparent_values = [None, False, '', 'none']
        self.black_values = ['#000000', '#000', (0, 0, 0, 255),
                             0, 0.0, (.0, .0, .0), (.0, .0, .0, 1.0)]
        self.colored_values = ['#BE3537', (190, 53, 55),
                               (0.7451, 0.20784, 0.21569)]
        self.invalid_values = ['wfaef', '#0SX#$S', (0.45, 0.3),
                               3.4, 344, (()), {}, True]

    def _evaluate_invalids(self, a):
        for values in self.invalid_values:
            try:
                a.color = values
                assert_true(False)
            except TraitError:
                assert_raises(TraitError)

    def test_noargs(self):

        class A(HasTraits):
            color = Color()

        a = A()
        for values in self.black_values:
            a.color = values
            assert_equal(a.color, (0.0, 0.0, 0.0, 1.0))

        for values in self.colored_values:
            a.color = values
            assert_equal(a.color, (0.7451, 0.20784, 0.21569, 1.0))
        self._evaluate_invalids(a)

    def test_hexcolor(self):

        class A(HasTraits):
            color = Color(as_hex=True)

        a = A()

        for values in self.black_values:
            a.color = values
            assert_equal(a.color, '#000000')

        for values in self.colored_values:
            a.color = values
            assert_equal(a.color, '#be3537')

        self._evaluate_invalids(a)

    def test_rgb(self):

        class A(HasTraits):
            color = Color(force_rgb=True)

        a = A()

        for values in self.black_values:
            a.color = values
            assert_equal(a.color, (0.0, 0.0, 0.0))

        for values in self.colored_values:
            a.color = values
            assert_equal(a.color, (0.7451, 0.20784, 0.21569))

        self._evaluate_invalids(a)

    def test_named(self):
        ncolors = {'hexblue': '#0000FF',
                   'floatbllue': (0.0, 0.0, 1.0),
                   'intblue': (0, 0, 255)}

        class A(HasTraits):
            color = Color()
            color.named_colors = ncolors

        a = A()

        for colorname in ncolors:
            a.color = colorname
            assert_equal(a.color, (0.0, 0.0, 1.0, 1.0))

    def test_alpha(self):

        class A(HasTraits):
            color = Color(default_alpha=0.4)

        a = A()

        assert_equal(a.color, (0.0, 0.0, 0.0, 1.0))

        for values in self.transparent_values:
            a.color = values
            assert_equal(a.color, (0.0, 0.0, 0.0, 0.0))

        for values in self.black_values:
            a.color = values
            if isinstance(values, (tuple, list)) and len(values) == 4:
                assert_equal(a.color, (0.0, 0.0, 0.0, 1.0))
            else:
                # User not provide alpha value so return default_alpha
                assert_equal(a.color, (0.0, 0.0, 0.0, 0.4))

        for values in self.colored_values:
            a.color = values
            assert_equal(a.color, (0.7451, 0.20784, 0.21569, 0.4))

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
