from __future__ import absolute_import

from nose.tools import *
from unittest import TestCase
from matplotlib.mpl_traitlets import Color, HasTraits

class ColorTestCase(TestCase):
    """Tests for the Color traits"""

    def setUp(self):
        self.transparent_values = [None, False, '', 'none']
        self.black_values = ['#000000', (0,0,0,0), 0, 0.0, (.0,.0,.0), (.0,.0,.0,.0)]
        self.colored_values = ['#BE3537', (190,53,55), (0.7451, 0.20784, 0.21569)]
        self.invalid_values = ['áfaef', '#FFF', '#0SX#$S', (0,0,0), (0.45,0.3), (()), {}, True]

    def _evaluate_unvalids(self, a):
        for values in self.invalid_values:
            try:
                a.color = values
            except:
                assert_raises(TypeError)

    def test_noargs(self):
        class A(HasTraits):
            color = Color()
        a = A()
        for values in self.black_values:
            a.color = values
            assert_equal(a.color, (0.0,0.0,0.0,0.0))

        for values in self.colored_values:
            a.color = values
            assert_equal(a.color, (0.7451, 0.20784, 0.21569, 0.0))
        self._evaluate_unvalids(a)
        

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

        self._evaluate_unvalids(a)

    def test_rgb(self):
        class A(HasTraits):
            color = Color(force_rgb=True)

        a = A()

        for values in self.black_values:
            a.color = values
            assert_equal(a.color, (0.0,0.0,0.0))

        for values in self.colored_values:
            a.color = values
            assert_equal(a.color, (0.7451, 0.20784, 0.21569))

        self._evaluate_unvalids(a)

    def test_named(self):
        ncolors = {'hexblue': '#0000FF',
                   'floatbllue': (0.0,0.0,1.0),
                   'intblue' : (0,0,255)}

        class A(HasTraits):
            color = Color()
            color.named_colors = ncolors

        a = A()

        for colorname in ncolors:
            a.color = colorname
            assert_equal(a.color, (0.0,0.0,1.0,0.0))

    def test_alpha(self):
        class A(HasTraits):
            color = Color(default_alpha=0.4)

        a = A()

        assert_equal(a.color, (0.0, 0.0, 0.0, 0.0))

        for values in self.transparent_values:
            a.color = values
            assert_equal(a.color, (0.0,0.0,0.0,1.0))

        for values in self.black_values:
            a.color = values
            if isinstance(values, (tuple,list)) and len(values) == 4:
                assert_equal(a.color, (0.0,0.0,0.0,0.0))
            else:
                assert_equal(a.color, (0.0,0.0,0.0,0.4))

        for values in self.colored_values:
            a.color = values
            assert_equal(a.color, (0.7451, 0.20784, 0.21569, 0.4))

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
